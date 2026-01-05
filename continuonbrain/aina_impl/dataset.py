"""
AINA Dataset Loader

Converts RLDS episodes to AINA training format for policy learning.
Adapted for SO-ARM101 (6-DOF arm with gripper) from human demonstration data.

Episode data flow:
    RLDS Episode → AINADataset → DataLoader → AINAPolicy Training

Input (from episodes):
    - rgb_image: (H, W, 3) uint8
    - depth_image: (H, W) uint16 mm
    - robot_state: [6] joint angles
    - pose_keypoints: [N_people, 17, 3] keypoints (if recorded)

Output (for training):
    - ee_trajectory: [T_obs, 3] end-effector position history
    - object_pcd: [T_obs, N_points, 3] object point cloud
    - target_joints: [T_pred, 7] future joint positions + gripper
"""
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import Dataset
    TORCH_AVAILABLE = True
except ImportError:
    Dataset = object  # Fallback base class
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available, AINADataset will be limited")


class AINADataset(Dataset):
    """
    Dataset for AINA policy training from RLDS manipulation episodes.

    Converts SO-ARM101 teleoperation recordings to:
    - End-effector trajectory history (from forward kinematics)
    - Object point clouds (from depth + segmentation)
    - Target joint positions (from demonstration actions)

    For human wrist tracking (when pose data available):
    - Maps human wrist position to robot end-effector target
    - Uses wrist trajectory as the "fingertip" equivalent
    """

    def __init__(
        self,
        episodes_dir: str,
        obs_horizon: int = 10,
        pred_horizon: int = 5,
        n_obj_points: int = 100,
        max_episodes: Optional[int] = None,
        use_pose_for_ee: bool = True,
    ):
        """
        Initialize AINA dataset.

        Args:
            episodes_dir: Directory containing RLDS episode folders
            obs_horizon: Number of observation timesteps (T_obs)
            pred_horizon: Number of prediction timesteps (T_pred)
            n_obj_points: Number of object points to sample
            max_episodes: Maximum episodes to load (for debugging)
            use_pose_for_ee: Use human wrist position as EE target if available
        """
        self.episodes_dir = Path(episodes_dir)
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        self.n_obj_points = n_obj_points
        self.max_episodes = max_episodes
        self.use_pose_for_ee = use_pose_for_ee

        # Load episode metadata
        self.episodes: List[Dict[str, Any]] = []
        self.samples: List[Tuple[int, int]] = []  # (episode_idx, start_step)

        self._load_episodes()
        self._build_sample_index()

        logger.info(
            f"AINADataset: {len(self.episodes)} episodes, "
            f"{len(self.samples)} samples, "
            f"obs_horizon={obs_horizon}, pred_horizon={pred_horizon}"
        )

    def _load_episodes(self) -> None:
        """Load all episode metadata from disk."""
        if not self.episodes_dir.exists():
            logger.warning(f"Episodes directory not found: {self.episodes_dir}")
            return

        episode_dirs = sorted(self.episodes_dir.iterdir())
        loaded = 0

        for ep_dir in episode_dirs:
            if not ep_dir.is_dir():
                continue

            metadata_file = ep_dir / "episode.json"
            if not metadata_file.exists():
                continue

            try:
                with open(metadata_file) as f:
                    data = json.load(f)

                self.episodes.append({
                    "path": ep_dir,
                    "metadata": data.get("metadata", {}),
                    "steps": data.get("steps", []),
                    "num_steps": len(data.get("steps", [])),
                })
                loaded += 1

                if self.max_episodes and loaded >= self.max_episodes:
                    break

            except Exception as e:
                logger.warning(f"Failed to load episode {ep_dir}: {e}")

        logger.info(f"Loaded {loaded} episodes from {self.episodes_dir}")

    def _build_sample_index(self) -> None:
        """Build index of valid training samples."""
        min_length = self.obs_horizon + self.pred_horizon

        for ep_idx, episode in enumerate(self.episodes):
            num_steps = episode["num_steps"]
            if num_steps < min_length:
                continue

            # Create sliding window samples
            for start_step in range(num_steps - min_length + 1):
                self.samples.append((ep_idx, start_step))

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get a training sample.

        Returns dict with:
            ee_trajectory: [T_obs, 3] end-effector history
            object_pcd: [T_obs, N_points, 3] object point cloud
            target_joints: [T_pred, 7] future joint + gripper targets
            robot_state_hist: [T_obs, 6] joint angle history
        """
        ep_idx, start_step = self.samples[idx]
        episode = self.episodes[ep_idx]

        # Load observation window
        ee_traj = []
        obj_pcds = []
        robot_states = []

        for t in range(self.obs_horizon):
            step_idx = start_step + t
            step_data = self._load_step(episode, step_idx)

            # Get end-effector position
            ee_pos = self._get_ee_position(step_data)
            ee_traj.append(ee_pos)

            # Get object point cloud
            obj_pcd = self._get_object_points(step_data)
            obj_pcds.append(obj_pcd)

            # Get robot state
            robot_state = step_data.get("robot_state", [0.0] * 6)
            robot_states.append(robot_state)

        # Load prediction targets
        target_joints = []
        for t in range(self.pred_horizon):
            step_idx = start_step + self.obs_horizon + t
            step_data = self._load_step(episode, step_idx)

            # Get target joint positions from action
            action = step_data.get("action", {}).get("command", [0.0] * 6)
            # Add gripper state (assume last action element or separate)
            gripper = 0.0  # Default closed
            target = list(action) + [gripper]
            target_joints.append(target)

        # Convert to tensors if torch available
        if TORCH_AVAILABLE:
            ee_traj = torch.tensor(np.array(ee_traj), dtype=torch.float32)
            obj_pcd = torch.tensor(np.array(obj_pcds), dtype=torch.float32)
            target_joints = torch.tensor(np.array(target_joints), dtype=torch.float32)
            robot_states = torch.tensor(np.array(robot_states), dtype=torch.float32)
        else:
            ee_traj = np.array(ee_traj)
            obj_pcd = np.array(obj_pcds)
            target_joints = np.array(target_joints)
            robot_states = np.array(robot_states)

        return {
            "ee_trajectory": ee_traj,       # [T_obs, 3]
            "object_pcd": obj_pcd,          # [T_obs, N_points, 3]
            "target_joints": target_joints, # [T_pred, 7]
            "robot_state_hist": robot_states,  # [T_obs, 6]
        }

    def _load_step(self, episode: Dict, step_idx: int) -> Dict[str, Any]:
        """Load step data including images if needed."""
        ep_path = episode["path"]
        step_meta = episode["steps"][step_idx]

        # Load images if needed
        rgb_path = ep_path / f"step_{step_idx:04d}_rgb.npy"
        depth_path = ep_path / f"step_{step_idx:04d}_depth.npy"

        result = {**step_meta}

        if rgb_path.exists():
            result["rgb_image"] = np.load(rgb_path)
        if depth_path.exists():
            result["depth_image"] = np.load(depth_path)

        return result

    def _get_ee_position(self, step_data: Dict) -> np.ndarray:
        """
        Get end-effector position from step data.

        Priority:
        1. Human wrist keypoint (if pose data available and use_pose_for_ee)
        2. Forward kinematics from robot state
        3. Fallback to zeros
        """
        # Try pose keypoints first
        if self.use_pose_for_ee:
            pose_data = step_data.get("observation", {}).get("pose_keypoints", [])
            if pose_data and len(pose_data) > 0:
                # Use first person's dominant wrist
                person = pose_data[0]
                keypoints = person.get("keypoints", [])
                for kp in keypoints:
                    if kp.get("name") in ("right_wrist", "left_wrist"):
                        if kp.get("conf", 0) > 0.3:
                            # Normalize to [-1, 1] range based on image size
                            # Assume 640x480 image
                            x = (kp["x"] / 320.0) - 1.0
                            y = (kp["y"] / 240.0) - 1.0
                            z = 0.5  # Estimated depth from wrist
                            return np.array([x, y, z], dtype=np.float32)

        # Fall back to forward kinematics (simplified)
        robot_state = step_data.get("observation", {}).get("robot_state", {})
        joint_positions = robot_state.get("joint_positions", [0.0] * 6)

        # Simplified FK for 6-DOF arm (would need actual kinematics)
        ee_pos = self._simple_fk(joint_positions)
        return ee_pos

    def _simple_fk(self, joint_angles: List[float]) -> np.ndarray:
        """
        Simplified forward kinematics for SO-ARM101.

        This is a placeholder - real FK would use DH parameters.
        Returns approximate EE position in normalized space.
        """
        if len(joint_angles) < 6:
            joint_angles = joint_angles + [0.0] * (6 - len(joint_angles))

        # Very simplified: map joint angles to approximate 3D position
        # Real implementation would use proper kinematics
        j1, j2, j3, j4, j5, j6 = joint_angles[:6]

        # Approximate reach based on joint angles
        x = 0.3 * np.sin(j1) * np.cos(j2)
        y = 0.3 * np.cos(j1) * np.cos(j2)
        z = 0.2 + 0.2 * np.sin(j2) + 0.1 * np.sin(j3)

        return np.array([x, y, z], dtype=np.float32)

    def _get_object_points(self, step_data: Dict) -> np.ndarray:
        """
        Extract object point cloud from depth image.

        Uses depth image to create a point cloud, optionally filtered
        by segmentation mask if available.
        """
        depth = step_data.get("depth_image")
        if depth is None:
            return np.zeros((self.n_obj_points, 3), dtype=np.float32)

        # Simple depth-to-points conversion
        h, w = depth.shape
        valid_mask = depth > 0

        # Create coordinate grids
        u, v = np.meshgrid(np.arange(w), np.arange(h))

        # Convert to 3D points (simplified pinhole camera model)
        fx, fy = 500.0, 500.0  # Approximate focal lengths
        cx, cy = w / 2, h / 2

        x = (u - cx) * depth / fx
        y = (v - cy) * depth / fy
        z = depth.astype(np.float32)

        # Stack into points
        points = np.stack([x, y, z], axis=-1)  # [H, W, 3]
        points = points[valid_mask]  # [N_valid, 3]

        # Normalize to unit cube
        if len(points) > 0:
            points = points / 1000.0  # Convert mm to meters
            points = points - points.mean(axis=0)  # Center
            max_dist = np.abs(points).max() + 1e-6
            points = points / max_dist  # Normalize to [-1, 1]

        # Sample fixed number of points
        if len(points) >= self.n_obj_points:
            indices = np.random.choice(len(points), self.n_obj_points, replace=False)
            sampled = points[indices]
        elif len(points) > 0:
            # Pad with repeated samples
            indices = np.random.choice(len(points), self.n_obj_points, replace=True)
            sampled = points[indices]
        else:
            sampled = np.zeros((self.n_obj_points, 3), dtype=np.float32)

        return sampled.astype(np.float32)

    def get_episode_info(self, idx: int) -> Dict[str, Any]:
        """Get episode metadata for a sample."""
        ep_idx, start_step = self.samples[idx]
        episode = self.episodes[ep_idx]
        return {
            "episode_id": episode["metadata"].get("episode_id"),
            "start_step": start_step,
            "language_instruction": episode["metadata"].get("language_instruction"),
            "robot_type": episode["metadata"].get("robot_type"),
        }


def create_dataloader(
    episodes_dir: str,
    batch_size: int = 4,
    obs_horizon: int = 10,
    pred_horizon: int = 5,
    shuffle: bool = True,
    num_workers: int = 0,
) -> Any:
    """
    Create a DataLoader for AINA training.

    Args:
        episodes_dir: Path to RLDS episodes
        batch_size: Training batch size
        obs_horizon: Observation window length
        pred_horizon: Prediction window length
        shuffle: Shuffle samples
        num_workers: DataLoader workers

    Returns:
        torch.utils.data.DataLoader
    """
    if not TORCH_AVAILABLE:
        raise ImportError("PyTorch required for DataLoader")

    from torch.utils.data import DataLoader

    dataset = AINADataset(
        episodes_dir=episodes_dir,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
    )

    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=True,
    )


def test_dataset():
    """Test AINA dataset loading."""
    import tempfile
    import os

    # Create a temporary episode for testing
    with tempfile.TemporaryDirectory() as tmp_dir:
        ep_dir = Path(tmp_dir) / "test_episode_001"
        ep_dir.mkdir()

        # Create mock episode data
        num_steps = 20
        steps = []
        for i in range(num_steps):
            steps.append({
                "step_index": i,
                "timestamp_ns": i * 50_000_000,
                "observation": {
                    "robot_state": {
                        "joint_positions": [0.1 * i] * 6,
                    },
                },
                "action": {
                    "command": [0.1 * (i + 1)] * 6,
                    "source": "human_teleop_xr",
                },
                "is_terminal": i == num_steps - 1,
            })

            # Save mock images
            np.save(ep_dir / f"step_{i:04d}_rgb.npy",
                   np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8))
            np.save(ep_dir / f"step_{i:04d}_depth.npy",
                   np.random.randint(500, 5000, (480, 640), dtype=np.uint16))

        episode_data = {
            "metadata": {
                "episode_id": "test_episode_001",
                "robot_type": "SO-ARM101",
                "total_steps": num_steps,
            },
            "steps": steps,
        }

        with open(ep_dir / "episode.json", "w") as f:
            json.dump(episode_data, f)

        # Test dataset
        dataset = AINADataset(
            episodes_dir=tmp_dir,
            obs_horizon=5,
            pred_horizon=3,
            n_obj_points=50,
        )

        print(f"Dataset length: {len(dataset)}")
        print(f"Episodes loaded: {len(dataset.episodes)}")

        if len(dataset) > 0:
            sample = dataset[0]
            print(f"\nSample shapes:")
            print(f"  ee_trajectory: {sample['ee_trajectory'].shape}")
            print(f"  object_pcd: {sample['object_pcd'].shape}")
            print(f"  target_joints: {sample['target_joints'].shape}")
            print(f"  robot_state_hist: {sample['robot_state_hist'].shape}")

            info = dataset.get_episode_info(0)
            print(f"\nEpisode info: {info}")

        print("\nDataset test complete")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    test_dataset()
