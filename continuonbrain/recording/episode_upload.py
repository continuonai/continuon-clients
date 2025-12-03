"""
Episode upload pipeline for ContinuonBrain → Cloud.
Prepares RLDS episodes for upload with provenance and validation.
"""
import json
import hashlib
import tarfile
import shutil
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
import time


@dataclass
class EpisodeManifest:
    """Manifest for episode upload with provenance."""
    episode_id: str
    total_steps: int
    robot_type: str
    xr_mode: str
    action_source: str
    created_timestamp_ns: int
    file_checksums: Dict[str, str]  # filename -> SHA256
    metadata_checksum: str  # SHA256 of episode.json
    total_size_bytes: int
    version: str = "1.0.0"
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class EpisodeUploadPipeline:
    """
    Prepares RLDS episodes for cloud upload with safety checks.
    Follows continuon-lifecycle-plan.md gating requirements.
    """
    
    def __init__(self, staging_dir: str = "/tmp/continuonbrain_staging"):
        self.staging_dir = Path(staging_dir)
        self.staging_dir.mkdir(parents=True, exist_ok=True)
        
    def _compute_file_checksum(self, file_path: Path) -> str:
        """Compute SHA256 checksum of a file."""
        sha256 = hashlib.sha256()
        with open(file_path, 'rb') as f:
            for chunk in iter(lambda: f.read(8192), b''):
                sha256.update(chunk)
        return sha256.hexdigest()
    
    def validate_episode(self, episode_path: Path) -> bool:
        """
        Validate episode structure and metadata.

        Required files:
        - episode.json (metadata)
        - step_*_rgb.npy (RGB frames)
        - step_*_depth.npy (depth frames)
        """
        if not episode_path.exists():
            print(f"ERROR: Episode path does not exist: {episode_path}")
            return False
        
        # Check required files
        metadata_file = episode_path / "episode.json"
        if not metadata_file.exists():
            print(f"ERROR: Missing episode.json in {episode_path}")
            return False
        
        # Load and validate metadata
        try:
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            required_fields = ['metadata', 'steps']
            for field in required_fields:
                if field not in metadata:
                    print(f"ERROR: Missing required field '{field}' in episode.json")
                    return False

            episode_metadata = metadata['metadata']
            required_metadata = ['episode_id', 'robot_type', 'xr_mode', 'action_source', 'total_steps']
            for field in required_metadata:
                if field not in episode_metadata:
                    print(f"ERROR: Missing required metadata field '{field}'")
                    return False

            steps = metadata.get('steps', [])
            if len(steps) != episode_metadata['total_steps']:
                print(
                    f"ERROR: Step count mismatch metadata={episode_metadata['total_steps']}"
                    f" actual={len(steps)}"
                )
                return False

            # Validate per-step alignment within 5ms tolerance
            for idx, step in enumerate(steps):
                obs = step.get('observation', {})
                video_frame_id = obs.get('video_frame_id')
                depth_frame_id = obs.get('depth_frame_id')
                frame_timestamp = obs.get('frame_timestamp_ns')

                if not video_frame_id or not depth_frame_id:
                    print(f"ERROR: Missing frame_id for step {idx}")
                    return False

                if video_frame_id != depth_frame_id:
                    print(f"ERROR: Frame ID mismatch at step {idx}")
                    return False

                robot_state = obs.get('robot_state', {})
                robot_frame_id = robot_state.get('frame_id')
                robot_timestamp = robot_state.get('timestamp_nanos')

                if robot_frame_id and robot_frame_id != video_frame_id:
                    print(f"ERROR: Robot state frame_id mismatch at step {idx}")
                    return False

                if frame_timestamp and robot_timestamp:
                    skew_ns = abs(frame_timestamp - robot_timestamp)
                    if skew_ns > 5_000_000:
                        print(
                            f"ERROR: Timestamp skew {skew_ns / 1_000_000:.2f}ms"
                            f" exceeds 5ms at step {idx}"
                        )
                        return False

            # Check step count matches files
            total_steps = episode_metadata['total_steps']
            rgb_files = list(episode_path.glob("step_*_rgb.npy"))
            depth_files = list(episode_path.glob("step_*_depth.npy"))

            if len(rgb_files) != total_steps:
                print(f"WARNING: Expected {total_steps} RGB files, found {len(rgb_files)}")
            
            if len(depth_files) != total_steps:
                print(f"WARNING: Expected {total_steps} depth files, found {len(depth_files)}")
            
            print(f"✅ Episode validation passed: {episode_metadata['episode_id']}")
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to validate episode: {e}")
            return False
    
    def create_upload_package(
        self,
        episode_path: Path,
        include_images: bool = True,
    ) -> Optional[Path]:
        """
        Create upload package (tar.gz) with manifest and checksums.
        
        Args:
            episode_path: Path to episode directory
            include_images: Whether to include RGB/depth images (can be large)
        
        Returns:
            Path to created tar.gz file, or None if failed
        """
        if not self.validate_episode(episode_path):
            return None
        
        try:
            # Load metadata
            metadata_file = episode_path / "episode.json"
            with open(metadata_file) as f:
                metadata = json.load(f)
            
            episode_metadata = metadata['metadata']
            episode_id = episode_metadata['episode_id']
            
            print(f"\nCreating upload package for: {episode_id}")
            
            # Create staging directory for this episode
            stage_dir = self.staging_dir / episode_id
            if stage_dir.exists():
                shutil.rmtree(stage_dir)
            stage_dir.mkdir(parents=True)
            
            # Copy metadata
            shutil.copy(metadata_file, stage_dir / "episode.json")
            
            # Compute checksums
            file_checksums = {}
            total_size = 0
            
            # Always include metadata
            metadata_checksum = self._compute_file_checksum(stage_dir / "episode.json")
            file_checksums["episode.json"] = metadata_checksum
            total_size += (stage_dir / "episode.json").stat().st_size
            
            # Optionally include images
            if include_images:
                print("  Including RGB and depth images...")
                for file_path in episode_path.glob("step_*.npy"):
                    dest_path = stage_dir / file_path.name
                    shutil.copy(file_path, dest_path)
                    
                    checksum = self._compute_file_checksum(dest_path)
                    file_checksums[file_path.name] = checksum
                    total_size += dest_path.stat().st_size
                
                print(f"  Total files: {len(file_checksums)}")
            else:
                print("  Skipping images (metadata only)")
            
            # Create manifest
            manifest = EpisodeManifest(
                episode_id=episode_id,
                total_steps=episode_metadata['total_steps'],
                robot_type=episode_metadata['robot_type'],
                xr_mode=episode_metadata['xr_mode'],
                action_source=episode_metadata['action_source'],
                created_timestamp_ns=episode_metadata.get('start_timestamp_ns', int(time.time_ns())),
                file_checksums=file_checksums,
                metadata_checksum=metadata_checksum,
                total_size_bytes=total_size,
            )
            
            # Save manifest
            manifest_file = stage_dir / "upload_manifest.json"
            with open(manifest_file, 'w') as f:
                json.dump(manifest.to_dict(), f, indent=2)
            
            print(f"  Total size: {total_size / 1024 / 1024:.1f} MB")
            
            # Create tar.gz
            tar_path = self.staging_dir / f"{episode_id}.tar.gz"
            with tarfile.open(tar_path, 'w:gz') as tar:
                tar.add(stage_dir, arcname=episode_id)
            
            tar_size = tar_path.stat().st_size
            compression_ratio = (1 - tar_size / total_size) * 100 if total_size > 0 else 0
            
            print(f"  Compressed: {tar_size / 1024 / 1024:.1f} MB ({compression_ratio:.1f}% reduction)")
            print(f"✅ Upload package created: {tar_path}")
            
            return tar_path
            
        except Exception as e:
            print(f"ERROR: Failed to create upload package: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def prepare_batch_upload(
        self,
        episodes_dir: Path,
        output_dir: Optional[Path] = None,
        max_episodes: int = 100,
    ) -> List[Path]:
        """
        Prepare multiple episodes for batch upload.
        
        Args:
            episodes_dir: Directory containing episode subdirectories
            output_dir: Where to save tar.gz files (default: staging_dir)
            max_episodes: Maximum number of episodes to process
        
        Returns:
            List of paths to created tar.gz files
        """
        if output_dir is None:
            output_dir = self.staging_dir
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Find all episode directories
        episode_dirs = [d for d in episodes_dir.iterdir() if d.is_dir()]
        episode_dirs = episode_dirs[:max_episodes]
        
        print(f"\nPreparing batch upload for {len(episode_dirs)} episodes...")
        print(f"Source: {episodes_dir}")
        print(f"Output: {output_dir}\n")
        
        packages = []
        for i, episode_dir in enumerate(episode_dirs, 1):
            print(f"[{i}/{len(episode_dirs)}] Processing {episode_dir.name}...")
            
            package_path = self.create_upload_package(episode_dir)
            if package_path:
                packages.append(package_path)
                print()
        
        print(f"\n{'='*60}")
        print(f"Batch upload preparation complete")
        print(f"  Successful: {len(packages)}/{len(episode_dirs)}")
        print(f"  Total size: {sum(p.stat().st_size for p in packages) / 1024 / 1024:.1f} MB")
        print(f"{'='*60}\n")
        
        return packages
    
    def verify_package(self, package_path: Path) -> bool:
        """
        Verify upload package integrity.
        
        Checks:
        - Tar.gz is valid
        - Manifest present
        - File checksums match
        """
        try:
            print(f"Verifying package: {package_path}")
            
            # Extract to temp
            temp_dir = self.staging_dir / "verify_temp"
            if temp_dir.exists():
                shutil.rmtree(temp_dir)
            temp_dir.mkdir()
            
            with tarfile.open(package_path, 'r:gz') as tar:
                tar.extractall(temp_dir)
            
            # Find episode directory (should be only one)
            episode_dirs = [d for d in temp_dir.iterdir() if d.is_dir()]
            if len(episode_dirs) != 1:
                print(f"ERROR: Expected 1 episode directory, found {len(episode_dirs)}")
                return False
            
            episode_dir = episode_dirs[0]
            
            # Load manifest
            manifest_file = episode_dir / "upload_manifest.json"
            if not manifest_file.exists():
                print("ERROR: Missing upload_manifest.json")
                return False
            
            with open(manifest_file) as f:
                manifest = json.load(f)
            
            # Verify checksums
            file_checksums = manifest['file_checksums']
            for filename, expected_checksum in file_checksums.items():
                file_path = episode_dir / filename
                if not file_path.exists():
                    print(f"ERROR: Missing file: {filename}")
                    return False
                
                actual_checksum = self._compute_file_checksum(file_path)
                if actual_checksum != expected_checksum:
                    print(f"ERROR: Checksum mismatch for {filename}")
                    print(f"  Expected: {expected_checksum}")
                    print(f"  Actual:   {actual_checksum}")
                    return False
            
            # Clean up
            shutil.rmtree(temp_dir)
            
            print(f"✅ Package verification passed")
            return True
            
        except Exception as e:
            print(f"ERROR: Verification failed: {e}")
            return False


def test_upload_pipeline():
    """Test the upload pipeline with recorded episodes."""
    pipeline = EpisodeUploadPipeline(staging_dir="/tmp/upload_test")
    
    # Use test episodes from integration test
    test_episodes = Path("/tmp/continuonbrain_test")
    
    if not test_episodes.exists():
        print("No test episodes found. Run integration_test.py first.")
        return
    
    # Prepare batch upload
    packages = pipeline.prepare_batch_upload(
        episodes_dir=test_episodes,
        max_episodes=5,
    )
    
    # Verify packages
    print("\nVerifying packages...")
    for package in packages:
        pipeline.verify_package(package)
    
    print("\n✅ Upload pipeline test complete")


if __name__ == "__main__":
    test_upload_pipeline()
