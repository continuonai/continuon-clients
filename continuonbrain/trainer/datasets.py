import json
import torch
from torch.utils.data import Dataset
from pathlib import Path
import numpy as np

class RLDSEpisodeDataset(Dataset):
    def __init__(self, episode_dir: str):
        self.episode_dir = Path(episode_dir)
        self.steps_path = self.episode_dir / "steps" / "000000.jsonl"
        self.metadata_path = self.episode_dir / "metadata.json"
        
        if not self.steps_path.exists():
            raise FileNotFoundError(f"Steps file not found: {self.steps_path}")
            
        self.steps = []
        with open(self.steps_path, 'r') as f:
            for line in f:
                if line.strip():
                    self.steps.append(json.loads(line))
        
        if self.metadata_path.exists():
            with open(self.metadata_path, 'r') as f:
                self.metadata = json.load(f)
        else:
            self.metadata = {}
            
    def __len__(self):
        return len(self.steps)

    def __getitem__(self, idx):
        return self.steps[idx]

class FastWindowDataset(Dataset):
    def __init__(self, root_dir: str, window_size: int = 5):
        """
        Dataset for Fast Loop training (short windows).
        Args:
            root_dir: Directory containing episode directories.
            window_size: Number of steps in the window (e.g., 5 steps @ 20ms = 100ms).
        """
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.episodes = []
        self.indices = [] # (episode_idx, start_step_idx)
        
        if not self.root_dir.exists():
             print(f"Warning: Root dir {root_dir} does not exist.")
             return

        # Load all episodes
        for ep_dir in self.root_dir.iterdir():
            if ep_dir.is_dir() and (ep_dir / "steps" / "000000.jsonl").exists():
                 try:
                     dataset = RLDSEpisodeDataset(str(ep_dir))
                     if len(dataset) >= window_size:
                         self.episodes.append(dataset)
                         for i in range(len(dataset) - window_size + 1):
                             self.indices.append((len(self.episodes) - 1, i))
                 except Exception as e:
                     print(f"Failed to load episode {ep_dir}: {e}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start_step = self.indices[idx]
        episode = self.episodes[ep_idx]
        window = [episode[i] for i in range(start_step, start_step + self.window_size)]
        
        # Helper to extract relevant fields for Fast Loop (e.g., proprioception, action)
        # This can be expanded later.
        return window

class MidTrajectoryDataset(Dataset):
    def __init__(self, root_dir: str, window_size: int = 50, stride: int = 10):
        """
        Dataset for Mid Loop training (longer trajectories).
        Args:
            root_dir: Directory containing episode directories.
            window_size: Number of steps (e.g., 50 steps @ 20ms = 1s).
            stride: Stride for sampling windows to reduce overlap/redundancy.
        """
        self.root_dir = Path(root_dir)
        self.window_size = window_size
        self.episodes = []
        self.indices = []
        
        if not self.root_dir.exists():
             print(f"Warning: Root dir {root_dir} does not exist.")
             return
        
        for ep_dir in self.root_dir.iterdir():
             if ep_dir.is_dir() and (ep_dir / "steps" / "000000.jsonl").exists():
                 try:
                     dataset = RLDSEpisodeDataset(str(ep_dir))
                     if len(dataset) >= window_size:
                         self.episodes.append(dataset)
                         for i in range(0, len(dataset) - window_size + 1, stride):
                             self.indices.append((len(self.episodes) - 1, i))
                 except Exception as e:
                     print(f"Failed to load episode {ep_dir}: {e}")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        ep_idx, start_step = self.indices[idx]
        episode = self.episodes[ep_idx]
        window = [episode[i] for i in range(start_step, start_step + self.window_size)]
        return window
