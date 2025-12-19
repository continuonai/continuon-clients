import unittest
import tempfile
import shutil
import json
import os
from pathlib import Path
from continuonbrain.trainer.datasets import FastWindowDataset, MidTrajectoryDataset

class TestDatasets(unittest.TestCase):
    def setUp(self):
        self.test_dir = Path(tempfile.mkdtemp())
        
        # Create a dummy episode
        self.ep_dir = self.test_dir / "ep_test_01"
        self.ep_dir.mkdir()
        (self.ep_dir / "steps").mkdir()
        
        self.episode_len = 20
        steps = []
        for i in range(self.episode_len):
            steps.append({
                "observation": {"val": i},
                "action": {"val": i},
                "is_terminal": i == self.episode_len - 1
            })
            
        with open(self.ep_dir / "steps" / "000000.jsonl", "w") as f:
            for step in steps:
                f.write(json.dumps(step) + "\n")
                
        with open(self.ep_dir / "metadata.json", "w") as f:
            json.dump({"tags": ["test"]}, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    def test_fast_window_slicing(self):
        window_size = 5
        dataset = FastWindowDataset(str(self.test_dir), window_size=window_size)
        
        # Expected windows: 20 - 5 + 1 = 16
        self.assertEqual(len(dataset), 16)
        
        # Check first window
        first_window = dataset[0]
        self.assertEqual(len(first_window), window_size)
        self.assertEqual(first_window[0]["observation"]["val"], 0)
        self.assertEqual(first_window[-1]["observation"]["val"], 4)
        
        # Check last window
        last_window = dataset[-1]
        self.assertEqual(len(last_window), window_size)
        self.assertEqual(last_window[0]["observation"]["val"], 15)
        self.assertEqual(last_window[-1]["observation"]["val"], 19)

    def test_mid_trajectory_slicing(self):
        window_size = 10
        stride = 5
        dataset = MidTrajectoryDataset(str(self.test_dir), window_size=window_size, stride=stride)
        
        # Expected windows:
        # 0: 0-9
        # 5: 5-14
        # 10: 10-19
        # 15: 15-24 (Exceeds length 20)
        # So 3 windows
        self.assertEqual(len(dataset), 3)
        
        # Check first window
        first_window = dataset[0]
        self.assertEqual(len(first_window), window_size)
        self.assertEqual(first_window[0]["observation"]["val"], 0)
        
        # Check second window (stride 5)
        second_window = dataset[1]
        self.assertEqual(second_window[0]["observation"]["val"], 5)

if __name__ == '__main__':
    unittest.main()
