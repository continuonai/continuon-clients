
import sys
import unittest
import json
from pathlib import Path
from unittest.mock import MagicMock, patch

# Ensure repo root on path
sys.path.insert(0, str(Path(__file__).resolve().parent))

from continuonbrain.resource_monitor import ResourceMonitor, ResourceStatus, ResourceLevel

class TestMonitoring(unittest.TestCase):
    def setUp(self):
        self.tmp_dir = Path("/tmp/test_monitoring")
        self.tmp_dir.mkdir(parents=True, exist_ok=True)
        self.ckpt_dir = self.tmp_dir / "checkpoints" / "autonomous"
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        
    def test_disk_usage(self):
        # Create dummy file
        with open(self.ckpt_dir / "test.ckpt", "wb") as f:
            f.write(b"0" * 1024 * 1024) # 1 MB
            
        monitor = ResourceMonitor(config_dir=self.tmp_dir)
        
        # Test get_directory_size directly
        size_bytes = monitor.get_directory_size(self.ckpt_dir)
        self.assertEqual(size_bytes, 1024 * 1024)
        
        # Test get_status_summary
        summary = monitor.get_status_summary()
        self.assertIn("checkpoints_mb", summary)
        self.assertEqual(summary["checkpoints_mb"], 1.0)
        
        print("\n✅ ResourceMonitor.get_directory_size and summary OK")

if __name__ == "__main__":
    unittest.main()
