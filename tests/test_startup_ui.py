import unittest
from unittest.mock import MagicMock, patch
import json
import shutil
from pathlib import Path
import tempfile
import sys
import os

# Add repo root to path
REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.startup_manager import StartupManager

class TestStartupUI(unittest.TestCase):
    def setUp(self):
        self.test_dir = tempfile.mkdtemp()
        self.config_dir = Path(self.test_dir)
        self.manager = StartupManager(config_dir=str(self.config_dir), start_services=False)
        # Mock discovery service
        self.manager.discovery_service = MagicMock()
        self.manager.discovery_service.get_robot_info.return_value = {'ip_address': '127.0.0.1'}
        
        # Force UI launch in config
        with open(self.config_dir / "ui_config.json", "w") as f:
            json.dump({"auto_launch": True}, f)

    def tearDown(self):
        shutil.rmtree(self.test_dir)

    @patch('webbrowser.open')
    @patch('shutil.which')
    def test_launch_ui_default(self, mock_which, mock_open):
        """Test that UI launches using fallback when no specific browser found."""
        mock_which.return_value = None
        self.manager.launch_ui()
        mock_open.assert_called_once_with("http://127.0.0.1:8080/ui")

    @patch('webbrowser.open')
    @patch('continuonbrain.startup_manager.subprocess.Popen')
    @patch('shutil.which')
    def test_launch_ui_specific_browser(self, mock_which, mock_popen, mock_webbrowser):
        """Test that specific browser is used if found."""
        mock_which.return_value = "/usr/bin/chromium-browser"
        
        self.manager.launch_ui()
        
        mock_which.assert_called()
        mock_popen.assert_called()
        args = mock_popen.call_args[0][0]
        self.assertIn("--password-store=basic", args)
        self.assertIn("--kiosk", args)
        self.assertIn("--noerrdialogs", args)
        self.assertIn("--disable-infobars", args)
        self.assertIn("http://127.0.0.1:8080/ui", args)
        # Verify fallback wasn't called
        mock_webbrowser.assert_not_called()

    @patch('webbrowser.open')
    @patch('continuonbrain.startup_manager.subprocess.Popen')
    @patch('shutil.which')
    def test_launch_ui_fallback(self, mock_which, mock_popen, mock_webbrowser):
        """Test fallback to default browser if no specific one found."""
        mock_which.return_value = None
        
        self.manager.launch_ui()
        
        mock_popen.assert_not_called()
        mock_webbrowser.assert_called_once_with("http://127.0.0.1:8080/ui")

    @patch('webbrowser.open')
    def test_launch_ui_disabled(self, mock_open):
        """Test that UI does NOT launch when disabled."""
        config_path = self.config_dir / "ui_config.json"
        with open(config_path, 'w') as f:
            json.dump({"auto_launch": False}, f)
            
        self.manager.launch_ui()
        mock_open.assert_not_called()

if __name__ == '__main__':
    unittest.main()
