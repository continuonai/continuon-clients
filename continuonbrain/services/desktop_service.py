"""
Desktop Service: Provides screen interaction capabilities.
"""
import os
import time
from pathlib import Path
from typing import Dict, Tuple

try:
    import pyautogui
    import mss
    # Safe defaults
    pyautogui.FAILSAFE = True
except ImportError:
    pyautogui = None
    mss = None

class DesktopService:
    def __init__(self, storage_dir: str = "/tmp/continuonbrain_demo"):
        self.screenshot_dir = Path(storage_dir) / "ui" / "screenshots"
        self.screenshot_dir.mkdir(parents=True, exist_ok=True)
        self.enabled = pyautogui is not None

    def get_status(self) -> Dict:
        if not self.enabled:
            return {"enabled": False, "error": "Dependencies missing (pyautogui/mss)"}
        
        try:
            size = pyautogui.size()
            pos = pyautogui.position()
            return {
                "enabled": True,
                "screen_width": size.width,
                "screen_height": size.height,
                "mouse_x": pos.x,
                "mouse_y": pos.y
            }
        except Exception as e:
            return {"enabled": False, "error": str(e)}

    def take_screenshot(self, filename: str = "latest.png") -> str:
        """Capture screen and save to disk. Returns absolute path."""
        if not self.enabled:
            return ""
        
        filepath = self.screenshot_dir / filename
        try:
            with mss.mss() as sct:
                # Capture primary monitor
                sct.shot(mon=-1, output=str(filepath))
            return str(filepath)
        except Exception as e:
            print(f"Screenshot failed: {e}")
            return ""

    def move_mouse(self, x: int, y: int):
        if self.enabled:
            pyautogui.moveTo(x, y, duration=0.5)

    def click(self, x: int = None, y: int = None):
        if self.enabled:
            pyautogui.click(x, y)

    def type_text(self, text: str):
        if self.enabled:
            pyautogui.write(text, interval=0.1)

    def press_key(self, key: str):
        if self.enabled:
            pyautogui.press(key)
