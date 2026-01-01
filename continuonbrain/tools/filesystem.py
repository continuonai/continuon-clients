"""FileSystem tool for environment awareness."""
from __future__ import annotations
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from .base import BaseBrainTool


class FileSystemTool(BaseBrainTool):
    """Read-only access to whitelisted project directories."""

    def __init__(self, whitelist_root: Optional[str] = None):
        super().__init__(
            name="filesystem",
            description="List and read permitted files in the project workspace."
        )
        self.root = Path(whitelist_root or os.getcwd()).resolve()

    def _is_safe(self, path: str) -> bool:
        """Ensure path is within the whitelisted root."""
        target = Path(self.root, path).resolve()
        return self.root in target.parents or target == self.root

    async def execute(self, action: str, path: str = ".", content: Optional[str] = None) -> Any:
        """Execute filesystem actions."""
        if not self._is_safe(path):
            return {"error": "Access denied: Path is outside the permitted workspace.", "path": path}

        target = (self.root / path).resolve()

        if action == "list":
            if not target.exists():
                return {"error": "Path does not exist.", "path": path}
            if not target.is_dir():
                return {"error": "Path is not a directory.", "path": path}
            
            items = []
            for item in target.iterdir():
                items.append({
                    "name": item.name,
                    "type": "dir" if item.is_dir() else "file",
                    "size": item.stat().st_size if item.is_file() else 0
                })
            return {"items": items, "path": path}

        elif action == "read":
            if not target.exists():
                return {"error": "File does not exist.", "path": path}
            if not target.is_file():
                return {"error": "Path is not a file.", "path": path}
            
            # Limit read size
            if target.stat().st_size > 1024 * 100: # 100KB limit
                return {"error": "File too large to read via tool.", "path": path}

            try:
                content = target.read_text(errors="replace")
                return {"content": content, "path": path}
            except Exception as e:
                return {"error": f"Read error: {str(e)}", "path": path}

        elif action == "write":
            if not content:
                return {"error": "Content is required for write action.", "path": path}
            
            try:
                # Ensure parent directory exists
                target.parent.mkdir(parents=True, exist_ok=True)
                target.write_text(content)
                return {"success": True, "path": path, "size": len(content)}
            except Exception as e:
                return {"error": f"Write error: {str(e)}", "path": path}

        return {"error": f"Unknown action: {action}"}

    def _get_params_spec(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "action": {
                    "type": "string",
                    "enum": ["list", "read", "write"],
                    "description": "The action to perform: 'list', 'read', or 'write'."
                },
                "path": {
                    "type": "string",
                    "description": "The relative path within the workspace."
                },
                "content": {
                    "type": "string",
                    "description": "Content to write (required for 'write' action)."
                }
            },
            "required": ["action"]
        }
