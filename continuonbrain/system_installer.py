"""
System-level package installer helpers for ContinuonBrain.
Supports best-effort detection of the host OS and preferred package manager.
Actual execution is gated by the allow_run flag and the environment variable
CONTINUON_ALLOW_SYSTEM_INSTALL=1 to avoid unintended privileged actions.
"""
from __future__ import annotations

import os
import platform
import shutil
import subprocess
from dataclasses import dataclass
from typing import List, Optional, Tuple, Dict


@dataclass
class InstallPlan:
    manager: Optional[str]
    command: List[str]
    executed: bool
    returncode: Optional[int]
    message: str


class SystemInstaller:
    def __init__(self):
        self.system = platform.system()
        self.manager = self._detect_manager()

    def _detect_manager(self) -> Optional[str]:
        if self.system == "Linux":
            if shutil.which("apt-get"):
                return "apt"
            if shutil.which("dnf"):
                return "dnf"
            if shutil.which("yum"):
                return "yum"
        if self.system == "Darwin":
            if shutil.which("brew"):
                return "brew"
        if self.system == "Windows":
            if shutil.which("choco"):
                return "choco"
            if shutil.which("winget"):
                return "winget"
        return None

    def build_command(self, packages: List[str], use_sudo: bool = True) -> Optional[List[str]]:
        if not packages:
            return None
        if not self.manager:
            return None
        mgr = self.manager
        cmd: List[str]

        if mgr == "apt":
            # Handled in install_packages (update + install)
            return None
        if mgr == "dnf":
            cmd = (["sudo"] if use_sudo else []) + ["dnf", "install", "-y", *packages]
            return cmd
        if mgr == "yum":
            cmd = (["sudo"] if use_sudo else []) + ["yum", "install", "-y", *packages]
            return cmd
        if mgr == "brew":
            cmd = ["brew", "install", *packages]
            return cmd
        if mgr == "choco":
            cmd = ["choco", "install", "-y", *packages]
            return cmd
        if mgr == "winget":
            cmd = ["winget", "install", "--accept-source-agreements", "--accept-package-agreements", *packages]
            return cmd
        return None

    def install_packages(self, packages: List[str], allow_run: bool = False, use_sudo: bool = True) -> List[InstallPlan]:
        """
        Attempt to install system-level packages. Execution is gated by allow_run and
        CONTINUON_ALLOW_SYSTEM_INSTALL.
        Returns a list of InstallPlan entries (even if nothing ran).
        """
        plans: List[InstallPlan] = []
        if not packages:
            return plans

        if not self.manager:
            plans.append(InstallPlan(None, [], False, None, "No supported package manager detected"))
            return plans

        env_allow = os.environ.get("CONTINUON_ALLOW_SYSTEM_INSTALL") == "1"
        effective_run = allow_run and env_allow

        # apt needs update + install; others can be single command
        if self.manager == "apt":
            cmds = [
                ((["sudo"] if use_sudo else []) + ["apt-get", "update"]),
                ((["sudo"] if use_sudo else []) + ["apt-get", "install", "-y", *packages]),
            ]
        else:
            cmd = self.build_command(packages, use_sudo=use_sudo)
            cmds = [cmd] if cmd else []

        if not cmds:
            plans.append(InstallPlan(self.manager, [], False, None, "Unable to build install command"))
            return plans

        for cmd in cmds:
            if cmd is None:
                continue
            if not effective_run:
                plans.append(InstallPlan(self.manager, cmd, False, None, "Pending manual approval (set CONTINUON_ALLOW_SYSTEM_INSTALL=1 and allow_run=True)"))
                continue
            try:
                result = subprocess.run(cmd, check=False)
                plans.append(InstallPlan(self.manager, cmd, True, result.returncode, "ok" if result.returncode == 0 else "failed"))
            except Exception as exc:
                plans.append(InstallPlan(self.manager, cmd, True, None, f"exception: {exc}"))

        return plans


__all__ = ["SystemInstaller", "InstallPlan"]
