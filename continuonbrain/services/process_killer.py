
import psutil
import logging
from typing import List, Optional

logger = logging.getLogger(__name__)

class ProcessKiller:
    """
    Utility to aggressively kill processes to free up memory for the Background Learner.
    """

    TARGET_PROCESS_NAMES = [
        "antigravity",
        "cursor",
        "code",         # VS Code
        "pycharm",
        "intellij",
        "webstorm",
        "goland",
        "rider",
        "clion",
        "datagrip",
        "rubymine",
        "phpstorm"
    ]

    @staticmethod
    def kill_target_processes(targets: Optional[List[str]] = None) -> int:
        """
        Kill standard IDE/Editor processes or specific targets.
        
        Args:
            targets: Optional list of process names to target (substring match).
            
        Returns:
            Number of processes killed.
        """
        if targets is None:
            targets = ProcessKiller.TARGET_PROCESS_NAMES

        killed_count = 0
        try:
            for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
                try:
                    pinfo = proc.info
                    name = (pinfo['name'] or "").lower()
                    cmd = " ".join(pinfo['cmdline'] or []).lower()
                    
                    found = False
                    for t in targets:
                        if t in name or t in cmd:
                            found = True
                            break
                    
                    if found:
                        # Avoid killing self or essential robot services if they happen to share a name (unlikely for "antigravity" binary vs python module)
                        # Actually 'antigravity' binary is the agent interface.
                        logger.warning(f"ProcessKiller: Terminating {name} (PID {pinfo['pid']}) to free memory.")
                        proc.terminate()
                        killed_count += 1
                except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                    pass
        except Exception as e:
            logger.error(f"ProcessKiller failed iteration: {e}")

        if killed_count > 0:
            logger.info(f"ProcessKiller: Terminated {killed_count} processes to prioritize Background Learner.")
        
        return killed_count
