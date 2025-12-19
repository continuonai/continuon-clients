import time
import argparse
import sys
import os
import subprocess
from pathlib import Path
from typing import Optional

try:
    from watchdog.observers import Observer
    from watchdog.events import FileSystemEventHandler
except ImportError:
    print("Error: watchdog package not installed. Run 'pip install watchdog'")
    sys.exit(1)

# Import discovery logic
sys.path.append(str(Path(__file__).parent))
try:
    from find_robot import find_robot
except ImportError:
    find_robot = None

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SOURCE_DIR = PROJECT_ROOT / "continuonbrain"

class SyncHandler(FileSystemEventHandler):
    def __init__(self, host: str, user: str, remote_base: str, key_path: Optional[str] = None):
        self.host = host
        self.user = user
        self.remote_base = remote_base
        self.key_path = key_path
        self.last_sync = 0.0

    def on_any_event(self, event):
        if event.is_directory:
            return
        
        # Debounce
        if time.time() - self.last_sync < 1.0:
            return
        
        # Filter interesting files
        if not event.src_path.endswith(".py") and not event.src_path.endswith(".json") and not event.src_path.endswith(".yaml"):
            return

        print(f"🔄 Change detected: {event.src_path}")
        self.sync()
        self.last_sync = time.time()

    def sync(self):
        # Construct rsync command
        # Exclude common junk
        exclude_flags = "--exclude '__pycache__' --exclude '*.pyc' --exclude '.git' --exclude '.venv' --exclude 'logs/'"
        
        ssh_cmd = "ssh"
        if self.key_path:
            ssh_cmd += f" -i {self.key_path}"
        
        # Adjust source to be the content of continuonbrain/ directory
        # We want PROJECT_ROOT/continuonbrain/* -> REMOTE/continuonbrain/
        
        cmd = f"rsync -avz {exclude_flags} -e \"{ssh_cmd}\" {SOURCE_DIR}/ {self.user}@{self.host}:{self.remote_base}/continuonbrain/"
        
        try:
            # On Windows, os.system might struggle with complex quotes. using subprocess.
            # However, rsync on Windows is usually via Cygwin/GitBash.
            # If rsync is missing, we fail gracefully.
            ret = os.system(cmd)
            if ret == 0:
                print("✅ Sync complete.")
            else:
                print("⚠️  Sync returned non-zero exit code.")
        except Exception as e:
            print(f"❌ Sync failed: {e}")

def resolve_host_args(args):
    host = args.host
    if not host and find_robot:
        print("🔍 Searching for robot...")
        robot = find_robot(timeout=2.0)
        if robot:
            host = robot["address"]
            print(f"   Found {robot['name']} at {host}")
    
    if not host:
        print("❌ Could not resolve robot host. Specify --host.")
        sys.exit(1)
    return host

def main():
    parser = argparse.ArgumentParser(description="Sync ContinuonBrain code to Robot")
    parser.add_argument("--host", help="Robot IP address")
    parser.add_argument("--user", default="pi", help="SSH username")
    parser.add_argument("--remote-path", default="/home/pi/continuonbrain", help="Remote base directory")
    parser.add_argument("--key", help="SSH private key path")
    parser.add_argument("--watch", action="store_true", help="Watch for changes and sync automatically")
    
    args = parser.parse_args()
    host = resolve_host_args(args)

    handler = SyncHandler(host, args.user, args.remote_path, args.key)
    
    # Initial sync
    print(f"🚀 Performing initial sync to {args.user}@{host}...")
    handler.sync()

    if args.watch:
        print("👀 Watching for changes... (Ctrl+C to stop)")
        observer = Observer()
        observer.schedule(handler, str(SOURCE_DIR), recursive=True)
        observer.start()
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            observer.stop()
        observer.join()

if __name__ == "__main__":
    main()
