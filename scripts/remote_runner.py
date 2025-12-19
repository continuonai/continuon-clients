import json
import argparse
import sys
import os
from pathlib import Path
from typing import Optional

try:
    from fabric import Connection, Config
    from invoke import UnexpectedExit
except ImportError:
    print("Error: fabric package not installed. Run 'pip install fabric'")
    sys.exit(1)

# Look for config in project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PROJECT_ROOT / ".continuon_config" / "discovery_cache.json"

class RemoteRunner:
    def __init__(self, host: Optional[str] = None, user: str = "pi", key_filename: Optional[str] = None):
        self.host = host
        self.user = user
        self.key_filename = key_filename
        self.connection = None

        if not self.host:
            self.host = self._resolve_host()
        
        if not self.host:
            raise ValueError("Could not resolve robot host. Run 'python scripts/find_robot.py' first or specify --host.")

        self._connect()

    def _resolve_host(self) -> Optional[str]:
        if CACHE_FILE.exists():
            try:
                cache = json.loads(CACHE_FILE.read_text())
                return cache.get("address")
            except Exception:
                pass
        return None

    def _connect(self):
        connect_kwargs = {}
        if self.key_filename:
            connect_kwargs["key_filename"] = self.key_filename
        
        # Load SSH config if available
        config = Config()
        config.load_system_configs()
        config.load_user_configs()

        self.connection = Connection(
            host=self.host,
            user=self.user,
            connect_kwargs=connect_kwargs,
            config=config
        )

    def run(self, command: str, hide: bool = False, warn: bool = False):
        try:
            return self.connection.run(command, hide=hide, warn=warn)
        except UnexpectedExit as e:
            print(f"Remote command failed: {e}", file=sys.stderr)
            if not warn:
                sys.exit(e.result.exited)
            return e.result

    def put(self, local_path: str, remote_path: str):
        self.connection.put(local_path, remote_path)

    def get(self, remote_path: str, local_path: str):
        self.connection.get(remote_path, local_path)

def main():
    parser = argparse.ArgumentParser(description="Run commands on remote Continuon Robot")
    parser.add_argument("command", nargs="?", help="Command to run (e.g., 'ls -la')")
    parser.add_argument("--host", help="Robot IP address (overrides discovery cache)")
    parser.add_argument("--user", default="pi", help="SSH username (default: pi)")
    parser.add_argument("--key", help="Path to private key file")
    parser.add_argument("--sync", action="store_true", help="Sync continuonbrain/ to robot before running")
    
    args = parser.parse_args()

    try:
        runner = RemoteRunner(host=args.host, user=args.user, key_filename=args.key)
        
        if args.sync:
            print(f"🔄 Syncing code to {runner.user}@{runner.host}...")
            # For a robust sync, we'd use rsync, but fabric/invoke doesn't have a native recursive put.
            # We'll shell out to rsync if available, or just warn for now as this is Phase 3.
            # Using rsync over ssh is standard.
            rsync_cmd = f"rsync -avz --exclude '.git' --exclude '__pycache__' --exclude '.venv' {PROJECT_ROOT}/continuonbrain {runner.user}@{runner.host}:/home/{runner.user}/"
            os.system(rsync_cmd) # Simple wrapper

        if args.command:
            print(f"🚀 Running: {args.command}")
            runner.run(args.command)
        else:
             # Interactive shell? Fabric doesn't do full PTY interaction easily in a script
             # without 'shell=True' logic which invoke handles differently.
             # For now, just print info.
             print(f"✅ Connected to {runner.user}@{runner.host}")
             print("   Usage: python scripts/remote_runner.py \"command\"")

    except ValueError as e:
        print(f"❌ Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
