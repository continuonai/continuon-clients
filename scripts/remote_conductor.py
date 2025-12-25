import os
import sys
from pathlib import Path
from fabric import Connection
import invoke

class RemoteRunner:
    def __init__(self, host=None, user="pi"):
        self.host = host or self._get_cached_ip()
        self.user = user
        self.conn = None
        
        if not self.host:
            print("Error: No robot IP found. Run 'python scripts/find_robot.py' first.")
            sys.exit(1)

    def _get_cached_ip(self):
        cache_path = Path(".robot_ip")
        if cache_path.exists():
            return cache_path.read_text().strip()
        return None

    def get_connection(self):
        if not self.conn:
            self.conn = Connection(host=self.host, user=self.user)
        return self.conn

    def run(self, command, hide=False):
        print(f"[{self.host}] Running: {command}")
        try:
            result = self.get_connection().run(command, hide=hide)
            return result
        except Exception as e:
            print(f"Remote command failed: {e}")
            return None

    def setup_ssh_keys(self):
        """Deploy local public key to robot."""
        pub_key_path = Path.home() / ".ssh" / "id_rsa.pub"
        if not pub_key_path.exists():
            print("Local public key not found. Generating...")
            # Use invoke to run local command
            invoke.run("ssh-keygen -t rsa -b 4096 -N '' -f ~/.ssh/id_rsa")
        
        pub_key = pub_key_path.read_text().strip()
        print(f"Deploying SSH key to {self.host}...")
        self.run(f"mkdir -p ~/.ssh && echo '{pub_key}' >> ~/.ssh/authorized_keys && chmod 700 ~/.ssh && chmod 600 ~/.ssh/authorized_keys")
        print("SSH keys deployed.")

    def sync(self, local_path="continuonbrain/", remote_path="~/continuonbrain/"):
        """Sync local directory to robot using rsync (if available) or scp."""
        print(f"Syncing {local_path} to {self.user}@{self.host}:{remote_path}...")
        # Use local rsync via subprocess
        import subprocess
        try:
            # -a: archive mode, -v: verbose, -z: compress, -e: specify shell
            cmd = ["rsync", "-avz", "-e", "ssh", local_path, f"{self.user}@{self.host}:{remote_path}"]
            subprocess.run(cmd, check=True)
            print("Sync complete.")
        except Exception as e:
            print(f"Sync failed: {e}. Ensure rsync is installed locally.")

    def tail_logs(self, log_file="~/continuonbrain/logs/robot_api_server.log"):
        """Tail remote logs."""
        print(f"Tailing {log_file} on {self.host}...")
        try:
            self.get_connection().run(f"tail -f {log_file}")
        except KeyboardInterrupt:
            print("\nStopped tailing.")

    def tunnel(self, remote_port, local_port=None):
        """Establish SSH tunnel for a port."""
        local_port = local_port or remote_port
        print(f"Tunneling {self.host}:{remote_port} to localhost:{local_port}...")
        print("Press Ctrl+C to stop.")
        # Note: fabric connection.forward_local is better but complex for CLI.
        # Use system ssh command for simplicity
        import subprocess
        try:
            subprocess.run(["ssh", "-L", f"{local_port}:localhost:{remote_port}", f"{self.user}@{self.host}", "-N"])
        except KeyboardInterrupt:
            print("\nTunnel closed.")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Remote Conductor Interface")
    parser.add_argument("command", nargs="?", help="Command to run on robot")
    parser.add_argument("--setup-keys", action="store_true", help="Deploy SSH keys")
    parser.add_argument("--sync", action="store_true", help="Sync code to robot")
    parser.add_argument("--tail", action="store_true", help="Tail remote logs")
    parser.add_argument("--tunnel", type=int, help="Tunnel a remote port (e.g., 8080)")
    parser.add_argument("--local-port", type=int, help="Local port for tunnel")
    args = parser.parse_args()

    runner = RemoteRunner()
    if args.setup_keys:
        runner.setup_ssh_keys()
    elif args.sync:
        runner.sync()
    elif args.tail:
        runner.tail_logs()
    elif args.tunnel:
        runner.tunnel(args.tunnel, args.local_port)
    elif args.command:
        runner.run(args.command)
    else:
        parser.print_help()
