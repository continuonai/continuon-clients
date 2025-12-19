import argparse
import sys
import subprocess
from pathlib import Path

# Import discovery logic
sys.path.append(str(Path(__file__).parent))
try:
    from find_robot import find_robot
except ImportError:
    find_robot = None

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
    parser = argparse.ArgumentParser(description="Stream logs from remote Robot")
    parser.add_argument("--host", help="Robot IP address")
    parser.add_argument("--user", default="pi", help="SSH username")
    parser.add_argument("--key", help="SSH private key path")
    parser.add_argument("--log-file", default="continuonbrain/logs/robot_api_server.log", help="Relative path to log file")
    
    args = parser.parse_args()
    host = resolve_host_args(args)

    print(f"📜 Streaming {args.log_file} from {args.user}@{host}...")
    
    ssh_cmd = ["ssh"]
    if args.key:
        ssh_cmd.extend(["-i", args.key])
    
    ssh_cmd.extend([
        f"{args.user}@{host}",
        f"tail -f {args.log_file}"
    ])

    try:
        # Use simple subprocess call to pipe stdout to console
        subprocess.run(ssh_cmd)
    except KeyboardInterrupt:
        print("\n🛑 Stream stopped.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
