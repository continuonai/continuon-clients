import argparse
import sys
import subprocess
import time
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
    parser = argparse.ArgumentParser(description="Tunnel Robot ports to localhost")
    parser.add_argument("--host", help="Robot IP address")
    parser.add_argument("--user", default="pi", help="SSH username")
    parser.add_argument("--key", help="SSH private key path")
    parser.add_argument("--ports", default="8080,8081,8082,50051", help="Comma-separated ports to forward")
    
    args = parser.parse_args()
    host = resolve_host_args(args)

    ports = [p.strip() for p in args.ports.split(",")]
    print(f"🚇 Establishing tunnel to {args.user}@{host} for ports: {ports}")
    
    ssh_cmd = ["ssh"]
    if args.key:
        ssh_cmd.extend(["-i", args.key])
    
    # Add port forwarding flags
    for port in ports:
        ssh_cmd.extend(["-L", f"{port}:localhost:{port}"])
    
    ssh_cmd.extend([
        "-N", # Do not execute a remote command
        f"{args.user}@{host}"
    ])

    print(f"   Command: {' '.join(ssh_cmd)}")
    print("✅ Tunnel active. Press Ctrl+C to stop.")
    print(f"   Access UI at: http://localhost:8080/ui")
    
    try:
        subprocess.run(ssh_cmd)
    except KeyboardInterrupt:
        print("\n🛑 Tunnel closed.")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
