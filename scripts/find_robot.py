import socket
import json
import time
import argparse
import sys
from pathlib import Path
from typing import Optional, Dict

try:
    from zeroconf import ServiceBrowser, Zeroconf, ServiceListener
except ImportError:
    print("Error: zeroconf package not installed. Run 'pip install zeroconf'")
    sys.exit(1)

# Look for config in project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CACHE_FILE = PROJECT_ROOT / ".continuon_config" / "discovery_cache.json"

class ContinuonListener(ServiceListener):
    def __init__(self):
        self.found_robots = {}

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        if name in self.found_robots:
            del self.found_robots[name]

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            address = socket.inet_ntoa(info.addresses[0])
            self.found_robots[name] = {
                "name": name,
                "address": address,
                "port": info.port,
                "properties": {k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v for k, v in info.properties.items()}
            }

def find_robot(timeout: float = 2.0, use_cache: bool = True) -> Optional[Dict]:
    if use_cache and CACHE_FILE.exists():
        try:
            cache = json.loads(CACHE_FILE.read_text())
            # Check if cache is still fresh (e.g., < 1 hour)
            if time.time() - cache.get("timestamp", 0) < 3600:
                return cache
        except Exception:
            pass

    zeroconf = Zeroconf()
    listener = ContinuonListener()
    browser = ServiceBrowser(zeroconf, "_continuon._tcp.local.", listener)
    
    try:
        # Incremental sleep to catch robots faster
        start_time = time.time()
        while time.time() - start_time < timeout:
            if listener.found_robots:
                break
            time.sleep(0.5)
    finally:
        zeroconf.close()

    if listener.found_robots:
        # For now, just return the first one found
        name = next(iter(listener.found_robots))
        robot = listener.found_robots[name]
        robot["timestamp"] = time.time()
        
        # Update cache
        try:
            CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
            CACHE_FILE.write_text(json.dumps(robot, indent=2))
        except Exception as e:
            print(f"Warning: Could not write cache: {e}", file=sys.stderr)
        
        return robot
    
    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Find Continuon Robot on LAN")
    parser.add_argument("--timeout", type=float, default=3.0, help="Search timeout in seconds")
    parser.add_argument("--no-cache", action="store_true", help="Ignore cache and perform fresh search")
    parser.add_argument("--quiet", action="store_true", help="Only output IP address")
    args = parser.parse_args()

    robot = find_robot(timeout=args.timeout, use_cache=not args.no_cache)
    if robot:
        if args.quiet:
            print(robot['address'])
        else:
            print(f"✅ Found robot: {robot['name']}")
            print(f"   Address: {robot['address']}:{robot['port']}")
            print(f"   Version: {robot['properties'].get('version', 'unknown')}")
    else:
        if not args.quiet:
            print("❌ No robot found.")
        sys.exit(1)
