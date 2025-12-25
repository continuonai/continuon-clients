import socket
import time
import json
from pathlib import Path
from zeroconf import ServiceBrowser, ServiceListener, Zeroconf

class ContinuonListener(ServiceListener):
    def __init__(self):
        self.found_robots = []

    def update_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

    def remove_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        pass

    def add_service(self, zc: Zeroconf, type_: str, name: str) -> None:
        info = zc.get_service_info(type_, name)
        if info:
            addresses = [socket.inet_ntoa(addr) for addr in info.addresses]
            self.found_robots.append({
                "name": name,
                "addresses": addresses,
                "port": info.port,
                "server": info.server
            })

def find_robot(timeout=5):
    zeroconf = Zeroconf()
    listener = ContinuonListener()
    browser = ServiceBrowser(zeroconf, "_continuon._tcp.local.", listener)
    
    print(f"Searching for Continuon robots (timeout={timeout}s)...")
    time.sleep(timeout)
    zeroconf.close()
    
    return listener.found_robots

def cache_ip(ip):
    cache_path = Path(".robot_ip")
    cache_path.write_text(ip)
    print(f"Cached robot IP: {ip}")

if __name__ == "__main__":
    robots = find_robot()
    if robots:
        print(f"Found {len(robots)} robot(s):")
        for i, r in enumerate(robots):
            ip = r['addresses'][0]
            print(f"  {i+1}. {r['name']} at {ip}:{r['port']}")
            if i == 0:
                cache_ip(ip)
    else:
        print("No robots found via mDNS.")