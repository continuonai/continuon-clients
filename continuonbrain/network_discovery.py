"""
LAN discovery service for ContinuonBrain robots.
Broadcasts robot availability on local network for iPhone/web browser discovery.
"""
import socket
import json
import threading
import time
from typing import Optional, Dict, Any
from pathlib import Path


class LANDiscoveryService:
    """
    Broadcasts robot presence on LAN using mDNS/Zeroconf.
    Allows iPhone Flutter app or web browser to find robots automatically.
    """
    
    def __init__(
        self,
        robot_name: str = "ContinuonBot",
        service_port: int = 8080,
        discovery_port: int = 5353
    ):
        self.robot_name = robot_name
        self.service_port = service_port
        self.discovery_port = discovery_port
        self.running = False
        self.broadcast_thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
    
    def get_robot_info(self) -> Dict[str, Any]:
        """Get robot information for broadcasting."""
        import platform
        
        # Get local IP address
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        try:
            s.connect(("8.8.8.8", 80))
            local_ip = s.getsockname()[0]
        except Exception:
            local_ip = "127.0.0.1"
        finally:
            s.close()
        
        return {
            "robot_name": self.robot_name,
            "service_type": "_continuonbrain._tcp",
            "hostname": socket.gethostname(),
            "ip_address": local_ip,
            "port": self.service_port,
            "platform": platform.machine(),
            "version": "0.1.0",
            "capabilities": [
                "arm_control",
                "depth_vision",
                "training_mode",
                "autonomous_mode"
            ],
            "endpoints": {
                "robot_api": f"http://{local_ip}:{self.service_port}",
                "web_ui": f"http://{local_ip}:{self.service_port}/ui",
                "status": f"http://{local_ip}:{self.service_port}/status"
            }
        }
    
    def start_simple_broadcast(self):
        """
        Start simple UDP broadcast (fallback if Zeroconf not available).
        Broadcasts robot info every 5 seconds on port 5555.
        """
        self.running = True
        self._stop_event.clear()
        
        def broadcast_loop():
            sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
            sock.setsockopt(socket.SOL_SOCKET, socket.SO_BROADCAST, 1)
            
            try:
                while not self._stop_event.is_set():
                    info = self.get_robot_info()
                    message = json.dumps(info).encode('utf-8')
                    
                    try:
                        sock.sendto(message, ('<broadcast>', 5555))
                        print(f"📡 Broadcast: {info['robot_name']} at {info['ip_address']}:{info['port']}")
                    except Exception as e:
                        print(f"Broadcast error: {e}")
                    
                    # Wait 5 seconds or until stop
                    self._stop_event.wait(5.0)
            
            finally:
                sock.close()
        
        self.broadcast_thread = threading.Thread(target=broadcast_loop, daemon=True)
        self.broadcast_thread.start()
        print(f"✅ Simple broadcast started on UDP port 5555")
    
    def start_zeroconf(self):
        """
        Start Zeroconf/mDNS service for proper LAN discovery.
        This allows iPhone apps to discover robots using Bonjour.
        """
        try:
            from zeroconf import ServiceInfo, Zeroconf
            
            info = self.get_robot_info()
            
            # Create mDNS service info
            service_info = ServiceInfo(
                "_continuonbrain._tcp.local.",
                f"{self.robot_name}._continuonbrain._tcp.local.",
                addresses=[socket.inet_aton(info['ip_address'])],
                port=self.service_port,
                properties={
                    'robot_name': self.robot_name,
                    'version': info['version'],
                    'capabilities': ','.join(info['capabilities'])
                },
                server=f"{socket.gethostname()}.local."
            )
            
            self.zeroconf = Zeroconf()
            self.zeroconf.register_service(service_info)
            self.running = True
            
            print(f"✅ Zeroconf service registered:")
            print(f"   Name: {self.robot_name}")
            print(f"   Type: _continuonbrain._tcp.local.")
            print(f"   Address: {info['ip_address']}:{self.service_port}")
            
            return True
        
        except ImportError:
            print("⚠️  Zeroconf not installed - using simple broadcast")
            print("   Install with: pip install zeroconf")
            return False
        except Exception as e:
            print(f"⚠️  Zeroconf failed: {e}")
            return False
    
    def start(self):
        """Start discovery service (tries Zeroconf first, falls back to UDP)."""
        print("🔍 Starting LAN discovery service...")
        
        # Try Zeroconf first (proper mDNS for iPhone)
        if not self.start_zeroconf():
            # Fallback to simple UDP broadcast
            self.start_simple_broadcast()
        
        info = self.get_robot_info()
        print()
        print("=" * 60)
        print("🌐 Robot Available on LAN")
        print("=" * 60)
        print(f"Robot Name: {info['robot_name']}")
        print(f"IP Address: {info['ip_address']}")
        print(f"Web UI: http://{info['ip_address']}:{self.service_port}/ui")
        print(f"Robot API: http://{info['ip_address']}:{self.service_port}")
        print()
        print("iPhone/Web Browser Discovery:")
        print(f"  • Browse to: http://{info['ip_address']}:{self.service_port}/ui")
        print(f"  • Or scan LAN for _continuonbrain._tcp services")
        print("=" * 60)
        print()
    
    def stop(self):
        """Stop discovery service."""
        print("🛑 Stopping LAN discovery...")
        
        self.running = False
        self._stop_event.set()
        
        # Stop Zeroconf if running
        if hasattr(self, 'zeroconf'):
            try:
                self.zeroconf.unregister_all_services()
                self.zeroconf.close()
            except Exception as e:
                print(f"Error stopping Zeroconf: {e}")
        
        # Wait for broadcast thread
        if self.broadcast_thread and self.broadcast_thread.is_alive():
            self.broadcast_thread.join(timeout=2)
        
        print("✅ LAN discovery stopped")


def main():
    """Test LAN discovery service."""
    import argparse
    
    parser = argparse.ArgumentParser(description="ContinuonBrain LAN Discovery")
    parser.add_argument("--name", default="ContinuonBot", help="Robot name")
    parser.add_argument("--port", type=int, default=8080, help="Service port")
    
    args = parser.parse_args()
    
    discovery = LANDiscoveryService(
        robot_name=args.name,
        service_port=args.port
    )
    
    try:
        discovery.start()
        
        print("Press Ctrl+C to stop...")
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        print("\n")
        discovery.stop()


if __name__ == "__main__":
    main()
