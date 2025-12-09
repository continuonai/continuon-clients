"""
ContinuonBrain Watchdog Service
Monitors system health and automatically recovers from failures.
"""
import time
import sys
import requests
import subprocess
import logging
from pathlib import Path
from datetime import datetime, timedelta

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger('watchdog')

class ContinuonWatchdog:
    def __init__(self, config_dir="/opt/continuonbrain/config"):
        self.config_dir = Path(config_dir)
        self.health_url = "http://localhost:8080/api/status"
        self.check_interval = 30  # seconds
        self.failure_threshold = 3
        self.consecutive_failures = 0
        self.last_restart = None
        self.restart_cooldown = timedelta(minutes=5)
        
    def check_health(self):
        """Check if the ContinuonBrain service is healthy."""
        try:
            response = requests.get(self.health_url, timeout=5)
            if response.status_code == 200:
                self.consecutive_failures = 0
                return True
            else:
                logger.warning(f"Health check failed: HTTP {response.status_code}")
                return False
        except requests.exceptions.RequestException as e:
            logger.warning(f"Health check failed: {e}")
            return False
    
    def restart_service(self):
        """Restart the ContinuonBrain service."""
        # Check cooldown
        if self.last_restart:
            time_since_restart = datetime.now() - self.last_restart
            if time_since_restart < self.restart_cooldown:
                logger.warning(f"Restart cooldown active ({time_since_restart.seconds}s since last restart)")
                return False
        
        logger.info("Restarting ContinuonBrain service...")
        try:
            subprocess.run(
                ["systemctl", "restart", "continuonbrain.service"],
                check=True,
                capture_output=True
            )
            self.last_restart = datetime.now()
            self.consecutive_failures = 0
            logger.info("Service restarted successfully")
            return True
        except subprocess.CalledProcessError as e:
            logger.error(f"Failed to restart service: {e}")
            return False
    
    def run(self):
        """Main watchdog loop."""
        logger.info("ContinuonBrain Watchdog started")
        logger.info(f"Monitoring {self.health_url} every {self.check_interval}s")
        
        while True:
            try:
                if not self.check_health():
                    self.consecutive_failures += 1
                    logger.warning(f"Health check failed ({self.consecutive_failures}/{self.failure_threshold})")
                    
                    if self.consecutive_failures >= self.failure_threshold:
                        logger.error("Failure threshold reached, attempting restart...")
                        self.restart_service()
                        # Wait longer after restart
                        time.sleep(60)
                        continue
                else:
                    if self.consecutive_failures > 0:
                        logger.info("Service recovered")
                
                time.sleep(self.check_interval)
                
            except KeyboardInterrupt:
                logger.info("Watchdog stopped by user")
                break
            except Exception as e:
                logger.error(f"Watchdog error: {e}")
                time.sleep(self.check_interval)

def main():
    import argparse
    parser = argparse.ArgumentParser(description="ContinuonBrain Watchdog")
    parser.add_argument("--config-dir", default="/opt/continuonbrain/config")
    args = parser.parse_args()
    
    watchdog = ContinuonWatchdog(config_dir=args.config_dir)
    watchdog.run()

if __name__ == "__main__":
    main()
