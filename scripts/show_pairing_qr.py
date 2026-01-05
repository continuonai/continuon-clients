#!/usr/bin/env python3
"""
Generate and display an RCAN-compatible QR code for pairing with ContinuonAI app.
Reads the current connection info and creates a scannable QR code with RCAN protocol data.
"""

import json
import os
import subprocess
import sys
import re
import uuid
from pathlib import Path

try:
    import qrcode
except ImportError:
    print("Installing qrcode library...")
    subprocess.run([sys.executable, "-m", "pip", "install", "qrcode[pil]", "-q"])
    import qrcode

try:
    import requests
except ImportError:
    print("Installing requests library...")
    subprocess.run([sys.executable, "-m", "pip", "install", "requests", "-q"])
    import requests


def get_tunnel_url() -> str | None:
    """Get current Cloudflare tunnel URL from journalctl."""
    try:
        result = subprocess.run(
            ["journalctl", "-u", "cloudflared-tunnel", "-n", "50", "--no-pager"],
            capture_output=True,
            text=True,
        )
        matches = re.findall(r'https://([a-z0-9-]+\.trycloudflare\.com)', result.stdout)
        if matches:
            return matches[-1]
    except Exception as e:
        print(f"Error reading tunnel logs: {e}")
    return None


def get_local_ip() -> str:
    """Get the local IP address."""
    try:
        result = subprocess.run(["hostname", "-I"], capture_output=True, text=True)
        return result.stdout.strip().split()[0]
    except Exception:
        return "192.168.1.1"


def get_rcan_status(host: str, port: int, use_https: bool = False) -> dict | None:
    """Fetch RCAN status from the robot."""
    scheme = "https" if use_https else "http"
    url = f"{scheme}://{host}:{port}/rcan/v1/status" if port not in (80, 443) else f"{scheme}://{host}/rcan/v1/status"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception as e:
        print(f"Could not fetch RCAN status: {e}")
    return None


def get_robot_info(host: str, port: int, use_https: bool = False) -> dict:
    """Get robot info from ping endpoint."""
    scheme = "https" if use_https else "http"
    url = f"{scheme}://{host}:{port}/api/ping" if port not in (80, 443) else f"{scheme}://{host}/api/ping"
    try:
        response = requests.get(url, timeout=5)
        if response.status_code == 200:
            return response.json()
    except Exception:
        pass
    return {}


def generate_pairing_token() -> str:
    """Generate a short-lived pairing token."""
    return str(uuid.uuid4())[:8]


def generate_rcan_pairing_data(use_tunnel: bool = True, pairing_token: str | None = None) -> dict:
    """Generate RCAN-compatible pairing data for QR code."""

    if use_tunnel:
        tunnel_host = get_tunnel_url()
        if tunnel_host:
            host = tunnel_host
            port = 443
            secure = True
        else:
            print("Warning: No tunnel found, falling back to local")
            host = get_local_ip()
            port = 8080
            secure = False
    else:
        host = get_local_ip()
        port = 8080
        secure = False

    # Get RCAN status from robot
    rcan_status = get_rcan_status(host, port, secure)
    robot_info = get_robot_info(host, port, secure)

    # Build RCAN pairing data
    data = {
        "v": 1,                          # QR format version
        "proto": "rcan",                 # Protocol identifier
        "h": host,                       # Host
        "p": port,                       # Port
        "s": secure,                     # Use HTTPS
    }

    # Add RCAN-specific fields if available
    if rcan_status:
        if "ruri" in rcan_status:
            data["ruri"] = rcan_status["ruri"]
        if "robot_name" in rcan_status or "friendly_name" in rcan_status:
            data["name"] = rcan_status.get("robot_name") or rcan_status.get("friendly_name")
        if "model" in rcan_status:
            data["model"] = rcan_status["model"]
        if "caps" in rcan_status or "capabilities" in rcan_status:
            data["caps"] = rcan_status.get("caps") or rcan_status.get("capabilities")
        if "version" in rcan_status:
            data["rcan_v"] = rcan_status["version"]

    # Fallback to ping info
    if "name" not in data and robot_info:
        data["name"] = robot_info.get("device_id", "ContinuonBrain")

    # Add pairing token for quick auth (optional)
    if pairing_token:
        data["token"] = pairing_token

    return data


def create_qr_code(data: dict, output_path: str | None = None):
    """Create QR code from pairing data."""
    json_data = json.dumps(data, separators=(',', ':'))

    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=2,
    )
    qr.add_data(json_data)
    qr.make(fit=True)

    if output_path:
        img = qr.make_image(fill_color="black", back_color="white")
        img.save(output_path)
        print(f"QR code saved to: {output_path}")

    qr.print_ascii(invert=True)
    return json_data


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Generate ContinuonAI RCAN pairing QR code")
    parser.add_argument("--local", action="store_true", help="Use local IP instead of tunnel")
    parser.add_argument("--output", "-o", help="Save QR code image to file")
    parser.add_argument("--token", "-t", action="store_true", help="Generate pairing token for quick auth")
    args = parser.parse_args()

    print("=" * 60)
    print("  ContinuonAI RCAN Robot Pairing QR Code")
    print("=" * 60)
    print()

    use_tunnel = not args.local
    pairing_token = generate_pairing_token() if args.token else None

    data = generate_rcan_pairing_data(use_tunnel=use_tunnel, pairing_token=pairing_token)

    # Display info
    if data.get("s"):
        print(f"  Mode:  Tunnel (accessible from anywhere)")
    else:
        print(f"  Mode:  Local Network")

    print(f"  Host:  {data['h']}")
    if data['p'] not in (80, 443):
        print(f"  Port:  {data['p']}")

    if data.get("ruri"):
        print(f"  RURI:  {data['ruri']}")
    if data.get("name"):
        print(f"  Name:  {data['name']}")
    if data.get("model"):
        print(f"  Model: {data['model']}")
    if data.get("caps"):
        print(f"  Caps:  {', '.join(data['caps']) if isinstance(data['caps'], list) else data['caps']}")
    if pairing_token:
        print(f"  Token: {pairing_token} (valid for this session)")

    print()
    print("  Scan this QR code with the ContinuonAI app:")
    print()

    json_str = create_qr_code(data, args.output)

    print()
    print(f"  Data: {json_str}")
    print()
    print("  In the app: Connect > Scan Robot QR Code")
    print("=" * 60)


if __name__ == "__main__":
    main()
