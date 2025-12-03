"""Hardware validation helpers for Pi 5 + OAK-D Lite + PCA9685.

This module keeps the Pi 5 bring-up checklist scriptable:
- Ensures the continuonos brain directories exist with writable permissions.
- Confirms UVC/depth camera profiles (OAK-D Lite) and I2C servo controller presence.
- Issues a short servo pulse and depth frame grab, logging timestamp alignment (target ≤5 ms).

Run as a CLI on the Pi:
    python -m continuonbrain.pi5_hardware_validation --log-json /tmp/pi5_check.json
"""
from __future__ import annotations

import importlib.util
import json
import subprocess
import time
from dataclasses import dataclass, asdict, field
from pathlib import Path
from typing import Dict, List, Optional

DEPTHAI_AVAILABLE = importlib.util.find_spec("depthai") is not None
SERVOKIT_AVAILABLE = importlib.util.find_spec("adafruit_servokit") is not None

if DEPTHAI_AVAILABLE:
    import depthai as dai  # type: ignore

if SERVOKIT_AVAILABLE:
    from adafruit_servokit import ServoKit  # type: ignore


@dataclass
class CommandResult:
    """Captured output from a shell command."""

    command: List[str]
    stdout: str
    stderr: str
    return_code: int


@dataclass
class ValidationStatus:
    """Aggregated validation results for a single check."""

    ok: bool
    details: Dict[str, object] = field(default_factory=dict)


@dataclass
class ValidationReport:
    """Top-level report for Pi 5 hardware readiness."""

    base_dir: str
    directories: ValidationStatus
    i2c: ValidationStatus
    uvc: ValidationStatus
    depth_capture: ValidationStatus
    servo_pulse: ValidationStatus
    timestamp_skew_ms: Optional[float] = None

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def run_command(command: List[str]) -> CommandResult:
    """Run a command and return stdout/stderr/exit code."""

    result = subprocess.run(command, capture_output=True, text=True)
    return CommandResult(command=command, stdout=result.stdout, stderr=result.stderr, return_code=result.returncode)


def ensure_continuonos_directories(base_dir: Path) -> ValidationStatus:
    """Create the continuonos brain directories with group-writable permissions."""

    expected_paths = [
        base_dir / "model" / "base_model",
        base_dir / "model" / "adapters" / "current",
        base_dir / "model" / "adapters" / "candidate",
        base_dir / "rlds" / "episodes",
        base_dir / "trainer" / "logs",
    ]

    created: List[str] = []
    for path in expected_paths:
        path.mkdir(parents=True, exist_ok=True)
        path.chmod(0o775)
        created.append(str(path))

    return ValidationStatus(ok=True, details={"paths": created})


def detect_pca9685(bus: int = 1, address: int = 0x40) -> ValidationStatus:
    """Check I2C bus output for the PCA9685 servo controller."""

    command = ["i2cdetect", "-y", str(bus)]
    result = run_command(command)
    found = f"{address:02x}" in result.stdout.lower()
    return ValidationStatus(
        ok=result.return_code == 0 and found,
        details={
            "bus": bus,
            "address": f"0x{address:02x}",
            "command": result.command,
            "stdout": result.stdout.strip(),
            "stderr": result.stderr.strip(),
        },
    )


def list_uvc_devices() -> ValidationStatus:
    """List V4L2 devices and expected OAK-D Lite profiles."""

    command = ["v4l2-ctl", "--list-devices"]
    result = run_command(command)
    if result.return_code != 0:
        return ValidationStatus(ok=False, details={"command": result.command, "stderr": result.stderr.strip()})

    devices: Dict[str, List[str]] = {}
    current_name: Optional[str] = None
    for line in result.stdout.splitlines():
        if not line.startswith("\t"):
            current_name = line.strip()
            devices[current_name] = []
            continue
        if current_name is not None:
            devices[current_name].append(line.strip())

    expected_profiles = {"640x480@30", "1280x720@30"}
    profile_hits: Dict[str, List[str]] = {}
    for name, nodes in devices.items():
        for node in nodes:
            formats = run_command(["v4l2-ctl", "--device", node, "--list-formats-ext"])
            found_profiles = [p for p in expected_profiles if p in formats.stdout]
            if found_profiles:
                profile_hits[node] = found_profiles

    ok = any(profile_hits.values())
    return ValidationStatus(
        ok=ok,
        details={"devices": devices, "profile_hits": profile_hits, "expected_profiles": sorted(expected_profiles)},
    )


def capture_depth_frame(timeout: float = 5.0) -> ValidationStatus:
    """Grab a single depth frame from an OAK-D Lite using DepthAI if available."""

    if not DEPTHAI_AVAILABLE:
        return ValidationStatus(ok=False, details={"error": "depthai module not available"})

    pipeline = dai.Pipeline()
    mono_left = pipeline.create(dai.node.MonoCamera)
    mono_right = pipeline.create(dai.node.MonoCamera)
    stereo = pipeline.create(dai.node.StereoDepth)
    xout = pipeline.create(dai.node.XLinkOut)

    mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
    mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
    mono_left.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
    mono_right.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)

    stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
    mono_left.out.link(stereo.left)
    mono_right.out.link(stereo.right)

    xout.setStreamName("depth")
    stereo.depth.link(xout.input)

    with dai.Device(pipeline) as device:
        queue = device.getOutputQueue(name="depth", maxSize=1, blocking=True)
        packet = queue.get(timeout=timeout)
        frame = packet.getFrame()
        timestamp = packet.getTimestamp().total_seconds()
        resolution = {"width": len(frame[0]), "height": len(frame)}

    return ValidationStatus(
        ok=True,
        details={"timestamp": timestamp, "resolution": resolution},
    )


def send_servo_pulse(channel: int = 0, angle: float = 90.0) -> ValidationStatus:
    """Send a bounded servo pulse to the PCA9685 using adafruit-servokit if present."""

    if not SERVOKIT_AVAILABLE:
        return ValidationStatus(ok=False, details={"error": "adafruit_servokit module not available"})

    kit = ServoKit(channels=16)
    angle = max(0.0, min(180.0, angle))
    kit.servo[channel].angle = angle
    return ValidationStatus(ok=True, details={"channel": channel, "angle": angle})


def compute_timestamp_skew(depth_status: ValidationStatus, servo_status: ValidationStatus, servo_ts: float, capture_ts: float) -> Optional[float]:
    """Compute host timestamp skew in milliseconds between servo command and depth frame."""

    if not depth_status.ok or not servo_status.ok:
        return None
    return abs(capture_ts - servo_ts) * 1000.0


def run_validation(base_dir: Path = Path("/opt/continuonos/brain")) -> ValidationReport:
    """Run the Pi 5 validation suite and return a structured report."""

    directories = ensure_continuonos_directories(base_dir)
    i2c_status = detect_pca9685()
    uvc_status = list_uvc_devices()

    servo_ts = time.time()
    servo_status = send_servo_pulse()

    depth_status = capture_depth_frame()
    capture_ts = depth_status.details.get("timestamp") if depth_status.ok else None

    skew_ms = None
    if capture_ts is not None:
        skew_ms = compute_timestamp_skew(depth_status, servo_status, servo_ts=servo_ts, capture_ts=capture_ts)

    return ValidationReport(
        base_dir=str(base_dir),
        directories=directories,
        i2c=i2c_status,
        uvc=uvc_status,
        depth_capture=depth_status,
        servo_pulse=servo_status,
        timestamp_skew_ms=skew_ms,
    )


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Validate Pi 5 camera + servo hardware")
    parser.add_argument(
        "--base-dir",
        type=Path,
        default=Path("/opt/continuonos/brain"),
        help="continuonos brain root (default: /opt/continuonos/brain)",
    )
    parser.add_argument("--log-json", type=Path, help="Optional path to write the JSON report", default=None)
    args = parser.parse_args()

    report = run_validation(args.base_dir)
    print(report.to_json())

    if args.log_json:
        args.log_json.write_text(report.to_json())
        print(f"Report written to {args.log_json}")


if __name__ == "__main__":
    main()
