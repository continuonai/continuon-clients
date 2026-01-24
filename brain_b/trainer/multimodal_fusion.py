"""
Multi-Modal Sensor Fusion for Brain B

Combines data from multiple sensors (LiDAR, camera, IMU, etc.) into a unified
perception representation. Uses attention mechanisms to weight sensor importance
and Kalman filtering for temporal fusion.

Key Components:
1. SensorEncoder: Encodes each modality into common feature space
2. CrossModalAttention: Attends across modalities for feature fusion
3. TemporalFusion: Kalman filter for state estimation over time
4. FusionModel: Full multi-modal fusion pipeline

This enables the robot to make robust decisions by leveraging complementary
sensor information.
"""

import math
import random
import json
from dataclasses import dataclass, field
from typing import List, Dict, Tuple, Optional, Any
from datetime import datetime


@dataclass
class LiDARData:
    """LiDAR point cloud data (simplified to distance readings)."""
    distances: List[float] = field(default_factory=lambda: [1.0] * 360)  # 360-degree scan
    intensities: List[float] = field(default_factory=lambda: [0.5] * 360)
    timestamp: float = 0.0

    def to_sectors(self, num_sectors: int = 8) -> List[float]:
        """Reduce to sector-based representation."""
        sector_size = len(self.distances) // num_sectors
        sectors = []
        for i in range(num_sectors):
            start = i * sector_size
            end = start + sector_size
            sectors.append(min(self.distances[start:end]))
        return sectors


@dataclass
class CameraData:
    """Camera visual data (simplified features)."""
    # Object detections: list of (class, confidence, bbox)
    detections: List[Tuple[str, float, Tuple[float, float, float, float]]] = field(default_factory=list)
    # Color histogram (RGB, 8 bins each = 24 values)
    color_histogram: List[float] = field(default_factory=lambda: [0.0] * 24)
    # Edge density (top, bottom, left, right quadrants)
    edge_density: List[float] = field(default_factory=lambda: [0.0] * 4)
    # Optical flow magnitude (motion detection)
    flow_magnitude: float = 0.0
    # Brightness
    brightness: float = 0.5
    timestamp: float = 0.0


@dataclass
class IMUData:
    """Inertial Measurement Unit data."""
    # Accelerometer (x, y, z) in m/s^2
    acceleration: Tuple[float, float, float] = (0.0, 0.0, 9.81)
    # Gyroscope (roll, pitch, yaw rate) in rad/s
    gyroscope: Tuple[float, float, float] = (0.0, 0.0, 0.0)
    # Orientation quaternion (w, x, y, z)
    orientation: Tuple[float, float, float, float] = (1.0, 0.0, 0.0, 0.0)
    timestamp: float = 0.0


@dataclass
class AudioData:
    """Audio sensor data."""
    # Sound level in dB
    sound_level: float = 30.0
    # Direction of loudest sound (angle in radians)
    sound_direction: float = 0.0
    # Frequency spectrum (8 bands)
    frequency_spectrum: List[float] = field(default_factory=lambda: [0.0] * 8)
    # Voice activity detected
    voice_detected: bool = False
    timestamp: float = 0.0


class SensorEncoder:
    """Encodes sensor data into common feature space."""

    def __init__(self, output_dim: int = 32):
        self.output_dim = output_dim

        # LiDAR encoder (8 sectors -> 32 features)
        self.lidar_W = [[random.gauss(0, 0.1) for _ in range(8)] for _ in range(output_dim)]
        self.lidar_b = [0.0] * output_dim

        # Camera encoder (24 color + 4 edge + 2 other = 30 -> 32 features)
        self.camera_W = [[random.gauss(0, 0.1) for _ in range(30)] for _ in range(output_dim)]
        self.camera_b = [0.0] * output_dim

        # IMU encoder (10 features -> 32)
        self.imu_W = [[random.gauss(0, 0.1) for _ in range(10)] for _ in range(output_dim)]
        self.imu_b = [0.0] * output_dim

        # Audio encoder (12 features -> 32)
        self.audio_W = [[random.gauss(0, 0.1) for _ in range(12)] for _ in range(output_dim)]
        self.audio_b = [0.0] * output_dim

    def _relu(self, x: float) -> float:
        return max(0, x)

    def _linear(self, x: List[float], W: List[List[float]], b: List[float]) -> List[float]:
        """Linear transformation with ReLU."""
        output = []
        for j in range(len(b)):
            total = b[j]
            for i in range(min(len(x), len(W[j]))):
                total += W[j][i] * x[i]
            output.append(self._relu(total))
        return output

    def encode_lidar(self, data: LiDARData) -> List[float]:
        """Encode LiDAR data."""
        sectors = data.to_sectors(8)
        return self._linear(sectors, self.lidar_W, self.lidar_b)

    def encode_camera(self, data: CameraData) -> List[float]:
        """Encode camera data."""
        features = data.color_histogram + data.edge_density + [data.flow_magnitude, data.brightness]
        while len(features) < 30:
            features.append(0.0)
        return self._linear(features, self.camera_W, self.camera_b)

    def encode_imu(self, data: IMUData) -> List[float]:
        """Encode IMU data."""
        features = list(data.acceleration) + list(data.gyroscope) + list(data.orientation)[:4]
        while len(features) < 10:
            features.append(0.0)
        # Normalize acceleration (divide by ~10 to get ~1 range)
        features = [f / 10.0 if i < 3 else f for i, f in enumerate(features)]
        return self._linear(features, self.imu_W, self.imu_b)

    def encode_audio(self, data: AudioData) -> List[float]:
        """Encode audio data."""
        features = [data.sound_level / 100.0, data.sound_direction / math.pi, float(data.voice_detected)]
        features.extend(data.frequency_spectrum)
        while len(features) < 12:
            features.append(0.0)
        return self._linear(features, self.audio_W, self.audio_b)


class CrossModalAttention:
    """
    Cross-modal attention mechanism.
    Each modality attends to all other modalities to enrich its representation.
    """

    def __init__(self, dim: int = 32, num_heads: int = 4):
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads

        # Query, Key, Value projections for each modality
        self.num_modalities = 4  # lidar, camera, imu, audio
        self.W_q = [[[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
                    for _ in range(self.num_modalities)]
        self.W_k = [[[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
                    for _ in range(self.num_modalities)]
        self.W_v = [[[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
                    for _ in range(self.num_modalities)]
        self.W_o = [[[random.gauss(0, 0.1) for _ in range(dim)] for _ in range(dim)]
                    for _ in range(self.num_modalities)]

    def _linear(self, x: List[float], W: List[List[float]]) -> List[float]:
        """Linear transformation."""
        output = []
        for j in range(len(W)):
            total = 0.0
            for i in range(min(len(x), len(W[j]))):
                total += W[j][i] * x[i]
            output.append(total)
        return output

    def _softmax(self, x: List[float]) -> List[float]:
        """Softmax over list."""
        max_val = max(x) if x else 0
        exp_vals = [math.exp(v - max_val) for v in x]
        total = sum(exp_vals)
        return [e / max(total, 1e-10) for e in exp_vals]

    def attend(self, modality_features: List[List[float]]) -> List[List[float]]:
        """
        Apply cross-modal attention.
        Input: list of 4 modality feature vectors (each 32-dim)
        Output: attended feature vectors
        """
        if len(modality_features) != self.num_modalities:
            # Pad with zeros if missing modalities
            while len(modality_features) < self.num_modalities:
                modality_features.append([0.0] * self.dim)

        # Compute Q, K, V for each modality
        queries = [self._linear(modality_features[i], self.W_q[i])
                   for i in range(self.num_modalities)]
        keys = [self._linear(modality_features[i], self.W_k[i])
                for i in range(self.num_modalities)]
        values = [self._linear(modality_features[i], self.W_v[i])
                  for i in range(self.num_modalities)]

        # Each modality attends to all modalities
        attended = []
        for i in range(self.num_modalities):
            q = queries[i]

            # Compute attention scores with all keys
            scores = []
            for j in range(self.num_modalities):
                score = sum(q[d] * keys[j][d] for d in range(self.dim))
                score /= math.sqrt(self.dim)
                scores.append(score)

            # Softmax
            attn_weights = self._softmax(scores)

            # Weighted sum of values
            attended_vec = [0.0] * self.dim
            for j in range(self.num_modalities):
                for d in range(self.dim):
                    attended_vec[d] += attn_weights[j] * values[j][d]

            # Output projection
            output = self._linear(attended_vec, self.W_o[i])

            # Residual connection
            output = [output[d] + modality_features[i][d] for d in range(self.dim)]
            attended.append(output)

        return attended


class KalmanFilter:
    """
    Kalman filter for temporal sensor fusion.
    Estimates robot state from noisy sensor measurements over time.
    """

    def __init__(self, state_dim: int = 6):
        """
        State: [x, y, theta, vx, vy, omega]
        """
        self.state_dim = state_dim

        # State estimate
        self.x = [0.0] * state_dim

        # State covariance (diagonal for simplicity)
        self.P = [1.0] * state_dim

        # Process noise
        self.Q = [0.01] * state_dim

        # Measurement noise (different for each sensor)
        self.R_lidar = [0.05] * 3  # position
        self.R_imu = [0.1, 0.1, 0.05]  # velocity
        self.R_camera = [0.2] * 3  # position (less accurate)

    def predict(self, dt: float = 0.1):
        """Predict step using motion model."""
        # Simple motion model: x += v * dt
        self.x[0] += self.x[3] * dt  # x += vx * dt
        self.x[1] += self.x[4] * dt  # y += vy * dt
        self.x[2] += self.x[5] * dt  # theta += omega * dt

        # Normalize theta to [-pi, pi]
        while self.x[2] > math.pi:
            self.x[2] -= 2 * math.pi
        while self.x[2] < -math.pi:
            self.x[2] += 2 * math.pi

        # Update covariance
        for i in range(self.state_dim):
            self.P[i] += self.Q[i]

    def update_from_lidar(self, lidar_position: Tuple[float, float, float]):
        """Update with LiDAR-derived position estimate."""
        for i in range(3):
            # Kalman gain
            K = self.P[i] / (self.P[i] + self.R_lidar[i])

            # Update estimate
            self.x[i] += K * (lidar_position[i] - self.x[i])

            # Update covariance
            self.P[i] = (1 - K) * self.P[i]

    def update_from_imu(self, imu_velocity: Tuple[float, float, float]):
        """Update with IMU velocity estimate."""
        for i in range(3):
            idx = i + 3  # velocity indices
            K = self.P[idx] / (self.P[idx] + self.R_imu[i])
            self.x[idx] += K * (imu_velocity[i] - self.x[idx])
            self.P[idx] = (1 - K) * self.P[idx]

    def update_from_camera(self, camera_position: Tuple[float, float, float]):
        """Update with camera-derived position (e.g., visual odometry)."""
        for i in range(3):
            K = self.P[i] / (self.P[i] + self.R_camera[i])
            self.x[i] += K * (camera_position[i] - self.x[i])
            self.P[i] = (1 - K) * self.P[i]

    def get_state(self) -> Dict[str, float]:
        """Get current state estimate."""
        return {
            "x": self.x[0],
            "y": self.x[1],
            "theta": self.x[2],
            "vx": self.x[3],
            "vy": self.x[4],
            "omega": self.x[5],
            "uncertainty": sum(self.P) / len(self.P)
        }


class MultiModalFusion:
    """
    Complete multi-modal sensor fusion system.
    Combines sensor encoding, cross-modal attention, and temporal filtering.
    """

    def __init__(self, feature_dim: int = 32):
        self.feature_dim = feature_dim

        self.encoder = SensorEncoder(output_dim=feature_dim)
        self.attention = CrossModalAttention(dim=feature_dim)
        self.kalman = KalmanFilter()

        # Fusion MLP to combine attended features
        self.fusion_W1 = [[random.gauss(0, 0.1) for _ in range(feature_dim * 4)]
                          for _ in range(feature_dim * 2)]
        self.fusion_b1 = [0.0] * (feature_dim * 2)
        self.fusion_W2 = [[random.gauss(0, 0.1) for _ in range(feature_dim * 2)]
                          for _ in range(feature_dim)]
        self.fusion_b2 = [0.0] * feature_dim

        # Sensor reliability scores (learned)
        self.sensor_reliability = {
            "lidar": 0.9,
            "camera": 0.7,
            "imu": 0.95,
            "audio": 0.5
        }

        # History for temporal consistency
        self.history: List[Dict] = []
        self.max_history = 10

    def _relu(self, x: float) -> float:
        return max(0, x)

    def _linear(self, x: List[float], W: List[List[float]], b: List[float]) -> List[float]:
        """Linear transformation with ReLU."""
        output = []
        for j in range(len(b)):
            total = b[j]
            for i in range(min(len(x), len(W[j]))):
                total += W[j][i] * x[i]
            output.append(self._relu(total))
        return output

    def fuse(self, lidar: Optional[LiDARData] = None,
             camera: Optional[CameraData] = None,
             imu: Optional[IMUData] = None,
             audio: Optional[AudioData] = None,
             dt: float = 0.1) -> Dict[str, Any]:
        """
        Fuse all available sensor data into unified perception.

        Returns:
            Dictionary with:
            - fused_features: 32-dim feature vector
            - state_estimate: Kalman filter state
            - sensor_contributions: How much each sensor contributed
            - confidence: Overall confidence in perception
        """
        # Encode available sensors
        modality_features = []
        available_sensors = []

        if lidar is not None:
            modality_features.append(self.encoder.encode_lidar(lidar))
            available_sensors.append("lidar")
        else:
            modality_features.append([0.0] * self.feature_dim)

        if camera is not None:
            modality_features.append(self.encoder.encode_camera(camera))
            available_sensors.append("camera")
        else:
            modality_features.append([0.0] * self.feature_dim)

        if imu is not None:
            modality_features.append(self.encoder.encode_imu(imu))
            available_sensors.append("imu")
        else:
            modality_features.append([0.0] * self.feature_dim)

        if audio is not None:
            modality_features.append(self.encoder.encode_audio(audio))
            available_sensors.append("audio")
        else:
            modality_features.append([0.0] * self.feature_dim)

        # Apply cross-modal attention
        attended = self.attention.attend(modality_features)

        # Concatenate and fuse
        concat = []
        for feat in attended:
            concat.extend(feat)

        # MLP fusion
        h = self._linear(concat, self.fusion_W1, self.fusion_b1)
        fused = self._linear(h, self.fusion_W2, self.fusion_b2)

        # Update Kalman filter
        self.kalman.predict(dt)

        if lidar is not None:
            # Estimate position from LiDAR (simplified: use front distance)
            sectors = lidar.to_sectors(8)
            lidar_pos = (sectors[0], sectors[2] - sectors[6], 0.0)  # Rough estimate
            self.kalman.update_from_lidar(lidar_pos)

        if imu is not None:
            # Use IMU for velocity
            vx = imu.acceleration[0] * dt
            vy = imu.acceleration[1] * dt
            omega = imu.gyroscope[2]
            self.kalman.update_from_imu((vx, vy, omega))

        # Get state estimate
        state = self.kalman.get_state()

        # Calculate confidence
        total_reliability = 0.0
        for sensor in available_sensors:
            total_reliability += self.sensor_reliability[sensor]
        confidence = total_reliability / len(self.sensor_reliability) if available_sensors else 0.0

        # Store in history
        self.history.append({
            "timestamp": datetime.now().timestamp(),
            "fused": fused[:5],  # Store subset for memory
            "sensors": available_sensors,
            "confidence": confidence
        })
        if len(self.history) > self.max_history:
            self.history.pop(0)

        return {
            "fused_features": fused,
            "state_estimate": state,
            "sensor_contributions": {s: self.sensor_reliability[s] for s in available_sensors},
            "confidence": confidence,
            "sensors_used": available_sensors
        }

    def detect_anomaly(self, current_fusion: Dict) -> Tuple[bool, str]:
        """
        Detect anomalies by comparing with history.
        Returns (is_anomaly, reason).
        """
        if len(self.history) < 3:
            return False, "insufficient_history"

        current_features = current_fusion["fused_features"]

        # Compare with recent history
        distances = []
        for h in self.history[-3:]:
            dist = sum((a - b) ** 2 for a, b in zip(current_features[:5], h["fused"]))
            distances.append(dist ** 0.5)

        avg_distance = sum(distances) / len(distances)

        # Check for sudden changes
        if avg_distance > 0.5:
            return True, "sudden_perception_change"

        # Check for sensor dropout
        current_sensors = set(current_fusion["sensors_used"])
        prev_sensors = set(self.history[-1]["sensors"])
        dropped = prev_sensors - current_sensors
        if dropped:
            return True, f"sensor_dropout: {dropped}"

        return False, "normal"

    def get_action_recommendation(self, fusion_result: Dict) -> Tuple[str, float]:
        """
        Recommend action based on fused perception.
        Returns (action, confidence).
        """
        state = fusion_result["state_estimate"]
        features = fusion_result["fused_features"]
        confidence = fusion_result["confidence"]

        # Simple heuristic based on state
        if state["uncertainty"] > 0.5:
            return "stop", 0.3  # Too uncertain, stop

        # Check if moving
        speed = (state["vx"] ** 2 + state["vy"] ** 2) ** 0.5

        # Use fused features for obstacle detection
        obstacle_score = sum(features[:4]) / 4  # First few features relate to obstacles

        if obstacle_score > 0.7:
            return "move_forward", confidence
        elif obstacle_score < 0.3:
            return "rotate_left", confidence * 0.8
        else:
            return "stop", confidence * 0.5

    def save(self, filepath: str):
        """Save fusion model state."""
        data = {
            "sensor_reliability": self.sensor_reliability,
            "kalman_state": self.kalman.x,
            "kalman_covariance": self.kalman.P
        }
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

    def load(self, filepath: str):
        """Load fusion model state."""
        with open(filepath) as f:
            data = json.load(f)
        self.sensor_reliability = data.get("sensor_reliability", self.sensor_reliability)
        self.kalman.x = data.get("kalman_state", self.kalman.x)
        self.kalman.P = data.get("kalman_covariance", self.kalman.P)


def demo_multimodal_fusion():
    """Demonstrate multi-modal sensor fusion."""
    print("Multi-Modal Sensor Fusion Demo")
    print("=" * 50)

    fusion = MultiModalFusion()

    # Simulate 20 time steps
    print("\nSimulating 20 sensor fusion steps...")

    for step in range(20):
        # Generate synthetic sensor data
        t = step * 0.1

        # LiDAR
        lidar = LiDARData(
            distances=[1.0 - 0.3 * math.sin(i * 0.1 + t) for i in range(360)],
            intensities=[0.5] * 360,
            timestamp=t
        )

        # Camera
        camera = CameraData(
            detections=[("obstacle", 0.9, (0.3, 0.4, 0.1, 0.1))] if step % 3 == 0 else [],
            color_histogram=[random.random() for _ in range(24)],
            edge_density=[0.3, 0.5, 0.2, 0.4],
            flow_magnitude=0.1 * step,
            brightness=0.5 + 0.1 * math.sin(t),
            timestamp=t
        )

        # IMU
        imu = IMUData(
            acceleration=(0.1 * math.sin(t), 0.05 * math.cos(t), 9.81),
            gyroscope=(0.0, 0.0, 0.1 * math.sin(t * 2)),
            timestamp=t
        )

        # Audio (available intermittently)
        audio = None
        if step % 5 == 0:
            audio = AudioData(
                sound_level=40 + 10 * random.random(),
                sound_direction=random.random() * math.pi,
                frequency_spectrum=[random.random() for _ in range(8)],
                voice_detected=step % 10 == 0,
                timestamp=t
            )

        # Fuse sensors
        result = fusion.fuse(lidar, camera, imu, audio, dt=0.1)

        # Check for anomalies
        is_anomaly, reason = fusion.detect_anomaly(result)

        # Get action recommendation
        action, conf = fusion.get_action_recommendation(result)

        if step % 5 == 0:
            print(f"\nStep {step}:")
            print(f"  Sensors: {result['sensors_used']}")
            print(f"  Confidence: {result['confidence']:.2f}")
            print(f"  State: x={result['state_estimate']['x']:.3f}, "
                  f"y={result['state_estimate']['y']:.3f}, "
                  f"theta={result['state_estimate']['theta']:.3f}")
            print(f"  Action: {action} (conf={conf:.2f})")
            if is_anomaly:
                print(f"  ANOMALY: {reason}")

    print("\n" + "=" * 50)
    print("Final sensor reliability scores:")
    for sensor, score in fusion.sensor_reliability.items():
        print(f"  {sensor}: {score:.2f}")

    print("\nFinal state estimate:")
    state = fusion.kalman.get_state()
    for key, value in state.items():
        print(f"  {key}: {value:.4f}")

    return fusion


if __name__ == "__main__":
    fusion = demo_multimodal_fusion()
    print("\nDemo completed successfully!")
