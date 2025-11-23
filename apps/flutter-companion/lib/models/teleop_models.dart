enum ControlMode { eeVelocity, jointDelta, gripper }

enum ReferenceFrame { base, tool }

enum GripperMode { position, velocity }

class Vector3 {
  const Vector3({required this.x, required this.y, required this.z});

  final double x;
  final double y;
  final double z;

  Map<String, dynamic> toJson() => {'x': x, 'y': y, 'z': z};
}

class EeVelocityCommand {
  const EeVelocityCommand({
    required this.linearMps,
    required this.angularRadS,
    this.referenceFrame = ReferenceFrame.base,
  });

  final Vector3 linearMps;
  final Vector3 angularRadS;
  final ReferenceFrame referenceFrame;

  Map<String, dynamic> toJson() => {
        'linear_mps': linearMps.toJson(),
        'angular_rad_s': angularRadS.toJson(),
        'reference_frame': referenceFrame.name,
      };
}

class JointDeltaCommand {
  const JointDeltaCommand({required this.deltaRadians});

  final List<double> deltaRadians;

  Map<String, dynamic> toJson() => {'delta_radians': deltaRadians};
}

class GripperCommand {
  const GripperCommand({
    required this.mode,
    this.positionM,
    this.velocityMps,
  });

  final GripperMode mode;
  final double? positionM;
  final double? velocityMps;

  Map<String, dynamic> toJson() => {
        'mode': mode.name,
        if (positionM != null) 'position_m': positionM,
        if (velocityMps != null) 'velocity_mps': velocityMps,
      };
}

class SafetyStatus {
  const SafetyStatus({this.estopReleasedAck = false, this.safetyToken});

  final bool estopReleasedAck;
  final String? safetyToken;

  Map<String, dynamic> toJson() => {
        'estop_released_ack': estopReleasedAck,
        if (safetyToken != null) 'safety_token': safetyToken,
      };
}

class ControlCommand {
  ControlCommand({
    required this.clientId,
    required this.controlMode,
    required this.targetFrequencyHz,
    this.safetyStatus = const SafetyStatus(),
    this.eeVelocity,
    this.jointDelta,
    this.gripperCommand,
  });

  final String clientId;
  final ControlMode controlMode;
  final double targetFrequencyHz;
  final SafetyStatus safetyStatus;
  final EeVelocityCommand? eeVelocity;
  final JointDeltaCommand? jointDelta;
  final GripperCommand? gripperCommand;

  Map<String, dynamic> toJson() {
    return {
      'client_id': clientId,
      'control_mode': controlMode.name,
      'target_frequency_hz': targetFrequencyHz,
      'safety': safetyStatus.toJson(),
      if (eeVelocity != null) 'ee_velocity': eeVelocity!.toJson(),
      if (jointDelta != null) 'joint_delta': jointDelta!.toJson(),
      if (gripperCommand != null) 'gripper': gripperCommand!.toJson(),
    };
  }
}

class RobotState {
  const RobotState({
    required this.frameId,
    required this.gripperOpen,
    required this.jointPositions,
    this.wallTimeMillis,
  });

  final String frameId;
  final bool gripperOpen;
  final List<double> jointPositions;
  final int? wallTimeMillis;
}
