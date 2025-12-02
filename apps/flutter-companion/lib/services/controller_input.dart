import 'dart:async';
import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

import '../models/teleop_models.dart';
import 'brain_client.dart';

class ControllerSnapshot {
  ControllerSnapshot({
    required this.leftX,
    required this.leftY,
    required this.rightX,
    required this.rightY,
    required this.shouldOpenGripper,
    required this.shouldCloseGripper,
  });

  final double leftX;
  final double leftY;
  final double rightX;
  final double rightY;
  final bool shouldOpenGripper;
  final bool shouldCloseGripper;

  factory ControllerSnapshot.fromMap(Map<dynamic, dynamic> map) {
    final normalized = Map<String, dynamic>.fromEntries(
      map.entries.map(
        (entry) => MapEntry(entry.key.toString(), entry.value),
      ),
    );
    return ControllerSnapshot(
      leftX: (normalized['left_x'] as num? ?? normalized['lx'] as num? ?? 0).toDouble(),
      leftY: (normalized['left_y'] as num? ?? normalized['ly'] as num? ?? 0).toDouble(),
      rightX: (normalized['right_x'] as num? ?? normalized['rx'] as num? ?? 0).toDouble(),
      rightY: (normalized['right_y'] as num? ?? normalized['ry'] as num? ?? 0).toDouble(),
      shouldOpenGripper: normalized['button_cross'] == true || normalized['button_x'] == true,
      shouldCloseGripper: normalized['button_circle'] == true || normalized['button_o'] == true,
    );
  }

  ControlCommand toVelocityCommand({
    required String clientId,
    required AccelerationProfile profile,
  }) {
    return ControlCommand(
      clientId: clientId,
      controlMode: ControlMode.eeVelocity,
      targetFrequencyHz: 30,
      eeVelocity: EeVelocityCommand(
        referenceFrame: ReferenceFrame.base,
        linearMps: Vector3(
          x: profile.linearScale * (-leftY),
          y: profile.linearScale * leftX,
          z: 0,
        ),
        angularRadS: Vector3(
          x: 0,
          y: 0,
          z: profile.angularScale * rightX,
        ),
      ),
    );
  }

  GripperCommand? toGripperCommand() {
    if (shouldOpenGripper == shouldCloseGripper) {
      return null;
    }
    return GripperCommand(
      mode: GripperMode.position,
      positionM: shouldOpenGripper ? 0.045 : 0.0,
    );
  }
}

class Ps3ControllerBridge {
  const Ps3ControllerBridge();

  static const _methodChannel = MethodChannel('ps3_controller');
  static const _eventChannel = EventChannel('ps3_controller/events');

  Future<bool> requestConnection() async {
    if (kIsWeb) return false;
    final connected = await _methodChannel.invokeMethod<bool>('connect');
    return connected ?? false;
  }

  Stream<ControllerSnapshot> snapshots() {
    if (kIsWeb) {
      return const Stream.empty();
    }
    return _eventChannel
        .receiveBroadcastStream()
        .map((event) => ControllerSnapshot.fromMap(event as Map<dynamic, dynamic>));
  }

  StreamSubscription<ControllerSnapshot> pipeToBrain({
    required BrainClient brainClient,
    required AccelerationProfile profile,
    String clientId = 'flutter-companion',
  }) {
    Timer? backoff;
    return snapshots().listen((snapshot) async {
      if (backoff != null && backoff!.isActive) {
        return;
      }
      backoff = Timer(const Duration(milliseconds: 40), () {});
      await brainClient.sendCommand(snapshot.toVelocityCommand(clientId: clientId, profile: profile));
      final gripper = snapshot.toGripperCommand();
      if (gripper != null) {
        await brainClient.sendCommand(
          ControlCommand(
            clientId: clientId,
            controlMode: ControlMode.gripper,
            targetFrequencyHz: 5,
            gripperCommand: gripper,
          ),
        );
      }
    });
  }
}
