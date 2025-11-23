import 'dart:async';
import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:grpc/grpc.dart' as grpc;

import '../models/teleop_models.dart';
import 'platform_channels.dart';

class BrainClient {
  BrainClient({grpc.ClientChannel? channel})
      : _channel = channel,
        _platformBridge = const PlatformBrainBridge();

  grpc.ClientChannel? _channel;
  final PlatformBrainBridge _platformBridge;
  StreamController<RobotState>? _stateController;

  bool get isConnected => _channel != null || _platformBridge.isConnected;

  Future<void> connect({
    required String host,
    required int port,
    bool useTls = false,
  }) async {
    if (_channel != null) {
      return;
    }
    final credentials = useTls ? const grpc.ChannelCredentials.secure() : const grpc.ChannelCredentials.insecure();
    _channel = grpc.ClientChannel(
      host,
      port: port,
      options: grpc.ChannelOptions(credentials: credentials),
    );
    await _channel!.getConnectionState();
  }

  Future<void> sendCommand(ControlCommand command) async {
    final payload = jsonEncode(command.toJson());
    try {
      if (_channel != null) {
        final method = grpc.ClientMethod<String, Map<String, dynamic>>(
          '/continuonxr.continuonbrain.v1.ContinuonBrainBridge/SendCommand',
          (String value) => utf8.encode(value),
          (List<int> value) => jsonDecode(utf8.decode(value)) as Map<String, dynamic>,
        );
        final client = grpc.Client(_channel!);
        await client.invoke<Map<String, dynamic>>(method, payload).first;
      } else {
        await _platformBridge.sendCommand(payload);
      }
    } on PlatformException catch (error) {
      throw StateError('Platform send failed: ${error.message}');
    } on grpc.GrpcError catch (error) {
      throw StateError('gRPC send failed: ${error.message}');
    }
  }

  Stream<RobotState> streamRobotState(String clientId) {
    _stateController ??= StreamController<RobotState>.broadcast();
    if (_channel != null) {
      final method = grpc.ClientMethod<String, Map<String, dynamic>>(
        '/continuonxr.continuonbrain.v1.ContinuonBrainBridge/StreamRobotState',
        (String value) => utf8.encode(value),
        (List<int> value) => jsonDecode(utf8.decode(value)) as Map<String, dynamic>,
      );
      final client = grpc.Client(_channel!);
      client.invoke<Map<String, dynamic>>(method, jsonEncode({'client_id': clientId})).listen((event) {
        _emitRobotState(event);
      });
    } else {
      _platformBridge.subscribeState(clientId).listen((stateJson) {
        final data = jsonDecode(stateJson) as Map<String, dynamic>;
        _emitRobotState(data);
      });
    }
    return _stateController!.stream;
  }

  void _emitRobotState(Map<String, dynamic> data) {
    final positions = (data['state']?['joint_positions'] as List<dynamic>? ?? []).cast<num>();
    _stateController?.add(
      RobotState(
        frameId: (data['state']?['frame_id'] as String?) ?? 'frame',
        gripperOpen: (data['state']?['gripper_open'] as bool?) ?? false,
        jointPositions: positions.map((e) => e.toDouble()).toList(),
        wallTimeMillis: data['state']?['wall_time_millis'] as int?,
      ),
    );
  }

  Future<void> dispose() async {
    await _stateController?.close();
    await _channel?.shutdown();
  }
}
