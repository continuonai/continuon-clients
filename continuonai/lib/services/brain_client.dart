import 'dart:async';
import 'dart:convert';

import 'package:flutter/services.dart';
import 'package:grpc/grpc.dart' as grpc;

import '../models/teleop_models.dart';
import 'platform_channels.dart';

class BrainClient {
  static const _servicePrefix = '/continuonxr.continuonbrain.v1.ContinuonBrainBridge';
  static final grpc.ClientMethod<String, Map<String, dynamic>> _sendCommandMethod =
      grpc.ClientMethod<String, Map<String, dynamic>>(
    '$_servicePrefix/SendCommand',
    utf8.encode,
    (List<int> value) => jsonDecode(utf8.decode(value)) as Map<String, dynamic>,
  );
  static final grpc.ClientMethod<String, Map<String, dynamic>> _streamRobotStateMethod =
      grpc.ClientMethod<String, Map<String, dynamic>>(
    '$_servicePrefix/StreamRobotState',
    utf8.encode,
    (List<int> value) => jsonDecode(utf8.decode(value)) as Map<String, dynamic>,
  );
  BrainClient({
    grpc.ClientChannel? channel,
    PlatformBrainBridge platformBridge = const PlatformBrainBridge(),
    this.clientId = 'flutter-companion',
  })  : _channel = channel,
        _platformBridge = platformBridge;

  grpc.ClientChannel? _channel;
  final PlatformBrainBridge _platformBridge;
  final String clientId;
  StreamController<RobotState>? _stateController;
  grpc.CallOptions _callOptions = grpc.CallOptions();
  bool _usePlatformBridge = false;

  bool get isConnected => _usePlatformBridge ? _platformBridge.isConnected : _channel != null;

  Future<void> connect({
    required String host,
    required int port,
    bool useTls = true,
    String? authToken,
    List<int>? trustedRootCertificates,
    bool preferPlatformBridge = false,
  }) async {
    _usePlatformBridge = preferPlatformBridge;
    _callOptions = authToken != null
        ? grpc.CallOptions(metadata: {'authorization': 'Bearer $authToken'})
        : grpc.CallOptions();

    if (_usePlatformBridge) {
      await _platformBridge.initConnection(host, port, useTls: useTls, authToken: authToken);
      return;
    }

    if (_channel != null) {
      return;
    }

    final credentials = useTls
        ? grpc.ChannelCredentials.secure(certificates: trustedRootCertificates)
        : const grpc.ChannelCredentials.insecure();
    _channel = grpc.ClientChannel(
      host,
      port: port,
      options: grpc.ChannelOptions(credentials: credentials),
    );
  }

  Future<void> sendCommand(ControlCommand command) async {
    final payload = jsonEncode(command.toJson());
    try {
      if (_usePlatformBridge) {
        await _platformBridge.sendCommand(payload);
        return;
      }
      final channel = _channel ?? (throw StateError('BrainClient not connected'));
      final stub = _JsonBrainClient(channel);
      await stub.unary(_sendCommandMethod, payload, _callOptions);
    } on PlatformException catch (error) {
      throw StateError('Platform send failed: ${error.message}');
    } on grpc.GrpcError catch (error) {
      throw StateError('gRPC send failed: ${error.message}');
    }
  }

  Future<Map<String, dynamic>> startRecording(String instruction) async {
    return _invokeJsonMethod(
      'StartRecording',
      {'instruction': instruction, 'client_id': clientId},
      errorLabel: 'start recording',
    );
  }

  Future<Map<String, dynamic>> stopRecording({bool success = true}) async {
    return _invokeJsonMethod(
      'StopRecording',
      {'success': success, 'client_id': clientId},
      errorLabel: 'stop recording',
    );
  }

  Stream<RobotState> streamRobotState(String clientId) {
    _stateController ??= StreamController<RobotState>.broadcast();
    if (_usePlatformBridge) {
      _platformBridge.subscribeState(clientId).listen((stateJson) {
        final data = jsonDecode(stateJson) as Map<String, dynamic>;
        _emitRobotState(data);
      });
      return _stateController!.stream;
    }

    final channel = _channel;
    if (channel == null) {
      _stateController!.addError(StateError('BrainClient not connected'));
      return _stateController!.stream;
    }
    final stub = _JsonBrainClient(channel);
    stub
        .serverStream(_streamRobotStateMethod, jsonEncode({'client_id': clientId}), _callOptions)
        .listen(
          _emitRobotState,
          onError: (error) => _stateController?.addError(error),
        );
    return _stateController!.stream;
  }

  Future<Map<String, dynamic>> _invokeJsonMethod(
    String methodName,
    Map<String, dynamic> payload, {
    required String errorLabel,
  }) async {
    if (_usePlatformBridge) {
      throw StateError('Method $methodName not available over platform bridge');
    }
    final channel = _channel ?? (throw StateError('BrainClient not connected'));
    try {
      final method = grpc.ClientMethod<String, Map<String, dynamic>>(
        '$_servicePrefix/$methodName',
        utf8.encode,
        (List<int> value) => jsonDecode(utf8.decode(value)) as Map<String, dynamic>,
      );
      final stub = _JsonBrainClient(channel);
      return await stub.unary(method, jsonEncode(payload), _callOptions);
    } on grpc.GrpcError catch (error) {
      throw StateError('Unable to $errorLabel: ${error.message}');
    }
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

class _JsonBrainClient extends grpc.Client {
  _JsonBrainClient(super.channel);

  grpc.ResponseFuture<Map<String, dynamic>> unary(
    grpc.ClientMethod<String, Map<String, dynamic>> method,
    String payload,
    grpc.CallOptions options,
  ) {
    return $createUnaryCall(method, payload, options: options);
  }

  grpc.ResponseStream<Map<String, dynamic>> serverStream(
    grpc.ClientMethod<String, Map<String, dynamic>> method,
    String payload,
    grpc.CallOptions options,
  ) {
    return $createStreamingCall(method, Stream.value(payload), options: options);
  }
}
