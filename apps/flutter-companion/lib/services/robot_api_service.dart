import 'dart:async';
import 'dart:convert';
import 'dart:io';
import '../models/teleop_models.dart';

/// Service for communicating with ContinuonBrain Robot API
/// Supports both mock (localhost) and real hardware connections
class RobotApiService {
  String _host;
  int _port;
  Socket? _socket;
  StreamController<RobotState>? _stateStreamController;
  bool _isConnected = false;
  
  RobotApiService({
    String host = 'localhost',
    int port = 8080,
  }) : _host = host, _port = port;

  /// Get connection status
  bool get isConnected => _isConnected;

  /// Connect to Robot API server
  Future<bool> connect() async {
    try {
      print('Connecting to Robot API at $_host:$_port...');
      
      _socket = await Socket.connect(_host, _port);
      _isConnected = true;
      
      print('✅ Connected to Robot API');
      return true;
    } catch (e) {
      print('❌ Failed to connect: $e');
      _isConnected = false;
      return false;
    }
  }

  /// Disconnect from server
  Future<void> disconnect() async {
    await _socket?.close();
    _socket = null;
    _isConnected = false;
    await _stateStreamController?.close();
    _stateStreamController = null;
    print('Disconnected from Robot API');
  }

  /// Send control command to robot
  Future<Map<String, dynamic>> sendCommand(ControlCommand command) async {
    if (!_isConnected || _socket == null) {
      return {
        'success': false,
        'message': 'Not connected to Robot API'
      };
    }

    try {
      final request = {
        'method': 'send_command',
        'params': command.toJson(),
      };

      // Send JSON command
      _socket!.writeln(jsonEncode(request));
      await _socket!.flush();

      // Read response (with timeout)
      final response = await _socket!
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .first
          .timeout(
            const Duration(seconds: 1),
            onTimeout: () => '{"success": false, "message": "Timeout"}',
          );

      return jsonDecode(response) as Map<String, dynamic>;
    } catch (e) {
      print('Error sending command: $e');
      return {
        'success': false,
        'message': 'Error: $e'
      };
    }
  }

  /// Stream robot state updates
  Stream<RobotState> streamRobotState(String clientId) {
    if (_stateStreamController != null) {
      return _stateStreamController!.stream;
    }

    _stateStreamController = StreamController<RobotState>();

    _startStateStream(clientId);

    return _stateStreamController!.stream;
  }

  Future<void> _startStateStream(String clientId) async {
    try {
      // Create separate connection for streaming
      final streamSocket = await Socket.connect(_host, _port);

      final request = {
        'method': 'stream_state',
        'params': {'client_id': clientId},
      };

      streamSocket.writeln(jsonEncode(request));
      await streamSocket.flush();

      // Listen to stream
      streamSocket
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .listen(
        (line) {
          try {
            final stateJson = jsonDecode(line) as Map<String, dynamic>;
            final state = RobotState.fromJson(stateJson);
            _stateStreamController?.add(state);
          } catch (e) {
            print('Error parsing state: $e');
          }
        },
        onError: (error) {
          print('State stream error: $error');
          _stateStreamController?.addError(error);
        },
        onDone: () {
          print('State stream closed');
          _stateStreamController?.close();
        },
      );
    } catch (e) {
      print('Error starting state stream: $e');
      _stateStreamController?.addError(e);
    }
  }

  /// Start RLDS episode recording
  Future<Map<String, dynamic>> startRecording(String languageInstruction) async {
    if (!_isConnected || _socket == null) {
      return {
        'success': false,
        'message': 'Not connected'
      };
    }

    try {
      final request = {
        'method': 'start_recording',
        'params': {'instruction': languageInstruction},
      };

      _socket!.writeln(jsonEncode(request));
      await _socket!.flush();

      final response = await _socket!
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .first
          .timeout(const Duration(seconds: 2));

      return jsonDecode(response) as Map<String, dynamic>;
    } catch (e) {
      return {
        'success': false,
        'message': 'Error: $e'
      };
    }
  }

  /// Stop RLDS episode recording
  Future<Map<String, dynamic>> stopRecording({bool success = true}) async {
    if (!_isConnected || _socket == null) {
      return {
        'success': false,
        'message': 'Not connected'
      };
    }

    try {
      final request = {
        'method': 'stop_recording',
        'params': {'success': success},
      };

      _socket!.writeln(jsonEncode(request));
      await _socket!.flush();

      final response = await _socket!
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .first
          .timeout(const Duration(seconds: 2));

      return jsonDecode(response) as Map<String, dynamic>;
    } catch (e) {
      return {
        'success': false,
        'message': 'Error: $e'
      };
    }
  }

  /// Get depth camera frame (metadata only for now)
  Future<Map<String, dynamic>?> getDepthFrame() async {
    if (!_isConnected || _socket == null) {
      return null;
    }

    try {
      final request = {
        'method': 'get_depth',
        'params': {},
      };

      _socket!.writeln(jsonEncode(request));
      await _socket!.flush();

      final response = await _socket!
          .transform(utf8.decoder)
          .transform(const LineSplitter())
          .first
          .timeout(const Duration(milliseconds: 500));

      return jsonDecode(response) as Map<String, dynamic>?;
    } catch (e) {
      print('Error getting depth frame: $e');
      return null;
    }
  }
}

extension RobotStateExtensions on RobotState {
  static RobotState fromJson(Map<String, dynamic> json) {
    return RobotState(
      frameId: json['frame_id'] as String,
      gripperOpen: json['gripper_open'] as bool,
      jointPositions: (json['joint_positions'] as List<dynamic>)
          .map((e) => (e as num).toDouble())
          .toList(),
      wallTimeMillis: json['wall_time_millis'] as int?,
    );
  }
}
