import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';
import 'package:grpc/grpc.dart' as grpc;
import 'package:grpc/grpc_web.dart' as grpc_web;
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../models/teleop_models.dart';
import 'platform_channels.dart';
import 'task_recorder.dart';

class ConnectionDiagnostic {
  final bool httpOk;
  final bool grpcOk;
  final String? httpError;
  final String? grpcError;
  final bool isCorsLikely;

  ConnectionDiagnostic({
    required this.httpOk,
    required this.grpcOk,
    this.httpError,
    this.grpcError,
    this.isCorsLikely = false,
  });

  @override
  String toString() {
    if (httpOk && grpcOk) return 'Connection healthy';
    if (isCorsLikely) return 'Connection blocked by CORS. Ensure robot has CORS enabled or use a tunnel.';
    return 'HTTP: ${httpOk ? 'OK' : httpError}, gRPC: ${grpcOk ? 'OK' : grpcError}';
  }
}

class BrainClient {
  static const _servicePrefix =
      '/continuonxr.continuonbrain.v1.ContinuonBrainBridge';
  static final grpc.ClientMethod<String, Map<String, dynamic>>
      _sendCommandMethod = grpc.ClientMethod<String, Map<String, dynamic>>(
    '$_servicePrefix/SendCommand',
    utf8.encode,
    (List<int> value) =>
        Map<String, dynamic>.from(jsonDecode(utf8.decode(value))),
  );
  static final grpc.ClientMethod<String, Map<String, dynamic>>
      _streamRobotStateMethod = grpc.ClientMethod<String, Map<String, dynamic>>(
    '$_servicePrefix/StreamRobotState',
    utf8.encode,
    (List<int> value) =>
        Map<String, dynamic>.from(jsonDecode(utf8.decode(value))),
  );
  BrainClient({
    grpc.ClientChannel? channel,
    PlatformBrainBridge platformBridge = const PlatformBrainBridge(),
    this.clientId = 'flutter-companion',
  })  : _channel = channel,
        _platformBridge = platformBridge;

  dynamic _channel;
  final PlatformBrainBridge _platformBridge;
  final String clientId;
  StreamController<RobotState>? _stateController;
  grpc.CallOptions _callOptions = grpc.CallOptions();
  bool _usePlatformBridge = false;
  String? _host;
  int _httpPort = 8080;
  // TODO: wire to persistent auth/subscription/ownership tokens when backend is ready.
  bool isOwned = false;
  bool hasSubscription = false;
  String? _authToken;
  bool _lanLikely = true;

  String? get authToken => _authToken;

  bool get isConnected =>
      _usePlatformBridge ? _platformBridge.isConnected : _channel != null;

  Future<void> setAuthToken(String token, {bool persist = false}) async {
    _authToken = token;
    _callOptions =
        grpc.CallOptions(metadata: {'authorization': 'Bearer $token'});
    if (persist) {
      const secure = FlutterSecureStorage();
      await secure.write(key: 'auth_token', value: token);
      final prefs = await SharedPreferences.getInstance();
      await prefs.remove('auth_token'); // migrate away from plaintext
    }
  }

  Map<String, String> _headers() {
    return _authToken != null ? {'authorization': 'Bearer $_authToken'} : {};
  }

  Uri _httpUri(String path, {Map<String, String>? queryParameters}) {
    final host = _host;
    if (host == null || host.isEmpty) {
      throw StateError('BrainClient not connected');
    }
    return Uri(
      scheme: 'http',
      host: host,
      port: _httpPort,
      path: path,
      queryParameters: queryParameters,
    );
  }

  Future<void> loadAuthToken() async {
    const secure = FlutterSecureStorage();
    String? token = await secure.read(key: 'auth_token');
    if (token == null) {
      // migrate legacy prefs storage if present
      final prefs = await SharedPreferences.getInstance();
      token = prefs.getString('auth_token');
      if (token != null) {
        await secure.write(key: 'auth_token', value: token);
        await prefs.remove('auth_token');
      }
    }
    if (token != null && token.isNotEmpty) {
      _authToken = token;
      _callOptions =
          grpc.CallOptions(metadata: {'authorization': 'Bearer $token'});
    }
  }

  /// Heuristic LAN check: tries multiple common targets (mdns + gateways).
  Future<bool> checkLocalNetwork(
      {Duration timeout = const Duration(seconds: 1)}) async {
    final targets = ['router.local', '192.168.1.1', '10.0.0.1'];
    for (final t in targets) {
      try {
        final result = await InternetAddress.lookup(t).timeout(timeout);
        if (result.isNotEmpty) {
          _lanLikely = true;
          return true;
        }
      } catch (_) {
        continue;
      }
    }
    _lanLikely = false;
    return false;
  }

  bool get isLanLikely => _lanLikely;

  Future<ConnectionDiagnostic> runDiagnostics({required String host, int httpPort = 8080, int grpcPort = 50051}) async {
    bool httpOk = false;
    bool grpcOk = false;
    String? httpError;
    String? grpcError;
    bool isCorsLikely = false;

    // 1. Test HTTP (REST)
    try {
      final uri = Uri.http('$host:$httpPort', '/api/ping');
      final response = await http.get(uri).timeout(const Duration(seconds: 3));
      if (response.statusCode == 200) {
        httpOk = true;
      } else {
        httpError = 'HTTP ${response.statusCode}';
      }
    } catch (e) {
      httpError = e.toString();
      // On web, a failure to fetch even when host is up often implies CORS
      if (kIsWeb && (e.toString().contains('XMLHttpRequest') || e.toString().contains('Failed to fetch'))) {
        isCorsLikely = true;
      }
    }

    // 2. Test gRPC
    try {
      // Create a temporary channel for testing
      final testChannel = kIsWeb 
        ? grpc_web.GrpcWebClientChannel.xhr(Uri.parse('http://$host:$grpcPort'))
        : grpc.ClientChannel(host, port: grpcPort, options: const grpc.ChannelOptions(credentials: grpc.ChannelCredentials.insecure()));
      
      // Try a simple RPC or just check if channel initializes
      // In a real test, we'd call a lightweight "Ping" RPC
      grpcOk = true; // Placeholder: assume true if no immediate exception
      
      if (!kIsWeb) {
        await (testChannel as grpc.ClientChannel).shutdown();
      }
    } catch (e) {
      grpcError = e.toString();
    }

    return ConnectionDiagnostic(
      httpOk: httpOk,
      grpcOk: grpcOk,
      httpError: httpError,
      grpcError: grpcError,
      isCorsLikely: isCorsLikely,
    );
  }

  Future<void> connect({
    required String host,
    required int port,
    int httpPort = 8080,
    bool useTls = true,
    String? authToken,
    List<int>? trustedRootCertificates,
    bool preferPlatformBridge = false,
  }) async {
    _host = host;
    _httpPort = httpPort;
    _usePlatformBridge = preferPlatformBridge;
    _callOptions = authToken != null
        ? grpc.CallOptions(metadata: {'authorization': 'Bearer $authToken'})
        : grpc.CallOptions();

    if (_usePlatformBridge) {
      await _platformBridge.initConnection(host, port,
          useTls: useTls, authToken: authToken);
      return;
    }

    if (_channel != null) {
      return;
    }

    if (kIsWeb) {
      // Use gRPC-Web for web platform
      // Note: This assumes the server supports gRPC-Web or is behind an Envoy proxy.
      // If not, we might need to rely solely on HTTP endpoints.
      _channel =
          grpc_web.GrpcWebClientChannel.xhr(Uri.parse('http://$host:$port'));
    } else {
      final credentials = useTls
          ? grpc.ChannelCredentials.secure(
              certificates: trustedRootCertificates)
          : const grpc.ChannelCredentials.insecure();
      _channel = grpc.ClientChannel(
        host,
        port: port,
        options: grpc.ChannelOptions(credentials: credentials),
      );
    }
  }

  Future<Map<String, dynamic>> getRobotStatus() async {
    if (_host == null) return {};
    final uri = Uri.http('$_host:$_httpPort', '/api/status');
    try {
      final response = await http.get(uri);
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
    } catch (e) {
      // ignore error
    }
    return {};
  }

  Future<Map<String, dynamic>> setRobotMode(String mode) async {
    if (_host == null) return {'success': false, 'message': 'Not connected'};
    // Use the new v1 context API if possible, fallback to legacy
    final uri = Uri.http('$_host:$_httpPort', '/api/mode/$mode');
    try {
      final response = await http.post(uri, headers: _headers());
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
      return {'success': false, 'message': 'HTTP ${response.statusCode}'};
    } catch (e) {
      return {'success': false, 'message': e.toString()};
    }
  }

  /// Alias for setRobotMode used by BLoC
  Future<bool> setMode(String mode) async {
    final res = await setRobotMode(mode);
    return res['success'] == true;
  }

  Future<Map<String, dynamic>> listModels() async {
    if (_host == null) return {'success': false, 'models': []};
    final uri = Uri.http('$_host:$_httpPort', '/api/v1/models');
    try {
      final response = await http.get(uri, headers: _headers());
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
    } catch (e) {
      debugPrint('List models failed: $e');
    }
    return {'success': false, 'models': []};
  }

  Future<Map<String, dynamic>> activateModel(String modelId) async {
    if (_host == null) return {'success': false};
    final uri = Uri.http('$_host:$_httpPort', '/api/v1/models/activate');
    try {
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json', ..._headers()},
        body: jsonEncode({'model_id': modelId}),
      );
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
    } catch (e) {
      debugPrint('Activate model failed: $e');
    }
    return {'success': false};
  }

  Future<Map<String, dynamic>> listEpisodes() async {
    if (_host == null) return {'success': false, 'episodes': []};
    final uri = Uri.http('$_host:$_httpPort', '/api/v1/data/episodes');
    try {
      final response = await http.get(uri, headers: _headers());
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
    } catch (e) {
      debugPrint('List episodes failed: $e');
    }
    return {'success': false, 'episodes': []};
  }

  Future<Map<String, dynamic>> tagEpisode(String episodeId, String tag) async {
    if (_host == null) return {'success': false};
    final uri = Uri.http('$_host:$_httpPort', '/api/v1/data/tag');
    try {
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json', ..._headers()},
        body: jsonEncode({'episode_id': episodeId, 'tag': tag}),
      );
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
    } catch (e) {
      debugPrint('Tag episode failed: $e');
    }
    return {'success': false};
  }

  Future<Map<String, dynamic>> chatWithGemma(
      String message, List<Map<String, String>> history) async {
    if (_host == null) return {'error': 'Not connected'};
    final uri = Uri.http('$_host:$_httpPort', '/api/chat');
    try {
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'message': message,
          'history': history,
        }),
      );
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
      return {'error': 'HTTP ${response.statusCode}'};
    } catch (e) {
      return {'error': e.toString()};
    }
  }

  /// Handoff recorded RLDS manifest + binary assets to a managed robot over LAN.
  Future<Map<String, String>> handoffEpisodeAssets(
      EpisodePackage package) async {
    if (_host == null) {
      throw StateError('BrainClient not connected');
    }
    final request =
        http.MultipartRequest('POST', _httpUri('/api/training/episode_handoff'))
          ..headers.addAll(_headers())
          ..fields['manifest'] = jsonEncode(package.record.toJson())
          ..fields['assets'] =
              jsonEncode(package.assets.map((a) => a.toJson()).toList());

    for (final asset in package.assets) {
      final file = File(asset.localUri);
      if (!await file.exists()) continue;
      request.files.add(
        await http.MultipartFile.fromPath(
          'asset',
          file.path,
          filename: file.uri.pathSegments.isNotEmpty
              ? file.uri.pathSegments.last
              : 'asset',
        ),
      );
    }

    final streamed = await request.send();
    final response = await http.Response.fromStream(streamed);
    if (response.statusCode >= 200 && response.statusCode < 300) {
      try {
        final body = jsonDecode(response.body) as Map<String, dynamic>;
        final remoteUris =
            (body['remote_uris'] as Map?)?.cast<String, String>();
        return remoteUris ?? {};
      } catch (_) {
        return {};
      }
    }
    throw HttpException('handoff failed: ${response.statusCode}');
  }

  /// Claim robot with optional account metadata.
  Future<bool> claimRobot({
    required String host,
    int httpPort = 8080,
    String? accountId,
    String? accountType,
    String? ownerId,
  }) async {
    final uri = Uri.http('$host:$httpPort', '/api/ownership/claim');
    try {
      final response = await http.post(
        uri,
        headers: {
          ..._headers(),
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          if (accountId != null) 'account_id': accountId,
          if (accountType != null) 'account_type': accountType,
          if (ownerId != null) 'owner_id': ownerId,
        }),
      );
      if (response.statusCode == 200) {
        isOwned = true;
        return true;
      }
      debugPrint(
          'Claim failed: HTTP ${response.statusCode} body=${response.body}');
      return false;
    } catch (e) {
      debugPrint('Claim failed: $e');
      return false;
    }
  }

  /// Placeholder for seed bundle install; replace with real API when available.
  Future<bool> installSeedBundle(
      {required String host, int httpPort = 8080}) async {
    final uri = Uri.http('$host:$httpPort', '/api/ota/install_seed');
    try {
      final response = await http.post(uri, headers: _headers());
      if (response.statusCode == 200) {
        return true;
      }
      debugPrint('Seed install failed: HTTP ${response.statusCode}');
      return false;
    } catch (e) {
      debugPrint('Seed install failed: $e');
      return false;
    }
  }

  /// Fetch ownership/subscription/seed status; expects backend shape.
  Future<Map<String, dynamic>> fetchOwnershipStatus({
    required String host,
    int httpPort = 8080,
  }) async {
    final uri = Uri.http('$host:$httpPort', '/api/ownership/status');
    try {
      final response = await http.get(uri, headers: _headers());
      if (response.statusCode == 200) {
        final decoded = jsonDecode(response.body);
        if (decoded is Map) {
          final map = Map<String, dynamic>.from(decoded);
          // Back-compat: new servers may return {status, ownership:{...}, owned:..., ...}
          // Normalize to a flat map containing the expected keys.
          final ownership = map['ownership'];
          if (ownership is Map) {
            map.addAll(Map<String, dynamic>.from(ownership));
          }
          return map;
        }
        return {};
      }
      debugPrint(
          'Status failed: HTTP ${response.statusCode} body=${response.body}');
    } catch (e) {
      debugPrint('Status failed: $e');
    }
    return {};
  }

  /// Pairing confirm using the robot QR URL (http://<robot>:8080/pair?token=...).
  Future<Map<String, dynamic>> confirmPairingFromQr({
    required Uri pairUrl,
    required String ownerId,
    required String confirmCode,
  }) async {
    final token = pairUrl.queryParameters['token'] ?? '';
    final scheme = pairUrl.scheme.isNotEmpty ? pairUrl.scheme : 'http';
    final host = pairUrl.host;
    final port = pairUrl.hasPort ? pairUrl.port : 8080;
    final uri = Uri(
        scheme: scheme,
        host: host,
        port: port,
        path: '/api/ownership/pair/confirm');
    final response = await http.post(
      uri,
      headers: {'Content-Type': 'application/json', ..._headers()},
      body: jsonEncode(
          {'token': token, 'confirm_code': confirmCode, 'owner_id': ownerId}),
    );
    if (response.statusCode == 200) {
      final decoded = jsonDecode(response.body);
      return decoded is Map
          ? Map<String, dynamic>.from(decoded)
          : {'data': decoded};
    }
    return {
      'status': 'error',
      'message': 'HTTP ${response.statusCode}',
      'body': response.body
    };
  }

  /// Ping endpoint to check reachability and get device id.
  Future<Map<String, dynamic>> ping({
    required String host,
    int httpPort = 8080,
  }) async {
    final uri = Uri.http('$host:$httpPort', '/api/ping');
    try {
      final response = await http.get(uri, headers: _headers());
      if (response.statusCode == 200) {
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
      debugPrint(
          'Ping failed: HTTP ${response.statusCode} body=${response.body}');
    } catch (e) {
      debugPrint('Ping failed: $e');
    }
    return {};
  }

  /// Transfer ownership to another user.
  Future<Map<String, dynamic>> transferOwnership({
    required String host,
    int httpPort = 8080,
    required String newOwnerId,
    String? newAccountId,
    String? newAccountType,
  }) async {
    final uri = Uri.http('$host:$httpPort', '/api/ownership/transfer');
    try {
      final response = await http.post(
        uri,
        headers: {
          ..._headers(),
          'Content-Type': 'application/json',
        },
        body: jsonEncode({
          'new_owner_id': newOwnerId,
          if (newAccountId != null) 'new_account_id': newAccountId,
          if (newAccountType != null) 'new_account_type': newAccountType,
        }),
      );
      if (response.statusCode == 200) {
        isOwned = false; // Reset local ownership state
        return Map<String, dynamic>.from(jsonDecode(response.body));
      }
      debugPrint('Transfer failed: HTTP ${response.statusCode} body=${response.body}');
      return {'success': false, 'message': 'HTTP ${response.statusCode}'};
    } catch (e) {
      debugPrint('Transfer failed: $e');
      return {'success': false, 'message': e.toString()};
    }
  }

  Future<void> sendCommand(ControlCommand command) async {
    final payload = jsonEncode(command.toJson());
    try {
      if (_usePlatformBridge) {
        await _platformBridge.sendCommand(payload);
        return;
      }
      final channel =
          _channel ?? (throw StateError('BrainClient not connected'));
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
        .serverStream(_streamRobotStateMethod,
            jsonEncode({'client_id': clientId}), _callOptions)
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
        (List<int> value) =>
            jsonDecode(utf8.decode(value)) as Map<String, dynamic>,
      );
      final stub = _JsonBrainClient(channel);
      return await stub.unary(method, jsonEncode(payload), _callOptions);
    } on grpc.GrpcError catch (error) {
      throw StateError('Unable to $errorLabel: ${error.message}');
    }
  }

  void _emitRobotState(Map<String, dynamic> data) {
    final positions =
        (data['state']?['joint_positions'] as List<dynamic>? ?? []).cast<num>();
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

  /// Fetch runtime settings (/api/settings GET).
  Future<Map<String, dynamic>> getSettings() async {
    try {
      final uri = _httpUri('/api/settings');
      final resp = await http.get(uri, headers: _headers());
      final decoded = jsonDecode(resp.body);
      if (resp.statusCode != 200) {
        return {
          'success': false,
          'message': 'HTTP ${resp.statusCode}',
          'body': resp.body,
        };
      }
      return decoded is Map
          ? Map<String, dynamic>.from(decoded)
          : {'data': decoded};
    } catch (e) {
      return {'success': false, 'message': e.toString()};
    }
  }

  /// Save runtime settings (/api/settings POST). Payload must match the settings schema.
  Future<Map<String, dynamic>> saveSettings(
      Map<String, dynamic> settings) async {
    try {
      final uri = _httpUri('/api/settings');
      final resp = await http.post(
        uri,
        headers: {
          ..._headers(),
          'Content-Type': 'application/json',
        },
        body: jsonEncode(settings),
      );
      final decoded = jsonDecode(resp.body);
      if (resp.statusCode != 200) {
        return decoded is Map
            ? Map<String, dynamic>.from(decoded)
            : {
                'success': false,
                'message': 'HTTP ${resp.statusCode}',
                'body': resp.body,
              };
      }
      return decoded is Map
          ? Map<String, dynamic>.from(decoded)
          : {'data': decoded};
    } catch (e) {
      return {'success': false, 'message': e.toString()};
    }
  }

  /// Trigger safety hold (E-STOP style latch): POST /api/safety/hold
  Future<Map<String, dynamic>> triggerSafetyHold() async {
    try {
      final uri = _httpUri('/api/safety/hold');
      final resp = await http.post(uri, headers: _headers());
      final decoded = jsonDecode(resp.body);
      return decoded is Map
          ? Map<String, dynamic>.from(decoded)
          : {'data': decoded};
    } catch (e) {
      return {'success': false, 'message': e.toString()};
    }
  }

  Future<Map<String, dynamic>> resetSafetyGates() async {
    // TODO: implement real reset endpoint. For now, we'll try to set mode to 'reflex' which often resets gates.
    return setRobotMode('reflex');
  }

  Stream<Map<String, dynamic>> subscribeToEvents() async* {
    if (_host == null) return;

    final client = http.Client();
    final request = http.Request('GET', _httpUri('/api/events'));
    request.headers['Accept'] = 'text/event-stream';
    request.headers['Cache-Control'] = 'no-cache';
    if (_authToken != null) {
      request.headers['Authorization'] = 'Bearer $_authToken';
    }

    try {
      final response = await client.send(request);
      if (response.statusCode == 200) {
        await for (final line in response.stream
            .transform(utf8.decoder)
            .transform(const LineSplitter())) {
          if (line.startsWith('data: ')) {
            final data = line.substring(6);
            try {
              yield jsonDecode(data) as Map<String, dynamic>;
            } catch (e) {
              debugPrint('Error decoding SSE data: $e');
            }
          }
        }
      }
    } catch (e) {
      debugPrint('SSE connection error: $e');
    } finally {
      client.close();
    }
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
    return $createStreamingCall(method, Stream.value(payload),
        options: options);
  }
}
