/// RCAN (Robot Communication & Addressing Network) Client
/// 
/// Implements the RCAN protocol for Flutter apps to discover, authenticate,
/// and communicate with robotic agents.
/// 
/// See: docs/rcan-protocol.md, docs/rcan-technical-spec.md
library rcan_client;

import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:http/http.dart' as http;
import 'package:flutter_secure_storage/flutter_secure_storage.dart';

import '../models/user_role.dart';

/// RCAN Message types
enum RCANMessageType {
  discover,
  status,
  command,
  stream,
  event,
  handoff,
  ack,
  error,
}

/// RCAN Message priority
enum RCANPriority {
  low,
  normal,
  high,
  safety,
}

/// Robot URI identity
class RCANIdentity {
  final String registry;
  final String manufacturer;
  final String model;
  final String deviceId;
  final int port;
  final String? capability;

  RCANIdentity({
    required this.registry,
    required this.manufacturer,
    required this.model,
    required this.deviceId,
    this.port = 8080,
    this.capability,
  });

  /// Parse RURI string into identity
  factory RCANIdentity.fromRuri(String ruri) {
    if (!ruri.startsWith('rcan://')) {
      throw ArgumentError('Invalid RURI: $ruri');
    }
    final parts = ruri.substring(7).split('/');
    if (parts.length < 4) {
      throw ArgumentError('Invalid RURI format: $ruri');
    }
    
    final deviceWithPort = parts[3].split(':');
    return RCANIdentity(
      registry: parts[0],
      manufacturer: parts[1],
      model: parts[2],
      deviceId: deviceWithPort[0],
      port: deviceWithPort.length > 1 ? int.tryParse(deviceWithPort[1]) ?? 8080 : 8080,
      capability: parts.length > 4 ? '/${parts.sublist(4).join('/')}' : null,
    );
  }

  /// Get the full Robot URI
  String get ruri {
    var s = 'rcan://$registry/$manufacturer/$model/$deviceId';
    if (port != 8080) s += ':$port';
    if (capability != null) s += capability!;
    return s;
  }

  @override
  String toString() => ruri;
}

/// RCAN protocol message
class RCANMessage {
  final String version;
  final String messageId;
  final String sourceRuri;
  final String targetRuri;
  final RCANMessageType type;
  final Map<String, dynamic> payload;
  final int timestampMs;
  final int ttlMs;
  final RCANPriority priority;
  final String? authToken;

  RCANMessage({
    this.version = '1.0.0',
    String? messageId,
    this.sourceRuri = '',
    this.targetRuri = '',
    this.type = RCANMessageType.status,
    Map<String, dynamic>? payload,
    int? timestampMs,
    this.ttlMs = 30000,
    this.priority = RCANPriority.normal,
    this.authToken,
  })  : messageId = messageId ?? _generateUuid(),
        payload = payload ?? {},
        timestampMs = timestampMs ?? DateTime.now().millisecondsSinceEpoch;

  static String _generateUuid() {
    // Simple UUID generation for messages
    final random = DateTime.now().microsecondsSinceEpoch;
    return random.toRadixString(16).padLeft(16, '0');
  }

  Map<String, dynamic> toJson() => {
        'version': version,
        'message_id': messageId,
        'source_ruri': sourceRuri,
        'target_ruri': targetRuri,
        'message_type': type.name.toUpperCase(),
        'payload': payload,
        'timestamp_ms': timestampMs,
        'ttl_ms': ttlMs,
        'priority': priority.name.toUpperCase(),
        if (authToken != null) 'auth_token': authToken,
      };

  factory RCANMessage.fromJson(Map<String, dynamic> json) {
    return RCANMessage(
      version: json['version'] ?? '1.0.0',
      messageId: json['message_id'],
      sourceRuri: json['source_ruri'] ?? '',
      targetRuri: json['target_ruri'] ?? '',
      type: RCANMessageType.values.firstWhere(
        (t) => t.name.toUpperCase() == (json['message_type'] ?? '').toString().toUpperCase(),
        orElse: () => RCANMessageType.status,
      ),
      payload: Map<String, dynamic>.from(json['payload'] ?? {}),
      timestampMs: json['timestamp_ms'],
      ttlMs: json['ttl_ms'] ?? 30000,
      priority: RCANPriority.values.firstWhere(
        (p) => p.name.toUpperCase() == (json['priority'] ?? '').toString().toUpperCase(),
        orElse: () => RCANPriority.normal,
      ),
      authToken: json['auth_token'],
    );
  }
}

/// RCAN authentication session
class RCANSession {
  final String sessionId;
  final String userId;
  final UserRole role;
  final String sourceRuri;
  final String targetRuri;
  final DateTime createdAt;
  final DateTime? expiresAt;
  final List<RobotCapability> capabilities;

  RCANSession({
    required this.sessionId,
    required this.userId,
    required this.role,
    required this.sourceRuri,
    required this.targetRuri,
    DateTime? createdAt,
    this.expiresAt,
    List<RobotCapability>? capabilities,
  })  : createdAt = createdAt ?? DateTime.now(),
        capabilities = capabilities ?? [];

  bool get isExpired {
    if (expiresAt == null) return false;
    return DateTime.now().isAfter(expiresAt!);
  }

  factory RCANSession.fromJson(Map<String, dynamic> json) {
    return RCANSession(
      sessionId: json['session_id'] ?? '',
      userId: json['user_id'] ?? '',
      role: UserRole.fromString(json['role'] ?? 'guest'),
      sourceRuri: json['source_ruri'] ?? '',
      targetRuri: json['target_ruri'] ?? '',
      expiresAt: json['expires_at'] != null
          ? DateTime.tryParse(json['expires_at'])
          : null,
      capabilities: (json['capabilities'] as List<dynamic>?)
              ?.map((c) => RobotCapability.values.firstWhere(
                    (cap) => cap.name == c.toString().toLowerCase(),
                    orElse: () => RobotCapability.viewStatus,
                  ))
              .toList() ??
          [],
    );
  }
}

/// RCAN command result
class RCANCommandResult {
  final String commandId;
  final bool success;
  final Map<String, dynamic>? result;
  final String? error;
  final int? executionTimeMs;

  RCANCommandResult({
    required this.commandId,
    required this.success,
    this.result,
    this.error,
    this.executionTimeMs,
  });

  factory RCANCommandResult.fromJson(Map<String, dynamic> json) {
    return RCANCommandResult(
      commandId: json['command'] ?? json['command_id'] ?? '',
      success: json['status'] == 'ok' || json['message_type'] == 'ACK',
      result: json['result'] as Map<String, dynamic>?,
      error: json['error'],
      executionTimeMs: json['execution_time_ms'],
    );
  }
}

/// Discovered robot via RCAN
class DiscoveredRobot {
  final String ruri;
  final String model;
  final String friendlyName;
  final List<String> capabilities;
  final List<String> supportedRoles;
  final String protocolVersion;
  final String host;
  final int port;
  final DateTime discoveredAt;

  DiscoveredRobot({
    required this.ruri,
    required this.model,
    this.friendlyName = '',
    this.capabilities = const [],
    this.supportedRoles = const [],
    this.protocolVersion = '1.0.0',
    required this.host,
    this.port = 8080,
    DateTime? discoveredAt,
  }) : discoveredAt = discoveredAt ?? DateTime.now();

  factory DiscoveredRobot.fromJson(Map<String, dynamic> json, {String? host, int? port}) {
    return DiscoveredRobot(
      ruri: json['ruri'] ?? '',
      model: json['model'] ?? '',
      friendlyName: json['friendly_name'] ?? json['robot_name'] ?? '',
      capabilities: List<String>.from(json['caps'] ?? json['capabilities'] ?? []),
      supportedRoles: List<String>.from(json['roles'] ?? json['supported_roles'] ?? []),
      protocolVersion: json['version'] ?? json['protocol_version'] ?? '1.0.0',
      host: host ?? json['hostname'] ?? 'localhost',
      port: port ?? json['port'] ?? 8080,
    );
  }

  /// Convert to a map for UI compatibility
  Map<String, dynamic> toMap() {
    return {
      'id': ruri,
      'name': friendlyName.isNotEmpty ? friendlyName : model,
      'host': host,
      'port': port,
      'ruri': ruri,
      'model': model,
      'capabilities': capabilities,
      'supportedRoles': supportedRoles,
    };
  }
}

/// RCAN Client for Flutter apps
/// 
/// Handles discovery, authentication, and communication with robots
/// using the RCAN protocol.
class RCANClient {
  final String clientId;
  final FlutterSecureStorage _secureStorage;
  
  String? _currentHost;
  int _currentPort = 8080;
  RCANSession? _session;
  
  // Discovered robots cache
  final List<DiscoveredRobot> _discoveredRobots = [];
  final StreamController<List<DiscoveredRobot>> _robotsController =
      StreamController<List<DiscoveredRobot>>.broadcast();

  RCANClient({
    this.clientId = 'continuonai-flutter',
    FlutterSecureStorage? secureStorage,
  }) : _secureStorage = secureStorage ?? const FlutterSecureStorage();

  /// Stream of discovered robots
  Stream<List<DiscoveredRobot>> get discoveredRobots => _robotsController.stream;

  /// Current session (if authenticated)
  RCANSession? get session => _session;

  /// Whether we have an active session
  bool get isAuthenticated => _session != null && !_session!.isExpired;

  /// Current connected host
  String? get currentHost => _currentHost;

  /// Current connected port
  int get currentPort => _currentPort;

  /// Build HTTP URI for RCAN endpoints
  Uri _rcanUri(String path, {Map<String, String>? queryParameters}) {
    if (_currentHost == null) {
      throw StateError('RCANClient not connected');
    }
    return Uri(
      scheme: 'http',
      host: _currentHost,
      port: _currentPort,
      path: path,
      queryParameters: queryParameters,
    );
  }

  /// Build headers with auth token if available
  Map<String, String> _headers() {
    final headers = <String, String>{
      'Content-Type': 'application/json',
    };
    if (_session?.sessionId != null) {
      headers['X-RCAN-Session'] = _session!.sessionId;
    }
    return headers;
  }

  /// Connect to a robot at the specified host
  Future<void> connect({required String host, int port = 8080}) async {
    _currentHost = host;
    _currentPort = port;
  }

  /// Disconnect from current robot
  void disconnect() {
    _currentHost = null;
    _session = null;
  }

  /// Discover robots on the network via HTTP probe
  /// 
  /// Unlike mDNS discovery, this probes known hosts directly.
  Future<List<DiscoveredRobot>> discover({
    Duration timeout = const Duration(seconds: 5),
    List<String>? hosts,
  }) async {
    _discoveredRobots.clear();
    
    // Default hosts to probe
    final hostsToProbe = hosts ?? [
      'localhost',
      'brain.local',
      'continuon.local',
      '192.168.1.1',
    ];

    // Probe each host in parallel
    final futures = hostsToProbe.map((host) => _probeHost(host, timeout: timeout));
    final results = await Future.wait(futures);
    
    for (final robot in results) {
      if (robot != null) {
        _discoveredRobots.add(robot);
      }
    }
    
    _robotsController.add(List.from(_discoveredRobots));
    return _discoveredRobots;
  }

  /// Probe a single host for RCAN support
  Future<DiscoveredRobot?> _probeHost(String host, {Duration timeout = const Duration(seconds: 3), int port = 8080}) async {
    try {
      final uri = Uri.http('$host:$port', '/rcan/v1/status');
      final response = await http.get(uri).timeout(timeout);
      
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final discovery = data['discovery'] as Map<String, dynamic>?;
        return DiscoveredRobot.fromJson(discovery ?? data, host: host, port: port);
      }
    } catch (e) {
      debugPrint('RCAN probe failed for $host: $e');
    }
    return null;
  }

  /// Get RCAN status from connected robot
  Future<Map<String, dynamic>> getStatus() async {
    final uri = _rcanUri('/rcan/v1/status');
    try {
      final response = await http.get(uri, headers: _headers());
      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
    } catch (e) {
      debugPrint('RCAN status failed: $e');
    }
    return {};
  }

  /// Claim control of the robot
  Future<RCANSession?> claim({
    required String userId,
    required UserRole role,
    String? sourceRuri,
  }) async {
    final uri = _rcanUri('/rcan/v1/auth/claim');
    try {
      final body = jsonEncode({
        'user_id': userId,
        'role': role.name,
        'source_ruri': sourceRuri ?? 'rcan://local/$clientId',
      });
      
      final response = await http.post(uri, headers: _headers(), body: body);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final payload = data['payload'] as Map<String, dynamic>?;
        
        if (data['message_type'] == 'GRANTED' && payload != null) {
          _session = RCANSession(
            sessionId: payload['session_id'] ?? '',
            userId: userId,
            role: role,
            sourceRuri: sourceRuri ?? 'rcan://local/$clientId',
            targetRuri: data['source_ruri'] ?? '',
            expiresAt: payload['expires_at'] != null
                ? DateTime.tryParse(payload['expires_at'])
                : null,
          );
          
          // Store session for persistence
          await _secureStorage.write(
            key: 'rcan_session',
            value: jsonEncode({
              'session_id': _session!.sessionId,
              'user_id': userId,
              'role': role.name,
              'target_ruri': _session!.targetRuri,
              'expires_at': _session!.expiresAt?.toIso8601String(),
            }),
          );
          
          return _session;
        }
      }
    } catch (e) {
      debugPrint('RCAN claim failed: $e');
    }
    return null;
  }

  /// Release control of the robot
  Future<bool> release() async {
    if (_session == null) return true;
    
    final uri = _rcanUri('/rcan/v1/auth/release');
    try {
      final body = jsonEncode({'session_id': _session!.sessionId});
      final response = await http.delete(uri, headers: _headers(), body: body);
      
      if (response.statusCode == 200) {
        _session = null;
        await _secureStorage.delete(key: 'rcan_session');
        return true;
      }
    } catch (e) {
      debugPrint('RCAN release failed: $e');
    }
    return false;
  }

  /// Send a command to the robot
  Future<RCANCommandResult> sendCommand({
    required String command,
    Map<String, dynamic>? parameters,
    RCANPriority priority = RCANPriority.normal,
    int? timeoutMs,
  }) async {
    if (!isAuthenticated) {
      return RCANCommandResult(
        commandId: '',
        success: false,
        error: 'Not authenticated',
      );
    }
    
    final uri = _rcanUri('/rcan/v1/command');
    try {
      final message = RCANMessage(
        type: RCANMessageType.command,
        payload: {
          'command': command,
          if (parameters != null) ...parameters,
        },
        priority: priority,
        ttlMs: timeoutMs ?? 30000,
      );
      
      final body = jsonEncode({
        'session_id': _session!.sessionId,
        'message': message.toJson(),
      });
      
      final response = await http.post(uri, headers: _headers(), body: body);
      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return RCANCommandResult.fromJson(data);
      }
      
      return RCANCommandResult(
        commandId: message.messageId,
        success: false,
        error: 'HTTP ${response.statusCode}',
      );
    } catch (e) {
      return RCANCommandResult(
        commandId: '',
        success: false,
        error: e.toString(),
      );
    }
  }

  /// Check if user has permission for a capability
  bool canPerform(RobotCapability capability) {
    if (!isAuthenticated) return false;
    return _session!.role.canPerform(capability);
  }

  /// Load persisted session
  Future<void> loadSession() async {
    try {
      final sessionJson = await _secureStorage.read(key: 'rcan_session');
      if (sessionJson != null) {
        final data = jsonDecode(sessionJson) as Map<String, dynamic>;
        _session = RCANSession(
          sessionId: data['session_id'] ?? '',
          userId: data['user_id'] ?? '',
          role: UserRole.fromString(data['role'] ?? 'guest'),
          sourceRuri: 'rcan://local/$clientId',
          targetRuri: data['target_ruri'] ?? '',
          expiresAt: data['expires_at'] != null
              ? DateTime.tryParse(data['expires_at'])
              : null,
        );
        
        // Clear if expired
        if (_session!.isExpired) {
          _session = null;
          await _secureStorage.delete(key: 'rcan_session');
        }
      }
    } catch (e) {
      debugPrint('Failed to load RCAN session: $e');
    }
  }

  /// Dispose resources
  void dispose() {
    _robotsController.close();
  }
}

