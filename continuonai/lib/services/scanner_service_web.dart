import 'dart:async';
import 'dart:convert';
// ignore: avoid_web_libraries_in_flutter, deprecated_member_use
import 'dart:html' as html;

class ScannedRobot {
  final String name;
  final String host;
  final int port;
  final int httpPort;
  final bool isBle;
  final String? bleId;

  ScannedRobot({
    required this.name,
    required this.host,
    required this.port,
    this.httpPort = 8080,
    this.isBle = false,
    this.bleId,
  });

  @override
  bool operator ==(Object other) =>
      identical(this, other) ||
      other is ScannedRobot &&
          runtimeType == other.runtimeType &&
          host == other.host &&
          port == other.port &&
          bleId == other.bleId;

  @override
  int get hashCode => host.hashCode ^ port.hashCode ^ bleId.hashCode;
}

class ScannerService {
  final StreamController<List<ScannedRobot>> _robotsController =
      StreamController<List<ScannedRobot>>.broadcast();
  Stream<List<ScannedRobot>> get scannedRobots => _robotsController.stream;

  final List<ScannedRobot> _foundRobots = [];
  Timer? _pollTimer;
  bool _isScanning = false;

  Future<void> startScan(
      {String? manualHost, bool forceRestart = false}) async {
    if (_isScanning && !forceRestart) return;
    if (forceRestart) {
      await stopScan();
    }

    _isScanning = true;
    _foundRobots.clear();
    _robotsController.add([]);

    final candidateHosts = _buildCandidateHosts(manualHost);
    if (candidateHosts.isEmpty) {
      _robotsController.add([]);
      return;
    }

    await _probeHosts(candidateHosts);
    if (_foundRobots.isEmpty && manualHost != null && manualHost.isNotEmpty) {
      // Add manual host with default ports if probe didn't find it
      _addRobot(
        ScannedRobot(
          name: 'Manual host',
          host: manualHost,
          port: 50051, // Default gRPC port
          httpPort: 8080, // Default HTTP port
        ),
      );
    }
    _pollTimer = Timer.periodic(const Duration(seconds: 12), (_) {
      _probeHosts(candidateHosts);
    });
  }

  Future<void> _probeHosts(Set<String> hosts) async {
    for (final host in hosts) {
      // Probe both common ports (8080 and 8081)
      await _probeHost(host, port: 8081);
      await _probeHost(host, port: 8080);
    }
  }

  Future<void> _probeHost(String host, {int port = 8080}) async {
    // Try RCAN status endpoint first (preferred, RCAN protocol)
    try {
      final rcanUrl = Uri.http('$host:$port', '/rcan/v1/status');
      final rcanResponse = await html.HttpRequest.request(
        rcanUrl.toString(),
        method: 'GET',
        requestHeaders: {'Accept': 'application/json'},
      ).timeout(const Duration(seconds: 3));

      if (rcanResponse.status == 200 && rcanResponse.responseText != null) {
        final payload = jsonDecode(rcanResponse.responseText!);
        if (payload is Map<String, dynamic>) {
          final robot = _parseRobotFromRcan(host, payload, defaultPort: port);
          if (robot != null) {
            _addRobot(robot);
            html.window.localStorage['continuon_last_host'] = robot.host;
            html.window.localStorage['continuon_last_port'] = robot.httpPort.toString();
            // Store robot name and RURI for .local hostname support
            final discovery = payload['discovery'] as Map<String, dynamic>?;
            if (discovery != null && discovery['ruri'] != null) {
              html.window.localStorage['continuon_last_ruri'] = discovery['ruri'].toString();
            }
            return; // Success with RCAN endpoint
          }
        }
      }
    } catch (_) {
      // Fall through to discovery endpoint
    }
    
    // Try discovery endpoint (legacy, still preferred over status)
    try {
      final discoveryUrl = Uri.http('$host:$port', '/api/discovery/info');
      final discoveryResponse = await html.HttpRequest.request(
        discoveryUrl.toString(),
        method: 'GET',
        requestHeaders: {'Accept': 'application/json'},
      ).timeout(const Duration(seconds: 3));

      if (discoveryResponse.status == 200 && discoveryResponse.responseText != null) {
        final payload = jsonDecode(discoveryResponse.responseText!);
        if (payload is Map<String, dynamic>) {
          final robot = _parseRobotFromDiscovery(host, payload, defaultPort: port);
          if (robot != null) {
            _addRobot(robot);
            html.window.localStorage['continuon_last_host'] = robot.host;
            html.window.localStorage['continuon_last_port'] = robot.httpPort.toString();
            // Store robot name for .local hostname support
            if (payload['robot_name'] != null) {
              final robotName = payload['robot_name'].toString().toLowerCase();
              html.window.localStorage['continuon_last_robot_name'] = robotName;
            }
            return; // Success with discovery endpoint
          }
        }
      }
    } catch (_) {
      // Fall through to /status endpoint fallback
    }

    // Fallback to /status endpoint for backwards compatibility
    try {
      final statusUrl = Uri.http('$host:$port', '/api/status');
      final response = await html.HttpRequest.request(
        statusUrl.toString(),
        method: 'GET',
        requestHeaders: {'Accept': 'application/json'},
      ).timeout(const Duration(seconds: 3));

      if (response.status != 200 || response.responseText == null) {
        return;
      }

      final payload = jsonDecode(response.responseText!);
      if (payload is! Map<String, dynamic>) return;

      final robot = _parseRobotFromStatus(host, payload);
      if (robot != null) {
        _addRobot(robot);
        // Remember last good host for follow-up scans.
        html.window.localStorage['continuon_last_host'] = robot.host;
      }
    } catch (_) {
      // Swallow network/CORS errors: discovery should be best-effort on web.
    }
  }

  ScannedRobot? _parseRobotFromRcan(
      String host, Map<String, dynamic> payload, {int defaultPort = 8080}) {
    // Parse RCAN /rcan/v1/status endpoint response format
    final discovery = payload['discovery'] as Map<String, dynamic>?;
    if (discovery == null) return null;

    // Note: ruri is available in discovery for future use
    final model = discovery['model']?.toString() ?? 'companion-v1';
    final hostname = discovery['hostname']?.toString() ?? host;
    final port = discovery['port'] as int? ?? defaultPort;
    
    // Extract friendly name from model or RURI
    final name = model.isNotEmpty ? model : 'ContinuonBrain';
    
    return ScannedRobot(
      name: name,
      host: hostname,
      port: port,
      httpPort: port,
    );
  }

  ScannedRobot? _parseRobotFromDiscovery(
      String host, Map<String, dynamic> payload, {int defaultPort = 8080}) {
    // Parse discovery endpoint response format
    if (payload['status'] != 'ok' || payload['product'] != 'continuon_brain_runtime') {
      return null;
    }

    final name = (payload['robot_name'] ?? 'ContinuonBrain').toString();

    // Extract base_url and parse host/port from it
    String? baseUrl = payload['base_url']?.toString();
    String resolvedHost = host;
    int httpPort = defaultPort;
    
    if (baseUrl != null && baseUrl.isNotEmpty) {
      try {
        final uri = Uri.parse(baseUrl);
        if (uri.host.isNotEmpty) {
          resolvedHost = uri.host;
        }
        if (uri.port > 0) {
          httpPort = uri.port;
        }
      } catch (_) {
        // Fall back to provided host if URL parsing fails
      }
    }

    // Use provided host if base_url didn't yield a host
    if (resolvedHost.isEmpty || resolvedHost == host) {
      resolvedHost = host;
    }

    // Default GRPC port same as HTTP for now
    final grpcPort = httpPort;

    return ScannedRobot(
      name: name,
      host: resolvedHost,
      port: grpcPort,
      httpPort: httpPort,
    );
  }

  ScannedRobot? _parseRobotFromStatus(
      String host, Map<String, dynamic> payload) {
    // Parse legacy /status endpoint response format
    final name = (payload['robot_name'] ?? payload['name'] ?? 'ContinuonBrain')
        .toString();
    final httpPort = (payload['port'] is int)
        ? payload['port'] as int
        : int.tryParse('${payload['port']}') ?? 8080;
    final grpcPort = (payload['grpc_port'] is int)
        ? payload['grpc_port'] as int
        : int.tryParse('${payload['grpc_port']}') ?? httpPort;

    final resolvedHost =
        (payload['ip_address'] ?? payload['host'] ?? host).toString();
    if (resolvedHost.isEmpty) return null;

    return ScannedRobot(
      name: name,
      host: resolvedHost,
      port: grpcPort,
      httpPort: httpPort,
    );
  }

  Set<String> _buildCandidateHosts(String? manualHost) {
    final hosts = <String>{};

    if (manualHost != null && manualHost.isNotEmpty) {
      hosts.add(manualHost);
    }

    final storedHost = html.window.localStorage['continuon_last_host'];
    if (storedHost != null && storedHost.isNotEmpty) {
      hosts.add(storedHost);
    }

    // Always probe localhost for local development
    hosts.add('localhost');
    hosts.add('127.0.0.1');

    // Add .local hostname from last known robot name (from discovery response)
    final storedRobotName = html.window.localStorage['continuon_last_robot_name'];
    if (storedRobotName != null && storedRobotName.isNotEmpty) {
      // Try both the stored name and common defaults
      final cleanName = storedRobotName.replaceAll(RegExp(r'[^a-z0-9-]'), '');
      hosts.add('$cleanName.local');
      hosts.add('continuonbot.local');
    } else {
      // Common fallback for devices advertising a .local mDNS hostname.
      hosts.add('continuonbot.local');
    }

    final locationHost = html.window.location.hostname;
    if (locationHost != null &&
        locationHost.isNotEmpty &&
        locationHost != 'localhost') {
      hosts.add(locationHost);
    }

    if (locationHost != null && locationHost.isNotEmpty) {
      final prefix = _localPrefix(locationHost);
      if (prefix != null) {
        // Probe a small set of neighbors on the same subnet instead of scanning the full /24.
        for (final candidate in [2, 10, 20, 42, 99, 150, 200, 220, 250]) {
          hosts.add('$prefix$candidate');
        }
      }
    }

    return hosts;
  }

  String? _localPrefix(String host) {
    final parts = host.split('.');
    if (parts.length != 4) return null;
    final octets = parts.map(int.tryParse).toList();
    if (octets.any((o) => o == null)) return null;
    if (octets[0]! == 10 ||
        (octets[0]! == 192 && octets[1] == 168) ||
        (octets[0]! == 172 && octets[1]! >= 16 && octets[1]! <= 31)) {
      return '${octets[0]}.${octets[1]}.${octets[2]}.';
    }
    return null;
  }

  Future<void> stopScan() async {
    _isScanning = false;
    _pollTimer?.cancel();
    _pollTimer = null;
  }

  void _addRobot(ScannedRobot robot) {
    if (!_foundRobots.contains(robot)) {
      _foundRobots.add(robot);
      _robotsController.add(List.from(_foundRobots));
    }
  }

  void dispose() {
    stopScan();
    _robotsController.close();
  }
}
