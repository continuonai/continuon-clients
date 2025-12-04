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
      _addRobot(
        ScannedRobot(
          name: 'Manual host',
          host: manualHost,
          port: 8080,
          httpPort: 8080,
        ),
      );
    }
    _pollTimer = Timer.periodic(const Duration(seconds: 12), (_) {
      _probeHosts(candidateHosts);
    });
  }

  Future<void> _probeHosts(Set<String> hosts) async {
    for (final host in hosts) {
      await _probeHost(host);
    }
  }

  Future<void> _probeHost(String host) async {
    try {
      final statusUrl = Uri.http('$host:8080', '/status');
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

  ScannedRobot? _parseRobotFromStatus(
      String host, Map<String, dynamic> payload) {
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

    // Common fallback for devices advertising a .local mDNS hostname.
    hosts.add('continuonbot.local');

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
