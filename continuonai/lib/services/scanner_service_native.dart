import 'dart:async';
import 'dart:io';
import 'package:flutter/foundation.dart';
import 'package:nsd/nsd.dart';
import 'package:flutter_blue_plus/flutter_blue_plus.dart';
import 'package:permission_handler/permission_handler.dart' hide ServiceStatus;

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
  Discovery? _discovery;
  StreamSubscription? _bleSubscription;
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

    if (kIsWeb) {
      // Web scanning limitations
      debugPrint('Network scanning is limited on Web.');
      // We could try to guess local IP or just return empty
      return;
    }

    await _requestPermissions();

    // Start mDNS Scan
    try {
      // Scan for generic HTTP or specific Continuon service
      // Note: '_continuon._tcp' would be ideal if the robot broadcasts it.
      // For now, we might scan for _http._tcp and filter, or assume the user knows.
      // Let's try scanning for a specific service type if possible, or generic.
      _discovery = await startDiscovery('_continuon._tcp',
          ipLookupType: IpLookupType.v4);
      _discovery!.addServiceListener((service, status) {
        if (status == ServiceStatus.found) {
          _parseMdnsService(service);
        }
      });
    } catch (e) {
      debugPrint('mDNS Error: $e');
    }

    // Start BLE Scan
    try {
      // Check if BLE is supported
      if (await FlutterBluePlus.isSupported == false) {
        debugPrint("Bluetooth not supported by this device");
        return;
      }

      // Turn on Bluetooth if off (Android only)
      if (Platform.isAndroid) {
        await FlutterBluePlus.turnOn();
      }

      // Listen to scan results
      _bleSubscription = FlutterBluePlus.scanResults.listen((results) {
        for (ScanResult r in results) {
          _parseBleDevice(r);
        }
      });

      // Start scanning
      await FlutterBluePlus.startScan(timeout: const Duration(seconds: 15));
    } catch (e) {
      debugPrint('BLE Error: $e');
    }
  }

  Future<void> stopScan() async {
    _isScanning = false;

    if (kIsWeb) return;

    try {
      if (_discovery != null) {
        await stopDiscovery(_discovery!);
        _discovery = null;
      }
    } catch (e) {
      debugPrint('Error stopping mDNS: $e');
    }

    try {
      await FlutterBluePlus.stopScan();
      _bleSubscription?.cancel();
      _bleSubscription = null;
    } catch (e) {
      debugPrint('Error stopping BLE: $e');
    }
  }

  Future<void> _requestPermissions() async {
    if (Platform.isAndroid) {
      await [
        Permission.location,
        Permission.bluetoothScan,
        Permission.bluetoothConnect,
      ].request();
    }
  }

  void _parseMdnsService(Service service) {
    // Extract IP and Port
    final name = service.name ?? 'Unknown Robot';
    final host = service.host ?? '';
    final port = service.port ?? 8080;

    // Simple validation
    if (host.isEmpty) return;

    // Extract HTTP port from TXT records if available
    int httpPort = 8080;
    final txt = service.txt;
    if (txt != null) {
      if (txt.containsKey('http_port')) {
        final val = txt['http_port'];
        if (val != null) {
          httpPort = int.tryParse(String.fromCharCodes(val)) ?? 8080;
        }
      }
    }

    // If the main port is 8080 or 8081, it's likely the HTTP port
    if (port == 8080 || port == 8081) {
      httpPort = port;
    }

    final robot = ScannedRobot(
      name: name,
      host: host,
      port: port == httpPort ? 50051 : port, // Fallback to 50051 if only HTTP found
      httpPort: httpPort,
    );

    _addRobot(robot);
  }

  void _parseBleDevice(ScanResult result) {
    if (result.device.platformName.isEmpty) return;

    // Filter by name prefix if needed, e.g., "Continuon" or "ContinuonAI"
    // For now, let's just show devices with "Robot" or "Continuon" in name
    final name = result.device.platformName;
    if (!name.toLowerCase().contains('robot') &&
        !name.toLowerCase().contains('continuon')) {
      return;
    }

    final robot = ScannedRobot(
      name: name,
      host: 'BLE', // Special marker
      port: 0,
      isBle: true,
      bleId: result.device.remoteId.str,
    );

    _addRobot(robot);
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
