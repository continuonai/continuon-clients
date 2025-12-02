import 'dart:async';

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
}

class ScannerService {
  final StreamController<List<ScannedRobot>> _robotsController = StreamController<List<ScannedRobot>>.broadcast();
  Stream<List<ScannedRobot>> get scannedRobots => _robotsController.stream;

  Future<void> startScan() async {
    print('Scanning not supported on Web');
    _robotsController.add([]);
  }

  Future<void> stopScan() async {}

  void dispose() {
    _robotsController.close();
  }
}
