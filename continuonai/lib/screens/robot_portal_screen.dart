import 'package:flutter/material.dart';

import '../services/brain_client.dart';
import 'dashboard_screen.dart';

class RobotPortalScreen extends StatefulWidget {
  final String host;
  final int httpPort;
  final int port;
  final String robotName;

  const RobotPortalScreen({
    super.key,
    required this.host,
    required this.httpPort,
    this.port = 50051,
    required this.robotName,
  });

  static const routeName = '/robot-portal';

  @override
  State<RobotPortalScreen> createState() => _RobotPortalScreenState();
}

class _RobotPortalScreenState extends State<RobotPortalScreen> {
  final BrainClient _brainClient = BrainClient();
  bool _isConnected = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _connect();
  }

  Future<void> _connect() async {
    setState(() {
      _error = null;
      _isConnected = false;
    });

    try {
      // Connects via BrainClient (uses HTTP for status/mode, gRPC/Web for commands)
      await _brainClient.connect(
        host: widget.host,
        port: widget.port,
        httpPort: widget.httpPort,
        useTls: false, // Local connection usually plain text
      );
      if (mounted) {
        setState(() {
          _isConnected = true;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _error = e.toString();
        });
      }
    }
  }

  @override
  void dispose() {
    _brainClient.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    if (_error != null) {
      return Scaffold(
        appBar: AppBar(title: Text(widget.robotName)),
        body: Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              const Icon(Icons.error_outline, color: Colors.red, size: 48),
              const SizedBox(height: 16),
              Text('Failed to connect to ${widget.robotName}'),
              const SizedBox(height: 8),
              Text(_error!, style: const TextStyle(color: Colors.grey)),
              const SizedBox(height: 24),
              ElevatedButton.icon(
                onPressed: _connect,
                icon: const Icon(Icons.refresh),
                label: const Text('Retry'),
              ),
            ],
          ),
        ),
      );
    }

    if (!_isConnected) {
      return Scaffold(
        appBar: AppBar(title: Text(widget.robotName)),
        body: const Center(
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              CircularProgressIndicator(),
              SizedBox(height: 16),
              Text('Connecting to robot...'),
            ],
          ),
        ),
      );
    }

    // Pass true BrainClient to the native Dashboard
    return DashboardScreen(brainClient: _brainClient);
  }
}
