import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../services/brain_client.dart';
import '../services/scanner_service.dart';
import 'dashboard_screen.dart';
import 'record_screen.dart';

class ConnectScreen extends StatefulWidget {
  const ConnectScreen({super.key, required this.brainClient});

  static const routeName = '/connect';

  final BrainClient brainClient;

  @override
  State<ConnectScreen> createState() => _ConnectScreenState();
}

class _ConnectScreenState extends State<ConnectScreen> {
  final _formKey = GlobalKey<FormState>();
  // Default to localhost:8081 for local development (common case)
  final _hostController = TextEditingController(text: 'localhost');
  final _portController = TextEditingController(text: '8081');
  final _authTokenController = TextEditingController();
  final ScannerService _scanner = ScannerService();

  // Default TLS off for local development
  bool _useTls = false;
  bool _usePlatformBridge = false;
  bool _connecting = false;
  bool _scanning = false;
  String? _error;
  List<ScannedRobot> _scannedRobots = [];

  @override
  void initState() {
    super.initState();
    if (kIsWeb) {
      _scanner.scannedRobots.listen((robots) {
        if (!mounted) return;
        setState(() => _scannedRobots = robots);
      });
    }
  }

  @override
  void dispose() {
    _scanner.dispose();
    _hostController.dispose();
    _portController.dispose();
    _authTokenController.dispose();
    super.dispose();
  }

  Future<void> _connect() async {
    if (!_formKey.currentState!.validate()) {
      return;
    }
    setState(() {
      _connecting = true;
      _error = null;
    });
    final host = _hostController.text.trim();
    final port = int.tryParse(_portController.text.trim()) ?? 50051;
    final authToken = _authTokenController.text.trim().isNotEmpty
        ? _authTokenController.text.trim()
        : null;
    try {
      await widget.brainClient.connect(
        host: host,
        port: port,
        useTls: _useTls,
        authToken: authToken,
        preferPlatformBridge: _usePlatformBridge,
      );
      if (mounted) {
        Navigator.pushReplacementNamed(context, DashboardScreen.routeName);
      }
    } catch (error) {
      setState(() => _error = error.toString());
    } finally {
      setState(() => _connecting = false);
    }
  }

  Future<void> _startWebScan() async {
    if (!kIsWeb) return;
    setState(() {
      _scanning = true;
      _error = null;
    });
    await _scanner.startScan(
      manualHost: _hostController.text.trim(),
      forceRestart: true,
    );
    if (!mounted) return;
    setState(() => _scanning = false);
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('ContinuonBrain connection')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Form(
          key: _formKey,
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              if (kIsWeb) ...[
                Row(
                  children: [
                    ElevatedButton.icon(
                      onPressed: _scanning ? null : _startWebScan,
                      icon: _scanning
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.wifi_find),
                      label: Text(_scanning ? 'Scanning' : 'Scan LAN'),
                    ),
                    const SizedBox(width: 12),
                    Expanded(
                      child: Text(
                        'Discover robots advertising _continuonbrain._tcp on your LAN.',
                        style: Theme.of(context).textTheme.bodySmall,
                      ),
                    ),
                  ],
                ),
                if (_scannedRobots.isNotEmpty) ...[
                  const SizedBox(height: 12),
                  Text(
                    'Discovered robots',
                    style: Theme.of(context).textTheme.titleMedium,
                  ),
                  const SizedBox(height: 8),
                  Wrap(
                    spacing: 8,
                    runSpacing: 8,
                    children: _scannedRobots
                        .map(
                          (robot) => ChoiceChip(
                            label: Column(
                              crossAxisAlignment: CrossAxisAlignment.start,
                              mainAxisSize: MainAxisSize.min,
                              children: [
                                Text(robot.name),
                                Text('${robot.host}:${robot.port}',
                                    style: Theme.of(context)
                                        .textTheme
                                        .bodySmall
                                        ?.copyWith(
                                            color: Theme.of(context)
                                                .colorScheme
                                                .onSurfaceVariant)),
                              ],
                            ),
                            selected: _hostController.text.trim() == robot.host,
                            onSelected: (_) {
                              setState(() {
                                _hostController.text = robot.host;
                                _portController.text =
                                    robot.httpPort.toString();
                                _useTls = false;
                              });
                            },
                          ),
                        )
                        .toList(),
                  ),
                ],
                const SizedBox(height: 12),
              ],
              TextFormField(
                controller: _hostController,
                decoration:
                    const InputDecoration(labelText: 'ContinuonBrain host'),
                validator: (value) =>
                    (value == null || value.isEmpty) ? 'Host required' : null,
              ),
              const SizedBox(height: 8),
              TextFormField(
                controller: _portController,
                decoration: const InputDecoration(labelText: 'Port'),
                keyboardType: TextInputType.number,
                validator: (value) =>
                    int.tryParse(value ?? '') != null ? null : 'Port required',
              ),
              const SizedBox(height: 8),
              TextFormField(
                controller: _authTokenController,
                decoration: const InputDecoration(
                  labelText: 'Bearer token (optional)',
                  helperText: 'Passed as Authorization header for gRPC/WebRTC',
                ),
              ),
              SwitchListTile(
                title: const Text('Use TLS'),
                value: _useTls,
                onChanged: (value) => setState(() => _useTls = value),
              ),
              SwitchListTile(
                title: const Text('Use platform WebRTC bridge'),
                value: _usePlatformBridge,
                onChanged: (value) =>
                    setState(() => _usePlatformBridge = value),
              ),
              if (_error != null) ...[
                const SizedBox(height: 8),
                Text(_error!, style: const TextStyle(color: Colors.red)),
              ],
              const Spacer(),
              Row(
                children: [
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _connecting ? null : _connect,
                      icon: _connecting
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2),
                            )
                          : const Icon(Icons.cable),
                      label: Text(_connecting ? 'Connecting' : 'Connect'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  OutlinedButton(
                    onPressed: () => Navigator.pushReplacementNamed(
                      context,
                      RecordScreen.routeName,
                    ),
                    child: const Text('Skip to record'),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
    );
  }
}
