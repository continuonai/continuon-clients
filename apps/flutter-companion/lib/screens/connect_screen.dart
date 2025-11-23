import 'package:flutter/material.dart';

import '../services/brain_client.dart';
import '../services/platform_channels.dart';
import 'control_screen.dart';
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
  final _hostController = TextEditingController(text: 'localhost');
  final _portController = TextEditingController(text: '50051');
  bool _useTls = false;
  bool _connecting = false;
  String? _error;

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
    try {
      await widget.brainClient.connect(host: host, port: port, useTls: _useTls);
      await const PlatformBrainBridge().initConnection(host, port, useTls: _useTls);
      if (mounted) {
        Navigator.pushReplacementNamed(context, ControlScreen.routeName);
      }
    } catch (error) {
      setState(() => _error = error.toString());
    } finally {
      setState(() => _connecting = false);
    }
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
              TextFormField(
                controller: _hostController,
                decoration: const InputDecoration(labelText: 'ContinuonBrain host'),
                validator: (value) => (value == null || value.isEmpty) ? 'Host required' : null,
              ),
              const SizedBox(height: 8),
              TextFormField(
                controller: _portController,
                decoration: const InputDecoration(labelText: 'Port'),
                keyboardType: TextInputType.number,
                validator: (value) => int.tryParse(value ?? '') != null ? null : 'Port required',
              ),
              SwitchListTile(
                title: const Text('Use TLS'),
                value: _useTls,
                onChanged: (value) => setState(() => _useTls = value),
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
