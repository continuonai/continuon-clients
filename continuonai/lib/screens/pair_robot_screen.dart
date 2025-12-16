import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;
import 'package:mobile_scanner/mobile_scanner.dart';

/// Pair a robot via the QR flow served by the robot:
/// - Robot UI starts pairing and shows QR + 6-digit confirm code.
/// - This screen scans the QR (pair URL), then calls /api/ownership/pair/confirm with the code.
class PairRobotScreen extends StatefulWidget {
  const PairRobotScreen({super.key});

  static const routeName = '/pair_robot';

  @override
  State<PairRobotScreen> createState() => _PairRobotScreenState();
}

class _PairRobotScreenState extends State<PairRobotScreen> {
  final _ownerController = TextEditingController();
  final _codeController = TextEditingController();
  final MobileScannerController _scannerController = MobileScannerController(
    detectionSpeed: DetectionSpeed.noDuplicates,
    formats: const [BarcodeFormat.qrCode],
  );

  Uri? _pairUrl;
  bool _busy = false;
  String? _error;
  Map<String, dynamic>? _result;

  @override
  void initState() {
    super.initState();
    final user = FirebaseAuth.instance.currentUser;
    _ownerController.text = (user?.displayName?.trim().isNotEmpty ?? false)
        ? user!.displayName!.trim()
        : '';
  }

  @override
  void dispose() {
    _scannerController.dispose();
    _ownerController.dispose();
    _codeController.dispose();
    super.dispose();
  }

  void _onDetect(BarcodeCapture capture) {
    if (_pairUrl != null) return;
    final barcodes = capture.barcodes;
    if (barcodes.isEmpty) return;
    final raw = barcodes.first.rawValue;
    if (raw == null || raw.isEmpty) return;
    try {
      final uri = Uri.parse(raw);
      if (uri.host.isEmpty || !uri.path.contains('/pair')) return;
      setState(() {
        _pairUrl = uri;
        _error = null;
      });
      _scannerController.stop();
    } catch (_) {}
  }

  Future<void> _confirm() async {
    final url = _pairUrl;
    if (url == null) return;
    final token = url.queryParameters['token'] ?? '';
    if (token.isEmpty) {
      setState(() => _error = 'Pair QR missing token');
      return;
    }
    final ownerId = _ownerController.text.trim();
    final code = _codeController.text.trim();
    if (ownerId.isEmpty) {
      setState(() => _error = 'Owner ID required');
      return;
    }
    if (code.length != 6) {
      setState(() => _error = 'Confirm code must be 6 digits');
      return;
    }

    setState(() {
      _busy = true;
      _error = null;
      _result = null;
    });
    try {
      final confirmUri = Uri(
        scheme: url.scheme.isEmpty ? 'http' : url.scheme,
        host: url.host,
        port: url.hasPort ? url.port : 8080,
        path: '/api/ownership/pair/confirm',
      );
      final resp = await http.post(
        confirmUri,
        headers: const {'Content-Type': 'application/json'},
        body: jsonEncode({'token': token, 'confirm_code': code, 'owner_id': ownerId}),
      );
      final data = jsonDecode(resp.body);
      if (resp.statusCode != 200) {
        throw StateError('HTTP ${resp.statusCode}: ${resp.body}');
      }
      setState(() => _result = data is Map ? Map<String, dynamic>.from(data) : {'data': data});
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _busy = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final url = _pairUrl;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Pair robot (QR)'),
        actions: [
          IconButton(
            tooltip: 'Restart scan',
            onPressed: _busy
                ? null
                : () async {
                    setState(() {
                      _pairUrl = null;
                      _error = null;
                      _result = null;
                    });
                    await _scannerController.start();
                  },
            icon: const Icon(Icons.refresh),
          )
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            if (url == null) ...[
              Text(
                'Scan the QR code shown on the robot UI (Agent Details → Pair phone).',
                style: Theme.of(context).textTheme.bodyMedium,
              ),
              const SizedBox(height: 12),
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(12),
                  child: MobileScanner(
                    controller: _scannerController,
                    onDetect: _onDetect,
                  ),
                ),
              ),
            ] else ...[
              Text('Pair URL:', style: Theme.of(context).textTheme.titleSmall),
              const SizedBox(height: 6),
              Text(url.toString(), style: Theme.of(context).textTheme.bodySmall),
              const SizedBox(height: 12),
              TextField(
                controller: _ownerController,
                decoration: const InputDecoration(
                  labelText: 'Owner ID',
                  helperText: 'Example: Craig Merry',
                ),
              ),
              const SizedBox(height: 10),
              TextField(
                controller: _codeController,
                keyboardType: TextInputType.number,
                decoration: const InputDecoration(
                  labelText: '6-digit confirm code',
                  helperText: 'Shown on the robot UI',
                ),
              ),
              const SizedBox(height: 12),
              if (_error != null) ...[
                Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
                const SizedBox(height: 8),
              ],
              if (_result != null) ...[
                Text('Result:', style: Theme.of(context).textTheme.titleSmall),
                const SizedBox(height: 6),
                Expanded(
                  child: SingleChildScrollView(
                    child: Text(const JsonEncoder.withIndent('  ').convert(_result)),
                  ),
                ),
              ] else ...[
                ElevatedButton.icon(
                  onPressed: _busy ? null : _confirm,
                  icon: _busy
                      ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
                      : const Icon(Icons.verified_user),
                  label: Text(_busy ? 'Pairing...' : 'Pair now'),
                ),
              ],
            ]
          ],
        ),
      ),
    );
  }
}


