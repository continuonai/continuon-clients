import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:http/http.dart' as http;
import 'package:mobile_scanner/mobile_scanner.dart';

import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';

/// Detected QR code type
enum QRCodeType {
  pairing,      // /pair?token=...
  rcanRegister, // /rcan/v1/cloud/register?...
  unknown,
}

/// Unified QR scanner that handles both robot pairing AND RCAN cloud registration
class UnifiedQRScannerScreen extends StatefulWidget {
  static const routeName = '/unified_scanner';

  final BrainClient? brainClient;

  const UnifiedQRScannerScreen({
    super.key,
    this.brainClient,
  });

  @override
  State<UnifiedQRScannerScreen> createState() => _UnifiedQRScannerScreenState();
}

class _UnifiedQRScannerScreenState extends State<UnifiedQRScannerScreen> {
  final _ownerController = TextEditingController();
  final _codeController = TextEditingController();
  final MobileScannerController _scannerController = MobileScannerController(
    detectionSpeed: DetectionSpeed.noDuplicates,
    formats: const [BarcodeFormat.qrCode],
  );

  Uri? _scannedUrl;
  QRCodeType _detectedType = QRCodeType.unknown;
  bool _busy = false;
  String? _error;
  Map<String, dynamic>? _result;

  @override
  void initState() {
    super.initState();
    final user = FirebaseAuth.instance.currentUser;
    _ownerController.text = user?.displayName?.trim() ?? user?.email ?? '';
  }

  @override
  void dispose() {
    _scannerController.dispose();
    _ownerController.dispose();
    _codeController.dispose();
    super.dispose();
  }

  QRCodeType _detectQRType(Uri uri) {
    final path = uri.path.toLowerCase();

    if (path.contains('/pair')) {
      return QRCodeType.pairing;
    } else if (path.contains('/rcan/v1/cloud/register') ||
        path.contains('/rcan/') && uri.queryParameters.containsKey('registry')) {
      return QRCodeType.rcanRegister;
    }

    return QRCodeType.unknown;
  }

  void _onDetect(BarcodeCapture capture) {
    if (_scannedUrl != null) return;
    final barcodes = capture.barcodes;
    if (barcodes.isEmpty) return;
    final raw = barcodes.first.rawValue;
    if (raw == null || raw.isEmpty) return;

    try {
      final uri = Uri.parse(raw);
      if (uri.host.isEmpty) return;

      final type = _detectQRType(uri);

      setState(() {
        _scannedUrl = uri;
        _detectedType = type;
        _error = null;
      });
      _scannerController.stop();
    } catch (_) {}
  }

  Future<void> _confirmPairing() async {
    final url = _scannedUrl;
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
        body: jsonEncode({
          'token': token,
          'confirm_code': code,
          'owner_id': ownerId,
        }),
      );

      final data = jsonDecode(resp.body);
      if (resp.statusCode != 200) {
        throw StateError('HTTP ${resp.statusCode}: ${resp.body}');
      }

      setState(() =>
          _result = data is Map ? Map<String, dynamic>.from(data) : {'data': data});
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _busy = false);
    }
  }

  Future<void> _registerWithCloud() async {
    final url = _scannedUrl;
    if (url == null) return;

    setState(() {
      _busy = true;
      _error = null;
      _result = null;
    });

    try {
      // Extract registration parameters from QR
      final registryUrl = url.queryParameters['registry'] ?? url.toString();
      final capabilities = url.queryParameters['capabilities']?.split(',') ??
          ['arm', 'vision', 'chat'];

      // If we have a BrainClient, use the RCAN client to register
      if (widget.brainClient != null) {
        final brainClient = widget.brainClient!;

        // Call the robot's cloud register endpoint
        final registerUri = Uri(
          scheme: url.scheme.isEmpty ? 'http' : url.scheme,
          host: brainClient.rcan.currentHost ?? url.host,
          port: brainClient.rcan.currentPort,
          path: '/rcan/v1/cloud/register',
        );

        final resp = await http.post(
          registerUri,
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({
            'registry_url': registryUrl,
            'capabilities': capabilities,
            'firmware_version': '0.1.0',
          }),
        );

        final data = jsonDecode(resp.body);
        if (resp.statusCode != 200) {
          throw StateError('HTTP ${resp.statusCode}: ${resp.body}');
        }

        setState(() => _result =
            data is Map ? Map<String, dynamic>.from(data) : {'data': data});
      } else {
        // Direct call without BrainClient
        final resp = await http.post(
          url,
          headers: const {'Content-Type': 'application/json'},
          body: jsonEncode({
            'capabilities': capabilities,
          }),
        );

        final data = jsonDecode(resp.body);
        if (resp.statusCode != 200) {
          throw StateError('HTTP ${resp.statusCode}: ${resp.body}');
        }

        setState(() => _result =
            data is Map ? Map<String, dynamic>.from(data) : {'data': data});
      }
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _busy = false);
    }
  }

  void _resetScanner() async {
    setState(() {
      _scannedUrl = null;
      _detectedType = QRCodeType.unknown;
      _error = null;
      _result = null;
    });
    await _scannerController.start();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Scan QR Code'),
        actions: [
          IconButton(
            tooltip: 'Restart scan',
            onPressed: _busy ? null : _resetScanner,
            icon: const Icon(Icons.refresh),
          ),
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(ContinuonTokens.s16),
        child: _scannedUrl == null
            ? _buildScannerView(isDark)
            : _buildResultView(isDark),
      ),
    );
  }

  Widget _buildScannerView(bool isDark) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // Instructions
        Container(
          padding: const EdgeInsets.all(ContinuonTokens.s12),
          decoration: BoxDecoration(
            color: ContinuonColors.primaryBlue.withOpacity(0.1),
            borderRadius: BorderRadius.circular(ContinuonTokens.r8),
          ),
          child: Row(
            children: [
              Icon(
                Icons.info_outline,
                color: ContinuonColors.primaryBlue,
              ),
              const SizedBox(width: ContinuonTokens.s12),
              Expanded(
                child: Text(
                  'Scan a robot QR code to pair or register with cloud.',
                  style: TextStyle(
                    color: isDark ? ContinuonColors.gray200 : ContinuonColors.gray900,
                  ),
                ),
              ),
            ],
          ),
        ),
        const SizedBox(height: ContinuonTokens.s16),

        // Scanner
        Expanded(
          child: ClipRRect(
            borderRadius: BorderRadius.circular(ContinuonTokens.r16),
            child: Stack(
              children: [
                MobileScanner(
                  controller: _scannerController,
                  onDetect: _onDetect,
                ),
                // Scan overlay
                Center(
                  child: Container(
                    width: 250,
                    height: 250,
                    decoration: BoxDecoration(
                      border: Border.all(
                        color: ContinuonColors.primaryBlue,
                        width: 3,
                      ),
                      borderRadius: BorderRadius.circular(ContinuonTokens.r16),
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
        const SizedBox(height: ContinuonTokens.s16),

        // Supported QR types
        Text(
          'Supported QR codes:',
          style: TextStyle(
            color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            fontSize: 12,
          ),
        ),
        const SizedBox(height: ContinuonTokens.s8),
        Wrap(
          spacing: ContinuonTokens.s8,
          children: [
            Chip(
              avatar: const Icon(Icons.link, size: 16),
              label: const Text('Robot Pairing'),
              backgroundColor: isDark
                  ? ContinuonColors.gray800
                  : ContinuonColors.gray200,
            ),
            Chip(
              avatar: const Icon(Icons.cloud, size: 16),
              label: const Text('Cloud Registration'),
              backgroundColor: isDark
                  ? ContinuonColors.gray800
                  : ContinuonColors.gray200,
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildResultView(bool isDark) {
    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Detected type badge
          _buildTypeBadge(isDark),
          const SizedBox(height: ContinuonTokens.s16),

          // Scanned URL
          Container(
            padding: const EdgeInsets.all(ContinuonTokens.s12),
            decoration: BoxDecoration(
              color: isDark
                  ? ContinuonColors.gray800.withOpacity(0.5)
                  : ContinuonColors.gray200.withOpacity(0.5),
              borderRadius: BorderRadius.circular(ContinuonTokens.r8),
            ),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Scanned URL',
                  style: TextStyle(
                    color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
                    fontSize: 12,
                  ),
                ),
                const SizedBox(height: 4),
                Text(
                  _scannedUrl.toString(),
                  style: const TextStyle(fontSize: 12),
                ),
              ],
            ),
          ),
          const SizedBox(height: ContinuonTokens.s16),

          // Type-specific form
          if (_result == null) ...[
            if (_detectedType == QRCodeType.pairing) _buildPairingForm(isDark),
            if (_detectedType == QRCodeType.rcanRegister)
              _buildRcanForm(isDark),
            if (_detectedType == QRCodeType.unknown)
              _buildUnknownTypeMessage(isDark),
          ],

          // Error message
          if (_error != null) ...[
            const SizedBox(height: ContinuonTokens.s12),
            Container(
              padding: const EdgeInsets.all(ContinuonTokens.s12),
              decoration: BoxDecoration(
                color: Theme.of(context).colorScheme.errorContainer,
                borderRadius: BorderRadius.circular(ContinuonTokens.r8),
              ),
              child: Row(
                children: [
                  Icon(
                    Icons.error_outline,
                    color: Theme.of(context).colorScheme.error,
                  ),
                  const SizedBox(width: ContinuonTokens.s8),
                  Expanded(
                    child: Text(
                      _error!,
                      style: TextStyle(color: Theme.of(context).colorScheme.error),
                    ),
                  ),
                ],
              ),
            ),
          ],

          // Result display
          if (_result != null) ...[
            const SizedBox(height: ContinuonTokens.s16),
            Container(
              padding: const EdgeInsets.all(ContinuonTokens.s12),
              decoration: BoxDecoration(
                color: Colors.green.withOpacity(0.1),
                borderRadius: BorderRadius.circular(ContinuonTokens.r8),
                border: Border.all(color: Colors.green.withOpacity(0.3)),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    children: [
                      const Icon(Icons.check_circle, color: Colors.green),
                      const SizedBox(width: ContinuonTokens.s8),
                      const Text(
                        'Success!',
                        style: TextStyle(
                          fontWeight: FontWeight.w600,
                          color: Colors.green,
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: ContinuonTokens.s12),
                  Text(
                    const JsonEncoder.withIndent('  ').convert(_result),
                    style: TextStyle(
                      fontSize: 12,
                      fontFamily: 'monospace',
                      color: isDark ? ContinuonColors.gray200 : ContinuonColors.gray900,
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: ContinuonTokens.s16),
            SizedBox(
              width: double.infinity,
              child: ElevatedButton(
                onPressed: () => Navigator.of(context).pop(),
                child: const Text('Done'),
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildTypeBadge(bool isDark) {
    IconData icon;
    String label;
    Color color;

    switch (_detectedType) {
      case QRCodeType.pairing:
        icon = Icons.link;
        label = 'Robot Pairing';
        color = ContinuonColors.primaryBlue;
        break;
      case QRCodeType.rcanRegister:
        icon = Icons.cloud;
        label = 'Cloud Registration';
        color = ContinuonColors.cmsViolet;
        break;
      case QRCodeType.unknown:
        icon = Icons.help_outline;
        label = 'Unknown QR Type';
        color = ContinuonColors.particleOrange;
        break;
    }

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: ContinuonTokens.s12,
        vertical: ContinuonTokens.s8,
      ),
      decoration: BoxDecoration(
        color: color.withOpacity(0.1),
        borderRadius: BorderRadius.circular(ContinuonTokens.rFull),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 18, color: color),
          const SizedBox(width: ContinuonTokens.s8),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildPairingForm(bool isDark) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Enter the details shown on your robot:',
          style: TextStyle(
            color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
          ),
        ),
        const SizedBox(height: ContinuonTokens.s16),

        TextField(
          controller: _ownerController,
          decoration: const InputDecoration(
            labelText: 'Owner ID',
            helperText: 'Your name or identifier',
            prefixIcon: Icon(Icons.person),
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: ContinuonTokens.s12),

        TextField(
          controller: _codeController,
          keyboardType: TextInputType.number,
          maxLength: 6,
          decoration: const InputDecoration(
            labelText: '6-digit confirm code',
            helperText: 'Shown on the robot screen',
            prefixIcon: Icon(Icons.pin),
            border: OutlineInputBorder(),
          ),
        ),
        const SizedBox(height: ContinuonTokens.s16),

        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: _busy ? null : _confirmPairing,
            icon: _busy
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.verified_user),
            label: Text(_busy ? 'Pairing...' : 'Confirm Pairing'),
          ),
        ),
      ],
    );
  }

  Widget _buildRcanForm(bool isDark) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Register this robot with the cloud registry.',
          style: TextStyle(
            color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
          ),
        ),
        const SizedBox(height: ContinuonTokens.s16),

        Container(
          padding: const EdgeInsets.all(ContinuonTokens.s12),
          decoration: BoxDecoration(
            color: isDark
                ? ContinuonColors.gray800.withOpacity(0.5)
                : ContinuonColors.gray200.withOpacity(0.5),
            borderRadius: BorderRadius.circular(ContinuonTokens.r8),
          ),
          child: Column(
            children: [
              _buildInfoRow('Registry', _scannedUrl?.queryParameters['registry'] ?? 'Default'),
              const SizedBox(height: 8),
              _buildInfoRow(
                'Capabilities',
                _scannedUrl?.queryParameters['capabilities'] ?? 'arm, vision, chat',
              ),
            ],
          ),
        ),
        const SizedBox(height: ContinuonTokens.s16),

        SizedBox(
          width: double.infinity,
          child: ElevatedButton.icon(
            onPressed: _busy ? null : _registerWithCloud,
            icon: _busy
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.cloud_upload),
            label: Text(_busy ? 'Registering...' : 'Register with Cloud'),
          ),
        ),
      ],
    );
  }

  Widget _buildUnknownTypeMessage(bool isDark) {
    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: ContinuonColors.particleOrange.withOpacity(0.1),
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
        border: Border.all(color: ContinuonColors.particleOrange.withOpacity(0.3)),
      ),
      child: Column(
        children: [
          Icon(
            Icons.warning_amber,
            size: 48,
            color: ContinuonColors.particleOrange,
          ),
          const SizedBox(height: ContinuonTokens.s12),
          const Text(
            'Unknown QR Code Type',
            style: TextStyle(fontWeight: FontWeight.w600),
          ),
          const SizedBox(height: ContinuonTokens.s8),
          Text(
            'This QR code is not recognized as a robot pairing or cloud registration code.',
            textAlign: TextAlign.center,
            style: TextStyle(
              color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
          const SizedBox(height: ContinuonTokens.s16),
          OutlinedButton(
            onPressed: _resetScanner,
            child: const Text('Scan Again'),
          ),
        ],
      ),
    );
  }

  Widget _buildInfoRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(
          label,
          style: const TextStyle(fontWeight: FontWeight.w500),
        ),
        Text(value),
      ],
    );
  }
}
