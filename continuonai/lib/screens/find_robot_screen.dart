import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:mobile_scanner/mobile_scanner.dart';
import 'package:shared_preferences/shared_preferences.dart';

import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';
import 'dashboard_screen.dart';

// Status colors not in theme
const _statusGreen = Color(0xFF10b981);
const _statusRed = Color(0xFFef4444);

/// Find My Robot - Easy pairing screen for owners
/// Scan a QR code from robot sticker or robot web UI to connect
class FindRobotScreen extends StatefulWidget {
  static const routeName = '/find_robot';

  final BrainClient brainClient;

  const FindRobotScreen({
    super.key,
    required this.brainClient,
  });

  @override
  State<FindRobotScreen> createState() => _FindRobotScreenState();
}

class _FindRobotScreenState extends State<FindRobotScreen> {
  List<Map<String, dynamic>> _recentRobots = [];
  bool _isConnecting = false;
  Map<String, dynamic>? _foundRobot;
  String? _error;

  @override
  void initState() {
    super.initState();
    _loadRecentRobots();
  }

  Future<void> _loadRecentRobots() async {
    final prefs = await SharedPreferences.getInstance();
    final recentJson = prefs.getStringList('recent_robots') ?? [];
    setState(() {
      _recentRobots = recentJson
          .map((json) => jsonDecode(json) as Map<String, dynamic>)
          .toList();
    });
  }

  Future<void> _saveRecentRobot(Map<String, dynamic> robot) async {
    final prefs = await SharedPreferences.getInstance();

    // Remove duplicate if exists
    _recentRobots.removeWhere((r) =>
        (r['ruri'] == robot['ruri'] && robot['ruri'] != null) ||
        (r['h'] == robot['h'] && r['p'] == robot['p']));

    // Add to front with timestamp
    robot['lastConnected'] = DateTime.now().toIso8601String();
    _recentRobots.insert(0, robot);

    // Keep only last 5
    if (_recentRobots.length > 5) {
      _recentRobots = _recentRobots.sublist(0, 5);
    }

    await prefs.setStringList(
      'recent_robots',
      _recentRobots.map((r) => jsonEncode(r)).toList(),
    );
    setState(() {});
  }

  Future<void> _clearRecentRobots() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.remove('recent_robots');
    setState(() => _recentRobots = []);
  }

  Future<void> _startQrScan() async {
    final result = await showModalBottomSheet<Map<String, dynamic>>(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => const _EnhancedQrScannerSheet(),
    );

    if (result != null && mounted) {
      setState(() {
        _foundRobot = result;
        _error = null;
      });
    }
  }

  Future<void> _connectToRobot(Map<String, dynamic> robot) async {
    setState(() {
      _isConnecting = true;
      _error = null;
    });

    try {
      final host = robot['h'] ?? robot['host'] ?? '';
      final port = robot['p'] ?? robot['port'] ?? 8080;
      final secure = robot['s'] == true || robot['secure'] == true;

      await widget.brainClient.connect(
        host: host,
        port: port is int ? port : int.tryParse(port.toString()) ?? 8080,
        httpPort: secure ? 443 : (port is int ? port : int.tryParse(port.toString()) ?? 8080),
        useTls: secure,
        useHttps: secure,
        authToken: robot['token'],
      );

      // Save to recent robots
      await _saveRecentRobot(robot);

      if (mounted) {
        // Show success and navigate to dashboard
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Row(
              children: [
                const Icon(Icons.check_circle, color: Colors.white),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    mainAxisSize: MainAxisSize.min,
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text('Connected to ${robot['name'] ?? 'Robot'}'),
                      const Text(
                        'Say hi to HOPE!',
                        style: TextStyle(fontSize: 12, color: Colors.white70),
                      ),
                    ],
                  ),
                ),
              ],
            ),
            backgroundColor: _statusGreen,
            duration: const Duration(seconds: 2),
          ),
        );

        Navigator.pushReplacementNamed(context, DashboardScreen.routeName);
      }
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      setState(() => _isConnecting = false);
    }
  }

  void _dismissFoundRobot() {
    setState(() {
      _foundRobot = null;
      _error = null;
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;
    final user = FirebaseAuth.instance.currentUser;

    return Scaffold(
      body: SafeArea(
        child: CustomScrollView(
          slivers: [
            // App Bar
            SliverAppBar(
              floating: true,
              title: const Text('Find My Robot'),
              actions: [
                if (user != null)
                  Padding(
                    padding: const EdgeInsets.only(right: 16),
                    child: CircleAvatar(
                      radius: 16,
                      backgroundImage: user.photoURL != null
                          ? NetworkImage(user.photoURL!)
                          : null,
                      child: user.photoURL == null
                          ? Text(user.displayName?.substring(0, 1).toUpperCase() ?? 'U')
                          : null,
                    ),
                  ),
              ],
            ),

            SliverPadding(
              padding: const EdgeInsets.all(16),
              sliver: SliverList(
                delegate: SliverChildListDelegate([
                  // Hero Section
                  _buildHeroSection(isDark),
                  const SizedBox(height: 24),

                  // Robot Found Card
                  if (_foundRobot != null) ...[
                    _buildFoundRobotCard(isDark),
                    const SizedBox(height: 24),
                  ],

                  // QR Scanner Button
                  if (_foundRobot == null) ...[
                    _buildScannerButton(isDark),
                    const SizedBox(height: 24),
                  ],

                  // Error Message
                  if (_error != null) ...[
                    _buildErrorCard(isDark),
                    const SizedBox(height: 16),
                  ],

                  // Recent Robots
                  if (_recentRobots.isNotEmpty && _foundRobot == null) ...[
                    _buildRecentRobotsSection(isDark),
                    const SizedBox(height: 24),
                  ],

                  // Help Tips
                  if (_foundRobot == null) _buildHelpTips(isDark),
                ]),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHeroSection(bool isDark) {
    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            ContinuonColors.primaryBlue.withOpacity(0.15),
            ContinuonColors.cmsViolet.withOpacity(0.1),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: ContinuonColors.primaryBlue.withOpacity(0.2),
        ),
      ),
      child: Column(
        children: [
          // Animated robot icon
          TweenAnimationBuilder<double>(
            tween: Tween(begin: 0, end: 1),
            duration: const Duration(milliseconds: 800),
            builder: (context, value, child) {
              return Transform.translate(
                offset: Offset(0, -8 * (1 - value)),
                child: Opacity(opacity: value, child: child),
              );
            },
            child: Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: ContinuonColors.primaryBlue,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: ContinuonColors.primaryBlue.withOpacity(0.3),
                    blurRadius: 20,
                    offset: const Offset(0, 8),
                  ),
                ],
              ),
              child: const Center(
                child: Text('🤖', style: TextStyle(fontSize: 40)),
              ),
            ),
          ),
          const SizedBox(height: 20),
          Text(
            'Connect to Your Robot',
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: isDark ? Colors.white : ContinuonColors.gray900,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            'Scan the QR code on your robot to pair and start chatting with HOPE',
            textAlign: TextAlign.center,
            style: TextStyle(
              fontSize: 14,
              color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildScannerButton(bool isDark) {
    return GestureDetector(
      onTap: _startQrScan,
      child: Container(
        padding: const EdgeInsets.all(24),
        decoration: BoxDecoration(
          color: isDark ? ContinuonColors.gray800 : Colors.white,
          borderRadius: BorderRadius.circular(16),
          border: Border.all(
            color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
          ),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.05),
              blurRadius: 10,
              offset: const Offset(0, 4),
            ),
          ],
        ),
        child: Row(
          children: [
            Container(
              width: 64,
              height: 64,
              decoration: BoxDecoration(
                color: ContinuonColors.primaryBlue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(16),
              ),
              child: Icon(
                Icons.qr_code_scanner,
                size: 32,
                color: ContinuonColors.primaryBlue,
              ),
            ),
            const SizedBox(width: 16),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Scan Robot QR Code',
                    style: TextStyle(
                      fontSize: 16,
                      fontWeight: FontWeight.w600,
                      color: isDark ? Colors.white : ContinuonColors.gray900,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    'Point camera at your robot\'s QR sticker',
                    style: TextStyle(
                      fontSize: 13,
                      color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
                    ),
                  ),
                ],
              ),
            ),
            Icon(
              Icons.arrow_forward_ios,
              size: 20,
              color: isDark ? ContinuonColors.gray500 : ContinuonColors.gray400,
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildFoundRobotCard(bool isDark) {
    final robot = _foundRobot!;
    final name = robot['name'] ?? robot['ruri'] ?? 'Robot Found';
    final ruri = robot['ruri'] ?? 'rcan://${robot['h']}:${robot['p']}';

    return Container(
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            _statusGreen.withOpacity(0.15),
            _statusGreen.withOpacity(0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: _statusGreen.withOpacity(0.3),
          width: 2,
        ),
      ),
      child: Column(
        children: [
          // Success icon
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: _statusGreen,
              shape: BoxShape.circle,
              boxShadow: [
                BoxShadow(
                  color: _statusGreen.withOpacity(0.3),
                  blurRadius: 20,
                  offset: const Offset(0, 8),
                ),
              ],
            ),
            child: const Icon(
              Icons.smart_toy,
              color: Colors.white,
              size: 32,
            ),
          ),
          const SizedBox(height: 16),
          Text(
            'Robot Found!',
            style: TextStyle(
              fontSize: 14,
              fontWeight: FontWeight.w600,
              color: _statusGreen,
            ),
          ),
          const SizedBox(height: 8),
          Text(
            name,
            style: TextStyle(
              fontSize: 24,
              fontWeight: FontWeight.bold,
              color: isDark ? Colors.white : ContinuonColors.gray900,
            ),
          ),
          const SizedBox(height: 4),
          Text(
            ruri,
            style: TextStyle(
              fontSize: 12,
              fontFamily: 'monospace',
              color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
          if (robot['caps'] != null) ...[
            const SizedBox(height: 12),
            Wrap(
              spacing: 6,
              runSpacing: 6,
              alignment: WrapAlignment.center,
              children: (robot['caps'] as List?)?.map<Widget>((cap) {
                return Container(
                  padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 4),
                  decoration: BoxDecoration(
                    color: ContinuonColors.primaryBlue.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                    border: Border.all(
                      color: ContinuonColors.primaryBlue.withOpacity(0.3),
                    ),
                  ),
                  child: Text(
                    cap.toString().toUpperCase(),
                    style: TextStyle(
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                      color: ContinuonColors.primaryBlue,
                    ),
                  ),
                );
              }).toList() ?? [],
            ),
          ],
          const SizedBox(height: 20),
          Row(
            children: [
              Expanded(
                child: OutlinedButton(
                  onPressed: _dismissFoundRobot,
                  child: const Text('Scan Again'),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 2,
                child: FilledButton.icon(
                  onPressed: _isConnecting ? null : () => _connectToRobot(robot),
                  icon: _isConnecting
                      ? const SizedBox(
                          width: 20,
                          height: 20,
                          child: CircularProgressIndicator(
                            strokeWidth: 2,
                            color: Colors.white,
                          ),
                        )
                      : const Icon(Icons.link),
                  label: Text(_isConnecting ? 'Connecting...' : 'Connect & Chat'),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildErrorCard(bool isDark) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: _statusRed.withOpacity(0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: _statusRed.withOpacity(0.3),
        ),
      ),
      child: Row(
        children: [
          Icon(
            Icons.error_outline,
            color: _statusRed,
          ),
          const SizedBox(width: 12),
          Expanded(
            child: Text(
              _error!,
              style: TextStyle(
                fontSize: 13,
                color: _statusRed,
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRecentRobotsSection(bool isDark) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Row(
          mainAxisAlignment: MainAxisAlignment.spaceBetween,
          children: [
            Text(
              'Recently Connected',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: isDark ? Colors.white : ContinuonColors.gray900,
              ),
            ),
            TextButton(
              onPressed: _clearRecentRobots,
              child: Text(
                'Clear',
                style: TextStyle(
                  fontSize: 13,
                  color: ContinuonColors.gray500,
                ),
              ),
            ),
          ],
        ),
        const SizedBox(height: 12),
        ...(_recentRobots.map((robot) => _buildRecentRobotTile(robot, isDark))),
      ],
    );
  }

  Widget _buildRecentRobotTile(Map<String, dynamic> robot, bool isDark) {
    final name = robot['name'] ?? 'Robot';
    final host = robot['h'] ?? robot['host'] ?? '';
    final port = robot['p'] ?? robot['port'] ?? 8080;

    return GestureDetector(
      onTap: () => _connectToRobot(robot),
      child: Container(
        margin: const EdgeInsets.only(bottom: 8),
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: isDark ? ContinuonColors.gray800 : Colors.white,
          borderRadius: BorderRadius.circular(12),
          border: Border.all(
            color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
          ),
        ),
        child: Row(
          children: [
            Container(
              width: 44,
              height: 44,
              decoration: BoxDecoration(
                color: ContinuonColors.primaryBlue.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
              ),
              child: const Center(
                child: Text('🤖', style: TextStyle(fontSize: 24)),
              ),
            ),
            const SizedBox(width: 12),
            Expanded(
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    name,
                    style: TextStyle(
                      fontSize: 14,
                      fontWeight: FontWeight.w600,
                      color: isDark ? Colors.white : ContinuonColors.gray900,
                    ),
                  ),
                  Text(
                    '$host:$port',
                    style: TextStyle(
                      fontSize: 12,
                      fontFamily: 'monospace',
                      color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
                    ),
                  ),
                ],
              ),
            ),
            Container(
              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
              decoration: BoxDecoration(
                color: _statusGreen.withOpacity(0.1),
                borderRadius: BorderRadius.circular(20),
              ),
              child: Text(
                'Connect',
                style: TextStyle(
                  fontSize: 12,
                  fontWeight: FontWeight.w600,
                  color: _statusGreen,
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildHelpTips(bool isDark) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: isDark
            ? ContinuonColors.gray800.withOpacity(0.5)
            : ContinuonColors.gray200.withOpacity(0.5),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(
                Icons.lightbulb_outline,
                size: 20,
                color: ContinuonColors.particleOrange,
              ),
              const SizedBox(width: 8),
              Text(
                'Tips',
                style: TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                  color: ContinuonColors.particleOrange,
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _buildTipRow(Icons.qr_code, 'Look for the QR sticker on your robot', isDark),
          _buildTipRow(Icons.wifi, 'Make sure you\'re on the same WiFi network', isDark),
          _buildTipRow(Icons.chat_bubble_outline, 'After connecting, chat with HOPE to control your robot', isDark),
        ],
      ),
    );
  }

  Widget _buildTipRow(IconData icon, String text, bool isDark) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(
            icon,
            size: 16,
            color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
          ),
          const SizedBox(width: 10),
          Expanded(
            child: Text(
              text,
              style: TextStyle(
                fontSize: 13,
                color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

/// Enhanced QR Scanner bottom sheet
class _EnhancedQrScannerSheet extends StatefulWidget {
  const _EnhancedQrScannerSheet();

  @override
  State<_EnhancedQrScannerSheet> createState() => _EnhancedQrScannerSheetState();
}

class _EnhancedQrScannerSheetState extends State<_EnhancedQrScannerSheet> {
  final MobileScannerController _controller = MobileScannerController(
    detectionSpeed: DetectionSpeed.noDuplicates,
    formats: const [BarcodeFormat.qrCode],
  );
  bool _hasScanned = false;
  String? _error;
  bool _torchOn = false;

  @override
  void dispose() {
    _controller.dispose();
    super.dispose();
  }

  void _handleBarcode(BarcodeCapture capture) {
    if (_hasScanned) return;

    final barcode = capture.barcodes.firstOrNull;
    if (barcode == null || barcode.rawValue == null) return;

    final rawValue = barcode.rawValue!;
    debugPrint('Scanned QR: $rawValue');

    try {
      Map<String, dynamic> data;

      // Try parsing as JSON first
      try {
        data = jsonDecode(rawValue) as Map<String, dynamic>;
      } catch (_) {
        // Try parsing as URL
        final uri = Uri.tryParse(rawValue);
        if (uri != null && uri.host.isNotEmpty) {
          data = {
            'h': uri.host,
            'p': uri.hasPort ? uri.port : 8080,
            's': uri.scheme == 'https',
            'name': uri.queryParameters['name'] ?? 'Robot',
            'token': uri.queryParameters['token'],
          };
        } else {
          throw const FormatException('Invalid QR format');
        }
      }

      // Validate required fields
      if (data['h'] == null && data['host'] == null) {
        setState(() => _error = 'Invalid QR code: missing host');
        return;
      }

      _hasScanned = true;
      Navigator.pop(context, data);
    } catch (e) {
      setState(() => _error = 'Invalid QR code. Please scan a robot pairing code.');
      Future.delayed(const Duration(seconds: 2), () {
        if (mounted) setState(() => _error = null);
      });
    }
  }

  void _toggleTorch() {
    _controller.toggleTorch();
    setState(() => _torchOn = !_torchOn);
  }

  @override
  Widget build(BuildContext context) {
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final screenHeight = MediaQuery.of(context).size.height;

    return Container(
      height: screenHeight * 0.8,
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : Colors.white,
        borderRadius: const BorderRadius.vertical(top: Radius.circular(24)),
      ),
      child: Column(
        children: [
          // Handle
          Container(
            margin: const EdgeInsets.symmetric(vertical: 12),
            width: 40,
            height: 4,
            decoration: BoxDecoration(
              color: ContinuonColors.gray400,
              borderRadius: BorderRadius.circular(2),
            ),
          ),

          // Header
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 20),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(10),
                  decoration: BoxDecoration(
                    color: ContinuonColors.primaryBlue.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Icon(
                    Icons.qr_code_scanner,
                    color: ContinuonColors.primaryBlue,
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        'Scan Robot QR Code',
                        style: TextStyle(
                          fontSize: 18,
                          fontWeight: FontWeight.bold,
                          color: isDark ? Colors.white : ContinuonColors.gray900,
                        ),
                      ),
                      Text(
                        'Point at the QR sticker on your robot',
                        style: TextStyle(
                          fontSize: 13,
                          color: ContinuonColors.gray500,
                        ),
                      ),
                    ],
                  ),
                ),
                IconButton(
                  onPressed: () => Navigator.pop(context),
                  icon: Icon(
                    Icons.close,
                    color: ContinuonColors.gray500,
                  ),
                ),
              ],
            ),
          ),
          const SizedBox(height: 20),

          // Scanner
          Expanded(
            child: Padding(
              padding: const EdgeInsets.symmetric(horizontal: 20),
              child: ClipRRect(
                borderRadius: BorderRadius.circular(20),
                child: Stack(
                  children: [
                    MobileScanner(
                      controller: _controller,
                      onDetect: _handleBarcode,
                    ),
                    // Scan frame overlay
                    Center(
                      child: Container(
                        width: 250,
                        height: 250,
                        decoration: BoxDecoration(
                          border: Border.all(
                            color: ContinuonColors.primaryBlue,
                            width: 3,
                          ),
                          borderRadius: BorderRadius.circular(20),
                        ),
                      ),
                    ),
                    // Torch button
                    Positioned(
                      bottom: 20,
                      right: 20,
                      child: GestureDetector(
                        onTap: _toggleTorch,
                        child: Container(
                          padding: const EdgeInsets.all(12),
                          decoration: BoxDecoration(
                            color: _torchOn
                                ? ContinuonColors.particleOrange
                                : Colors.black54,
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            _torchOn ? Icons.flash_on : Icons.flash_off,
                            color: Colors.white,
                          ),
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),

          // Error message
          if (_error != null)
            Padding(
              padding: const EdgeInsets.all(20),
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: _statusRed.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(12),
                  border: Border.all(
                    color: _statusRed.withOpacity(0.3),
                  ),
                ),
                child: Row(
                  children: [
                    Icon(
                      Icons.error_outline,
                      color: _statusRed,
                      size: 20,
                    ),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _error!,
                        style: TextStyle(
                          color: _statusRed,
                          fontSize: 13,
                        ),
                      ),
                    ),
                  ],
                ),
              ),
            )
          else
            const SizedBox(height: 20),

          // Bottom padding
          SizedBox(height: MediaQuery.of(context).padding.bottom + 20),
        ],
      ),
    );
  }
}
