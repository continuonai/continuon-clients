import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'dart:async';

import '../theme/continuon_theme.dart';
import '../services/scanner_service_native.dart';
import '../utils/platform_info.dart';
import 'package:url_launcher/url_launcher.dart';

import 'dashboard_screen.dart';
import 'robot_portal_screen.dart';
import 'unified_qr_scanner_screen.dart';
import '../services/brain_client.dart';
import '../widgets/layout/continuon_layout.dart';
import '../widgets/layout/continuon_card.dart';

enum DiscoveryStatus { searching, found, error, idle }

class RobotListScreen extends StatefulWidget {
  const RobotListScreen({super.key});

  static const routeName = '/robots';

  @override
  State<RobotListScreen> createState() => _RobotListScreenState();
}

class _RobotListScreenState extends State<RobotListScreen> {
  final User? _user = FirebaseAuth.instance.currentUser;
  final BrainClient _brainClient = BrainClient();
  final ScannerService _scannerService = ScannerService();
  StreamSubscription? _scannerSubscription;
  DiscoveryStatus _discoveryStatus = DiscoveryStatus.idle;
  List<ScannedRobot> _scannedRobots = [];

  // Per-robot state caches
  final Map<String, bool> _ownedByHost = {};
  final Map<String, bool> _subByHost = {};
  final Map<String, bool> _seedByHost = {};
  final Map<String, Map<String, dynamic>> _deviceInfoByHost = {};
  final Map<String, String> _errorByHost = {};
  String? _accountId;
  String? _accountType;
  String? _ownerId;
  // Busy per robot host to avoid blocking all cards
  final Set<String> _busyHosts = {};
  bool _isOwned = false;
  bool _hasSubscription = false;
  bool _hasSeedInstalled = false;
  String? _authToken;
  final TextEditingController _tokenController = TextEditingController();
  final TextEditingController _accountIdController = TextEditingController();
  final TextEditingController _accountTypeController = TextEditingController();
  final TextEditingController _ownerIdController = TextEditingController();
  
  // Controllers for prominent manual connect
  final TextEditingController _hostController = TextEditingController();
  final TextEditingController _httpPortController = TextEditingController(text: '8080');
  
  bool _tokenLoaded = false;
  bool _lanLikely = true;
  bool _stateLoaded = false;
  Timer? _pollTimer;

  // Local list for guest mode
  final List<Map<String, dynamic>> _guestRobots = [
    {
      'name': 'Demo Robot',
      'host': '192.168.1.100',
      'port': 50051,
      'httpPort': 8080
    },
  ];

  @override
  void initState() {
    super.initState();
    _startPolling();
    _startDiscovery();
  }

  @override
  void dispose() {
    _pollTimer?.cancel();
    _scannerSubscription?.cancel();
    _scannerService.dispose();
    _tokenController.dispose();
    _accountIdController.dispose();
    _accountTypeController.dispose();
    _ownerIdController.dispose();
    _hostController.dispose();
    _httpPortController.dispose();
    super.dispose();
  }

  void _startDiscovery() {
    setState(() {
      _discoveryStatus = DiscoveryStatus.searching;
    });
    
    _scannerSubscription = _scannerService.scannedRobots.listen((robots) {
      if (mounted) {
        setState(() {
          _scannedRobots = robots;
          _discoveryStatus = robots.isEmpty ? DiscoveryStatus.searching : DiscoveryStatus.found;
          
          // Merge scanned robots into guest list if they aren't already there
          for (final robot in robots) {
            final exists = _guestRobots.any((r) => r['host'] == robot.host);
            if (!exists) {
              _guestRobots.add({
                'name': robot.name,
                'host': robot.host,
                'port': robot.port,
                'httpPort': robot.httpPort,
                'isScanned': true,
              });
            }
          }
        });
      }
    }, onError: (e) {
      if (mounted) {
        setState(() {
          _discoveryStatus = DiscoveryStatus.error;
        });
      }
    });

    _scannerService.startScan();
  }

  void _startPolling() {
    _pollTimer = Timer.periodic(const Duration(seconds: 30), (timer) {
      if (!mounted) return;
      _refreshAllCachedHosts();
    });
  }

  Future<void> _refreshAllCachedHosts() async {
    final hosts = _deviceInfoByHost.keys.toList();
    for (final host in hosts) {
      try {
        await _brainClient.ping(host: host, httpPort: 8080);
      } catch (_) {}
    }
  }

  void _addRobot() {
    showDialog(
      context: context,
      builder: (context) => _AddRobotDialog(
        userId: _user?.uid,
        onGuestAdd: (robot) {
          setState(() {
            _guestRobots.add(robot);
          });
          _refreshStatus(robot);
        },
      ),
    ).then((result) {
      if (result != null && result is Map<String, dynamic>) {
        _refreshStatus(result);
      }
    });
  }

  Future<void> _quickConnect() async {
    final host = _hostController.text.trim();
    final httpPortText = _httpPortController.text.trim();
    
    if (host.isEmpty) {
      _showSnack('Please enter a host IP');
      return;
    }
    
    final httpPort = int.tryParse(httpPortText) ?? 8080;
    
    // Add to guest list if not already there
    final exists = _guestRobots.any((r) => r['host'] == host);
    final robotData = {
      'name': 'Manual Robot',
      'host': host,
      'port': 50051,
      'httpPort': httpPort,
    };
    
    if (!exists) {
      setState(() {
        _guestRobots.add(robotData);
      });
    }
    
    await _connectToRobot(robotData);
  }

  Future<void> _connectToRobot(Map<String, dynamic> data) async {
    final host = data['host'] as String;
    final port = data['port'] as int;
    final httpPort = data['httpPort'] as int? ?? 8080;

    if (_authToken != null) {
      _brainClient.setAuthToken(_authToken!);
    }

    setState(() => _busyHosts.add(host));

    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => const Center(child: CircularProgressIndicator()),
    );

    try {
      await _brainClient.connect(
        host: host,
        port: port,
        httpPort: httpPort,
        useTls: false,
      );

      if (mounted) {
        Navigator.pop(context);
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => DashboardScreen(brainClient: _brainClient),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        Navigator.pop(context);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed to connect: $e'),
            backgroundColor: Theme.of(context).colorScheme.error,
          ),
        );
      }
    } finally {
      if (mounted) setState(() => _busyHosts.remove(host));
    }
  }

  @override
  Widget build(BuildContext context) {
    if (!_tokenLoaded) {
      _brainClient.loadAuthToken().then((_) {
        setState(() {
          _authToken = _brainClient.authToken;
          _tokenLoaded = true;
        });
      });
      _brainClient.checkLocalNetwork().then((lan) {
        if (!mounted) return;
        setState(() {
          _lanLikely = lan;
        });
        _persistLanLikely(lan);
      });
      _loadCachedState();
    }

    return ContinuonLayout(
      body: Column(
        children: [
          _buildDiscoveryStatusHeader(),
          _buildWebDiscoveryNotice(),
          _buildProminentAddRobot(),
          Expanded(
              child: _user == null ? _buildGuestList() : _buildFirestoreList()),
        ],
      ),
    );
  }

  Widget _buildWebDiscoveryNotice() {
    if (!PlatformInfo.isWeb) return const SizedBox.shrink();
    
    return Container(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.amber.withOpacity(0.1),
        border: Border.all(color: Colors.amber.withOpacity(0.3)),
        borderRadius: BorderRadius.circular(12),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              const Icon(Icons.lightbulb_outline, color: Colors.amber, size: 20),
              const SizedBox(width: 8),
              Text(
                'Web Discovery Tip',
                style: TextStyle(color: Colors.amber.shade800, fontWeight: FontWeight.bold),
              ),
            ],
          ),
          const SizedBox(height: 8),
          const Text(
            'Browsers restrict automatic mDNS scanning. If your robot isn\'t found, connect via IP below. Ensure your robot has CORS enabled or use the tunnel utility.',
            style: TextStyle(fontSize: 13),
          ),
          TextButton(
            onPressed: () => _launchURL('https://github.com/continuon-ai/continuon-xr/blob/main/docs/PI5_EDGE_BRAIN_INSTRUCTIONS.md#11-unified-discovery-and-remote-conductor-cli'),
            child: const Text('Read docs for tunnel and discovery'),
          ),
        ],
      ),
    );
  }

  Future<void> _launchURL(String url) async {
    final uri = Uri.parse(url);
    if (await canLaunchUrl(uri)) {
      await launchUrl(uri);
    }
  }

  Widget _buildProminentAddRobot() {
    return ContinuonCard(
      margin: const EdgeInsets.all(24),
      padding: const EdgeInsets.all(24),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text('Quick Connect', style: Theme.of(context).textTheme.titleMedium?.copyWith(fontWeight: FontWeight.bold)),
              ElevatedButton.icon(
                onPressed: _openQrScanner,
                icon: const Icon(Icons.qr_code_scanner, size: 20),
                label: const Text('Scan QR Code'),
                style: ElevatedButton.styleFrom(
                  backgroundColor: Colors.deepPurple,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
              ),
            ],
          ),
          const SizedBox(height: 8),
          Text(
            'Scan the QR code on your robot or enter IP manually',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(color: Colors.grey),
          ),
          const SizedBox(height: 16),
          Row(
            children: [
              Expanded(
                flex: 3,
                child: TextField(
                  controller: _hostController,
                  decoration: const InputDecoration(
                    labelText: 'Robot IP Address',
                    hintText: 'e.g. 192.168.1.50',
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                flex: 1,
                child: TextField(
                  controller: _httpPortController,
                  decoration: const InputDecoration(
                    labelText: 'Port',
                    hintText: '8080',
                    border: OutlineInputBorder(),
                    isDense: true,
                  ),
                  keyboardType: TextInputType.number,
                ),
              ),
              const SizedBox(width: 12),
              ElevatedButton(
                onPressed: _quickConnect,
                style: ElevatedButton.styleFrom(
                  backgroundColor: ContinuonColors.primaryBlue,
                  foregroundColor: Colors.white,
                  padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 18),
                  shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                ),
                child: const Text('Connect'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  void _openQrScanner() async {
    final result = await Navigator.push<Map<String, dynamic>>(
      context,
      MaterialPageRoute(
        builder: (context) => const UnifiedQRScannerScreen(),
      ),
    );

    if (result != null && mounted) {
      // Handle the scanned result - could be pairing data or connection info
      final host = result['host'] as String?;
      final httpPort = result['httpPort'] as int? ?? 8080;
      final port = result['port'] as int? ?? 50051;
      final name = result['name'] as String? ?? 'Scanned Robot';

      if (host != null && host.isNotEmpty) {
        // Add to guest list if not already there
        final exists = _guestRobots.any((r) => r['host'] == host);
        final robotData = {
          'name': name,
          'host': host,
          'port': port,
          'httpPort': httpPort,
        };

        if (!exists) {
          setState(() {
            _guestRobots.add(robotData);
          });
        }

        // Attempt to connect
        await _connectToRobot(robotData);
      } else {
        _showSnack('QR scan completed. Check result for pairing info.');
      }
    }
  }

  Widget _buildDiscoveryStatusHeader() {
    String message;
    Color color;
    IconData icon;

    switch (_discoveryStatus) {
      case DiscoveryStatus.searching:
        message = 'Searching for robots on local network...';
        color = Colors.blue;
        icon = Icons.search;
        break;
      case DiscoveryStatus.found:
        message = 'Discovery active. ${_scannedRobots.length} robot(s) found via mDNS.';
        color = Colors.green;
        icon = Icons.radar;
        break;
      case DiscoveryStatus.error:
        message = 'Discovery error. Check your Wi-Fi connection.';
        color = Colors.red;
        icon = Icons.error_outline;
        break;
      case DiscoveryStatus.idle:
      default:
        message = 'Discovery idle.';
        color = Colors.grey;
        icon = Icons.pause_circle_outline;
    }

    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
      color: color.withOpacity(0.1),
      child: Row(
        children: [
          Icon(icon, color: color, size: 16),
          const SizedBox(width: 8),
          Expanded(
            child: Text(
              message,
              style: TextStyle(color: color, fontSize: 12, fontWeight: FontWeight.w500),
            ),
          ),
          if (_discoveryStatus == DiscoveryStatus.searching)
            const SizedBox(
              width: 12,
              height: 12,
              child: CircularProgressIndicator(strokeWidth: 2, color: Colors.blue),
            ),
          IconButton(
            icon: const Icon(Icons.refresh, size: 16),
            onPressed: () {
              _scannerService.startScan(forceRestart: true);
              setState(() {
                _discoveryStatus = DiscoveryStatus.searching;
              });
            },
            tooltip: 'Rescan network',
          ),
        ],
      ),
    );
  }

  void _showAuthTokenDialog() {
    _tokenController.text = _authToken ?? '';
    _accountIdController.text = _accountId ?? '';
    _accountTypeController.text = _accountType ?? '';
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Connection credentials'),
        content: SizedBox(
          width: 520,
          child: SingleChildScrollView(
            child: Column(
              mainAxisSize: MainAxisSize.min,
              children: [
                TextField(
                  controller: _tokenController,
                  decoration: const InputDecoration(
                    labelText: 'Bearer token',
                    helperText:
                        'Optional. Stored securely on-device when you tap Save.',
                  ),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _accountIdController,
                  decoration: const InputDecoration(
                    labelText: 'Account ID',
                    helperText:
                        'Used for ownership/subscription checks (optional).',
                  ),
                ),
                const SizedBox(height: 12),
                TextField(
                  controller: _accountTypeController,
                  decoration: const InputDecoration(
                    labelText: 'Account type',
                    helperText: 'Example: personal | org (optional).',
                  ),
                ),
              ],
            ),
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              setState(() {
                _authToken = _tokenController.text.isNotEmpty
                    ? _tokenController.text
                    : null;
                if (_authToken != null) {
                  _brainClient.setAuthToken(_authToken!, persist: true);
                }
                _accountId = _accountIdController.text;
                _accountType = _accountTypeController.text;
                _ownerId = _ownerIdController.text;
              });
              Navigator.pop(context);
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  Widget _buildActionRow() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          OutlinedButton.icon(
            onPressed: _showAuthTokenDialog,
            icon: const Icon(Icons.vpn_key, size: 18),
            label: const Text('Credentials'),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusBanner() {
    final messages = <String>[];
    if (_user == null) {
      messages
          .add('Guest mode: control/record/OTA disabled. Sign in to unlock.');
    } else if (!_lanLikely) {
      messages.add(
          'Not on robot LAN. Join the same Wi‑Fi/hotspot as the robot to claim.');
    } else if (!_isOwned) {
      messages.add('Local claim required before remote control/OTA.');
    } else if (!_hasSeedInstalled) {
      messages.add('Initial seed install required (local-only).');
    } else if (!_hasSubscription) {
      messages.add('Subscription required for remote control/OTA.');
    } else {
      messages.add('Remote control/OTA allowed (owned + subscribed).');
    }
    return ContinuonCard(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
      backgroundColor: Colors.blue.withOpacity(0.1),
      child: Row(
        children: [
          const Icon(Icons.info_outline, color: ContinuonColors.primaryBlue),
          const SizedBox(width: 12),
          Expanded(child: Text(messages.join(' '))),
        ],
      ),
    );
  }

  Widget _buildGuestList() {
    if (_guestRobots.isEmpty) {
      return _buildEmptyState();
    }
    return ListView.builder(
      padding: const EdgeInsets.all(20),
      itemCount: _guestRobots.length,
      itemBuilder: (context, index) {
        return _buildAnimatedRobotCard(_guestRobots[index], index);
      },
    );
  }

  Widget _buildFirestoreList() {
    return StreamBuilder<QuerySnapshot>(
      stream: FirebaseFirestore.instance
          .collection('users')
          .doc(_user!.uid)
          .collection('robots')
          .snapshots(),
      builder: (context, snapshot) {
        if (snapshot.hasError) {
          return Center(child: Text('Error: ${snapshot.error}'));
        }

        if (snapshot.connectionState == ConnectionState.waiting) {
          return const Center(child: CircularProgressIndicator());
        }

        final docs = snapshot.data?.docs ?? [];

        if (docs.isEmpty) {
          return _buildEmptyState();
        }

        return ListView.builder(
          padding: const EdgeInsets.all(20),
          itemCount: docs.length,
          itemBuilder: (context, index) {
            final doc = docs[index];
            final data = doc.data() as Map<String, dynamic>;
            // Include document ID for management operations
            final dataWithId = Map<String, dynamic>.from(data);
            dataWithId['_documentId'] = doc.id;
            return _buildAnimatedRobotCard(dataWithId, index);
          },
        );
      },
    );
  }

  Widget _buildManualConnectSection() {
    return Padding(
      padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 8),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.end,
        children: [
          OutlinedButton.icon(
            onPressed: _addRobot,
            icon: const Icon(Icons.add, size: 18),
            label: const Text('Add Robot'),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: ContinuonColors.primaryBlue.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: Image.asset(
              'assets/branding/continuon_ai_logo_text_transparent.png',
              height: 64,
              width: 64,
            ),
          ),
          const SizedBox(height: 24),
          Text('No robots added yet',
              style: Theme.of(context).textTheme.bodyLarge),
          const SizedBox(height: 32),
          ElevatedButton.icon(
            onPressed: _addRobot,
            icon: const Icon(Icons.add),
            label: const Text('Add Robot'),
            style: ElevatedButton.styleFrom(
              backgroundColor: ContinuonColors.primaryBlue,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12)),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildAnimatedRobotCard(Map<String, dynamic> data, int index) {
    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.0, end: 1.0),
      duration: Duration(milliseconds: 500 + (index * 100)),
      curve: Curves.easeOutQuart,
      builder: (context, value, child) {
        return Transform.translate(
          offset: Offset(0, 50 * (1 - value)),
          child: Opacity(
            opacity: value,
            child: child,
          ),
        );
      },
      child: _buildRobotCard(data),
    );
  }

  Widget _buildRobotCard(Map<String, dynamic> data) {
    final name = data['name'] ?? 'Unnamed Robot';
    final host = data['host'] ?? 'unknown';
    final isBusy = _busyHosts.contains(host);
    final isGuest = _user == null;
    final owned = _ownedByHost[host] ?? _isOwned;
    final hasSeed = _seedByHost[host] ?? _hasSeedInstalled;
    final hasSubscription = _subByHost[host] ?? _hasSubscription;
    final deviceInfo = _deviceInfoByHost[host] ?? {};
    final deviceId = deviceInfo['device_id'] as String? ?? '';
    final statusAccountId = _deviceInfoByHost[host]?['account_id'] as String?;
    final statusAccountType =
        _deviceInfoByHost[host]?['account_type'] as String?;
    final mismatch = (statusAccountId != null &&
            _accountId != null &&
            statusAccountId != _accountId) ||
        (statusAccountType != null &&
            _accountType != null &&
            statusAccountType != _accountType);

    return ContinuonCard(
      margin: const EdgeInsets.only(bottom: 16),
      padding: EdgeInsets.zero,
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          // Header section with robot info
          Padding(
            padding: const EdgeInsets.all(16),
            child: Row(
              children: [
                // Robot icon with online indicator
                Stack(
                  children: [
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        color: ContinuonColors.primaryBlue.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Icon(Icons.smart_toy,
                          color: ContinuonColors.primaryBlue, size: 28),
                    ),
                    Positioned(
                      right: 0,
                      bottom: 0,
                      child: Container(
                        width: 14,
                        height: 14,
                        decoration: BoxDecoration(
                          color: owned ? Colors.green : Colors.orange,
                          shape: BoxShape.circle,
                          border: Border.all(color: Colors.white, width: 2),
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(width: 16),
                // Robot name and host
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        name,
                        style: Theme.of(context)
                            .textTheme
                            .titleMedium
                            ?.copyWith(fontSize: 18, fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          Text(host, style: Theme.of(context).textTheme.bodyMedium?.copyWith(color: Colors.grey.shade600)),
                          if (deviceId.isNotEmpty) ...[
                            const SizedBox(width: 8),
                            Container(
                              padding: const EdgeInsets.symmetric(horizontal: 6, vertical: 2),
                              decoration: BoxDecoration(
                                color: Colors.grey.shade200,
                                borderRadius: BorderRadius.circular(4),
                              ),
                              child: Text('ID: $deviceId',
                                  style: Theme.of(context)
                                      .textTheme
                                      .bodySmall
                                      ?.copyWith(color: Colors.grey.shade700, fontSize: 10)),
                            ),
                          ],
                        ],
                      ),
                    ],
                  ),
                ),
                // Loading indicator or refresh button
                if (isBusy)
                  const SizedBox(
                    width: 24,
                    height: 24,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                else
                  IconButton(
                    icon: const Icon(Icons.refresh, size: 20),
                    tooltip: 'Refresh status',
                    onPressed: isGuest ? null : () => _refreshStatus(data),
                  ),
              ],
            ),
          ),

          // Status badges row
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                _buildStatusBadge(
                  icon: Icons.verified_user,
                  label: owned ? 'Owned' : 'Not Claimed',
                  color: owned ? Colors.green : Colors.orange,
                  filled: owned,
                ),
                _buildStatusBadge(
                  icon: Icons.memory,
                  label: hasSeed ? 'Seed Installed' : 'No Seed',
                  color: hasSeed ? Colors.blue : Colors.grey,
                  filled: hasSeed,
                ),
                _buildStatusBadge(
                  icon: Icons.cloud_done,
                  label: hasSubscription ? 'Subscribed' : 'No Subscription',
                  color: hasSubscription ? Colors.purple : Colors.grey,
                  filled: hasSubscription,
                ),
                if (mismatch)
                  _buildStatusBadge(
                    icon: Icons.warning,
                    label: 'Account Mismatch',
                    color: Colors.red,
                    filled: true,
                  ),
              ],
            ),
          ),

          const SizedBox(height: 12),

          // Error message if any
          if (_errorByHost[host] != null)
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16),
              child: Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: Colors.red.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.red.shade200),
                ),
                child: Row(
                  children: [
                    Icon(Icons.error_outline, size: 16, color: Colors.red.shade700),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        _errorByHost[host]!,
                        style: TextStyle(color: Colors.red.shade700, fontSize: 12),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 12),

          // Primary action buttons
          Padding(
            padding: const EdgeInsets.symmetric(horizontal: 16),
            child: Row(
              children: [
                if (!owned)
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: (isGuest || isBusy || mismatch)
                          ? null
                          : () => _claimRobot(data),
                      icon: const Icon(Icons.how_to_reg, size: 18),
                      label: const Text('Claim Ownership'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: Colors.green,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                if (owned) ...[
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: (isGuest || isBusy || mismatch || !hasSeed || !hasSubscription)
                          ? null
                          : () => _connectOrGateRemote(data),
                      icon: const Icon(Icons.play_arrow, size: 18),
                      label: const Text('Connect'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: ContinuonColors.primaryBlue,
                        foregroundColor: Colors.white,
                        padding: const EdgeInsets.symmetric(vertical: 12),
                      ),
                    ),
                  ),
                  const SizedBox(width: 8),
                  OutlinedButton.icon(
                    onPressed: isBusy ? null : () => _openRobotWebUI(data),
                    icon: const Icon(Icons.open_in_browser, size: 18),
                    label: const Text('Web UI'),
                    style: OutlinedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 16),
                    ),
                  ),
                ],
              ],
            ),
          ),

          // Guest mode notice
          if (isGuest)
            Padding(
              padding: const EdgeInsets.fromLTRB(16, 12, 16, 0),
              child: Container(
                padding: const EdgeInsets.all(12),
                decoration: BoxDecoration(
                  color: Colors.orange.shade50,
                  borderRadius: BorderRadius.circular(8),
                  border: Border.all(color: Colors.orange.shade200),
                ),
                child: Row(
                  children: [
                    Icon(Icons.info_outline, size: 16, color: Colors.orange.shade700),
                    const SizedBox(width: 8),
                    Expanded(
                      child: Text(
                        'Sign in to claim ownership and manage this robot.',
                        style: TextStyle(color: Colors.orange.shade700, fontSize: 12),
                      ),
                    ),
                  ],
                ),
              ),
            ),

          const SizedBox(height: 8),

          // Expandable Management Section (only for owned robots)
          if (owned && !isGuest)
            Theme(
              data: Theme.of(context).copyWith(dividerColor: Colors.transparent),
              child: ExpansionTile(
                tilePadding: const EdgeInsets.symmetric(horizontal: 16),
                childrenPadding: const EdgeInsets.fromLTRB(16, 0, 16, 16),
                title: Text(
                  'Robot Management',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: Colors.grey.shade700,
                  ),
                ),
                leading: Icon(Icons.settings, size: 20, color: Colors.grey.shade600),
                children: [
                  _buildManagementSection(data, isBusy, mismatch, hasSeed),
                ],
              ),
            ),

          if (!owned || isGuest)
            const SizedBox(height: 8),
        ],
      ),
    );
  }

  Widget _buildStatusBadge({
    required IconData icon,
    required String label,
    required Color color,
    required bool filled,
  }) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
      decoration: BoxDecoration(
        color: filled ? color.withValues(alpha: 0.15) : Colors.grey.shade100,
        borderRadius: BorderRadius.circular(20),
        border: Border.all(
          color: filled ? color.withValues(alpha: 0.5) : Colors.grey.shade300,
        ),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: filled ? color : Colors.grey),
          const SizedBox(width: 6),
          Text(
            label,
            style: TextStyle(
              fontSize: 12,
              fontWeight: FontWeight.w500,
              color: filled ? color : Colors.grey.shade600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildManagementSection(Map<String, dynamic> data, bool isBusy, bool mismatch, bool hasSeed) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        // AI Model Management
        Text(
          'AI Model',
          style: Theme.of(context).textTheme.labelMedium?.copyWith(
            fontWeight: FontWeight.w600,
            color: Colors.grey.shade800,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _buildManagementButton(
                icon: Icons.download,
                label: hasSeed ? 'Update Seed Model' : 'Install Seed Model',
                color: Colors.blue,
                onPressed: (isBusy || mismatch) ? null : () => _installSeed(data),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildManagementButton(
                icon: Icons.model_training,
                label: 'Manage Models',
                color: Colors.indigo,
                onPressed: isBusy ? null : () => _showModelManagerDialog(data),
              ),
            ),
          ],
        ),

        const SizedBox(height: 16),

        // OTA Updates
        Text(
          'Software Updates (OTA)',
          style: Theme.of(context).textTheme.labelMedium?.copyWith(
            fontWeight: FontWeight.w600,
            color: Colors.grey.shade800,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _buildManagementButton(
                icon: Icons.system_update,
                label: 'Update Slow Loop',
                color: Colors.teal,
                onPressed: (isBusy || mismatch) ? null : () => _updateSlowLoop(data),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildManagementButton(
                icon: Icons.update,
                label: 'Check for Updates',
                color: Colors.green,
                onPressed: isBusy ? null : () => _checkForUpdates(data),
              ),
            ),
          ],
        ),

        const SizedBox(height: 16),

        // Robot Settings
        Text(
          'Settings & Ownership',
          style: Theme.of(context).textTheme.labelMedium?.copyWith(
            fontWeight: FontWeight.w600,
            color: Colors.grey.shade800,
          ),
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _buildManagementButton(
                icon: Icons.edit,
                label: 'Rename Robot',
                color: Colors.grey.shade700,
                onPressed: () => _showRenameDialog(data),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildManagementButton(
                icon: Icons.swap_horiz,
                label: 'Transfer Ownership',
                color: Colors.orange,
                onPressed: isBusy ? null : () => _showTransferOwnershipDialog(data),
              ),
            ),
          ],
        ),
        const SizedBox(height: 8),
        Row(
          children: [
            Expanded(
              child: _buildManagementButton(
                icon: Icons.business_center,
                label: 'Leasing & Rental',
                color: Colors.purple,
                onPressed: () => _showLeasingDialog(data),
              ),
            ),
            const SizedBox(width: 8),
            Expanded(
              child: _buildManagementButton(
                icon: Icons.delete_outline,
                label: 'Remove Robot',
                color: Colors.red,
                onPressed: () => _showDeleteDialog(data),
              ),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildManagementButton({
    required IconData icon,
    required String label,
    required Color color,
    VoidCallback? onPressed,
  }) {
    return OutlinedButton(
      onPressed: onPressed,
      style: OutlinedButton.styleFrom(
        foregroundColor: color,
        side: BorderSide(color: color.withValues(alpha: 0.5)),
        padding: const EdgeInsets.symmetric(vertical: 12, horizontal: 12),
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(icon, size: 16),
          const SizedBox(width: 6),
          Flexible(
            child: Text(
              label,
              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.w500),
              overflow: TextOverflow.ellipsis,
            ),
          ),
        ],
      ),
    );
  }

  void _showModelManagerDialog(Map<String, dynamic> data) {
    final host = data['host'] as String;
    final httpPort = data['httpPort'] as int? ?? 8080;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Manage AI Models'),
        content: SizedBox(
          width: 400,
          child: Column(
            mainAxisSize: MainAxisSize.min,
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              const Text('Available actions:'),
              const SizedBox(height: 16),
              ListTile(
                leading: const Icon(Icons.list),
                title: const Text('List Installed Models'),
                subtitle: const Text('View all models on the robot'),
                onTap: () async {
                  Navigator.pop(context);
                  await _brainClient.connect(host: host, port: 50051, httpPort: httpPort, useTls: false);
                  final models = await _brainClient.listModels();
                  if (mounted) {
                    _showSnack('Models: ${models['models']?.length ?? 0} found');
                  }
                },
              ),
              ListTile(
                leading: const Icon(Icons.cloud_download),
                title: const Text('Download New Model'),
                subtitle: const Text('Fetch latest models from cloud'),
                onTap: () {
                  Navigator.pop(context);
                  _showSnack('Model download feature coming soon');
                },
              ),
            ],
          ),
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  Future<void> _updateSlowLoop(Map<String, dynamic> data) async {
    final host = data['host'] as String;
    final httpPort = data['httpPort'] as int? ?? 8080;

    final confirm = await showDialog<bool>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Update Slow Loop'),
        content: const Text(
          'This will update the slow loop (training/inference) component on your robot. '
          'The robot may restart during this process.\n\n'
          'Do you want to continue?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context, false),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.pop(context, true),
            style: ElevatedButton.styleFrom(backgroundColor: Colors.teal),
            child: const Text('Update'),
          ),
        ],
      ),
    );

    if (confirm != true) return;

    setState(() => _busyHosts.add(host));

    try {
      // Call the OTA update endpoint
      final uri = Uri.http('$host:$httpPort', '/api/ota/update_slow_loop');
      final response = await http.post(uri, headers: _authToken != null ? {'authorization': 'Bearer $_authToken'} : {});

      if (response.statusCode == 200) {
        _showSnack('Slow loop update initiated successfully');
      } else {
        _showSnack('Update failed: HTTP ${response.statusCode}');
      }
    } catch (e) {
      _showSnack('Update failed: $e');
    } finally {
      if (mounted) setState(() => _busyHosts.remove(host));
    }
  }

  Future<void> _checkForUpdates(Map<String, dynamic> data) async {
    final host = data['host'] as String;
    final httpPort = data['httpPort'] as int? ?? 8080;

    setState(() => _busyHosts.add(host));

    try {
      final uri = Uri.http('$host:$httpPort', '/api/ota/check');
      final response = await http.get(uri, headers: _authToken != null ? {'authorization': 'Bearer $_authToken'} : {});

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body);
        final hasUpdates = data['updates_available'] == true;
        if (hasUpdates) {
          _showSnack('Updates available! Check Robot Management to install.');
        } else {
          _showSnack('Robot is up to date');
        }
      } else {
        _showSnack('Could not check for updates: HTTP ${response.statusCode}');
      }
    } catch (e) {
      _showSnack('Could not check for updates: $e');
    } finally {
      if (mounted) setState(() => _busyHosts.remove(host));
    }
  }

  Future<void> _refreshStatus(Map<String, dynamic> data) async {
    final host = data['host'] as String? ?? '';
    final httpPort = data['httpPort'] as int? ?? 8080;
    if (_authToken != null) {
      _brainClient.setAuthToken(_authToken!);
    }
    setState(() => _busyHosts.add(host));
    try {
      final status = await _brainClient.fetchOwnershipStatus(
          host: host, httpPort: httpPort);
      final ping = await _brainClient.ping(host: host, httpPort: httpPort);
      if (mounted) {
        setState(() {
          final owned = status['owned'] == true;
          final sub = status['subscription_active'] == true;
          final seed = status['seed_installed'] == true;
          _ownedByHost[host] = owned;
          _subByHost[host] = sub;
          _seedByHost[host] = seed;
          if (ping.isNotEmpty) {
            _deviceInfoByHost[host] = ping;
          }
          _errorByHost.remove(host);
          // Update global flags based on any known robot status
          _isOwned = _ownedByHost.values.any((v) => v);
          _hasSubscription = _subByHost.values.any((v) => v);
          _hasSeedInstalled = _seedByHost.values.any((v) => v);
          _busyHosts.remove(host);
        });
        _saveCachedState();
        if (status.isEmpty) {
          _errorByHost[host] = 'Status fetch failed';
          _showSnack('Status fetch failed for $host');
        } else if (ping.isEmpty) {
          _errorByHost[host] = 'Ping failed';
          _showSnack('Ping failed for $host');
        } else if (_accountId != null &&
            status['account_id'] != null &&
            status['account_id'] != _accountId) {
          _errorByHost[host] = 'Account mismatch';
          _showSnack('Account mismatch for $host');
        }
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _errorByHost[host] = 'Connection error: ${e.toString()}';
          _busyHosts.remove(host);
        });
        _showSnack('Failed to refresh status: $e');
      }
    }
  }

  Future<void> _connectOrGateRemote(Map<String, dynamic> data) async {
    if (_user == null) {
      _showSnack('Sign in to control robots. Guest mode is view-only.');
      return;
    }
    final host = data['host'] as String? ?? '';
    final port = data['port'] as int? ?? 50051;
    final httpPort = data['httpPort'] as int? ?? 8080;

    // Refresh status, but don't hard block on LAN-ownership + subscription + seed are required.
    await _refreshStatus(data);

    final owned = _ownedByHost[host] ?? _isOwned;
    final hasSubscription = _subByHost[host] ?? _hasSubscription;
    final hasSeed = _seedByHost[host] ?? _hasSeedInstalled;

    if (owned && hasSubscription && hasSeed) {
      _connectToRobot({
        'name': data['name'] ?? 'Unnamed Robot',
        'host': host,
        'port': port,
        'httpPort': httpPort,
      });
    } else {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: const Text(
                'Remote control requires ownership + subscription + seed install.'),
            backgroundColor: Theme.of(context).colorScheme.error,
          ),
        );
      }
    }
  }

  Future<void> _claimRobot(Map<String, dynamic> data) async {
    if (_user == null) {
      _showSnack('Sign in to claim robots. Guest mode is view-only.');
      return;
    }
    final host = data['host'] as String? ?? '';
    if (_authToken != null) {
      _brainClient.setAuthToken(_authToken!);
    }
    setState(() => _busyHosts.add(host));
    final ok = await _brainClient.claimRobot(
      host: host,
      httpPort: data['httpPort'] as int? ?? 8080,
      accountId: _accountId,
      accountType: _accountType,
      ownerId: _ownerId,
    );
    if (mounted) {
      setState(() {
        if (ok) {
          _ownedByHost[host] = true;
          _isOwned = _ownedByHost.values.any((v) => v);
          _saveCachedState();
        }
        _busyHosts.remove(host);
      });
      _showSnack(ok ? 'Claimed robot successfully.' : 'Claim failed.');
    }
  }

  Future<void> _installSeed(Map<String, dynamic> data) async {
    final host = data['host'] as String? ?? '';
    if (_authToken != null) {
      _brainClient.setAuthToken(_authToken!);
    }
    setState(() => _busyHosts.add(host));
    final ok = await _brainClient.installSeedBundle(
        host: host, httpPort: data['httpPort'] as int? ?? 8080);
    if (mounted) {
      setState(() {
        if (ok) {
          _seedByHost[host] = true;
          _hasSeedInstalled = _seedByHost.values.any((v) => v);
          _saveCachedState();
        }
        _busyHosts.remove(host);
      });
      _showSnack(ok ? 'Seed bundle installed.' : 'Seed install failed.');
    }
  }

  void _openRobotWebUI(Map<String, dynamic> data) {
    final host = data['host'] as String;
    final httpPort = data['httpPort'] as int? ?? 8080;
    final port = data['port'] as int? ?? 50051;
    final robotName = data['name'] as String? ?? 'Robot';

    Navigator.push(
      context,
      MaterialPageRoute(
        builder: (context) => RobotPortalScreen(
          host: host,
          httpPort: httpPort,
          port: port,
          robotName: robotName,
        ),
      ),
    );
  }

  void _showRenameDialog(Map<String, dynamic> data) {
    final currentName = data['name'] as String? ?? 'Unnamed Robot';
    final nameController = TextEditingController(text: currentName);
    final documentId = data['_documentId'] as String?;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Rename Robot'),
        content: TextField(
          controller: nameController,
          decoration: const InputDecoration(
            labelText: 'Robot Name',
            hintText: 'Enter a new name for this robot',
          ),
          autofocus: true,
          onSubmitted: (_) {
            if (nameController.text.trim().isNotEmpty) {
              Navigator.pop(context);
              _renameRobot(data, nameController.text.trim(), documentId);
            }
          },
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              if (nameController.text.trim().isNotEmpty) {
                Navigator.pop(context);
                _renameRobot(data, nameController.text.trim(), documentId);
              }
            },
            child: const Text('Save'),
          ),
        ],
      ),
    );
  }

  Future<void> _renameRobot(
      Map<String, dynamic> data, String newName, String? documentId) async {
    if (_user == null) {
      // Guest mode: update local list
      setState(() {
        final index = _guestRobots.indexWhere(
            (r) => r['host'] == data['host'] && r['port'] == data['port']);
        if (index >= 0) {
          _guestRobots[index]['name'] = newName;
        }
      });
      _showSnack('Robot renamed to "$newName"');
      return;
    }

    if (documentId == null) {
      _showSnack('Cannot rename: robot not found in database');
      return;
    }

    // _user is guaranteed to be non-null here due to early return above
    final userId = _user.uid;

    try {
      await FirebaseFirestore.instance
          .collection('users')
          .doc(userId)
          .collection('robots')
          .doc(documentId)
          .update({'name': newName});
      _showSnack('Robot renamed to "$newName"');
    } catch (e) {
      _showSnack('Failed to rename robot: $e');
    }
  }

  void _showDeleteDialog(Map<String, dynamic> data) {
    final name = data['name'] as String? ?? 'this robot';
    final documentId = data['_documentId'] as String?;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Delete Robot'),
        content: Text(
            'Are you sure you want to delete "$name"? This will remove it from your list but will not affect the robot itself.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              _deleteRobot(data, documentId);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.red,
              foregroundColor: Colors.white,
            ),
            child: const Text('Delete'),
          ),
        ],
      ),
    );
  }

  Future<void> _deleteRobot(
      Map<String, dynamic> data, String? documentId) async {
    if (_user == null) {
      // Guest mode: remove from local list
      setState(() {
        _guestRobots.removeWhere(
            (r) => r['host'] == data['host'] && r['port'] == data['port']);
      });
      _showSnack('Robot removed from list');
      return;
    }

    if (documentId == null) {
      _showSnack('Cannot delete: robot not found in database');
      return;
    }

    // _user is guaranteed to be non-null here due to early return above
    final userId = _user.uid;

    try {
      await FirebaseFirestore.instance
          .collection('users')
          .doc(userId)
          .collection('robots')
          .doc(documentId)
          .delete();
      _showSnack('Robot deleted');
    } catch (e) {
      _showSnack('Failed to delete robot: $e');
    }
  }

  void _showTransferOwnershipDialog(Map<String, dynamic> data) {
    final name = data['name'] as String? ?? 'this robot';
    final host = data['host'] as String? ?? '';
    final httpPort = data['httpPort'] as int? ?? 8080;
    final emailController = TextEditingController();
    final documentId = data['_documentId'] as String?;

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Transfer Ownership'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Transfer ownership of "$name" to another user.'),
            const SizedBox(height: 16),
            TextField(
              controller: emailController,
              decoration: const InputDecoration(
                labelText: 'New Owner Email',
                hintText: 'user@example.com',
              ),
              keyboardType: TextInputType.emailAddress,
            ),
            const SizedBox(height: 8),
            Text(
              'Note: The new owner must accept the transfer. This action cannot be undone.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.orange.shade700,
                  ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              final email = emailController.text.trim();
              if (email.isNotEmpty && email.contains('@')) {
                Navigator.pop(context);
                _transferOwnership(data, email, host, httpPort, documentId);
              } else {
                ScaffoldMessenger.of(context).showSnackBar(
                  const SnackBar(content: Text('Please enter a valid email')),
                );
              }
            },
            child: const Text('Transfer'),
          ),
        ],
      ),
    );
  }

  Future<void> _transferOwnership(
      Map<String, dynamic> data,
      String newOwnerEmail,
      String host,
      int httpPort,
      String? documentId) async {
    if (_user == null) {
      _showSnack('Sign in to transfer ownership');
      return;
    }

    setState(() => _busyHosts.add(host));

    try {
      // First, call the robot's API to transfer ownership
      if (_authToken != null) {
        _brainClient.setAuthToken(_authToken!);
      }

      // Note: This would require a new API endpoint on the robot
      // For now, we'll show a message that this feature needs robot-side support
      _showSnack(
          'Transfer ownership requires robot API support. Please use the robot web UI to transfer ownership.');

      // TODO: Implement actual transfer via robot API when available
      // await _brainClient.transferOwnership(
      //   host: host,
      //   httpPort: httpPort,
      //   newOwnerEmail: newOwnerEmail,
      // );

      // Optionally remove from current user's list after successful transfer
      // if (documentId != null) {
      //   await FirebaseFirestore.instance
      //       .collection('users')
      //       .doc(_user!.uid)
      //       .collection('robots')
      //       .doc(documentId)
      //       .delete();
      // }
    } catch (e) {
      _showSnack('Failed to transfer ownership: $e');
    } finally {
      if (mounted) setState(() => _busyHosts.remove(host));
    }
  }

  void _showLeasingDialog(Map<String, dynamic> data) {
    final name = data['name'] as String? ?? 'this robot';

    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Leasing & Rental'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Manage leasing and rental options for "$name".'),
            const SizedBox(height: 16),
            const Text(
              'Available Options:',
              style: TextStyle(fontWeight: FontWeight.bold),
            ),
            const SizedBox(height: 8),
            const Text('• Set rental rates per hour/day'),
            const Text('• Enable/disable public job listings'),
            const Text('• Manage active leases'),
            const Text('• View rental history'),
            const SizedBox(height: 16),
            Text(
              'This feature requires robot API support and will be available in a future update.',
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    color: Colors.grey,
                    fontStyle: FontStyle.italic,
                  ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              _openRobotWebUI(data); // Open robot web UI for now
            },
            child: const Text('Open Robot Web UI'),
          ),
        ],
      ),
    );
  }

  void _showSnack(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  Future<void> _persistLanLikely(bool lanLikely) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setBool('lan_likely', lanLikely);
  }

  Future<void> _loadCachedState() async {
    if (_stateLoaded) return;
    final prefs = await SharedPreferences.getInstance();
    final owned = prefs.getString('robot_owned_map');
    final sub = prefs.getString('robot_sub_map');
    final seed = prefs.getString('robot_seed_map');
    final cachedLan = prefs.getBool('lan_likely');
    final cachedOwnedFlag = prefs.getBool('robot_any_owned');
    final cachedSubFlag = prefs.getBool('robot_any_sub');
    final cachedSeedFlag = prefs.getBool('robot_any_seed');
    if (owned != null) {
      _ownedByHost
        ..clear()
        ..addAll(Map<String, dynamic>.from(jsonDecode(owned))
            .map((k, v) => MapEntry(k, v == true)));
    }
    if (sub != null) {
      _subByHost
        ..clear()
        ..addAll(Map<String, dynamic>.from(jsonDecode(sub))
            .map((k, v) => MapEntry(k, v == true)));
    }
    if (seed != null) {
      _seedByHost
        ..clear()
        ..addAll(Map<String, dynamic>.from(jsonDecode(seed))
            .map((k, v) => MapEntry(k, v == true)));
    }
    final derivedOwned = cachedOwnedFlag ?? _ownedByHost.values.any((v) => v);
    final derivedSub = cachedSubFlag ?? _subByHost.values.any((v) => v);
    final derivedSeed = cachedSeedFlag ?? _seedByHost.values.any((v) => v);
    if (!mounted) {
      _isOwned = derivedOwned;
      _hasSubscription = derivedSub;
      _hasSeedInstalled = derivedSeed;
      if (cachedLan != null) {
        _lanLikely = cachedLan;
      }
      _stateLoaded = true;
      return;
    }
    setState(() {
      _isOwned = derivedOwned;
      _hasSubscription = derivedSub;
      _hasSeedInstalled = derivedSeed;
      if (cachedLan != null) {
        _lanLikely = cachedLan;
      }
      _stateLoaded = true;
    });
  }

  Future<void> _saveCachedState() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('robot_owned_map', jsonEncode(_ownedByHost));
    await prefs.setString('robot_sub_map', jsonEncode(_subByHost));
    await prefs.setString('robot_seed_map', jsonEncode(_seedByHost));
    await prefs.setBool(
        'robot_any_owned', _ownedByHost.values.any((v) => v) || _isOwned);
    await prefs.setBool('robot_any_sub',
        _subByHost.values.any((v) => v) || _hasSubscription);
    await prefs.setBool('robot_any_seed',
        _seedByHost.values.any((v) => v) || _hasSeedInstalled);
    await prefs.setBool('lan_likely', _lanLikely);
  }
}

class _AddRobotDialog extends StatefulWidget {
  final String? userId; // Nullable for Guest Mode
  final Function(Map<String, dynamic>)? onGuestAdd;

  const _AddRobotDialog({this.userId, this.onGuestAdd});

  @override
  State<_AddRobotDialog> createState() => _AddRobotDialogState();
}

class _AddRobotDialogState extends State<_AddRobotDialog> {
  final _formKey = GlobalKey<FormState>();
  late final TextEditingController _nameController;
  late final TextEditingController _hostController;
  late final TextEditingController _portController;
  late final TextEditingController _httpPortController;
  bool _saving = false;

  @override
  void initState() {
    super.initState();
    _nameController = TextEditingController(text: '');
    _hostController = TextEditingController(text: '');
    _portController = TextEditingController(text: '50051');
    _httpPortController = TextEditingController(text: '8080');
  }

  @override
  void dispose() {
    _nameController.dispose();
    _hostController.dispose();
    _portController.dispose();
    _httpPortController.dispose();
    super.dispose();
  }

  Future<void> _save() async {
    if (!_formKey.currentState!.validate()) return;

    setState(() => _saving = true);

    try {
      final robotData = {
        'name': _nameController.text.trim(),
        'host': _hostController.text.trim(),
        'port': int.parse(_portController.text.trim()),
        'httpPort': int.parse(_httpPortController.text.trim()),
        'createdAt': FieldValue.serverTimestamp(),
      };

      if (widget.userId != null) {
        // Save to Firestore
        await FirebaseFirestore.instance
            .collection('users')
            .doc(widget.userId!)
            .collection('robots')
            .add(robotData);
      } else {
        // Guest mode: Callback to parent
        widget.onGuestAdd?.call(robotData);
      }

      if (mounted) {
        Navigator.pop(context,
            robotData); // Return robotData so parent can refresh status
      }
    } catch (e) {
      if (mounted) {
        setState(() => _saving = false);
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Error saving: $e')),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Add Robot'),
      content: Form(
        key: _formKey,
        child: SingleChildScrollView(
          child: Column(
            mainAxisSize: MainAxisSize.min,
            children: [
              TextFormField(
                controller: _nameController,
                decoration:
                    const InputDecoration(labelText: 'Name (e.g. Home Robot)'),
                validator: (v) => v?.isEmpty == true ? 'Required' : null,
              ),
              TextFormField(
                controller: _hostController,
                decoration: const InputDecoration(
                    labelText: 'Host IP (e.g. 192.168.1.x)'),
                validator: (v) => v?.isEmpty == true ? 'Required' : null,
              ),
              Row(
                children: [
                  Expanded(
                    child: TextFormField(
                      controller: _portController,
                      decoration: const InputDecoration(labelText: 'gRPC Port'),
                      keyboardType: TextInputType.number,
                      validator: (v) => v?.isEmpty == true ? 'Required' : null,
                    ),
                  ),
                  const SizedBox(width: 16),
                  Expanded(
                    child: TextFormField(
                      controller: _httpPortController,
                      decoration: const InputDecoration(labelText: 'HTTP Port'),
                      keyboardType: TextInputType.number,
                      validator: (v) => v?.isEmpty == true ? 'Required' : null,
                    ),
                  ),
                ],
              ),
              const SizedBox(height: 24),
              const ExpansionTile(
                title: Text('How to find connection details?',
                    style: TextStyle(fontSize: 14)),
                children: [
                  Padding(
                    padding: EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                            '• Host IP: Local IP of your robot (e.g., 192.168.1.x).',
                            style: TextStyle(
                                fontSize: 12, fontWeight: FontWeight.bold)),
                        Text(
                            '  - On robot UI: Settings → Network shows the IP.\n  - Router admin page: find connected devices.\n  - Network scanner (Fing): look for "raspberrypi" or "continuon".',
                            style: TextStyle(fontSize: 12, color: Colors.grey)),
                        SizedBox(height: 12),
                        Text('• gRPC Port: Default 50051.',
                            style: TextStyle(
                                fontSize: 12, fontWeight: FontWeight.bold)),
                        Text(
                            '  - Real-time control/video. If 50051 fails, try 50052 or check robot config.',
                            style: TextStyle(fontSize: 12, color: Colors.grey)),
                        SizedBox(height: 12),
                        Text('• HTTP Port: Default 8080.',
                            style: TextStyle(
                                fontSize: 12, fontWeight: FontWeight.bold)),
                        Text(
                            '  - Status/settings. Try http://<robot-ip>:8080 in a browser.',
                            style: TextStyle(fontSize: 12, color: Colors.grey)),
                        SizedBox(height: 12),
                        Text(
                            '• Make sure your phone is on the same Wi‑Fi/hotspot as the robot before claiming.',
                            style: TextStyle(
                                fontSize: 12,
                                fontWeight: FontWeight.bold,
                                color: ContinuonColors.primaryBlue)),
                      ],
                    ),
                  ),
                ],
              ),
            ],
          ),
        ),
      ),
      actions: [
        TextButton(
          onPressed: _saving ? null : () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
        ElevatedButton(
          onPressed: _saving ? null : _save,
          child: _saving
              ? const SizedBox(
                  width: 16,
                  height: 16,
                  child: CircularProgressIndicator(strokeWidth: 2))
              : const Text('Save'),
        ),
      ],
    );
  }
}
