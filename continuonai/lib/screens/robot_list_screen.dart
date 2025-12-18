import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import 'package:shared_preferences/shared_preferences.dart';
import 'dart:convert';

import '../theme/continuon_theme.dart';

import 'dashboard_screen.dart';

import 'pair_robot_screen.dart';
import '../services/brain_client.dart';
import '../services/scanner_service.dart';
import '../widgets/layout/continuon_layout.dart';
import '../widgets/layout/continuon_card.dart';

class RobotListScreen extends StatefulWidget {
  const RobotListScreen({super.key});

  static const routeName = '/robots';

  @override
  State<RobotListScreen> createState() => _RobotListScreenState();
}

class _RobotListScreenState extends State<RobotListScreen> {
  final User? _user = FirebaseAuth.instance.currentUser;
  final BrainClient _brainClient = BrainClient();
  // TODO: wire real LAN detection + persisted ownership/subscription/seed state
  bool _isLocalNetwork = true;
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
  bool _tokenLoaded = false;
  bool _lanLikely = true;
  bool _stateLoaded = false;

  // Local list for guest mode
  final List<Map<String, dynamic>> _guestRobots = [
    {
      'name': 'Demo Robot',
      'host': '192.168.1.100',
      'port': 50051,
      'httpPort': 8080
    },
  ];

  // _signOut handled by ContinuonAppBar now

  Future<void> _openPairing() async {
    await Navigator.pushNamed(context, PairRobotScreen.routeName);
    if (mounted) {
      // Best-effort refresh after pairing.
      setState(() {});
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
        },
      ),
    );
  }

  Future<void> _connectToRobot(Map<String, dynamic> data) async {
    final host = data['host'] as String;
    final port = data['port'] as int;
    final httpPort = data['httpPort'] as int? ?? 8080;

    // Propagate auth token if present
    if (_authToken != null) {
      _brainClient.setAuthToken(_authToken!);
    }

    setState(() => _busyHosts.add(host));

    // Show loading dialog
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
        useTls: false, // Assuming local connection for now
      );

      if (mounted) {
        Navigator.pop(context); // Dismiss loading
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => DashboardScreen(brainClient: _brainClient),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        Navigator.pop(context); // Dismiss loading
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

  void _scanForRobots() {
    showDialog(
      context: context,
      builder: (context) => const _ScanRobotsDialog(),
    ).then((result) {
      if (!mounted) return;
      if (result != null && result is ScannedRobot) {
        // Pre-fill add dialog
        showDialog(
          context: context,
          builder: (context) => _AddRobotDialog(
            userId: _user?.uid,
            initialRobot: result,
            onGuestAdd: (robot) {
              setState(() {
                _guestRobots.add(robot);
              });
            },
          ),
        );
      }
    });
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
        if (mounted) {
          setState(() {
            _lanLikely = lan;
            _isLocalNetwork = lan;
          });
        }
      });
      _loadCachedState();
    }

    return ContinuonLayout(
      // 100% Consistent Nav: No screen-specific actions in Top Bar
      body: Column(
        children: [
          _buildStatusBanner(),
          _buildActionRow(), // New local action bar
          _buildHelpCard(),
          Expanded(
              child: _user == null ? _buildGuestList() : _buildFirestoreList()),
        ],
      ),
      floatingActionButton: TweenAnimationBuilder<double>(
        tween: Tween(begin: 0.0, end: 1.0),
        duration: const Duration(milliseconds: 500),
        curve: Curves.elasticOut,
        builder: (context, value, child) {
          return Transform.scale(
            scale: value,
            child: FloatingActionButton.extended(
              onPressed: _addRobot,
              backgroundColor: ContinuonColors.primaryBlue,
              icon: const Icon(Icons.add),
              label: const Text('Add Robot'),
              elevation: 4,
            ),
          );
        },
      ),
    );
  }

  void _showAuthTokenDialog() {
    _tokenController.text = _authToken ?? '';
    _accountIdController.text = _accountId ?? '';
    _accountTypeController.text = _accountType ?? '';
    _ownerIdController.text = _ownerId ?? '';
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
                const SizedBox(height: 12),
                TextField(
                  controller: _ownerIdController,
                  decoration: const InputDecoration(
                    labelText: 'Owner ID (display name)',
                    helperText: 'Shown in pairing/claim flows (optional).',
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
          const SizedBox(width: 12),
          OutlinedButton.icon(
            onPressed: _openPairing,
            icon: const Icon(Icons.qr_code_scanner, size: 18),
            label: const Text('Pair'),
          ),
          const SizedBox(width: 12),
          OutlinedButton.icon(
            onPressed: _scanForRobots,
            icon: const Icon(Icons.radar, size: 18),
            label: const Text('Scan'),
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
          Icon(Icons.info_outline, color: ContinuonColors.primaryBlue),
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
            final data = docs[index].data() as Map<String, dynamic>;
            return _buildAnimatedRobotCard(data, index);
          },
        );
      },
    );
  }

  Widget _buildHelpCard() {
    final show = !_lanLikely || !_isOwned || !_hasSeedInstalled;
    if (!show) return const SizedBox.shrink();
    return ContinuonCard(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.help_outline, size: 20, color: Colors.grey),
              const SizedBox(width: 8),
              Text(
                'How to connect a new robot',
                style: Theme.of(context)
                    .textTheme
                    .titleSmall
                    ?.copyWith(fontWeight: FontWeight.w600),
              ),
            ],
          ),
          const SizedBox(height: 12),
          _helpText(
              '1) Join the same Wi‑Fi/LAN as the robot (or the robot’s hotspot).'),
          _helpText(
              '2) Find the robot IP (status screen or router). Defaults: gRPC 50051, HTTP 8080.'),
          _helpText('3) Tap "Add Robot" and enter IP/ports.'),
          _helpText(
              '4) When on robot LAN, tap Claim (local), then Seed install. Remote control/OTA works after that.'),
        ],
      ),
    );
  }

  Widget _helpText(String text) => Padding(
        padding: const EdgeInsets.symmetric(vertical: 2),
        child: Text(text, style: Theme.of(context).textTheme.bodySmall),
      );

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
      padding: const EdgeInsets.all(20),
      onTap: isGuest ? null : () => _connectOrGateRemote(data),
      child: Row(
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
          const SizedBox(width: 16),
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
                    Container(
                      width: 8,
                      height: 8,
                      decoration: const BoxDecoration(
                        color: Colors.green, // Success green
                        shape: BoxShape.circle,
                      ),
                    ),
                    const SizedBox(width: 8),
                    Text(host, style: Theme.of(context).textTheme.bodyMedium),
                    if (deviceId.isNotEmpty) ...[
                      const SizedBox(width: 8),
                      Text('id:$deviceId',
                          style: Theme.of(context)
                              .textTheme
                              .bodySmall
                              ?.copyWith(color: Colors.grey)),
                    ],
                  ],
                ),
              ],
            ),
          ),
          Column(
            crossAxisAlignment: CrossAxisAlignment.end,
            children: [
              Row(
                mainAxisSize: MainAxisSize.min,
                children: [
                  IconButton(
                    icon: const Icon(Icons.refresh),
                    tooltip: 'Refresh status',
                    onPressed:
                        (isBusy || isGuest) ? null : () => _refreshStatus(data),
                  ),
                  if (isBusy)
                    const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    ),
                  if (isBusy) const SizedBox(width: 12),
                  if (!_isOwned)
                    ElevatedButton.icon(
                      onPressed:
                          (isGuest || !_isLocalNetwork || isBusy || mismatch)
                              ? null
                              : () => _claimRobot(data),
                      icon: const Icon(Icons.how_to_reg),
                      label: const Text('Claim (local)'),
                    ),
                  if (_isOwned && !_hasSeedInstalled) const SizedBox(width: 8),
                  if (_isOwned && !_hasSeedInstalled)
                    ElevatedButton.icon(
                      onPressed:
                          (isGuest || !_isLocalNetwork || isBusy || mismatch)
                              ? null
                              : () => _installSeed(data),
                      icon: const Icon(Icons.system_update),
                      label: const Text('Seed install'),
                    ),
                  if (_isOwned && _hasSeedInstalled) const SizedBox(width: 8),
                  if (_isOwned && _hasSeedInstalled)
                    ElevatedButton.icon(
                      onPressed:
                          (isGuest || !_hasSubscription || isBusy || mismatch)
                              ? null
                              : () => _connectOrGateRemote(data),
                      icon: const Icon(Icons.power_settings_new),
                      label: const Text('Connect'),
                      style: ElevatedButton.styleFrom(
                        backgroundColor: ContinuonColors.primaryBlue,
                        foregroundColor: Colors.white,
                        elevation: 2,
                      ),
                    ),
                ],
              ),
              if (isGuest)
                Padding(
                  padding: const EdgeInsets.only(top: 8.0),
                  child: Text(
                    'Guest preview: sign in to enable control, recording, and OTA.',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.orange.shade700),
                  ),
                ),
              if (mismatch && !isBusy)
                Padding(
                  padding: const EdgeInsets.only(left: 8.0),
                  child: Text(
                    'Account mismatch',
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.red),
                  ),
                ),
              if (_errorByHost[host] != null)
                Padding(
                  padding: const EdgeInsets.only(top: 4),
                  child: Text(
                    _errorByHost[host]!,
                    style: Theme.of(context)
                        .textTheme
                        .bodySmall
                        ?.copyWith(color: Colors.red),
                  ),
                ),
            ],
          ),
        ],
      ),
    );
  }

  Future<void> _refreshStatus(Map<String, dynamic> data) async {
    final host = data['host'] as String? ?? '';
    final httpPort = data['httpPort'] as int? ?? 8080;
    if (_authToken != null) {
      _brainClient.setAuthToken(_authToken!);
    }
    setState(() => _busyHosts.add(host));
    final status =
        await _brainClient.fetchOwnershipStatus(host: host, httpPort: httpPort);
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
        _isOwned = owned;
        _hasSubscription = sub;
        _hasSeedInstalled = seed;
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
  }

  Future<void> _connectOrGateRemote(Map<String, dynamic> data) async {
    if (_user == null) {
      _showSnack('Sign in to control robots. Guest mode is view-only.');
      return;
    }
    final host = data['host'] as String? ?? '';
    final port = data['port'] as int? ?? 50051;
    final httpPort = data['httpPort'] as int? ?? 8080;

    // Refresh status, but don't hard block on LAN—ownership + subscription + seed are required.
    await _refreshStatus(data);

    if (_isOwned && _hasSubscription && _hasSeedInstalled) {
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
          _isOwned = true;
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
          _hasSeedInstalled = true;
          _saveCachedState();
        }
        _busyHosts.remove(host);
      });
      _showSnack(ok ? 'Seed bundle installed.' : 'Seed install failed.');
    }
  }

  void _showSnack(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }

  Future<void> _loadCachedState() async {
    if (_stateLoaded) return;
    final prefs = await SharedPreferences.getInstance();
    final owned = prefs.getString('robot_owned_map');
    final sub = prefs.getString('robot_sub_map');
    final seed = prefs.getString('robot_seed_map');
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
    setState(() => _stateLoaded = true);
  }

  Future<void> _saveCachedState() async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString('robot_owned_map', jsonEncode(_ownedByHost));
    await prefs.setString('robot_sub_map', jsonEncode(_subByHost));
    await prefs.setString('robot_seed_map', jsonEncode(_seedByHost));
  }
}

class _ScanRobotsDialog extends StatefulWidget {
  const _ScanRobotsDialog();

  @override
  State<_ScanRobotsDialog> createState() => _ScanRobotsDialogState();
}

class _ScanRobotsDialogState extends State<_ScanRobotsDialog> {
  final _scanner = ScannerService();
  List<ScannedRobot> _robots = [];

  @override
  void initState() {
    super.initState();
    _scanner.startScan();
    _scanner.scannedRobots.listen((robots) {
      if (mounted) setState(() => _robots = robots);
    });
  }

  @override
  void dispose() {
    _scanner.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    return AlertDialog(
      title: const Text('Scanning for Robots...'),
      content: SizedBox(
        width: double.maxFinite,
        height: 300,
        child: Column(
          children: [
            const LinearProgressIndicator(),
            const SizedBox(height: 16),
            Expanded(
              child: _robots.isEmpty
                  ? const Center(
                      child: Text('Searching via WiFi (mDNS) & Bluetooth...'))
                  : ListView.builder(
                      itemCount: _robots.length,
                      itemBuilder: (context, index) {
                        final robot = _robots[index];
                        return ListTile(
                          leading:
                              Icon(robot.isBle ? Icons.bluetooth : Icons.wifi),
                          title: Text(robot.name),
                          subtitle: Text(robot.isBle
                              ? 'Bluetooth'
                              : '${robot.host}:${robot.port}'),
                          onTap: () => Navigator.pop(context, robot),
                        );
                      },
                    ),
            ),
          ],
        ),
      ),
      actions: [
        TextButton(
          onPressed: () => Navigator.pop(context),
          child: const Text('Cancel'),
        ),
      ],
    );
  }
}

class _AddRobotDialog extends StatefulWidget {
  final String? userId; // Nullable for Guest Mode
  final Function(Map<String, dynamic>)? onGuestAdd;
  final ScannedRobot? initialRobot;

  const _AddRobotDialog({this.userId, this.onGuestAdd, this.initialRobot});

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
    _nameController =
        TextEditingController(text: widget.initialRobot?.name ?? '');
    _hostController =
        TextEditingController(text: widget.initialRobot?.host ?? '');
    _portController = TextEditingController(
        text: widget.initialRobot?.port.toString() ?? '50051');
    _httpPortController = TextEditingController(
        text: widget.initialRobot?.httpPort.toString() ?? '8080');
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
        Navigator.pop(context);
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
              ExpansionTile(
                title: const Text('How to find connection details?',
                    style: TextStyle(fontSize: 14)),
                children: [
                  Padding(
                    padding: const EdgeInsets.all(16.0),
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: const [
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
