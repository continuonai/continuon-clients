import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';
import 'package:cloud_firestore/cloud_firestore.dart';
import '../theme/app_theme.dart';
import 'connect_screen.dart';
import 'dashboard_screen.dart';
import 'login_screen.dart';
import '../services/brain_client.dart';
import '../services/scanner_service.dart';

class RobotListScreen extends StatefulWidget {
  const RobotListScreen({super.key});

  static const routeName = '/robots';

  @override
  State<RobotListScreen> createState() => _RobotListScreenState();
}

class _RobotListScreenState extends State<RobotListScreen> {
  final User? _user = FirebaseAuth.instance.currentUser;
  final BrainClient _brainClient = BrainClient();
  
  // Local list for guest mode
  final List<Map<String, dynamic>> _guestRobots = [
    {'name': 'Demo Robot', 'host': '192.168.1.100', 'port': 50051, 'httpPort': 8080},
  ];

  Future<void> _signOut() async {
    await FirebaseAuth.instance.signOut();
    if (mounted) {
      Navigator.pushReplacementNamed(context, LoginScreen.routeName);
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
          SnackBar(content: Text('Failed to connect: $e'), backgroundColor: AppColors.dangerRed),
        );
      }
    }
  }



  void _scanForRobots() {
    showDialog(
      context: context,
      builder: (context) => const _ScanRobotsDialog(),
    ).then((result) {
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
    return Scaffold(
      backgroundColor: AppColors.background,
      appBar: AppBar(
        title: Text(_user == null ? 'My Robots (Guest)' : 'My Robots', style: const TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Colors.white,
        foregroundColor: AppColors.textPrimary,
        elevation: 0,
        actions: [
          IconButton(
            icon: const Icon(Icons.radar),
            tooltip: 'Scan for Robots',
            onPressed: _scanForRobots,
          ),
          IconButton(
            icon: const Icon(Icons.logout),
            onPressed: _signOut,
          ),
        ],
      ),
      body: _user == null ? _buildGuestList() : _buildFirestoreList(),
      floatingActionButton: TweenAnimationBuilder<double>(
        tween: Tween(begin: 0.0, end: 1.0),
        duration: const Duration(milliseconds: 500),
        curve: Curves.elasticOut,
        builder: (context, value, child) {
          return Transform.scale(
            scale: value,
            child: FloatingActionButton.extended(
              onPressed: _addRobot,
              backgroundColor: AppColors.primaryBlue,
              icon: const Icon(Icons.add),
              label: const Text('Add Robot'),
              elevation: 4,
            ),
          );
        },
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

  Widget _buildEmptyState() {
    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Container(
            padding: const EdgeInsets.all(24),
            decoration: BoxDecoration(
              color: AppColors.primaryBlue.withOpacity(0.1),
              shape: BoxShape.circle,
            ),
            child: const Icon(Icons.smart_toy_outlined, size: 64, color: AppColors.primaryBlue),
          ),
          const SizedBox(height: 24),
          const Text('No robots added yet', style: AppTextStyles.label),
          const SizedBox(height: 32),
          ElevatedButton.icon(
            onPressed: _addRobot,
            icon: const Icon(Icons.add),
            label: const Text('Add Robot'),
            style: ElevatedButton.styleFrom(
              backgroundColor: AppColors.primaryBlue,
              foregroundColor: Colors.white,
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(12)),
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

    return Container(
      margin: const EdgeInsets.only(bottom: 16),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(16),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          onTap: () => _connectToRobot(data),
          borderRadius: BorderRadius.circular(16),
          child: Padding(
            padding: const EdgeInsets.all(20),
            child: Row(
              children: [
                Container(
                  padding: const EdgeInsets.all(12),
                  decoration: BoxDecoration(
                    color: AppColors.primaryBlue.withOpacity(0.1),
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: const Icon(Icons.smart_toy, color: AppColors.primaryBlue, size: 28),
                ),
                const SizedBox(width: 16),
                Expanded(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(
                        name,
                        style: AppTextStyles.value.copyWith(fontSize: 18, fontWeight: FontWeight.w600),
                      ),
                      const SizedBox(height: 4),
                      Row(
                        children: [
                          Container(
                            width: 8,
                            height: 8,
                            decoration: const BoxDecoration(
                              color: AppColors.successGreen,
                              shape: BoxShape.circle,
                            ),
                          ),
                          const SizedBox(width: 8),
                          Text(host, style: AppTextStyles.label),
                        ],
                      ),
                    ],
                  ),
                ),
                Container(
                  padding: const EdgeInsets.all(8),
                  decoration: BoxDecoration(
                    color: Colors.grey[100],
                    borderRadius: BorderRadius.circular(8),
                  ),
                  child: const Icon(Icons.arrow_forward_ios, size: 16, color: AppColors.textSecondary),
                ),
              ],
            ),
          ),
        ),
      ),
    );
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
                  ? const Center(child: Text('Searching via WiFi (mDNS) & Bluetooth...'))
                  : ListView.builder(
                      itemCount: _robots.length,
                      itemBuilder: (context, index) {
                        final robot = _robots[index];
                        return ListTile(
                          leading: Icon(robot.isBle ? Icons.bluetooth : Icons.wifi),
                          title: Text(robot.name),
                          subtitle: Text(robot.isBle ? 'Bluetooth' : '${robot.host}:${robot.port}'),
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
    _nameController = TextEditingController(text: widget.initialRobot?.name ?? '');
    _hostController = TextEditingController(text: widget.initialRobot?.host ?? '');
    _portController = TextEditingController(text: widget.initialRobot?.port.toString() ?? '50051');
    _httpPortController = TextEditingController(text: widget.initialRobot?.httpPort.toString() ?? '8080');
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
                decoration: const InputDecoration(labelText: 'Name (e.g. Home Robot)'),
                validator: (v) => v?.isEmpty == true ? 'Required' : null,
              ),
              TextFormField(
                controller: _hostController,
                decoration: const InputDecoration(labelText: 'Host IP (e.g. 192.168.1.x)'),
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
              ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
              : const Text('Save'),
        ),
      ],
    );
  }
}
