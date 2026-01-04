import 'dart:async';

import 'package:flutter/material.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';
import 'control_screen.dart';
import '../widgets/creator_dashboard.dart';
import '../widgets/chat_interaction_widget.dart';

class DashboardScreen extends StatefulWidget {
  const DashboardScreen({super.key, required this.brainClient});

  static const routeName = '/dashboard';

  final BrainClient brainClient;

  @override
  State<DashboardScreen> createState() => _DashboardScreenState();
}

class _DashboardScreenState extends State<DashboardScreen> {
  Timer? _timer;
  Map<String, dynamic> _status = {};
  bool _loading = false;
  bool _refreshing = false;
  bool _openingSettings = false;
  bool _showControls = false;

  @override
  void initState() {
    super.initState();
    _refreshStatus();
    _timer =
        Timer.periodic(const Duration(seconds: 5), (_) => _refreshStatus());
  }

  @override
  void dispose() {
    _timer?.cancel();
    super.dispose();
  }

  Future<void> _refreshStatus() async {
    final status = await widget.brainClient.getRobotStatus();
    if (mounted) {
      setState(() => _status = status);
    }
  }

  /// Full refresh - fetches comprehensive status including chat and training
  Future<void> _fullRefresh() async {
    if (_refreshing) return;
    setState(() => _refreshing = true);
    try {
      final data = await widget.brainClient.refresh();
      if (mounted) {
        setState(() => _status = data);
      }
    } finally {
      if (mounted) setState(() => _refreshing = false);
    }
  }

  Future<void> _setMode(String mode) async {
    setState(() => _loading = true);
    final result = await widget.brainClient.setRobotMode(mode);
    if (mounted) {
      setState(() => _loading = false);
      if (result['success'] == true) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Mode changed to $mode')),
        );
        _refreshStatus();
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text('Failed: ${result['message'] ?? 'Unknown error'}'),
            backgroundColor: Theme.of(context).colorScheme.error,
          ),
        );
      }
    }
  }

  Future<void> _openStartupAndFlags() async {
    if (_openingSettings) return;
    setState(() => _openingSettings = true);
    try {
      final res = await widget.brainClient.getSettings();
      if (!mounted) return;
      if (res['success'] != true) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(
                'Failed to load settings: ${res['message'] ?? 'unknown error'}'),
            backgroundColor: Theme.of(context).colorScheme.error,
          ),
        );
        return;
      }

      final raw = res['settings'];
      final settings =
          raw is Map ? Map<String, dynamic>.from(raw) : <String, dynamic>{};

      await showModalBottomSheet<void>(
        context: context,
        isScrollControlled: true,
        showDragHandle: true,
        builder: (context) => _StartupFlagsSheet(
          brainClient: widget.brainClient,
          initialSettings: settings,
        ),
      );
    } finally {
      if (mounted) setState(() => _openingSettings = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final statusData =
        (_status['status'] as Map?)?.cast<String, dynamic>() ?? {};
    final mode = statusData['mode'] ?? 'unknown';
    final isRecording = statusData['is_recording'] ?? false;
    final allowMotion = statusData['allow_motion'] ?? false;
    final hardware =
        (statusData['detected_hardware'] as Map?)?.cast<String, dynamic>() ??
            {};

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('ContinuonAI Robotic Control',
            style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Theme.of(context).appBarTheme.backgroundColor,
        foregroundColor: Theme.of(context).appBarTheme.foregroundColor,
        elevation: 0,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(color: Theme.of(context).dividerColor, height: 1),
        ),
        actions: [
          IconButton(
            tooltip: 'Refresh status',
            onPressed: _refreshing ? null : _fullRefresh,
            icon: _refreshing
                ? const SizedBox(
                    width: 20,
                    height: 20,
                    child: CircularProgressIndicator(strokeWidth: 2),
                  )
                : const Icon(Icons.refresh),
          ),
          IconButton(
            tooltip: 'Startup & flags (GET/POST /api/settings)',
            onPressed: _openingSettings ? null : _openStartupAndFlags,
            icon: const Icon(Icons.tune),
          ),
          if (_loading)
            const Padding(
              padding: EdgeInsets.all(16.0),
              child: SizedBox(
                width: 16,
                height: 16,
                child: CircularProgressIndicator(strokeWidth: 2),
              ),
            ),
        ],
      ),
      body: Column(
        children: [
          // Health Signals Header
          Container(
            padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
            color: Theme.of(context).colorScheme.surface,
            child: Row(
              children: [
                _buildHealthBadge(
                  label: mode.toUpperCase(),
                  color: _getModeColor(mode),
                  icon: Icons.psychology,
                ),
                const SizedBox(width: 8),
                if (isRecording)
                  _buildHealthBadge(
                      label: 'REC',
                      color: Colors.red,
                      icon: Icons.fiber_manual_record),
                const Spacer(),
                if (allowMotion)
                  Tooltip(
                    message: 'Motion Allowed',
                    child: Icon(Icons.check_circle,
                        color: Colors.orange, size: 20),
                  ),
              ],
            ),
          ),
          const Divider(height: 1),
          // Creator Dashboard (Hidden behind expander or just removed for basic users?
          // Requirements said "main interaction" is chat. Let's keep it clean.)

          // Main Chat Area
          Expanded(
            child: ChatInteractionWidget(brainClient: widget.brainClient),
          ),

          // Bottom Controls Expander
          ExpansionTile(
            title:
                const Text('Advanced Controls', style: TextStyle(fontSize: 14)),
            dense: true,
            initiallyExpanded: _showControls,
            onExpansionChanged: (val) => setState(() => _showControls = val),
            children: [
              Container(
                height: 300,
                padding: const EdgeInsets.all(16),
                child: SingleChildScrollView(
                  child: Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      const CreatorDashboard(),
                      const SizedBox(height: 12),
                      Text('Hardware Sensors',
                          style: Theme.of(context).textTheme.titleSmall),
                      const SizedBox(height: 8),
                      _buildSensorsCard(hardware),
                      const SizedBox(height: 12),
                      _buildActionButtons(context),
                    ],
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildHealthBadge(
      {required String label, required Color color, required IconData icon}) {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
      decoration: BoxDecoration(
        color: color.withValues(alpha: 0.1),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: color.withValues(alpha: 0.3)),
      ),
      child: Row(
        children: [
          Icon(icon, size: 14, color: color),
          const SizedBox(width: 4),
          Text(
            label,
            style: TextStyle(
              color: color,
              fontWeight: FontWeight.bold,
              fontSize: 12,
            ),
          ),
        ],
      ),
    );
  }

  Color _getModeColor(String mode) {
    switch (mode) {
      case 'idle':
        return ContinuonColors.gray700;
      case 'manual_training':
        return ContinuonColors.primaryBlue;
      case 'autonomous':
        return ContinuonColors.cmsViolet;
      case 'sleep_learning':
        return ContinuonColors.particleOrange;
      default:
        return ContinuonColors.gray700;
    }
  }

  Widget _buildSensorsCard(Map<String, dynamic> hardware) {
    if (hardware.isEmpty) {
      return Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
            border: Border.all(color: Colors.grey.withValues(alpha: 0.3)),
            borderRadius: BorderRadius.circular(8)),
        width: double.infinity,
        child: const Text('No hardware detected',
            style: TextStyle(fontSize: 12, color: Colors.grey)),
      );
    }
    return Column(
      children: [
        if (hardware['depth_camera'] != null)
          _buildSensorRow('Depth Cam', hardware['depth_camera']),
      ],
    );
  }

  Widget _buildSensorRow(String label, String value) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 4.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(fontSize: 12)),
          Text(value,
              style:
                  const TextStyle(fontSize: 12, fontWeight: FontWeight.bold)),
        ],
      ),
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Wrap(
      spacing: 8,
      runSpacing: 8,
      children: [
        _buildMiniActionButton('Manual', ContinuonColors.primaryBlue, () async {
          await _setMode('manual_control');
          if (context.mounted) {
            Navigator.pushNamed(context, ControlScreen.routeName);
          }
        }),
        _buildMiniActionButton('Train', ContinuonColors.primaryBlue,
            () => _setMode('manual_training')),
        _buildMiniActionButton(
            'Auto', ContinuonColors.cmsViolet, () => _setMode('autonomous')),
        _buildMiniActionButton(
            'Idle', ContinuonColors.gray700, () => _setMode('idle')),
        _buildMiniActionButton('STOP', Theme.of(context).colorScheme.error,
            () => _setMode('emergency_stop')),
      ],
    );
  }

  Widget _buildMiniActionButton(String label, Color color, VoidCallback onTap) {
    return ElevatedButton(
      onPressed: onTap,
      style: ElevatedButton.styleFrom(
        backgroundColor: color,
        foregroundColor: Colors.white,
        padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
        minimumSize: const Size(0, 36),
      ),
      child: Text(label, style: const TextStyle(fontSize: 12)),
    );
  }
}

class _StartupFlagsSheet extends StatefulWidget {
  const _StartupFlagsSheet({
    required this.brainClient,
    required this.initialSettings,
  });

  final BrainClient brainClient;
  final Map<String, dynamic> initialSettings;

  @override
  State<_StartupFlagsSheet> createState() => _StartupFlagsSheetState();
}

class _StartupFlagsSheetState extends State<_StartupFlagsSheet> {
  late Map<String, dynamic> _settings;
  bool _saving = false;
  String? _error;

  final TextEditingController _creatorController = TextEditingController();

  @override
  void initState() {
    super.initState();
    _settings = Map<String, dynamic>.from(widget.initialSettings);
    _creatorController.text =
        ((_settings['identity'] as Map?)?['creator_display_name'] as String?) ??
            '';
  }

  @override
  void dispose() {
    _creatorController.dispose();
    super.dispose();
  }

  Map<String, dynamic> _subMap(String key) {
    final raw = _settings[key];
    if (raw is Map) return Map<String, dynamic>.from(raw);
    return <String, dynamic>{};
  }

  void _setSubKey(String root, String key, dynamic value) {
    final m = _subMap(root);
    m[key] = value;
    setState(() {
      _settings[root] = m;
    });
  }

  Future<void> _save() async {
    setState(() {
      _saving = true;
      _error = null;
    });
    try {
      _setSubKey(
          'identity', 'creator_display_name', _creatorController.text.trim());
      final res = await widget.brainClient.saveSettings(_settings);
      if (!mounted) return;
      if (res['success'] == true) {
        Navigator.pop(context);
        return;
      }
      setState(() => _error =
          (res['message'] as String?) ?? 'Settings rejected by runtime');
    } finally {
      if (mounted) setState(() => _saving = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final safety = _subMap('safety');
    final chat = _subMap('chat');
    final training = _subMap('training');
    final agentMgr = _subMap('agent_manager');
    final chatLearn = (agentMgr['chat_learn'] is Map)
        ? Map<String, dynamic>.from(agentMgr['chat_learn'] as Map)
        : <String, dynamic>{};
    final orchestrator = (agentMgr['autonomy_orchestrator'] is Map)
        ? Map<String, dynamic>.from(agentMgr['autonomy_orchestrator'] as Map)
        : <String, dynamic>{};

    return SafeArea(
      child: Padding(
        padding: EdgeInsets.only(
          left: 16,
          right: 16,
          bottom: 16 + MediaQuery.of(context).viewInsets.bottom,
        ),
        child: SingleChildScrollView(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text('Startup & flags',
                  style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 8),
              Text(
                'These settings persist on the runtime (GET/POST /api/settings).',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 12),
              if (_error != null) ...[
                Text(_error!,
                    style:
                        TextStyle(color: Theme.of(context).colorScheme.error)),
                const SizedBox(height: 8),
              ],
              TextField(
                controller: _creatorController,
                decoration: const InputDecoration(
                  labelText: 'Creator display name',
                  helperText: 'Used for prompts/UI alignment (non-biometric).',
                ),
              ),
              const SizedBox(height: 12),
              const Divider(),
              SwitchListTile(
                title: const Text('Allow motion'),
                subtitle: const Text('Safety gate: allow robot motion'),
                value: (safety['allow_motion'] as bool?) ?? true,
                onChanged: (v) => _setSubKey('safety', 'allow_motion', v),
              ),
              SwitchListTile(
                title: const Text('Record episodes'),
                subtitle: const Text('Enable RLDS episode recording'),
                value: (safety['record_episodes'] as bool?) ?? true,
                onChanged: (v) => _setSubKey('safety', 'record_episodes', v),
              ),
              SwitchListTile(
                title: const Text('Require supervision'),
                subtitle: const Text(
                    'If enabled, autonomous actions should be gated'),
                value: (safety['require_supervision'] as bool?) ?? false,
                onChanged: (v) =>
                    _setSubKey('safety', 'require_supervision', v),
              ),
              const Divider(),
              SwitchListTile(
                title: const Text('Log chat to RLDS'),
                subtitle: const Text(
                    'Opt-in: persist chat turns for later training/eval'),
                value: (chat['log_rlds'] as bool?) ?? false,
                onChanged: (v) => _setSubKey('chat', 'log_rlds', v),
              ),
              const Divider(),
              SwitchListTile(
                title: const Text('Enable sleep learning'),
                subtitle: const Text(
                    'Allow background learning in sleep_learning mode'),
                value: (training['enable_sleep_learning'] as bool?) ?? true,
                onChanged: (v) =>
                    _setSubKey('training', 'enable_sleep_learning', v),
              ),
              SwitchListTile(
                title: const Text('Enable sidecar trainer'),
                subtitle: const Text('Start trainer sidecar (resource heavy)'),
                value: (training['enable_sidecar_trainer'] as bool?) ?? false,
                onChanged: (v) =>
                    _setSubKey('training', 'enable_sidecar_trainer', v),
              ),
              const Divider(),
              SwitchListTile(
                title: const Text('Enable autonomous learning'),
                subtitle: const Text('Agent manager learning loop'),
                value:
                    (agentMgr['enable_autonomous_learning'] as bool?) ?? true,
                onChanged: (v) => _setSubKey(
                    'agent_manager', 'enable_autonomous_learning', v),
              ),
              SwitchListTile(
                title: const Text('Enable scheduled chat learning'),
                subtitle: const Text(
                    'Runs periodic chat_learn turns (offline-first)'),
                value: (chatLearn['enabled'] as bool?) ?? false,
                onChanged: (v) {
                  chatLearn['enabled'] = v;
                  _setSubKey('agent_manager', 'chat_learn', chatLearn);
                },
              ),
              SwitchListTile(
                title: const Text('Enable autonomy orchestrator'),
                subtitle: const Text(
                    'Runs bounded maintenance tasks in allowed modes'),
                value: (orchestrator['enabled'] as bool?) ?? false,
                onChanged: (v) {
                  orchestrator['enabled'] = v;
                  _setSubKey(
                      'agent_manager', 'autonomy_orchestrator', orchestrator);
                },
              ),
              const SizedBox(height: 12),
              Row(
                children: [
                  Expanded(
                    child: OutlinedButton.icon(
                      onPressed: _saving ? null : () => Navigator.pop(context),
                      icon: const Icon(Icons.close),
                      label: const Text('Cancel'),
                    ),
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: ElevatedButton.icon(
                      onPressed: _saving ? null : _save,
                      icon: _saving
                          ? const SizedBox(
                              width: 16,
                              height: 16,
                              child: CircularProgressIndicator(strokeWidth: 2))
                          : const Icon(Icons.save),
                      label: Text(_saving ? 'Saving...' : 'Save to runtime'),
                    ),
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
