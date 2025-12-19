import 'dart:async';

import 'package:flutter/material.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';
import 'control_screen.dart';
import '../widgets/creator_dashboard.dart';

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
  bool _openingSettings = false;

  @override
  void initState() {
    super.initState();
    _refreshStatus();
    _timer =
        Timer.periodic(const Duration(seconds: 2), (_) => _refreshStatus());
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
            content: Text('Failed to load settings: ${res['message'] ?? 'unknown error'}'),
            backgroundColor: Theme.of(context).colorScheme.error,
          ),
        );
        return;
      }

      final raw = res['settings'];
      final settings = raw is Map ? Map<String, dynamic>.from(raw) : <String, dynamic>{};

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
        title: const Text('CraigBot Control',
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
      body: Center(
        child: Container(
          constraints: const BoxConstraints(maxWidth: 600),
          child: SingleChildScrollView(
            padding: const EdgeInsets.all(24.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.0, end: 1.0),
                  duration: const Duration(milliseconds: 600),
                  curve: Curves.easeOut,
                  builder: (context, value, child) {
                    return Opacity(
                      opacity: value,
                      child: Transform.translate(
                        offset: Offset(0, 20 * (1 - value)),
                        child: child,
                      ),
                    );
                  },
                  child: Row(
                    children: [
                      const Text('🤖', style: TextStyle(fontSize: 32)),
                      const SizedBox(width: 12),
                      Text('ContinuonAI',
                          style: Theme.of(context)
                              .textTheme
                              .headlineMedium
                              ?.copyWith(fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
                const SizedBox(height: 12),
                const CreatorDashboard(),
                const SizedBox(height: 12),
                TweenAnimationBuilder<double>(
                  tween: Tween(begin: 0.0, end: 1.0),
                  duration: const Duration(milliseconds: 800),
                  curve: Curves.easeOut,
                  builder: (context, value, child) {
                    return Opacity(
                      opacity: value,
                      child: Transform.translate(
                        offset: Offset(0, 30 * (1 - value)),
                        child: child,
                      ),
                    );
                  },
                  child: Container(
                    decoration: BoxDecoration(
                      color: Theme.of(context).cardTheme.color,
                      borderRadius: BorderRadius.circular(ContinuonTokens.r16),
                      boxShadow: ContinuonTokens.midShadow,
                    ),
                    padding: const EdgeInsets.all(24),
                    child: Column(
                      children: [
                        _buildStatusCard(mode, isRecording, allowMotion),
                        const SizedBox(height: 24),
                        Align(
                          alignment: Alignment.centerLeft,
                          child: Text('Hardware Sensors',
                              style: Theme.of(context)
                                  .textTheme
                                  .titleLarge
                                  ?.copyWith(fontWeight: FontWeight.bold)),
                        ),
                        const SizedBox(height: 12),
                        _buildSensorsCard(hardware),
                        const SizedBox(height: 24),
                        _buildActionButtons(context),
                      ],
                    ),
                  ),
                ),
                const SizedBox(height: 24),
                Center(
                  child: Text(
                    'ContinuonAI Robot Control Interface',
                    style: Theme.of(context).textTheme.bodySmall,
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildStatusCard(String mode, bool isRecording, bool allowMotion) {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context)
            .colorScheme
            .surfaceContainerHighest
            .withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
      ),
      child: Column(
        children: [
          _buildStatusRow('Mode', mode.toUpperCase(),
              isBadge: true, badgeColor: _getModeColor(mode)),
          const SizedBox(height: 8),
          _buildStatusRow('Recording', isRecording ? 'Yes' : 'No'),
          const SizedBox(height: 8),
          _buildStatusRow('Motion Allowed', allowMotion ? 'Yes' : 'No'),
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
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Theme.of(context)
              .colorScheme
              .surfaceContainerHighest
              .withValues(alpha: 0.3),
          borderRadius: BorderRadius.circular(ContinuonTokens.r8),
        ),
        width: double.infinity,
        child: Text('No hardware detected or status unavailable',
            style: Theme.of(context).textTheme.bodyMedium),
      );
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context)
            .colorScheme
            .surfaceContainerHighest
            .withValues(alpha: 0.3),
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
      ),
      child: Column(
        children: [
          if (hardware['depth_camera'] != null)
            _buildSensorRow('📷 Depth Camera', hardware['depth_camera']),
          if (hardware['depth_camera_driver'] != null) ...[
            const SizedBox(height: 8),
            _buildSensorRow('Camera Driver', hardware['depth_camera_driver']),
          ],
          if (hardware['servo_controller'] != null) ...[
            const SizedBox(height: 8),
            _buildSensorRow(
                '🦾 Servo Controller', hardware['servo_controller']),
          ],
          if (hardware['servo_controller_address'] != null) ...[
            const SizedBox(height: 8),
            _buildSensorRow(
                'I2C Address', hardware['servo_controller_address']),
          ],
        ],
      ),
    );
  }

  Widget _buildStatusRow(String label, String value,
      {bool isBadge = false, Color badgeColor = Colors.grey}) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: Theme.of(context).textTheme.bodyMedium),
        isBadge
            ? Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
                decoration: BoxDecoration(
                  color: badgeColor,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Text(
                  value.replaceAll('_', ' '),
                  style: const TextStyle(
                    color: Colors.white,
                    fontWeight: FontWeight.bold,
                    fontSize: 12,
                  ),
                ),
              )
            : Text(value,
                style: Theme.of(context)
                    .textTheme
                    .bodyLarge
                    ?.copyWith(fontWeight: FontWeight.w600)),
      ],
    );
  }

  Widget _buildSensorRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: Theme.of(context).textTheme.bodyMedium),
        Text(value,
            style: Theme.of(context)
                .textTheme
                .bodyLarge
                ?.copyWith(fontWeight: FontWeight.w600)),
      ],
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Column(
      children: [
        _buildActionButton(
          title: 'Manual control',
          subtitle: 'Set mode to manual_control (GET /api/mode/manual_control)',
          color: ContinuonColors.primaryBlue,
          onPressed: () async {
            await _setMode('manual_control');
            if (context.mounted) {
              Navigator.pushNamed(context, ControlScreen.routeName);
            }
          },
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          title: 'Manual training',
          subtitle: 'Set mode to manual_training (GET /api/mode/manual_training)',
          color: ContinuonColors.primaryBlue,
          onPressed: () => _setMode('manual_training'),
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          title: 'Autonomous',
          subtitle: 'Set mode to autonomous (GET /api/mode/autonomous)',
          color: ContinuonColors.cmsViolet,
          onPressed: () => _setMode('autonomous'),
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          title: 'Sleep learning',
          subtitle: 'Set mode to sleep_learning (GET /api/mode/sleep_learning)',
          color: ContinuonColors.particleOrange,
          onPressed: () => _setMode('sleep_learning'),
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          title: 'Idle',
          subtitle: 'Set mode to idle (GET /api/mode/idle)',
          color: ContinuonColors.gray700,
          onPressed: () => _setMode('idle'),
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          title: 'Emergency stop',
          subtitle: 'Set mode to emergency_stop (GET /api/mode/emergency_stop)',
          color: Theme.of(context).colorScheme.error,
          onPressed: () => _setMode('emergency_stop'),
        ),
      ],
    );
  }

  Widget _buildActionButton(
      {required String title,
      required String subtitle,
      required Color color,
      required VoidCallback? onPressed}) {
    return Tooltip(
      message: subtitle,
      child: SizedBox(
        width: double.infinity,
        height: 56,
        child: ElevatedButton(
          onPressed: onPressed,
          style: ElevatedButton.styleFrom(
            backgroundColor: color,
            foregroundColor: Colors.white,
            shape: RoundedRectangleBorder(
              borderRadius: BorderRadius.circular(8),
            ),
            elevation: 0,
            textStyle: const TextStyle(fontSize: 16, fontWeight: FontWeight.w600),
          ),
          child: Row(
            children: [
              Expanded(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(title),
                    const SizedBox(height: 2),
                    Text(
                      subtitle,
                      maxLines: 1,
                      overflow: TextOverflow.ellipsis,
                      style: TextStyle(
                        fontSize: 11,
                        color: Colors.white.withValues(alpha: 0.85),
                        fontWeight: FontWeight.w400,
                      ),
                    ),
                  ],
                ),
              ),
              const Icon(Icons.chevron_right),
            ],
          ),
        ),
      ),
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
        ((_settings['identity'] as Map?)?['creator_display_name'] as String?) ?? '';
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
      _setSubKey('identity', 'creator_display_name', _creatorController.text.trim());
      final res = await widget.brainClient.saveSettings(_settings);
      if (!mounted) return;
      if (res['success'] == true) {
        Navigator.pop(context);
        return;
      }
      setState(() => _error = (res['message'] as String?) ?? 'Settings rejected by runtime');
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
              Text('Startup & flags', style: Theme.of(context).textTheme.titleLarge),
              const SizedBox(height: 8),
              Text(
                'These settings persist on the runtime (GET/POST /api/settings).',
                style: Theme.of(context).textTheme.bodySmall,
              ),
              const SizedBox(height: 12),
              if (_error != null) ...[
                Text(_error!, style: TextStyle(color: Theme.of(context).colorScheme.error)),
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
                subtitle: const Text('If enabled, autonomous actions should be gated'),
                value: (safety['require_supervision'] as bool?) ?? false,
                onChanged: (v) => _setSubKey('safety', 'require_supervision', v),
              ),
              const Divider(),
              SwitchListTile(
                title: const Text('Log chat to RLDS'),
                subtitle: const Text('Opt-in: persist chat turns for later training/eval'),
                value: (chat['log_rlds'] as bool?) ?? false,
                onChanged: (v) => _setSubKey('chat', 'log_rlds', v),
              ),
              const Divider(),
              SwitchListTile(
                title: const Text('Enable sleep learning'),
                subtitle: const Text('Allow background learning in sleep_learning mode'),
                value: (training['enable_sleep_learning'] as bool?) ?? true,
                onChanged: (v) => _setSubKey('training', 'enable_sleep_learning', v),
              ),
              SwitchListTile(
                title: const Text('Enable sidecar trainer'),
                subtitle: const Text('Start trainer sidecar (resource heavy)'),
                value: (training['enable_sidecar_trainer'] as bool?) ?? false,
                onChanged: (v) => _setSubKey('training', 'enable_sidecar_trainer', v),
              ),
              const Divider(),
              SwitchListTile(
                title: const Text('Enable autonomous learning'),
                subtitle: const Text('Agent manager learning loop'),
                value: (agentMgr['enable_autonomous_learning'] as bool?) ?? true,
                onChanged: (v) => _setSubKey('agent_manager', 'enable_autonomous_learning', v),
              ),
              SwitchListTile(
                title: const Text('Enable scheduled chat learning'),
                subtitle: const Text('Runs periodic chat_learn turns (offline-first)'),
                value: (chatLearn['enabled'] as bool?) ?? false,
                onChanged: (v) {
                  chatLearn['enabled'] = v;
                  _setSubKey('agent_manager', 'chat_learn', chatLearn);
                },
              ),
              SwitchListTile(
                title: const Text('Enable autonomy orchestrator'),
                subtitle: const Text('Runs bounded maintenance tasks in allowed modes'),
                value: (orchestrator['enabled'] as bool?) ?? false,
                onChanged: (v) {
                  orchestrator['enabled'] = v;
                  _setSubKey('agent_manager', 'autonomy_orchestrator', orchestrator);
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
                          ? const SizedBox(width: 16, height: 16, child: CircularProgressIndicator(strokeWidth: 2))
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
