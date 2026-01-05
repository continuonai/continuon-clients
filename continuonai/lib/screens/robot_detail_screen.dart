import 'dart:async';

import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../blocs/learning/learning_bloc.dart';
import '../blocs/learning/learning_event.dart';
import '../blocs/learning/learning_state.dart';
import '../blocs/ota/ota_bloc.dart';
import '../blocs/ota/ota_event.dart';
import '../blocs/ota/ota_state.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';
import 'control_screen.dart';
import 'robot_init_wizard_screen.dart';
import 'seed_model_update_screen.dart';
import 'slow_loop_dashboard_screen.dart';
import 'unified_qr_scanner_screen.dart';

/// Per-robot detail screen with tabs for different management features
class RobotDetailScreen extends StatefulWidget {
  static const routeName = '/robot_detail';

  final BrainClient brainClient;
  final String? robotId;
  final String? robotName;

  const RobotDetailScreen({
    super.key,
    required this.brainClient,
    this.robotId,
    this.robotName,
  });

  @override
  State<RobotDetailScreen> createState() => _RobotDetailScreenState();
}

class _RobotDetailScreenState extends State<RobotDetailScreen>
    with SingleTickerProviderStateMixin {
  late TabController _tabController;
  Map<String, dynamic> _status = {};
  bool _isLoading = true;
  Timer? _refreshTimer;

  @override
  void initState() {
    super.initState();
    _tabController = TabController(length: 5, vsync: this);
    _refreshStatus();
    _refreshTimer = Timer.periodic(const Duration(seconds: 10), (_) {
      _refreshStatus();
    });
  }

  @override
  void dispose() {
    _tabController.dispose();
    _refreshTimer?.cancel();
    super.dispose();
  }

  Future<void> _refreshStatus() async {
    if (!mounted) return;

    try {
      final status = await widget.brainClient.getRobotStatus();
      if (mounted) {
        setState(() {
          _status = status;
          _isLoading = false;
        });
      }
    } catch (e) {
      if (mounted) {
        setState(() {
          _isLoading = false;
        });
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return MultiBlocProvider(
      providers: [
        BlocProvider(
          create: (context) => LearningBloc(brainClient: widget.brainClient),
        ),
        BlocProvider(
          create: (context) => OTABloc(brainClient: widget.brainClient),
        ),
      ],
      child: Scaffold(
        appBar: AppBar(
          title: Text(widget.robotName ?? 'Robot'),
          actions: [
            IconButton(
              icon: const Icon(Icons.refresh),
              onPressed: _refreshStatus,
              tooltip: 'Refresh',
            ),
            IconButton(
              icon: const Icon(Icons.gamepad),
              onPressed: () {
                Navigator.of(context).pushNamed(
                  ControlScreen.routeName,
                );
              },
              tooltip: 'Control',
            ),
          ],
          bottom: TabBar(
            controller: _tabController,
            isScrollable: true,
            tabs: const [
              Tab(icon: Icon(Icons.dashboard), text: 'Overview'),
              Tab(icon: Icon(Icons.rocket_launch), text: 'Setup'),
              Tab(icon: Icon(Icons.system_update), text: 'Updates'),
              Tab(icon: Icon(Icons.psychology), text: 'Learning'),
              Tab(icon: Icon(Icons.settings), text: 'Settings'),
            ],
          ),
        ),
        body: TabBarView(
          controller: _tabController,
          children: [
            _buildOverviewTab(isDark),
            _buildSetupTab(isDark),
            _buildUpdatesTab(isDark),
            _buildLearningTab(isDark),
            _buildSettingsTab(isDark),
          ],
        ),
      ),
    );
  }

  Widget _buildOverviewTab(bool isDark) {
    if (_isLoading) {
      return const Center(child: CircularProgressIndicator());
    }

    final mode = _status['mode'] as String? ?? 'unknown';
    final hardware = _status['hardware_mode'] as String? ?? 'unknown';
    final surprise = (_status['surprise'] as num?)?.toDouble() ?? 0.0;
    final learning = _status['learning'] as Map<String, dynamic>?;
    final battery = _status['battery'] as Map<String, dynamic>?;

    return RefreshIndicator(
      onRefresh: _refreshStatus,
      child: ListView(
        padding: const EdgeInsets.all(ContinuonTokens.s16),
        children: [
          // Status cards
          _buildStatusCard(
            title: 'Mode',
            value: mode.replaceAll('_', ' ').toUpperCase(),
            icon: Icons.tune,
            color: ContinuonColors.primaryBlue,
            isDark: isDark,
          ),
          const SizedBox(height: ContinuonTokens.s12),

          Row(
            children: [
              Expanded(
                child: _buildStatusCard(
                  title: 'Hardware',
                  value: hardware,
                  icon: Icons.memory,
                  color: ContinuonColors.cmsViolet,
                  isDark: isDark,
                  compact: true,
                ),
              ),
              const SizedBox(width: ContinuonTokens.s12),
              Expanded(
                child: _buildStatusCard(
                  title: 'Surprise',
                  value: surprise.toStringAsFixed(2),
                  icon: Icons.lightbulb_outline,
                  color: ContinuonColors.particleOrange,
                  isDark: isDark,
                  compact: true,
                ),
              ),
            ],
          ),
          const SizedBox(height: ContinuonTokens.s12),

          if (battery != null)
            _buildStatusCard(
              title: 'Battery',
              value: '${battery['percent'] ?? '--'}%',
              icon: Icons.battery_full,
              color: Colors.green,
              isDark: isDark,
            ),

          if (learning != null) ...[
            const SizedBox(height: ContinuonTokens.s12),
            _buildStatusCard(
              title: 'Learning',
              value: learning['running'] == true ? 'Active' : 'Idle',
              icon: Icons.school,
              color: learning['running'] == true
                  ? Colors.green
                  : ContinuonColors.gray500,
              isDark: isDark,
            ),
          ],

          const SizedBox(height: ContinuonTokens.s24),

          // Quick actions
          Text(
            'Quick Actions',
            style: Theme.of(context).textTheme.titleMedium,
          ),
          const SizedBox(height: ContinuonTokens.s12),

          Wrap(
            spacing: ContinuonTokens.s8,
            runSpacing: ContinuonTokens.s8,
            children: [
              ActionChip(
                avatar: const Icon(Icons.gamepad, size: 18),
                label: const Text('Control'),
                onPressed: () {
                  Navigator.of(context).pushNamed(ControlScreen.routeName);
                },
              ),
              ActionChip(
                avatar: const Icon(Icons.qr_code_scanner, size: 18),
                label: const Text('Scan QR'),
                onPressed: () {
                  Navigator.of(context).pushNamed(UnifiedQRScannerScreen.routeName);
                },
              ),
              ActionChip(
                avatar: const Icon(Icons.system_update, size: 18),
                label: const Text('Check Updates'),
                onPressed: () {
                  _tabController.animateTo(2);
                  context.read<OTABloc>().add(const CheckForUpdates());
                },
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildSetupTab(bool isDark) {
    return ListView(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      children: [
        _buildMenuCard(
          title: 'Initialize Robot',
          subtitle: 'Run the setup wizard for new robots',
          icon: Icons.rocket_launch,
          color: ContinuonColors.primaryBlue,
          isDark: isDark,
          onTap: () {
            Navigator.of(context).pushNamed(
              RobotInitWizardScreen.routeName,
            );
          },
        ),
        const SizedBox(height: ContinuonTokens.s12),

        _buildMenuCard(
          title: 'Scan QR Code',
          subtitle: 'Pair robot or register with cloud',
          icon: Icons.qr_code_scanner,
          color: ContinuonColors.cmsViolet,
          isDark: isDark,
          onTap: () {
            Navigator.of(context).pushNamed(
              UnifiedQRScannerScreen.routeName,
            );
          },
        ),
        const SizedBox(height: ContinuonTokens.s12),

        _buildMenuCard(
          title: 'Claim Ownership',
          subtitle: 'Register as the owner of this robot',
          icon: Icons.verified_user,
          color: Colors.green,
          isDark: isDark,
          onTap: () async {
            final success = await widget.brainClient.claimRobot(
              host: widget.brainClient.rcan.currentHost ?? 'localhost',
            );
            if (mounted) {
              ScaffoldMessenger.of(context).showSnackBar(
                SnackBar(
                  content: Text(success
                      ? 'Robot claimed successfully!'
                      : 'Failed to claim robot'),
                ),
              );
            }
          },
        ),
      ],
    );
  }

  Widget _buildUpdatesTab(bool isDark) {
    return BlocBuilder<OTABloc, OTAState>(
      builder: (context, state) {
        return ListView(
          padding: const EdgeInsets.all(ContinuonTokens.s16),
          children: [
            _buildMenuCard(
              title: 'Update Seed Model',
              subtitle: 'Check and install AI model updates',
              icon: Icons.cloud_download,
              color: ContinuonColors.primaryBlue,
              isDark: isDark,
              trailing: state.updateAvailable
                  ? Container(
                      padding: const EdgeInsets.symmetric(
                        horizontal: 8,
                        vertical: 4,
                      ),
                      decoration: BoxDecoration(
                        color: Colors.orange,
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: const Text(
                        'NEW',
                        style: TextStyle(
                          color: Colors.white,
                          fontSize: 10,
                          fontWeight: FontWeight.bold,
                        ),
                      ),
                    )
                  : null,
              onTap: () {
                Navigator.of(context).pushNamed(
                  SeedModelUpdateScreen.routeName,
                );
              },
            ),
            const SizedBox(height: ContinuonTokens.s12),

            // Current version info
            Container(
              padding: const EdgeInsets.all(ContinuonTokens.s16),
              decoration: BoxDecoration(
                color: isDark
                    ? ContinuonColors.gray800.withOpacity(0.5)
                    : ContinuonColors.gray200.withOpacity(0.5),
                borderRadius: BorderRadius.circular(ContinuonTokens.r12),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Current Version',
                    style: TextStyle(
                      color:
                          isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
                      fontSize: 12,
                    ),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    state.currentVersion ?? 'Unknown',
                    style: const TextStyle(
                      fontWeight: FontWeight.w600,
                      fontSize: 18,
                    ),
                  ),
                  if (state.lastChecked != null) ...[
                    const SizedBox(height: 8),
                    Text(
                      'Last checked: ${_formatTime(state.lastChecked!)}',
                      style: TextStyle(
                        color: isDark
                            ? ContinuonColors.gray400
                            : ContinuonColors.gray500,
                        fontSize: 12,
                      ),
                    ),
                  ],
                ],
              ),
            ),
            const SizedBox(height: ContinuonTokens.s16),

            ElevatedButton.icon(
              onPressed: state.status == OTAStateStatus.checking
                  ? null
                  : () {
                      context.read<OTABloc>().add(const CheckForUpdates());
                    },
              icon: state.status == OTAStateStatus.checking
                  ? const SizedBox(
                      width: 20,
                      height: 20,
                      child: CircularProgressIndicator(strokeWidth: 2),
                    )
                  : const Icon(Icons.refresh),
              label: Text(state.status == OTAStateStatus.checking
                  ? 'Checking...'
                  : 'Check for Updates'),
            ),
          ],
        );
      },
    );
  }

  Widget _buildLearningTab(bool isDark) {
    return BlocBuilder<LearningBloc, LearningState>(
      builder: (context, state) {
        return ListView(
          padding: const EdgeInsets.all(ContinuonTokens.s16),
          children: [
            _buildMenuCard(
              title: 'Slow Loop Dashboard',
              subtitle: 'View learning metrics and controls',
              icon: Icons.insights,
              color: ContinuonColors.cmsViolet,
              isDark: isDark,
              onTap: () {
                Navigator.of(context).pushNamed(
                  SlowLoopDashboardScreen.routeName,
                );
              },
            ),
            const SizedBox(height: ContinuonTokens.s12),

            // Quick status
            Container(
              padding: const EdgeInsets.all(ContinuonTokens.s16),
              decoration: BoxDecoration(
                color: isDark
                    ? ContinuonColors.gray800.withOpacity(0.5)
                    : ContinuonColors.gray200.withOpacity(0.5),
                borderRadius: BorderRadius.circular(ContinuonTokens.r12),
              ),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Learning Status'),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: 8,
                          vertical: 4,
                        ),
                        decoration: BoxDecoration(
                          color: state.isLearning
                              ? Colors.green.withOpacity(0.2)
                              : ContinuonColors.gray500.withOpacity(0.2),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: Text(
                          state.isLearning
                              ? 'Active'
                              : (state.isPaused ? 'Paused' : 'Idle'),
                          style: TextStyle(
                            color: state.isLearning
                                ? Colors.green
                                : ContinuonColors.gray500,
                            fontSize: 12,
                            fontWeight: FontWeight.w600,
                          ),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: ContinuonTokens.s12),
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceAround,
                    children: [
                      _buildMetricItem(
                        'Episodes',
                        state.totalEpisodes.toString(),
                        isDark,
                      ),
                      _buildMetricItem(
                        'Steps',
                        state.totalSteps.toString(),
                        isDark,
                      ),
                      _buildMetricItem(
                        'Curiosity',
                        state.curiosity.toStringAsFixed(2),
                        isDark,
                      ),
                    ],
                  ),
                ],
              ),
            ),
            const SizedBox(height: ContinuonTokens.s16),

            // Quick controls
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: state.isLearning || state.isPaused
                        ? () {
                            if (state.isPaused) {
                              context.read<LearningBloc>().add(const ResumeLearning());
                            } else {
                              context.read<LearningBloc>().add(const PauseLearning());
                            }
                          }
                        : null,
                    icon: Icon(state.isPaused ? Icons.play_arrow : Icons.pause),
                    label: Text(state.isPaused ? 'Resume' : 'Pause'),
                  ),
                ),
                const SizedBox(width: ContinuonTokens.s12),
                Expanded(
                  child: OutlinedButton.icon(
                    onPressed: () {
                      context.read<LearningBloc>().add(const FetchLearningMetrics());
                    },
                    icon: const Icon(Icons.refresh),
                    label: const Text('Refresh'),
                  ),
                ),
              ],
            ),
          ],
        );
      },
    );
  }

  Widget _buildSettingsTab(bool isDark) {
    return ListView(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      children: [
        _buildMenuCard(
          title: 'Robot Name',
          subtitle: _status['robot_name'] as String? ?? 'Tap to change',
          icon: Icons.edit,
          color: ContinuonColors.primaryBlue,
          isDark: isDark,
          onTap: _showNameDialog,
        ),
        const SizedBox(height: ContinuonTokens.s12),

        _buildMenuCard(
          title: 'RCAN Configuration',
          subtitle: 'Robot communication settings',
          icon: Icons.settings_ethernet,
          color: ContinuonColors.cmsViolet,
          isDark: isDark,
          onTap: () {
            // TODO: Navigate to RCAN settings
          },
        ),
        const SizedBox(height: ContinuonTokens.s12),

        _buildMenuCard(
          title: 'Transfer Ownership',
          subtitle: 'Transfer this robot to another user',
          icon: Icons.swap_horiz,
          color: ContinuonColors.particleOrange,
          isDark: isDark,
          onTap: _showTransferDialog,
        ),
      ],
    );
  }

  Widget _buildStatusCard({
    required String title,
    required String value,
    required IconData icon,
    required Color color,
    required bool isDark,
    bool compact = false,
  }) {
    return Container(
      padding: EdgeInsets.all(compact ? ContinuonTokens.s12 : ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark
            ? ContinuonColors.gray800.withOpacity(0.5)
            : ContinuonColors.gray200.withOpacity(0.5),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(color: color.withOpacity(0.3)),
      ),
      child: Row(
        children: [
          Container(
            padding: const EdgeInsets.all(8),
            decoration: BoxDecoration(
              color: color.withOpacity(0.1),
              borderRadius: BorderRadius.circular(8),
            ),
            child: Icon(icon, color: color, size: compact ? 20 : 24),
          ),
          SizedBox(width: compact ? ContinuonTokens.s8 : ContinuonTokens.s12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  title,
                  style: TextStyle(
                    color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
                    fontSize: compact ? 11 : 12,
                  ),
                ),
                Text(
                  value,
                  style: TextStyle(
                    fontWeight: FontWeight.w600,
                    fontSize: compact ? 14 : 16,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildMenuCard({
    required String title,
    required String subtitle,
    required IconData icon,
    required Color color,
    required bool isDark,
    Widget? trailing,
    VoidCallback? onTap,
  }) {
    return Material(
      color: isDark
          ? ContinuonColors.gray800.withOpacity(0.5)
          : ContinuonColors.gray200.withOpacity(0.5),
      borderRadius: BorderRadius.circular(ContinuonTokens.r12),
      child: InkWell(
        onTap: onTap,
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        child: Padding(
          padding: const EdgeInsets.all(ContinuonTokens.s16),
          child: Row(
            children: [
              Container(
                padding: const EdgeInsets.all(10),
                decoration: BoxDecoration(
                  color: color.withOpacity(0.1),
                  borderRadius: BorderRadius.circular(10),
                ),
                child: Icon(icon, color: color),
              ),
              const SizedBox(width: ContinuonTokens.s12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      title,
                      style: const TextStyle(fontWeight: FontWeight.w600),
                    ),
                    Text(
                      subtitle,
                      style: TextStyle(
                        color: isDark
                            ? ContinuonColors.gray400
                            : ContinuonColors.gray500,
                        fontSize: 13,
                      ),
                    ),
                  ],
                ),
              ),
              if (trailing != null) trailing,
              Icon(
                Icons.chevron_right,
                color: isDark ? ContinuonColors.gray500 : ContinuonColors.gray400,
              ),
            ],
          ),
        ),
      ),
    );
  }

  Widget _buildMetricItem(String label, String value, bool isDark) {
    return Column(
      children: [
        Text(
          value,
          style: const TextStyle(
            fontWeight: FontWeight.w600,
            fontSize: 18,
          ),
        ),
        Text(
          label,
          style: TextStyle(
            color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            fontSize: 12,
          ),
        ),
      ],
    );
  }

  String _formatTime(DateTime time) {
    final now = DateTime.now();
    final diff = now.difference(time);

    if (diff.inMinutes < 1) {
      return 'Just now';
    } else if (diff.inHours < 1) {
      return '${diff.inMinutes}m ago';
    } else if (diff.inDays < 1) {
      return '${diff.inHours}h ago';
    } else {
      return '${diff.inDays}d ago';
    }
  }

  Future<void> _showNameDialog() async {
    final controller = TextEditingController(
      text: _status['robot_name'] as String? ?? '',
    );

    final result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Robot Name'),
        content: TextField(
          controller: controller,
          decoration: const InputDecoration(
            labelText: 'Name',
            border: OutlineInputBorder(),
          ),
          autofocus: true,
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(context).pop(controller.text),
            child: const Text('Save'),
          ),
        ],
      ),
    );

    if (result != null && result.isNotEmpty) {
      await widget.brainClient.setRobotName(result);
      _refreshStatus();
    }
  }

  Future<void> _showTransferDialog() async {
    final controller = TextEditingController();

    final result = await showDialog<String>(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Transfer Ownership'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            const Text(
              'Enter the user ID of the new owner. This action cannot be undone.',
            ),
            const SizedBox(height: 16),
            TextField(
              controller: controller,
              decoration: const InputDecoration(
                labelText: 'New Owner ID',
                border: OutlineInputBorder(),
              ),
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(context).pop(),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () => Navigator.of(context).pop(controller.text),
            style: ElevatedButton.styleFrom(
              backgroundColor: Colors.orange,
            ),
            child: const Text('Transfer'),
          ),
        ],
      ),
    );

    if (result != null && result.isNotEmpty) {
      final response = await widget.brainClient.transferOwnership(
        host: widget.brainClient.rcan.currentHost ?? 'localhost',
        newOwnerId: result,
      );

      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(
            content: Text(response['success'] == true
                ? 'Ownership transferred successfully!'
                : 'Failed to transfer ownership'),
          ),
        );
      }
    }
  }
}
