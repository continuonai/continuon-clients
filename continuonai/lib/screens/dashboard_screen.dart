import 'dart:async';

import 'package:flutter/material.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';
import 'control_screen.dart';

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

  @override
  void initState() {
    super.initState();
    _refreshStatus();
    _timer = Timer.periodic(const Duration(seconds: 2), (_) => _refreshStatus());
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

  @override
  Widget build(BuildContext context) {
    final statusData = (_status['status'] as Map?)?.cast<String, dynamic>() ?? {};
    final mode = statusData['mode'] ?? 'unknown';
    final isRecording = statusData['is_recording'] ?? false;
    final allowMotion = statusData['allow_motion'] ?? false;
    final hardware = (statusData['detected_hardware'] as Map?)?.cast<String, dynamic>() ?? {};

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      appBar: AppBar(
        title: const Text('CraigBot Control', style: TextStyle(fontWeight: FontWeight.bold)),
        backgroundColor: Theme.of(context).appBarTheme.backgroundColor,
        foregroundColor: Theme.of(context).appBarTheme.foregroundColor,
        elevation: 0,
        bottom: PreferredSize(
          preferredSize: const Size.fromHeight(1),
          child: Container(color: Theme.of(context).dividerColor, height: 1),
        ),
        actions: [
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
                      Text('ContinuonAI', style: Theme.of(context).textTheme.headlineMedium?.copyWith(fontWeight: FontWeight.bold)),
                    ],
                  ),
                ),
                const SizedBox(height: 24),
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
                          child: Text('Hardware Sensors', style: Theme.of(context).textTheme.titleLarge?.copyWith(fontWeight: FontWeight.bold)),
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
        color: Theme.of(context).colorScheme.surfaceContainerHighest.withOpacity(0.3),
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
      ),
      child: Column(
        children: [
          _buildStatusRow('Mode', mode.toUpperCase(), isBadge: true, badgeColor: _getModeColor(mode)),
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
          color: Theme.of(context).colorScheme.surfaceContainerHighest.withOpacity(0.3),
          borderRadius: BorderRadius.circular(ContinuonTokens.r8),
        ),
        width: double.infinity,
        child: Text('No hardware detected or status unavailable', style: Theme.of(context).textTheme.bodyMedium),
      );
    }

    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Theme.of(context).colorScheme.surfaceContainerHighest.withOpacity(0.3),
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
            _buildSensorRow('🦾 Servo Controller', hardware['servo_controller']),
          ],
          if (hardware['servo_controller_address'] != null) ...[
            const SizedBox(height: 8),
            _buildSensorRow('I2C Address', hardware['servo_controller_address']),
          ],
        ],
      ),
    );
  }

  Widget _buildStatusRow(String label, String value, {bool isBadge = false, Color badgeColor = Colors.grey}) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: Theme.of(context).textTheme.bodyMedium),
        isBadge
            ? Container(
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 4),
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
            : Text(value, style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w600)),
      ],
    );
  }

  Widget _buildSensorRow(String label, String value) {
    return Row(
      mainAxisAlignment: MainAxisAlignment.spaceBetween,
      children: [
        Text(label, style: Theme.of(context).textTheme.bodyMedium),
        Text(value, style: Theme.of(context).textTheme.bodyLarge?.copyWith(fontWeight: FontWeight.w600)),
      ],
    );
  }

  Widget _buildActionButtons(BuildContext context) {
    return Column(
      children: [
        _buildActionButton(
          '🎮 Manual Control',
          ContinuonColors.primaryBlue,
          () async {
            await _setMode('manual_control');
            if (context.mounted) {
              Navigator.pushNamed(context, ControlScreen.routeName);
            }
          },
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          '📝 Manual Training',
          ContinuonColors.primaryBlue,
          () => _setMode('manual_training'),
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          '🚀 Autonomous',
          ContinuonColors.cmsViolet,
          () => _setMode('autonomous'),
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          '💤 Sleep Learning',
          ContinuonColors.particleOrange,
          () => _setMode('sleep_learning'),
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          '⏸️ Idle',
          ContinuonColors.gray700,
          () => _setMode('idle'),
        ),
        const SizedBox(height: 12),
        _buildActionButton(
          '🛑 Emergency Stop',
          Theme.of(context).colorScheme.error,
          () => _setMode('emergency_stop'),
        ),
      ],
    );
  }

  Widget _buildActionButton(String label, Color color, VoidCallback? onPressed) {
    return SizedBox(
      width: double.infinity,
      height: 50,
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
        child: Text(label),
      ),
    );
  }
}
