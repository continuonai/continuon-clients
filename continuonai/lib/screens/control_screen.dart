import 'dart:async';
import 'package:flutter/material.dart';
import '../models/teleop_models.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';
import '../widgets/chat_overlay.dart';

import 'record_screen.dart';

class ControlScreen extends StatefulWidget {
  const ControlScreen({super.key, required this.brainClient});

  static const routeName = '/control';

  final BrainClient brainClient;

  @override
  State<ControlScreen> createState() => _ControlScreenState();
}

class _ControlScreenState extends State<ControlScreen> {
  late StreamSubscription<RobotState> _subscription;
  String _frameId = 'unknown';
  bool _gripperOpen = false;

  bool _sending = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _subscription = widget.brainClient
        .streamRobotState(widget.brainClient.clientId)
        .listen((state) {
      if (mounted) {
        setState(() {
          _frameId = state.frameId;
          _gripperOpen = state.gripperOpen;

          _error = null;
        });
      }
    }, onError: (error) {
      if (mounted) setState(() => _error = error.toString());
    });
  }

  @override
  void dispose() {
    _subscription.cancel();
    super.dispose();
  }

  Future<void> _sendBaseTwist(
      {required double linearMps, required double yawRadS}) async {
    final command = ControlCommand(
      clientId: widget.brainClient.clientId,
      controlMode: ControlMode.eeVelocity,
      targetFrequencyHz: 30,
      eeVelocity: EeVelocityCommand(
        referenceFrame: ReferenceFrame.base,
        linearMps: Vector3(x: linearMps, y: 0, z: 0),
        angularRadS: Vector3(x: 0, y: 0, z: yawRadS),
      ),
    );
    await _safeSendCommand(command);
  }

  Future<void> _safeSendCommand(ControlCommand command) async {
    setState(() => _sending = true);
    try {
      await widget.brainClient.sendCommand(command);
      setState(() => _error = null);
    } catch (error) {
      setState(() => _error = error.toString());
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Command failed: $error')),
        );
      }
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _toggleGripper() async {
    final command = ControlCommand(
      clientId: widget.brainClient.clientId,
      controlMode: ControlMode.gripper,
      targetFrequencyHz: 5,
      gripperCommand: GripperCommand(
        mode: GripperMode.position,
        positionM: _gripperOpen ? 0.0 : 0.04,
      ),
    );
    await _safeSendCommand(command);
  }

  Future<void> _triggerEStop() async {
    setState(() => _sending = true);
    try {
      final result = await widget.brainClient.triggerSafetyHold();
      if (mounted) {
        if (result['success'] == true || result['ok'] == true) {
          ScaffoldMessenger.of(context).showSnackBar(
            const SnackBar(
              content: Text('SAFETY HOLD TRIGGERED (E-STOP)'),
              backgroundColor: Colors.red,
              duration: Duration(seconds: 5),
            ),
          );
        } else {
          setState(() => _error =
              'Safety hold failed: ${result['message'] ?? 'unknown error'}');
        }
      }
    } catch (e) {
      if (mounted) setState(() => _error = 'Safety hold error: $e');
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor: ContinuonColors.black,
      body: Stack(
        children: [
          Column(
            children: [
              _buildHeader(),
              Expanded(
                child: Row(
                  children: [
                    Expanded(
                      flex: 2,
                      child: _buildVideoPanel(),
                    ),
                    Container(
                      width: 1,
                      color: Colors.grey[900],
                    ),
                    Expanded(
                      flex: 1,
                      child: _buildStatusPanel(),
                    ),
                  ],
                ),
              ),
            ],
          ),
          ChatOverlay(brainClient: widget.brainClient),
        ],
      ),
    );
  }

  Widget _buildHeader() {
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 12),
      decoration: BoxDecoration(
        color: ContinuonColors.gray900,
        border: const Border(bottom: BorderSide(color: Color(0xFF333333))),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: 0.3),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Row(
            children: [
              Container(
                padding: const EdgeInsets.all(8),
                decoration: BoxDecoration(
                  color: ContinuonColors.primaryBlue.withValues(alpha: 0.2),
                  borderRadius: BorderRadius.circular(8),
                ),
                child: const Icon(Icons.gamepad,
                    color: ContinuonColors.primaryBlue, size: 20),
              ),
              const SizedBox(width: 12),
              const Text(
                'Manual Control',
                style: TextStyle(
                  color: Colors.white,
                  fontSize: 18,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 0.5,
                ),
              ),
            ],
          ),
          Row(
            children: [
              IconButton(
                onPressed: () =>
                    Navigator.pushNamed(context, RecordScreen.routeName),
                icon: const Icon(Icons.fiber_smart_record, color: Colors.white),
                tooltip: 'Record',
              ),
              const SizedBox(width: 12),
              ElevatedButton.icon(
                onPressed: () => Navigator.pop(context),
                style: ElevatedButton.styleFrom(
                  backgroundColor: const Color(0xFF333333),
                  foregroundColor: Colors.white,
                  elevation: 0,
                  shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8),
                    side:
                        BorderSide(color: Colors.white.withValues(alpha: 0.1)),
                  ),
                ),
                icon: const Icon(Icons.arrow_back, size: 18),
                label: const Text('Back'),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildVideoPanel() {
    return Container(
      color: Colors.black,
      child: Stack(
        children: [
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Container(
                  padding: const EdgeInsets.all(24),
                  decoration: BoxDecoration(
                    color: Colors.white.withValues(alpha: 0.05),
                    shape: BoxShape.circle,
                  ),
                  child: Icon(Icons.videocam_off,
                      size: 64, color: Colors.grey[800]),
                ),
                const SizedBox(height: 24),
                Text(
                  'Video Feed Unavailable',
                  style: TextStyle(
                    color: Colors.grey[600],
                    fontSize: 18,
                    fontWeight: FontWeight.w500,
                  ),
                ),
                const SizedBox(height: 8),
                Container(
                  padding:
                      const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                  decoration: BoxDecoration(
                    color: Colors.grey[900],
                    borderRadius: BorderRadius.circular(12),
                  ),
                  child: Text(
                    'Frame: $_frameId',
                    style: TextStyle(
                        color: Colors.grey[500],
                        fontSize: 12,
                        fontFamily: 'Monospace'),
                  ),
                ),
              ],
            ),
          ),
          // Add a subtle grid overlay for a "tech" feel
          IgnorePointer(
            child: Container(
              decoration: BoxDecoration(
                gradient: LinearGradient(
                  begin: Alignment.topCenter,
                  end: Alignment.bottomCenter,
                  colors: [
                    Colors.transparent,
                    Colors.black.withValues(alpha: 0.2),
                  ],
                ),
              ),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusPanel() {
    return Container(
      color: ContinuonColors.gray900,
      padding: const EdgeInsets.all(20),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          _buildStatusSection('System Status', [
            _buildDarkStatusItem('Mode', 'MANUAL_CONTROL', color: Colors.green),
            _buildDarkStatusItem('Gripper', _gripperOpen ? 'OPEN' : 'CLOSED',
                color: _gripperOpen
                    ? ContinuonColors.particleOrange
                    : Colors.green),
            if (_error != null)
              _buildDarkStatusItem('Error', 'Active',
                  color: Theme.of(context).colorScheme.error),
          ]),
          const SizedBox(height: 24),
          _buildStatusSection('Controls', [
            ElevatedButton.icon(
              onPressed: _sending ? null : _toggleGripper,
              icon: Icon(_gripperOpen ? Icons.pan_tool_alt : Icons.back_hand),
              label: Text(_gripperOpen ? 'Close Gripper' : 'Open Gripper'),
              style: ElevatedButton.styleFrom(
                backgroundColor: ContinuonColors.primaryBlue,
                foregroundColor: Colors.white,
                minimumSize: const Size(double.infinity, 44),
              ),
            ),
          ]),
          const SizedBox(height: 12),
          _buildArrowControls(),
          const Spacer(),
          SizedBox(
            width: double.infinity,
            child: OutlinedButton.icon(
              onPressed: _sending
                  ? null
                  : () async {
                      setState(() => _sending = true);
                      try {
                        final res = await widget.brainClient.resetSafetyGates();
                        final ok = res['success'] == true || res['ok'] == true;
                        if (!mounted) return;
                        ScaffoldMessenger.of(context).showSnackBar(
                          SnackBar(
                            content: Text(ok
                                ? 'Safety gates reset.'
                                : 'Reset failed: ${res['message'] ?? 'unknown'}'),
                          ),
                        );
                      } finally {
                        if (mounted) setState(() => _sending = false);
                      }
                    },
              icon: const Icon(Icons.lock_open),
              label: const Text('Reset safety gates'),
            ),
          ),
          const SizedBox(height: 10),
          SizedBox(
            width: double.infinity,
            child: ElevatedButton.icon(
              onPressed: _triggerEStop, // Trigger E-Stop
              style: ElevatedButton.styleFrom(
                backgroundColor: Theme.of(context).colorScheme.error,
                foregroundColor: Colors.white,
                padding: const EdgeInsets.all(16),
                shape: RoundedRectangleBorder(
                    borderRadius: BorderRadius.circular(8)),
              ),
              icon: const Icon(Icons.stop_circle),
              label: const Text('SAFETY HOLD (E-STOP)'),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildStatusSection(String title, List<Widget> children) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          title.toUpperCase(),
          style: const TextStyle(
            color: ContinuonColors.gray500,
            fontSize: 12,
            fontWeight: FontWeight.bold,
            letterSpacing: 0.5,
          ),
        ),
        const SizedBox(height: 12),
        ...children,
      ],
    );
  }

  Widget _buildDarkStatusItem(String label, String value, {Color? color}) {
    return Container(
      margin: const EdgeInsets.only(bottom: 8),
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: ContinuonColors.gray800,
        borderRadius: BorderRadius.circular(8),
      ),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(label, style: const TextStyle(color: ContinuonColors.gray400)),
          Text(
            value,
            style: TextStyle(
                color: color ?? Colors.white, fontWeight: FontWeight.bold),
          ),
        ],
      ),
    );
  }

  Widget _buildArrowControls() {
    return Center(
      child: Container(
        padding: const EdgeInsets.all(12),
        decoration: BoxDecoration(
          color: ContinuonColors.gray800,
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          children: [
            const Text(
              'BASE CONTROL',
              style: TextStyle(color: ContinuonColors.gray500, fontSize: 10),
            ),
            const SizedBox(height: 8),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const SizedBox(width: 50),
                _buildArrowButton(
                  icon: Icons.arrow_upward,
                  label: 'Forward',
                  tooltip: 'Drive forward (gRPC SendCommand)',
                  onPressed: _sending
                      ? null
                      : () => _sendBaseTwist(linearMps: 0.12, yawRadS: 0.0),
                ),
                const SizedBox(width: 50),
              ],
            ),
            const SizedBox(height: 4),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                _buildArrowButton(
                  icon: Icons.turn_left,
                  label: 'Left',
                  tooltip: 'Turn left (gRPC SendCommand)',
                  onPressed: _sending
                      ? null
                      : () => _sendBaseTwist(linearMps: 0.0, yawRadS: 0.35),
                ),
                const SizedBox(width: 4),
                _buildArrowButton(
                  icon: Icons.stop_circle,
                  label: 'Stop',
                  tooltip: 'Stop motion (gRPC SendCommand)',
                  isCenter: true,
                  onPressed: _sending
                      ? null
                      : () => _sendBaseTwist(linearMps: 0.0, yawRadS: 0.0),
                ),
                const SizedBox(width: 4),
                _buildArrowButton(
                  icon: Icons.turn_right,
                  label: 'Right',
                  tooltip: 'Turn right (gRPC SendCommand)',
                  onPressed: _sending
                      ? null
                      : () => _sendBaseTwist(linearMps: 0.0, yawRadS: -0.35),
                ),
              ],
            ),
            const SizedBox(height: 4),
            Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                const SizedBox(width: 50),
                _buildArrowButton(
                  icon: Icons.arrow_downward,
                  label: 'Reverse',
                  tooltip: 'Drive backward (gRPC SendCommand)',
                  onPressed: _sending
                      ? null
                      : () => _sendBaseTwist(linearMps: -0.12, yawRadS: 0.0),
                ),
                const SizedBox(width: 50),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildArrowButton({
    required IconData icon,
    required String label,
    required String tooltip,
    bool isCenter = false,
    VoidCallback? onPressed,
  }) {
    return Tooltip(
      message: tooltip,
      child: SizedBox(
        width: 64,
        height: 56,
        child: ElevatedButton(
          onPressed: onPressed,
          style: ElevatedButton.styleFrom(
            backgroundColor: isCenter
                ? const Color(0xFF333333)
                : ContinuonColors.primaryBlue,
            foregroundColor: Colors.white,
            padding: const EdgeInsets.symmetric(vertical: 6, horizontal: 4),
            shape:
                RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
          ),
          child: Column(
            mainAxisAlignment: MainAxisAlignment.center,
            children: [
              Icon(icon, color: Colors.white, size: 20),
              const SizedBox(height: 2),
              Text(
                label,
                textAlign: TextAlign.center,
                style:
                    const TextStyle(fontSize: 10, fontWeight: FontWeight.w600),
              ),
            ],
          ),
        ),
      ),
    );
  }
}
