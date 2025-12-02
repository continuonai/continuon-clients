import 'dart:async';

import 'package:flutter/material.dart';

import '../models/teleop_models.dart';
import '../services/brain_client.dart';
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
  List<double> _joints = const [];
  bool _sending = false;
  String? _error;

  @override
  void initState() {
    super.initState();
    _subscription = widget.brainClient.streamRobotState(widget.brainClient.clientId).listen((state) {
      setState(() {
        _frameId = state.frameId;
        _gripperOpen = state.gripperOpen;
        _joints = state.jointPositions;
        _error = null;
      });
    }, onError: (error) {
      setState(() => _error = error.toString());
    });
  }

  @override
  void dispose() {
    _subscription.cancel();
    super.dispose();
  }

  Future<void> _sendVelocity() async {
    final command = ControlCommand(
      clientId: widget.brainClient.clientId,
      controlMode: ControlMode.eeVelocity,
      targetFrequencyHz: 30,
      eeVelocity: const EeVelocityCommand(
        linearMps: Vector3(x: 0.1, y: 0, z: 0),
        angularRadS: Vector3(x: 0, y: 0.1, z: 0),
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

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('Teleop control'),
        actions: [
          IconButton(
            onPressed: () => Navigator.pushNamed(context, RecordScreen.routeName),
            icon: const Icon(Icons.fiber_smart_record),
          )
        ],
      ),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text('Frame: $_frameId'),
            Text('Gripper open: $_gripperOpen'),
            Text('Joints: ${_joints.join(', ')}'),
            if (_error != null)
              Padding(
                padding: const EdgeInsets.only(top: 8.0),
                child: Text(
                  _error!,
                  style: const TextStyle(color: Colors.red),
                ),
              ),
            const SizedBox(height: 16),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    onPressed: _sending ? null : _sendVelocity,
                    icon: const Icon(Icons.directions_run),
                    label: Text(_sending ? 'Sending' : 'Send twist'),
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton.icon(
                  onPressed: _sending ? null : _toggleGripper,
                  icon: Icon(_gripperOpen ? Icons.pan_tool_alt : Icons.back_hand),
                  label: Text(_gripperOpen ? 'Close gripper' : 'Open gripper'),
                ),
              ],
            ),
            const SizedBox(height: 24),
            const Text('Quick tips'),
            const Text('• Commands are throttled to maintain controller safety.'),
            const Text('• Robot state events stream continuously while connected.'),
          ],
        ),
      ),
    );
  }
}
