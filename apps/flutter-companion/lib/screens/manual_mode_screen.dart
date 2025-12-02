import 'dart:async';

import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';

import '../models/teleop_models.dart';
import '../services/brain_client.dart';
import '../services/controller_input.dart';
import 'record_screen.dart';

class ManualModeScreen extends StatefulWidget {
  const ManualModeScreen({super.key, required this.brainClient});

  static const routeName = '/manual';

  final BrainClient brainClient;

  @override
  State<ManualModeScreen> createState() => _ManualModeScreenState();
}

class _ManualModeScreenState extends State<ManualModeScreen> {
  final _ps3Controller = const Ps3ControllerBridge();
  late StreamSubscription<RobotState> _stateSubscription;
  StreamSubscription<ControllerSnapshot>? _controllerSubscription;

  AccelerationProfile _profile = AccelerationProfile.nominal;
  double _linear = 0.0;
  double _angular = 0.0;
  double _gripperTarget = 0.02;
  bool _gripperOpen = false;
  String _frameId = 'unknown';
  bool _sending = false;
  bool _controllerConnected = false;

  @override
  void initState() {
    super.initState();
    _stateSubscription = widget.brainClient.streamRobotState('flutter-companion').listen((state) {
      setState(() {
        _frameId = state.frameId;
        _gripperOpen = state.gripperOpen;
      });
    });
  }

  @override
  void dispose() {
    _controllerSubscription?.cancel();
    _stateSubscription.cancel();
    super.dispose();
  }

  Future<void> _sendManualTwist() async {
    setState(() => _sending = true);
    try {
      await widget.brainClient.sendCommand(
        ControlCommand(
          clientId: 'flutter-companion',
          controlMode: ControlMode.eeVelocity,
          targetFrequencyHz: 30,
          eeVelocity: EeVelocityCommand(
            referenceFrame: ReferenceFrame.base,
            linearMps: Vector3(x: _linear, y: 0, z: 0),
            angularRadS: Vector3(x: 0, y: 0, z: _angular),
          ),
        ),
      );
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _sendGripper() async {
    setState(() => _sending = true);
    try {
      await widget.brainClient.sendCommand(
        ControlCommand(
          clientId: 'flutter-companion',
          controlMode: ControlMode.gripper,
          targetFrequencyHz: 5,
          gripperCommand: GripperCommand(
            mode: GripperMode.position,
            positionM: _gripperTarget,
          ),
        ),
      );
    } finally {
      if (mounted) setState(() => _sending = false);
    }
  }

  Future<void> _toggleController() async {
    if (_controllerSubscription != null) {
      await _controllerSubscription?.cancel();
      setState(() {
        _controllerSubscription = null;
        _controllerConnected = false;
      });
      return;
    }
    final ok = await _ps3Controller.requestConnection();
    if (!mounted) return;
    setState(() => _controllerConnected = ok);
    if (!ok) return;
    _controllerSubscription = _ps3Controller.pipeToBrain(
      brainClient: widget.brainClient,
      profile: _profile,
    );
  }

  void _updateProfile(AccelerationProfile profile) {
    setState(() => _profile = profile);
    if (_controllerSubscription != null) {
      _controllerSubscription?.cancel();
      _controllerSubscription = _ps3Controller.pipeToBrain(
        brainClient: widget.brainClient,
        profile: profile,
      );
    }
  }

  Widget _buildModeChips() {
    return Wrap(
      spacing: 12,
      children: [
        ChoiceChip(
          label: const Text('Automatic'),
          selected: false,
          onSelected: (_) => Navigator.pop(context),
        ),
        ChoiceChip(
          label: const Text('Manual driving'),
          selected: true,
          onSelected: (_) {},
        ),
        ChoiceChip(
          label: const Text('Record'),
          selected: false,
          onSelected: (_) => Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => RecordScreen(
                brainClient: widget.brainClient,
                initialControlRole: 'manual_driver',
              ),
            ),
          ),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    final linearMax = _profile.linearScale;
    final angularMax = _profile.angularScale;
    return Scaffold(
      appBar: AppBar(
        title: const Text('Manual mode / driving surface'),
        actions: [
          IconButton(
            onPressed: () => Navigator.push(
              context,
              MaterialPageRoute(
                builder: (_) => RecordScreen(
                  brainClient: widget.brainClient,
                  initialControlRole: 'manual_driver',
                ),
              ),
            ),
            icon: const Icon(Icons.fiber_manual_record),
            tooltip: 'Record manual session',
          ),
        ],
      ),
      body: ListView(
        padding: const EdgeInsets.all(16),
        children: [
          _buildModeChips(),
          const SizedBox(height: 12),
          Text('Frame: $_frameId • Gripper open: $_gripperOpen'),
          const SizedBox(height: 16),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Acceleration profile'),
                  Wrap(
                    spacing: 8,
                    children: AccelerationProfile.values
                        .map(
                          (profile) => ChoiceChip(
                            label: Text(profile.label),
                            selected: profile == _profile,
                            onSelected: (_) => _updateProfile(profile),
                          ),
                        )
                        .toList(),
                  ),
                  const SizedBox(height: 12),
                  Text('Linear velocity (${linearMax.toStringAsFixed(2)} m/s)'),
                  Slider(
                    value: _linear,
                    min: -linearMax,
                    max: linearMax,
                    onChanged: (value) => setState(() => _linear = value),
                  ),
                  Text('Yaw velocity (${angularMax.toStringAsFixed(2)} rad/s)'),
                  Slider(
                    value: _angular,
                    min: -angularMax,
                    max: angularMax,
                    onChanged: (value) => setState(() => _angular = value),
                  ),
                  Row(
                    children: [
                      ElevatedButton.icon(
                        onPressed: _sending ? null : _sendManualTwist,
                        icon: const Icon(Icons.send),
                        label: Text(_sending ? 'Sending...' : 'Send command'),
                      ),
                      const SizedBox(width: 12),
                      OutlinedButton.icon(
                        onPressed: _sending
                            ? null
                            : () {
                                setState(() {
                                  _linear = 0;
                                  _angular = 0;
                                });
                                _sendManualTwist();
                              },
                        icon: const Icon(Icons.stop_circle),
                        label: const Text('Stop'),
                      ),
                    ],
                  )
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('Gripper control'),
                      Text(_gripperOpen ? 'Open' : 'Closed'),
                    ],
                  ),
                  Slider(
                    value: _gripperTarget,
                    min: 0,
                    max: 0.05,
                    divisions: 5,
                    label: '${_gripperTarget.toStringAsFixed(3)} m',
                    onChanged: (value) => setState(() => _gripperTarget = value),
                  ),
                  Row(
                    children: [
                      ElevatedButton.icon(
                        onPressed: _sending ? null : _sendGripper,
                        icon: const Icon(Icons.handshake),
                        label: const Text('Set gripper'),
                      ),
                      const SizedBox(width: 12),
                      OutlinedButton.icon(
                        onPressed: _sending
                            ? null
                            : () {
                                setState(() => _gripperTarget = _gripperOpen ? 0.0 : 0.045);
                                _sendGripper();
                              },
                        icon: const Icon(Icons.sync_alt),
                        label: const Text('Toggle'),
                      ),
                    ],
                  )
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      const Text('PlayStation 3 controller'),
                      Icon(_controllerConnected ? Icons.gamepad : Icons.gamepad_outlined),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    kIsWeb
                        ? 'Connect a controller from a native shell; browser surfaces forward input via the platform channel.'
                        : 'Left stick drives translation, right stick controls yaw. Cross opens the gripper, Circle closes.',
                  ),
                  const SizedBox(height: 8),
                  ElevatedButton.icon(
                    onPressed: kIsWeb ? null : _toggleController,
                    icon: Icon(_controllerSubscription != null ? Icons.close : Icons.play_arrow),
                    label: Text(_controllerSubscription != null ? 'Stop listening' : 'Start listening'),
                  ),
                ],
              ),
            ),
          ),
          const SizedBox(height: 12),
          Card(
            child: Padding(
              padding: const EdgeInsets.all(12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  const Text('Browser control surface'),
                  const SizedBox(height: 8),
                  const Text(
                    'Build/run with `flutter run -d chrome --web-port 8080` and open /#/manual to expose this driving surface for operators.',
                  ),
                  const SizedBox(height: 8),
                  OutlinedButton.icon(
                    onPressed: () => Navigator.push(
                      context,
                      MaterialPageRoute(
                        builder: (_) => ManualModeScreen(brainClient: widget.brainClient),
                      ),
                    ),
                    icon: const Icon(Icons.open_in_new),
                    label: const Text('Open embedded surface'),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}
