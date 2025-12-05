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
  BaseModePreset _basePreset = BaseModePreset.balanced;
  double _linear = 0.0;
  double _angular = 0.0;
  double _turningRadius = 1.0;
  double _accelerationBias = 0.6;
  double _brakeBias = 0.3;
  double _gripperTarget = 0.02;
  bool _gripperOpen = false;
  final List<double> _jointPositions = List<double>.filled(6, 0.0);
  final List<String> _jointLabels = const [
    'Base',
    'Shoulder',
    'Elbow',
    'Wrist Pitch',
    'Wrist Roll',
    'Gripper'
  ];
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
        if (state.jointPositions.length == _jointPositions.length) {
          for (int i = 0; i < _jointPositions.length; i++) {
            _jointPositions[i] = state.jointPositions[i];
          }
        }
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
      final double throttleScale = 0.4 + (_accelerationBias * 0.6);
      final double brakeScale = 1.0 - (_brakeBias * 0.35);
      final double linearCommand = _linear * throttleScale * brakeScale;
      final double angularCandidate = _linear.abs() > 0.01
          ? (linearCommand / _turningRadius) * (_linear.isNegative ? -1 : 1)
          : _angular;
      final double angularCommand = (angularCandidate.abs() > _angular.abs() && _angular != 0)
          ? _angular * brakeScale
          : angularCandidate * brakeScale;
      await widget.brainClient.sendCommand(
        ControlCommand(
          clientId: 'flutter-companion',
          controlMode: ControlMode.eeVelocity,
          targetFrequencyHz: 30,
          eeVelocity: EeVelocityCommand(
            referenceFrame: ReferenceFrame.base,
            linearMps: Vector3(x: linearCommand, y: 0, z: 0),
            angularRadS: Vector3(x: 0, y: 0, z: angularCommand),
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

  void _applyBasePreset(BaseModePreset preset) {
    setState(() {
      _basePreset = preset;
      _profile = preset.profile;
      _turningRadius = preset.turningRadiusMeters;
      _accelerationBias = preset.accelerationBias;
      _brakeBias = preset.brakeBias;
    });
  }

  void _applyPosturePreset(List<double> joints) {
    setState(() {
      for (int i = 0; i < _jointPositions.length; i++) {
        _jointPositions[i] = joints[i];
      }
    });
    _sendArmJointTargets();
  }

  Future<void> _sendArmJointTargets() async {
    setState(() => _sending = true);
    try {
      await widget.brainClient.sendCommand(
        ControlCommand(
          clientId: 'flutter-companion',
          controlMode: ControlMode.armJointAngles,
          targetFrequencyHz: 20.0,
          armJointAngles: ArmJointAnglesCommand(normalizedAngles: List.from(_jointPositions)),
        ),
      );
    } finally {
      if (mounted) setState(() => _sending = false);
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
      body: DefaultTabController(
        length: 2,
        child: Column(
          children: [
            Padding(
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  _buildModeChips(),
                  const SizedBox(height: 12),
                  Text('Frame: $_frameId • Gripper open: $_gripperOpen'),
                  const SizedBox(height: 16),
                  Semantics(
                    label: 'Manual control tabs',
                    hint: 'Use arrow keys or swipe to switch between Base and Arm controls',
                    child: TabBar(
                      isScrollable: true,
                      indicatorColor: Theme.of(context).colorScheme.primary,
                      tabs: const [
                        Tab(icon: Icon(Icons.directions_run), text: 'Base / Feet'),
                        Tab(icon: Icon(Icons.precision_manufacturing), text: 'Arm'),
                      ],
                    ),
                  ),
                ],
              ),
            ),
            Expanded(
              child: TabBarView(
                children: [
                  _buildBaseTab(linearMax, angularMax),
                  _buildArmTab(),
                ],
              ),
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildBaseTab(double linearMax, double angularMax) {
    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text('Driving modes'),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: BaseModePreset.values
                      .map(
                        (preset) => ChoiceChip(
                          label: Text(preset.label),
                          selected: _basePreset == preset,
                          tooltip: preset.description,
                          onSelected: (_) => _applyBasePreset(preset),
                        ),
                      )
                      .toList(),
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
                Text('Turning radius (${_turningRadius.toStringAsFixed(2)} m)'),
                Slider(
                  value: _turningRadius,
                  min: 0.25,
                  max: 2.5,
                  onChanged: (value) {
                    setState(() {
                      _turningRadius = value;
                      if (_linear.abs() > 0.01) {
                        _angular = (_linear / value) * (_linear.isNegative ? -1 : 1);
                      }
                    });
                  },
                ),
                Text('Acceleration bias (${(_accelerationBias * 100).round()}%)'),
                Slider(
                  value: _accelerationBias,
                  min: 0.0,
                  max: 1.0,
                  onChanged: (value) => setState(() => _accelerationBias = value),
                ),
                Text('Brake bias (${(_brakeBias * 100).round()}%)'),
                Slider(
                  value: _brakeBias,
                  min: 0.0,
                  max: 1.0,
                  onChanged: (value) => setState(() => _brakeBias = value),
                ),
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
                    const Text('PlayStation 3 controller'),
                    Icon(_controllerConnected ? Icons.gamepad : Icons.gamepad_outlined),
                  ],
                ),
                const SizedBox(height: 8),
                const Text(
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
        const SizedBox(height: 16),
      ],
    );
  }

  Widget _buildArmTab() {
    const presetPostures = <String, List<double>>{
      'Ready': [0.0, -0.2, 0.35, 0.15, 0.0, 0.4],
      'Carry': [0.1, -0.35, 0.2, 0.1, 0.0, 0.6],
      'Stow': [0.0, -0.5, 0.0, -0.1, 0.0, 0.0],
    };
    return ListView(
      padding: const EdgeInsets.symmetric(horizontal: 16),
      children: [
        Card(
          child: Padding(
            padding: const EdgeInsets.all(12),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                const Text(
                  'Quick postures',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 8),
                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: presetPostures.entries
                      .map(
                        (entry) => ElevatedButton(
                          onPressed: _sending ? null : () => _applyPosturePreset(entry.value),
                          child: Text(entry.key),
                        ),
                      )
                      .toList(),
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
                const Text(
                  'Arm joints',
                  style: TextStyle(fontWeight: FontWeight.bold),
                ),
                const SizedBox(height: 12),
                for (int i = 0; i < _jointPositions.length; i++) ...[
                  Row(
                    children: [
                      Expanded(
                        child: Text(_jointLabels[i], style: const TextStyle(fontWeight: FontWeight.w600)),
                      ),
                      Text('${(_jointPositions[i] * 100).toStringAsFixed(0)}%'),
                    ],
                  ),
                  Slider(
                    value: _jointPositions[i],
                    min: -1.0,
                    max: 1.0,
                    onChanged: (value) => setState(() => _jointPositions[i] = value),
                    onChangeEnd: (_) => _sendArmJointTargets(),
                  ),
                ],
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
        const SizedBox(height: 16),
      ],
    );
  }
}

enum BaseModePreset { dock, balanced, sport }

extension BaseModePresetDetails on BaseModePreset {
  String get label {
    switch (this) {
      case BaseModePreset.dock:
        return 'Docking';
      case BaseModePreset.balanced:
        return 'Balanced';
      case BaseModePreset.sport:
        return 'Sport';
    }
  }

  String get description {
    switch (this) {
      case BaseModePreset.dock:
        return 'Tighter turns, gentle acceleration';
      case BaseModePreset.balanced:
        return 'Everyday drive with medium responsiveness';
      case BaseModePreset.sport:
        return 'Widest turns with aggressive throttle';
    }
  }

  double get turningRadiusMeters {
    switch (this) {
      case BaseModePreset.dock:
        return 0.4;
      case BaseModePreset.balanced:
        return 1.0;
      case BaseModePreset.sport:
        return 1.8;
    }
  }

  AccelerationProfile get profile {
    switch (this) {
      case BaseModePreset.dock:
        return AccelerationProfile.precision;
      case BaseModePreset.balanced:
        return AccelerationProfile.nominal;
      case BaseModePreset.sport:
        return AccelerationProfile.aggressive;
    }
  }

  double get accelerationBias {
    switch (this) {
      case BaseModePreset.dock:
        return 0.4;
      case BaseModePreset.balanced:
        return 0.6;
      case BaseModePreset.sport:
        return 0.9;
    }
  }

  double get brakeBias {
    switch (this) {
      case BaseModePreset.dock:
        return 0.5;
      case BaseModePreset.balanced:
        return 0.3;
      case BaseModePreset.sport:
        return 0.15;
    }
  }
}
