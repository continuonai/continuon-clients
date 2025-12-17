import 'package:flutter/material.dart';
import '../models/teleop_models.dart';
import '../services/brain_client.dart';

/// SO-ARM101 6-DOF robot arm teleop control screen
/// Provides sliders for each joint + depth camera preview
class ArmTeleopScreen extends StatefulWidget {
  const ArmTeleopScreen({super.key, required this.brainClient});

  static const routeName = '/arm-teleop';

  final BrainClient brainClient;

  @override
  State<ArmTeleopScreen> createState() => _ArmTeleopScreenState();
}

class _ArmTeleopScreenState extends State<ArmTeleopScreen> {
  // Joint positions normalized [-1, 1]
  final List<double> _jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];

  // Joint names for SO-ARM101
  static const List<String> _jointNames = [
    'Base',
    'Shoulder',
    'Elbow',
    'Wrist Pitch',
    'Wrist Roll',
    'Gripper',
  ];

  // Icons for each joint
  static const List<IconData> _jointIcons = [
    Icons.rotate_right,
    Icons.rotate_90_degrees_ccw,
    Icons.gesture,
    Icons.pan_tool,
    Icons.rotate_left,
    Icons.back_hand,
  ];

  static const _defaultHost = 'brain.continuonai.com';
  static const _defaultPort = 443;

  bool _isRecording = false;
  bool _isConnected = false;
  String _episodeId = '';
  int _stepCount = 0;
  String? _error;

  @override
  void initState() {
    super.initState();
    _connectToRobot();
  }

  @override
  void dispose() {
    super.dispose();
  }

  Future<void> _connectToRobot() async {
    try {
      await widget.brainClient.connect(
        host: _defaultHost,
        port: _defaultPort,
        useTls: true,
      );
      widget.brainClient.streamRobotState(widget.brainClient.clientId).listen(
          (state) {
        setState(() {
          _error = null;
          _isConnected = true;
          if (state.jointPositions.length == 6) {
            for (int i = 0; i < 6; i++) {
              _jointPositions[i] = state.jointPositions[i];
            }
          }
        });
      }, onError: (error) {
        setState(() {
          _error = error.toString();
          _isConnected = false;
        });
      });
      setState(() {
        _isConnected = true;
      });
    } catch (error) {
      setState(() {
        _error = error.toString();
        _isConnected = false;
      });
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(
        title: const Text('SO-ARM101 Teleop'),
        actions: [
          // Connection status
          Padding(
            padding: const EdgeInsets.all(8.0),
            child: Center(
              child: Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                decoration: BoxDecoration(
                  color: _isConnected ? Colors.green : Colors.red,
                  borderRadius: BorderRadius.circular(12),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    Icon(
                      _isConnected ? Icons.check_circle : Icons.error,
                      size: 16,
                      color: Colors.white,
                    ),
                    const SizedBox(width: 4),
                    Text(
                      _isConnected ? 'Connected' : 'Disconnected',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 12,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ),
          if (_error != null)
            IconButton(
              tooltip: _error,
              onPressed: _connectToRobot,
              icon: const Icon(Icons.refresh),
            ),
        ],
      ),
      body: Column(
        children: [
          // Depth camera preview placeholder
          _buildCameraPreview(),

          // Joint controls
          Expanded(
            child: ListView(
              padding: const EdgeInsets.all(16),
              children: [
                // Joint sliders
                for (int i = 0; i < 6; i++) _buildJointControl(i),

                const SizedBox(height: 16),

                // Quick action buttons
                _buildQuickActions(),
              ],
            ),
          ),

          // Recording controls
          _buildRecordingControls(),
        ],
      ),
    );
  }

  Widget _buildCameraPreview() {
    return Container(
      height: 200,
      color: Colors.black,
      child: Stack(
        children: [
          // Placeholder for depth camera feed
          Center(
            child: Column(
              mainAxisAlignment: MainAxisAlignment.center,
              children: [
                Icon(
                  Icons.videocam_off,
                  size: 48,
                  color: Colors.white.withAlpha((255 * 0.3).round()),
                ),
                const SizedBox(height: 8),
                Text(
                  'OAK-D Lite Depth Camera',
                  style: TextStyle(
                    color: Colors.white.withAlpha((255 * 0.5).round()),
                    fontSize: 12,
                  ),
                ),
              ],
            ),
          ),

          // Recording indicator
          if (_isRecording)
            Positioned(
              top: 8,
              left: 8,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 8, vertical: 4),
                decoration: BoxDecoration(
                  color: Colors.red,
                  borderRadius: BorderRadius.circular(4),
                ),
                child: Row(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    const Icon(Icons.fiber_manual_record,
                        size: 12, color: Colors.white),
                    const SizedBox(width: 4),
                    Text(
                      'REC $_stepCount steps',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 10,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }

  Widget _buildJointControl(int index) {
    return Card(
      margin: const EdgeInsets.only(bottom: 12),
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(_jointIcons[index],
                    size: 20, color: Theme.of(context).primaryColor),
                const SizedBox(width: 8),
                Text(
                  _jointNames[index],
                  style: const TextStyle(
                    fontSize: 14,
                    fontWeight: FontWeight.bold,
                  ),
                ),
                const Spacer(),
                Text(
                  '${(_jointPositions[index] * 100).toStringAsFixed(0)}%',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[600],
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                // Quick min button
                IconButton(
                  icon: const Icon(Icons.first_page, size: 20),
                  onPressed: () => _setJointPosition(index, -1.0),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                  tooltip: 'Set ${_jointNames[index]} to minimum (-100%)',
                ),

                // Slider
                Expanded(
                  child: Slider(
                    value: _jointPositions[index],
                    min: -1.0,
                    max: 1.0,
                    divisions: 100,
                    onChanged: (value) {
                      _setJointPosition(index, value);
                    },
                  ),
                ),

                // Quick max button
                IconButton(
                  icon: const Icon(Icons.last_page, size: 20),
                  onPressed: () => _setJointPosition(index, 1.0),
                  padding: EdgeInsets.zero,
                  constraints: const BoxConstraints(),
                  tooltip: 'Set ${_jointNames[index]} to maximum (+100%)',
                ),

                // Center button
                IconButton(
                  icon: const Icon(Icons.center_focus_strong, size: 20),
                  onPressed: () => _setJointPosition(index, 0.0),
                  tooltip: 'Center ${_jointNames[index]} (0%)',
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildQuickActions() {
    return Card(
      child: Padding(
        padding: const EdgeInsets.all(12),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const Text(
              'Quick Actions',
              style: TextStyle(
                fontSize: 14,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: [
                ElevatedButton.icon(
                  icon: const Icon(Icons.home, size: 18),
                  label: const Text('Home'),
                  onPressed: _moveToHome,
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.back_hand, size: 18),
                  label: const Text('Open Gripper'),
                  onPressed: () => _setJointPosition(5, -1.0),
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.front_hand, size: 18),
                  label: const Text('Close Gripper'),
                  onPressed: () => _setJointPosition(5, 1.0),
                ),
                ElevatedButton.icon(
                  icon: const Icon(Icons.warning, size: 18),
                  label: const Text('Safety hold (E-STOP)'),
                  onPressed: _emergencyStop,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                  ),
                ),
                OutlinedButton.icon(
                  icon: const Icon(Icons.lock_open, size: 18),
                  label: const Text('Reset safety'),
                  onPressed: _resetSafety,
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  Widget _buildRecordingControls() {
    return Container(
      padding: const EdgeInsets.all(16),
      decoration: BoxDecoration(
        color: Colors.grey[100],
        boxShadow: [
          BoxShadow(
            color: Colors.black.withAlpha((255 * 0.1).round()),
            blurRadius: 4,
            offset: const Offset(0, -2),
          ),
        ],
      ),
      child: SafeArea(
        top: false,
        child: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            if (_isRecording)
              Padding(
                padding: const EdgeInsets.only(bottom: 8),
                child: Text(
                  'Episode: $_episodeId',
                  style: TextStyle(
                    fontSize: 12,
                    color: Colors.grey[600],
                  ),
                ),
              ),
            Row(
              children: [
                Expanded(
                  child: ElevatedButton.icon(
                    icon: Icon(
                        _isRecording ? Icons.stop : Icons.fiber_manual_record),
                    label: Text(
                        _isRecording ? 'Stop Recording' : 'Start Recording'),
                    onPressed: _toggleRecording,
                    style: ElevatedButton.styleFrom(
                      backgroundColor: _isRecording ? Colors.red : Colors.green,
                      foregroundColor: Colors.white,
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
                const SizedBox(width: 8),
                ElevatedButton.icon(
                  icon: const Icon(Icons.settings),
                  label: const Text('Startup & flags'),
                  onPressed: _showRuntimeSettings,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                        vertical: 16, horizontal: 16),
                  ),
                ),
              ],
            ),
          ],
        ),
      ),
    );
  }

  void _setJointPosition(int index, double value) {
    setState(() {
      _jointPositions[index] = value.clamp(-1.0, 1.0);
    });
    _sendCommand();
  }

  void _moveToHome() {
    setState(() {
      for (int i = 0; i < 6; i++) {
        _jointPositions[i] = 0.0;
      }
    });
    _sendCommand();
  }

  Future<void> _emergencyStop() async {
    try {
      final res = await widget.brainClient.triggerSafetyHold();
      final ok = res['success'] == true || res['ok'] == true;
      _showMessage(ok
          ? 'Safety hold triggered.'
          : 'Safety hold failed: ${res['message'] ?? 'unknown'}');
    } catch (e) {
      _showMessage('Safety hold failed: $e');
    }
  }

  Future<void> _resetSafety() async {
    try {
      final res = await widget.brainClient.resetSafetyGates();
      final ok = res['success'] == true || res['ok'] == true;
      _showMessage(ok
          ? 'Safety gates reset.'
          : 'Reset failed: ${res['message'] ?? 'unknown'}');
    } catch (e) {
      _showMessage('Reset failed: $e');
    }
  }

  Future<void> _showRuntimeSettings() async {
    try {
      final res = await widget.brainClient.getSettings();
      if (!mounted) return;
      if (res['success'] != true) {
        _showMessage('Failed to load settings: ${res['message'] ?? 'unknown'}');
        return;
      }
      final raw = res['settings'];
      final settings =
          raw is Map ? Map<String, dynamic>.from(raw) : <String, dynamic>{};
      final safety = (settings['safety'] is Map)
          ? Map<String, dynamic>.from(settings['safety'] as Map)
          : <String, dynamic>{};
      final chat = (settings['chat'] is Map)
          ? Map<String, dynamic>.from(settings['chat'] as Map)
          : <String, dynamic>{};

      await showDialog<void>(
        context: context,
        builder: (context) => StatefulBuilder(
          builder: (context, setLocal) {
            return AlertDialog(
              title: const Text('Startup & flags'),
              content: SingleChildScrollView(
                child: Column(
                  mainAxisSize: MainAxisSize.min,
                  children: [
                    SwitchListTile(
                      title: const Text('Allow motion'),
                      subtitle: const Text('Safety gate: allow robot motion'),
                      value: (safety['allow_motion'] as bool?) ?? true,
                      onChanged: (v) =>
                          setLocal(() => safety['allow_motion'] = v),
                    ),
                    SwitchListTile(
                      title: const Text('Record episodes'),
                      subtitle: const Text('Enable RLDS episode recording'),
                      value: (safety['record_episodes'] as bool?) ?? true,
                      onChanged: (v) =>
                          setLocal(() => safety['record_episodes'] = v),
                    ),
                    SwitchListTile(
                      title: const Text('Log chat to RLDS'),
                      subtitle: const Text('Opt-in: persist chat turns'),
                      value: (chat['log_rlds'] as bool?) ?? false,
                      onChanged: (v) => setLocal(() => chat['log_rlds'] = v),
                    ),
                  ],
                ),
              ),
              actions: [
                TextButton(
                  onPressed: () => Navigator.pop(context),
                  child: const Text('Cancel'),
                ),
                ElevatedButton.icon(
                  onPressed: () async {
                    settings['safety'] = safety;
                    settings['chat'] = chat;
                    final saved =
                        await widget.brainClient.saveSettings(settings);
                    if (!context.mounted) return;
                    final ok = saved['success'] == true;
                    Navigator.pop(context);
                    _showMessage(ok
                        ? 'Settings saved.'
                        : 'Settings rejected: ${saved['message'] ?? 'unknown'}');
                  },
                  icon: const Icon(Icons.save),
                  label: const Text('Save'),
                ),
              ],
            );
          },
        ),
      );
    } catch (e) {
      _showMessage('Failed to open settings: $e');
    }
  }

  Future<void> _sendCommand() async {
    if (!_isConnected) return;

    final command = ControlCommand(
      clientId: widget.brainClient.clientId,
      controlMode: ControlMode.armJointAngles,
      targetFrequencyHz: 20.0,
      armJointAngles: ArmJointAnglesCommand(
        normalizedAngles: List.from(_jointPositions),
      ),
    );

    try {
      await widget.brainClient.sendCommand(command);
      if (_isRecording) {
        setState(() {
          _stepCount++;
        });
      }
    } catch (error) {
      setState(() => _error = error.toString());
      _showMessage('Command failed: $error');
    }
  }

  void _toggleRecording() async {
    if (_isRecording) {
      // Stop recording
      try {
        final response = await widget.brainClient.stopRecording(success: true);
        setState(() {
          _isRecording = false;
        });

        if (response['success'] == true) {
          _showMessage('Episode saved: ${response['episode_id'] ?? ''}');
        } else {
          _showMessage('Failed to stop recording: ${response['message']}');
        }
      } catch (error) {
        setState(() => _error = error.toString());
        _showMessage('Failed to stop recording: $error');
      }
    } else {
      // Start recording
      final instruction = await _promptForInstruction();
      if (instruction != null && instruction.isNotEmpty) {
        try {
          final response = await widget.brainClient.startRecording(instruction);

          if (response['success'] == true) {
            setState(() {
              _isRecording = true;
              _episodeId = (response['episode_id'] as String?) ??
                  'episode_${DateTime.now().millisecondsSinceEpoch}';
              _stepCount = 0;
            });
            _showMessage('Recording started: $_episodeId');
          } else {
            _showMessage('Failed to start recording: ${response['message']}');
          }
        } catch (error) {
          setState(() => _error = error.toString());
          _showMessage('Failed to start recording: $error');
        }
      }
    }
  }

  Future<String?> _promptForInstruction() async {
    return showDialog<String>(
      context: context,
      builder: (context) {
        String instruction = '';
        return AlertDialog(
          title: const Text('Episode Instruction'),
          content: TextField(
            autofocus: true,
            decoration: const InputDecoration(
              hintText: 'e.g., Pick up the red cube',
              labelText: 'Task description',
            ),
            onChanged: (value) => instruction = value,
            onSubmitted: (value) {
              Navigator.pop(context, value);
            },
          ),
          actions: [
            TextButton(
              onPressed: () => Navigator.pop(context),
              child: const Text('Cancel'),
            ),
            TextButton(
              onPressed: () => Navigator.pop(context, instruction),
              child: const Text('Start'),
            ),
          ],
        );
      },
    );
  }

  void _showMessage(String message) {
    ScaffoldMessenger.of(context).showSnackBar(
      SnackBar(content: Text(message)),
    );
  }
}
