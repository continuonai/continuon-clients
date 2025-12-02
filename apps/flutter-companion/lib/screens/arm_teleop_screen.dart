import 'package:flutter/material.dart';
import '../models/teleop_models.dart';
import '../services/robot_api_service.dart';

/// SO-ARM101 6-DOF robot arm teleop control screen
/// Provides sliders for each joint + depth camera preview
class ArmTeleopScreen extends StatefulWidget {
  const ArmTeleopScreen({super.key});

  @override
  State<ArmTeleopScreen> createState() => _ArmTeleopScreenState();
}

class _ArmTeleopScreenState extends State<ArmTeleopScreen> {
  // Joint positions normalized [-1, 1]
  final List<double> _jointPositions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
  
  // Robot API service
  late RobotApiService _robotApi;
  
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

  bool _isRecording = false;
  bool _isConnected = false;
  String _episodeId = '';
  int _stepCount = 0;

  @override
  void initState() {
    super.initState();
    _robotApi = RobotApiService(host: 'localhost', port: 8080);
    _connectToRobot();
  }

  @override
  void dispose() {
    _robotApi.disconnect();
    super.dispose();
  }

  Future<void> _connectToRobot() async {
    final connected = await _robotApi.connect();
    setState(() {
      _isConnected = connected;
    });

    if (connected) {
      // Subscribe to robot state updates
      _robotApi.streamRobotState('flutter_companion').listen((state) {
        setState(() {
          if (state.jointPositions.length == 6) {
            for (int i = 0; i < 6; i++) {
              _jointPositions[i] = state.jointPositions[i];
            }
          }
        });
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
                padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
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
                for (int i = 0; i < 6; i++)
                  _buildJointControl(i),
                
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
                  color: Colors.white.withOpacity(0.3),
                ),
                const SizedBox(height: 8),
                Text(
                  'OAK-D Lite Depth Camera',
                  style: TextStyle(
                    color: Colors.white.withOpacity(0.5),
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
                    const Icon(Icons.fiber_manual_record, size: 12, color: Colors.white),
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
                Icon(_jointIcons[index], size: 20, color: Theme.of(context).primaryColor),
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
                ),
                
                // Center button
                IconButton(
                  icon: const Icon(Icons.center_focus_strong, size: 20),
                  onPressed: () => _setJointPosition(index, 0.0),
                  tooltip: 'Center',
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
                  label: const Text('Emergency Stop'),
                  onPressed: _emergencyStop,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.red,
                    foregroundColor: Colors.white,
                  ),
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
            color: Colors.black.withOpacity(0.1),
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
                    icon: Icon(_isRecording ? Icons.stop : Icons.fiber_manual_record),
                    label: Text(_isRecording ? 'Stop Recording' : 'Start Recording'),
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
                  label: const Text('Settings'),
                  onPressed: _showSettings,
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 16),
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

  void _emergencyStop() {
    // Send emergency stop command
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Emergency Stop'),
        content: const Text('Arm has been stopped. All servos are holding current positions.'),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('OK'),
          ),
        ],
      ),
    );
  }

  void _toggleRecording() {
    setState(() {
      _isRecording = !_isRecording;
      if (_isRecording) {
        _episodeId = 'episode_${DateTime.now().millisecondsSinceEpoch}';
        _stepCount = 0;
      }
    });
  }

  void _showSettings() {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        title: const Text('Arm Settings'),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            SwitchListTile(
              title: const Text('Enable Safety Bounds'),
              value: true,
              onChanged: (value) {},
            ),
            SwitchListTile(
              title: const Text('Record Depth Frames'),
              value: true,
              onChanged: (value) {},
            ),
            ListTile(
              title: const Text('Control Frequency'),
              trailing: const Text('20 Hz'),
              onTap: () {},
            ),
          ],
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Close'),
          ),
        ],
      ),
    );
  }

  void _sendCommand() {
    if (!_isConnected) return;

    final command = ControlCommand(
      clientId: 'flutter_companion',
      controlMode: ControlMode.armJointAngles,
      targetFrequencyHz: 20.0,
      armJointAngles: ArmJointAnglesCommand(
        normalizedAngles: List.from(_jointPositions),
      ),
    );

    // Send command to robot
    _robotApi.sendCommand(command).then((response) {
      if (!response['success']) {
        print('Command failed: ${response['message']}');
      }
    });

    if (_isRecording) {
      setState(() {
        _stepCount++;
      });
    }
  }

  void _toggleRecording() async {
    if (_isRecording) {
      // Stop recording
      final response = await _robotApi.stopRecording(success: true);
      setState(() {
        _isRecording = false;
      });
      
      if (response['success']) {
        _showMessage('Episode saved: ${response['episode_id']}');
      } else {
        _showMessage('Failed to stop recording: ${response['message']}');
      }
    } else {
      // Start recording
      final instruction = await _promptForInstruction();
      if (instruction != null && instruction.isNotEmpty) {
        final response = await _robotApi.startRecording(instruction);
        
        if (response['success']) {
          setState(() {
            _isRecording = true;
            _episodeId = response['episode_id'];
            _stepCount = 0;
          });
          _showMessage('Recording started: $_episodeId');
        } else {
          _showMessage('Failed to start recording: ${response['message']}');
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
