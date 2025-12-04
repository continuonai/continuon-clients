import 'package:flutter/material.dart';

import '../models/rlds_models.dart';
import '../models/teleop_models.dart';
import '../services/brain_client.dart';
import '../services/cloud_uploader.dart';
import '../services/multimodal_inputs.dart';
import '../services/task_recorder.dart';

class RecordScreen extends StatefulWidget {
  const RecordScreen({
    super.key,
    required this.brainClient,
    this.initialControlRole = 'human_teleop',
  });

  static const routeName = '/record';

  final BrainClient brainClient;
  final String initialControlRole;

  @override
  State<RecordScreen> createState() => _RecordScreenState();
}

class _RecordScreenState extends State<RecordScreen> {
  late TaskRecorder _recorder;
  late MultimodalInputs _multimodalInputs;
  final _notesController = TextEditingController();
  final _voiceCommandController = TextEditingController();
  final _cloudUploader = CloudUploader();
  bool _uploading = false;
  String? _status;
  late String _controlRole;
  bool _includeDeviceMic = true;
  bool _includeRobotMic = true;
  bool _includeEgocentricVideo = true;

  @override
  void initState() {
    super.initState();
    _controlRole = widget.initialControlRole;
    _recorder = _buildRecorder();
    _multimodalInputs = MultimodalInputs();
  }

  TaskRecorder _buildRecorder() {
    return TaskRecorder(
      metadata: EpisodeMetadata(
        xrMode: 'trainer',
        controlRole: _controlRole,
        environmentId: 'lab-mock',
        tags: const ['multimodal_audio_video'],
      ),
    );
  }

  @override
  void dispose() {
    _notesController.dispose();
    _voiceCommandController.dispose();
    super.dispose();
  }

  void _populateMultimodalInputs() {
    final frameId = DateTime.now().millisecondsSinceEpoch.toString();
    if (_includeDeviceMic) {
      _multimodalInputs.updateDeviceMic(
        uri: 'mic://device/$frameId.wav',
        frameId: frameId,
      );
    }
    if (_includeRobotMic) {
      _multimodalInputs.updateRobotMic(
        uri: 'mic://robot/$frameId.wav',
        frameId: frameId,
      );
    }
    if (_includeEgocentricVideo) {
      _multimodalInputs.updateEgocentricFrame(
        uri: 'video://egocentric/$frameId.mp4',
        frameId: frameId,
        depthUri: 'depth://egocentric/$frameId.bin',
      );
      _multimodalInputs.updateGaze(
        origin: const Vector3(x: 0, y: 0, z: 0),
        direction: const Vector3(x: 0, y: 0, z: -1),
        confidence: 0.8,
        targetId: 'ui:record',
      );
    } else {
      _multimodalInputs.clearEgocentricFrame();
      _multimodalInputs.clearGaze();
    }
    if (!_includeDeviceMic && !_includeRobotMic) {
      _multimodalInputs.clearAudio();
    }
  }

  void _recordStep() {
    _populateMultimodalInputs();
    _recorder.addStep(
      EpisodeStep(
        observation: _multimodalInputs.buildObservation(
          uiContext: {
            'notes': _notesController.text,
            'voice_hint': _voiceCommandController.text,
          },
        ),
        action: {
          'command': [],
          'source': 'human_teleop_xr',
          if (_voiceCommandController.text.isNotEmpty)
            'voice_command': _voiceCommandController.text,
        },
      ),
    );
    setState(() => _status = 'Recorded step ${DateTime.now().toIso8601String()}');
  }

  Future<void> _uploadRecord() async {
    setState(() => _uploading = true);
    final record = _recorder.buildRecord();
    try {
      final signedUrl = await _cloudUploader.fetchSignedUploadUrl(
        Uri.parse('https://upload.continuon.ai/episodes'),
        record.metadata,
      );
      await _cloudUploader.uploadEpisode(record, signedUrl);
      setState(() => _status = 'Uploaded to ${signedUrl.host}');
    } catch (error) {
      setState(() => _status = 'Upload failed: $error');
    } finally {
      setState(() => _uploading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Record task / RLDS episodes')),
      body: Padding(
        padding: const EdgeInsets.all(16),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            DropdownButtonFormField<String>(
              initialValue: _controlRole,
              decoration: const InputDecoration(
                labelText: 'Control role (RLDS metadata)',
                helperText: 'Manual mode sets control_role=manual_driver',
              ),
              items: const [
                DropdownMenuItem(value: 'human_teleop', child: Text('Automatic supervision')), 
                DropdownMenuItem(value: 'manual_driver', child: Text('Manual driver')), 
                DropdownMenuItem(value: 'autonomous_record', child: Text('Autonomous record')), 
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _controlRole = value;
                  _recorder = _buildRecorder();
                });
              },
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _notesController,
              decoration: const InputDecoration(
                labelText: 'Operator notes',
                helperText: 'Captured into RLDS UiContext',
              ),
            ),
            const SizedBox(height: 12),
            Wrap(
              spacing: 12,
              children: [
                ElevatedButton.icon(
                  onPressed: _recordStep,
                  icon: const Icon(Icons.note_add),
                  label: const Text('Record step'),
                ),
                ElevatedButton.icon(
                  onPressed: _uploading ? null : _uploadRecord,
                  icon: const Icon(Icons.cloud_upload),
                  label: Text(_uploading ? 'Uploading...' : 'Upload RLDS'),
                ),
              ],
            ),
            const SizedBox(height: 12),
            SwitchListTile(
              value: _includeDeviceMic,
              onChanged: (value) => setState(() => _includeDeviceMic = value),
              title: const Text('Include device mic audio (phone/TV)'),
              subtitle: const Text('Stamps sample_rate_hz/num_channels into observation.audio.device_mic'),
            ),
            SwitchListTile(
              value: _includeRobotMic,
              onChanged: (value) => setState(() => _includeRobotMic = value),
              title: const Text('Include robot mic audio'),
              subtitle: const Text('Keeps parity with XR app dual-mic traces for Gemma voice alignment'),
            ),
            SwitchListTile(
              value: _includeEgocentricVideo,
              onChanged: (value) => setState(() => _includeEgocentricVideo = value),
              title: const Text('Include egocentric video + depth + gaze'),
              subtitle: const Text('Aligns frame_id across video/depth/gaze for RLDS validation'),
            ),
            const SizedBox(height: 12),
            TextField(
              controller: _voiceCommandController,
              decoration: const InputDecoration(
                labelText: 'Voice command text',
                helperText: 'Logged to action.voice_command to mirror XR speech control',
              ),
            ),
            const SizedBox(height: 24),
            Text(_status ?? 'Pending'),
            const SizedBox(height: 12),
            const Text('Recording flows'),
            const Text('• Uses RLDS schema aligned with proto/rlds_episode.proto.'),
            const Text('• Upload broker expected to return signed storage URL.'),
            const Text('• Multimodal steps include audio (device + robot), egocentric video/depth, and gaze metadata.'),
          ],
        ),
      ),
    );
  }
}
