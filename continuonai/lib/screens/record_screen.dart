import 'package:flutter/material.dart';

import '../models/rlds_models.dart';
import '../models/teleop_models.dart';
import '../services/brain_client.dart';
import '../services/cloud_uploader.dart';
import '../services/media_capture_service.dart';
import '../services/multimodal_inputs.dart';
import '../services/task_recorder.dart';

enum SensorStatus { idle, recording, queued, error }

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
  final _captureService = MediaCaptureService();
  bool _uploading = false;
  bool _transferring = false;
  String? _status;
  late String _controlRole;
  bool _includeDeviceMic = true;
  bool _includeRobotMic = true;
  bool _includeEgocentricVideo = true;
  String _egocentricMount = 'chest';
  final Map<String, SensorStatus> _sensorStates = {
    'device_mic': SensorStatus.idle,
    'robot_mic': SensorStatus.idle,
    'egocentric': SensorStatus.idle,
  };

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

  String _sensorStatusLabel(String sensor, bool enabled) {
    if (!enabled) return 'Disabled';
    switch (_sensorStates[sensor]) {
      case SensorStatus.recording:
        return 'Recording…';
      case SensorStatus.queued:
        return 'Queued for upload';
      case SensorStatus.error:
        return 'Error – tap record to retry';
      case SensorStatus.idle:
      default:
        return 'Ready';
    }
  }

  void _applyCaptureToInputs(MediaCaptureResult capture) {
    if (_includeDeviceMic && capture.deviceMic != null) {
      final mic = capture.deviceMic!;
      _multimodalInputs.updateDeviceMic(
        uri: Uri.file(mic.path).toString(),
        frameId: mic.frameId,
        timestampMs: mic.timestampMs,
        sampleRateHz: mic.sampleRateHz,
        numChannels: mic.numChannels,
      );
    }
    if (_includeRobotMic && capture.robotMic != null) {
      final mic = capture.robotMic!;
      _multimodalInputs.updateRobotMic(
        uri: Uri.file(mic.path).toString(),
        frameId: mic.frameId,
        timestampMs: mic.timestampMs,
        sampleRateHz: mic.sampleRateHz,
        numChannels: mic.numChannels,
      );
    }
    if (!_includeDeviceMic) {
      _multimodalInputs.clearDeviceMic();
    }
    if (!_includeRobotMic) {
      _multimodalInputs.clearRobotMic();
    }
    if (!_includeDeviceMic && !_includeRobotMic) {
      _multimodalInputs.clearAudio();
    }

    if (_includeEgocentricVideo && capture.egocentric != null) {
      final frame = capture.egocentric!;
      _multimodalInputs.updateEgocentricFrame(
        uri: Uri.file(frame.path).toString(),
        frameId: frame.frameId,
        timestampMs: frame.timestampMs,
        mount: frame.mount,
        depthUri: frame.depthPath != null ? Uri.file(frame.depthPath!).toString() : null,
      );
      _multimodalInputs.updateGaze(
        origin: const Vector3(x: 0, y: 0, z: 0),
        direction: const Vector3(x: 0, y: 0, z: -1),
        confidence: 0.8,
        targetId: 'ui:record_${frame.mount}',
      );
    } else {
      _multimodalInputs.clearEgocentricFrame();
      _multimodalInputs.clearGaze();
    }
  }

  Future<void> _recordStep() async {
    setState(() {
      _status = 'Capturing sensors…';
      if (_includeDeviceMic) _sensorStates['device_mic'] = SensorStatus.recording;
      if (_includeRobotMic) _sensorStates['robot_mic'] = SensorStatus.recording;
      if (_includeEgocentricVideo) _sensorStates['egocentric'] = SensorStatus.recording;
    });
    try {
      final capture = await _captureService.capture(
        includeDeviceMic: _includeDeviceMic,
        includeRobotMic: _includeRobotMic,
        includeEgocentric: _includeEgocentricVideo,
        egocentricMount: _egocentricMount,
      );
      _applyCaptureToInputs(capture);
      _recorder.addAssets(capture.toAssets());
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

      setState(() {
        _status = 'Recorded step ${DateTime.now().toIso8601String()}';
        if (_includeDeviceMic) {
          _sensorStates['device_mic'] =
              capture.deviceMic != null ? SensorStatus.queued : SensorStatus.error;
        }
        if (_includeRobotMic) {
          _sensorStates['robot_mic'] =
              capture.robotMic != null ? SensorStatus.queued : SensorStatus.error;
        }
        if (_includeEgocentricVideo) {
          _sensorStates['egocentric'] =
              capture.egocentric != null ? SensorStatus.queued : SensorStatus.error;
        }
      });
    } catch (error) {
      setState(() {
        _status = 'Capture failed: $error';
        if (_includeDeviceMic) _sensorStates['device_mic'] = SensorStatus.error;
        if (_includeRobotMic) _sensorStates['robot_mic'] = SensorStatus.error;
        if (_includeEgocentricVideo) _sensorStates['egocentric'] = SensorStatus.error;
      });
    }
  }

  Future<void> _uploadRecord() async {
    setState(() {
      _uploading = true;
      _status = 'Uploading RLDS manifest…';
    });
    final package = _recorder.buildPackage();
    try {
      final signedUrl = await _cloudUploader.fetchSignedUploadUrl(
        Uri.parse('https://upload.continuon.ai/episodes'),
        package.record.metadata,
      );
      await _cloudUploader.uploadEpisode(package.record, signedUrl);
      setState(() => _status = 'Uploaded manifest to ${signedUrl.host}');

      if (package.assets.isNotEmpty && widget.brainClient.isConnected) {
        setState(() {
          _transferring = true;
          _status = 'Transferring media to managed robot…';
        });
        final remoteMap = await widget.brainClient.handoffEpisodeAssets(package);
        final updatedAssets = _recorder.markAssetsTransferred(remoteMap);
        final refreshedRecord = _recorder.buildRecord(assetsOverride: updatedAssets);
        await _cloudUploader.uploadEpisode(refreshedRecord, signedUrl);
        setState(() {
          _status = 'Uploaded + handed off to robot';
          _sensorStates.updateAll((key, value) =>
              value == SensorStatus.queued ? SensorStatus.idle : value);
        });
      } else if (package.assets.isNotEmpty) {
        setState(() => _status = 'Uploaded; waiting for robot connection to handoff media');
      }
    } catch (error) {
      setState(() {
        _status = 'Upload failed: $error';
        _sensorStates.updateAll((key, value) {
          if (value == SensorStatus.queued || value == SensorStatus.recording) {
            return SensorStatus.error;
          }
          return value;
        });
      });
    } finally {
      setState(() {
        _uploading = false;
        _transferring = false;
      });
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
              value: _controlRole,
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
                  onPressed: _uploading || _transferring ? null : _recordStep,
                  icon: const Icon(Icons.note_add),
                  label: const Text('Record step'),
                ),
                ElevatedButton.icon(
                  onPressed: _uploading || _transferring ? null : _uploadRecord,
                  icon: const Icon(Icons.cloud_upload),
                  label: Text(
                    _transferring
                        ? 'Transferring…'
                        : _uploading
                            ? 'Uploading...'
                            : 'Upload RLDS',
                  ),
                ),
              ],
            ),
            const SizedBox(height: 12),
            SwitchListTile(
              value: _includeDeviceMic,
              onChanged: (value) => setState(() {
                _includeDeviceMic = value;
                _sensorStates['device_mic'] = SensorStatus.idle;
              }),
              title: const Text('Include device mic audio (phone/TV)'),
              subtitle: Text(
                _sensorStatusLabel('device_mic', _includeDeviceMic),
              ),
            ),
            SwitchListTile(
              value: _includeRobotMic,
              onChanged: (value) => setState(() {
                _includeRobotMic = value;
                _sensorStates['robot_mic'] = SensorStatus.idle;
              }),
              title: const Text('Include robot mic audio'),
              subtitle: Text(
                _sensorStatusLabel('robot_mic', _includeRobotMic),
              ),
            ),
            SwitchListTile(
              value: _includeEgocentricVideo,
              onChanged: (value) => setState(() {
                _includeEgocentricVideo = value;
                _sensorStates['egocentric'] = SensorStatus.idle;
              }),
              title: const Text('Include egocentric video + depth + gaze'),
              subtitle: Text(
                '${_sensorStatusLabel('egocentric', _includeEgocentricVideo)} (${_egocentricMount.toUpperCase()} mount)',
              ),
            ),
            DropdownButtonFormField<String>(
              value: _egocentricMount,
              decoration: const InputDecoration(
                labelText: 'Mount',
                helperText: 'Select chest or hat mount for egocentric camera',
              ),
              items: const [
                DropdownMenuItem(value: 'chest', child: Text('Chest mount camera')),
                DropdownMenuItem(value: 'hat', child: Text('Hat/helmet mount camera')),
              ],
              onChanged: _includeEgocentricVideo
                  ? (value) {
                      if (value == null) return;
                      setState(() => _egocentricMount = value);
                    }
                  : null,
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
            const Text('• Recorded blobs are handed off to the managed robot when connected and mirrored into the RLDS manifest.'),
          ],
        ),
      ),
    );
  }
}
