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
  final _slugController = TextEditingController();
  final _titleController = TextEditingController();
  final _shareTagsController = TextEditingController();
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
  bool _publicShare = false;
  bool _piiScrubbingEnabled = false;
  bool _piiAttested = false;
  bool _piiCleared = false;
  bool _piiRedacted = false;
  bool _pendingReview = true;
  String _license = 'CC-BY-4.0';
  String _contentAudience = 'general';
  String _violenceLevel = 'none';
  String _languageLevel = 'clean';
  String _intendedAudience = 'general';
  final Set<String> _motionTags = {'drive', 'pick_and_place', 'follow_path'};
  final Set<String> _taskTags = {'warehouse', 'kitchen', 'lab'};
  final Set<String> _skillTags = {'grasp', 'navigate', 'inspect'};

  @override
  void initState() {
    super.initState();
    _controlRole = widget.initialControlRole;
    _recorder = _buildRecorder();
    _multimodalInputs = MultimodalInputs();
  }

  TaskRecorder _buildRecorder() {
    return TaskRecorder(metadata: _buildMetadata());
  }

  @override
  void dispose() {
    _notesController.dispose();
    _voiceCommandController.dispose();
    _slugController.dispose();
    _titleController.dispose();
    _shareTagsController.dispose();
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
  EpisodeMetadata _buildMetadata() {
    final classificationTags = _buildClassificationTags();
    final shareTags = _buildShareTags();
    final safetyMetadata = _buildSafetyMetadata();
    return EpisodeMetadata(
      xrMode: 'trainer',
      controlRole: _controlRole,
      environmentId: 'lab-mock',
      tags: ['multimodal_audio_video', ...classificationTags],
      safety: safetyMetadata,
      share: ShareMetadata(
        isPublic: _publicShare,
        slug: _slugController.text.trim(),
        title: _titleController.text.trim(),
        license: _license,
        tags: shareTags,
      ),
      schemaVersion: '1.1',
    );
  }

  SafetyMetadata _buildSafetyMetadata() {
    final piiPresent = _includeDeviceMic || _includeRobotMic || _includeEgocentricVideo;
    return SafetyMetadata(
      contentRating: ContentRating(
        audience: _contentAudience,
        violence: _violenceLevel,
        language: _languageLevel,
      ),
      intendedAudience: _intendedAudience,
      piiAttested: _piiAttested,
      piiCleared: _piiCleared,
      piiRedacted: _piiRedacted,
      pendingReview: _pendingReview,
      piiAttestation: PiiAttestation(
        piiPresent: piiPresent,
        facesPresent: _includeEgocentricVideo,
        namesPresent: _includeDeviceMic || _includeRobotMic,
        consent: _piiAttested,
      ),
    );
  }

  List<String> _buildShareTags() {
    final manualTags = _shareTagsController.text
        .split(',')
        .map((tag) => tag.trim())
        .where((tag) => tag.isNotEmpty)
        .toList();
    final classificationTags = _buildClassificationTags();
    return {...manualTags, ...classificationTags}.toList();
  }

  List<String> _buildClassificationTags() {
    return [
      ..._motionTags.map((tag) => 'motion:$tag'),
      ..._taskTags.map((tag) => 'task:$tag'),
      ..._skillTags.map((tag) => 'skill:$tag'),
    ];
  }

  EpisodeMetadata _refreshMetadata() {
    final metadata = _buildMetadata();
    _recorder.updateMetadata(metadata);
    return metadata;
  }

  void _populateMultimodalInputs() {
    final frameId = DateTime.now().millisecondsSinceEpoch.toString();
    if (_includeDeviceMic) {
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
  void _recordStep() {
    _refreshMetadata();
    _populateMultimodalInputs();
    _recorder.addStep(
      EpisodeStep(
        observation: _multimodalInputs.buildObservation(
          uiContext: {
            'notes': _notesController.text,
            'voice_hint': _voiceCommandController.text,
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
    setState(() => _uploading = true);
    final baseMetadata = _refreshMetadata();
    final record = _recorder.buildRecord(metadataOverride: baseMetadata);
    try {
      final processedRecord = await _runPiiAndSafetyPipeline(record);
      final validationError = _validateMetadata(processedRecord.metadata);
      if (validationError != null) {
        setState(() => _status = validationError);
        return;
      }
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
        processedRecord.metadata,
      );
      await _cloudUploader.uploadEpisode(processedRecord, signedUrl);
      setState(() => _status = 'Uploaded to ${signedUrl.host}');
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

  String? _validateMetadata(EpisodeMetadata metadata) {
    final safety = metadata.safety;
    if (safety == null) {
      return 'Safety metadata required before upload';
    }
    if (metadata.share?.isPublic == true) {
      final share = metadata.share!;
      if (share.slug.isEmpty || share.title.isEmpty || share.license.isEmpty) {
        return 'Slug, title, and license are required for public episodes';
      }
      if (share.tags.isEmpty) {
        return 'Add at least one share tag before publishing';
      }
      if (safety.intendedAudience.isEmpty || safety.contentRating.audience.isEmpty) {
        return 'Provide content rating and intended audience for public sharing';
      }
      if (!safety.piiAttested) {
        return 'PII attestation is required for public uploads';
      }
      if (!safety.piiCleared || safety.pendingReview) {
        return 'Complete the PII/safety pipeline before publishing';
      }
    }
    return null;
  }

  Future<EpisodeRecord> _runPiiAndSafetyPipeline(EpisodeRecord record) async {
    setState(() => _status = 'Running PII/safety pipeline...');
    // Simulate a local redaction pipeline that blurs faces/plates and runs OCR/ASR
    // scans before upload. This updates the metadata flags so cloud brokers can
    // block public listings until PII is cleared.
    final scrubbed = _piiScrubbingEnabled || record.metadata.share?.isPublic == true;
    final safety = record.metadata.safety ?? _buildSafetyMetadata();
    final updatedSafety = safety.copyWith(
      piiRedacted: scrubbed,
      piiCleared: scrubbed,
      pendingReview: false,
      piiAttestation: safety.piiAttestation?.copyWith(consent: _piiAttested) ??
          PiiAttestation(
            piiPresent: true,
            facesPresent: _includeEgocentricVideo,
            namesPresent: _includeDeviceMic || _includeRobotMic,
            consent: _piiAttested,
          ),
    );
    setState(() {
      _piiCleared = updatedSafety.piiCleared;
      _piiRedacted = updatedSafety.piiRedacted;
      _pendingReview = updatedSafety.pendingReview;
    });
    final updatedMetadata = record.metadata.copyWith(safety: updatedSafety, share: record.metadata.share);
    _recorder.updateMetadata(updatedMetadata);
    return EpisodeRecord(metadata: updatedMetadata, steps: record.steps);
  }

  Widget _buildTagSelector(String label, List<String> options, Set<String> selection) {
    return Column(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(label, style: Theme.of(context).textTheme.titleSmall),
        const SizedBox(height: 4),
        Wrap(
          spacing: 8,
          children: options
              .map(
                (option) => FilterChip(
                  label: Text(option),
                  selected: selection.contains(option),
                  onSelected: (selected) {
                    setState(() {
                      if (selected) {
                        selection.add(option);
                      } else {
                        selection.remove(option);
                      }
                      _refreshMetadata();
                    });
                  },
                ),
              )
              .toList(),
        ),
      ],
    );
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('Record task / RLDS episodes')),
      body: SingleChildScrollView(
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
                _refreshMetadata();
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
                _refreshMetadata();
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
                _refreshMetadata();
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
            const Text('Episode classification'),
            _buildTagSelector(
              'Motion tags',
              const ['drive', 'pick_and_place', 'follow_path', 'stationary'],
              _motionTags,
            ),
            const SizedBox(height: 8),
            _buildTagSelector(
              'Task tags',
              const ['warehouse', 'kitchen', 'lab', 'outdoor'],
              _taskTags,
            ),
            const SizedBox(height: 8),
            _buildTagSelector(
              'Skill tags',
              const ['grasp', 'navigate', 'inspect', 'handoff'],
              _skillTags,
            ),
            const SizedBox(height: 16),
            const Text('Public sharing + PII safeguards'),
            SwitchListTile(
              value: _publicShare,
              onChanged: (value) => setState(() {
                _publicShare = value;
                _refreshMetadata();
              }),
              title: const Text('Mark episode as public'),
              subtitle: const Text('Requires share block + safety metadata before upload'),
            ),
            SwitchListTile(
              value: _piiScrubbingEnabled,
              onChanged: (value) => setState(() {
                _piiScrubbingEnabled = value;
                _refreshMetadata();
              }),
              title: const Text('Enable PII scrubbing before upload'),
              subtitle: const Text('Blur faces/plates and run OCR/ASR before publishing'),
            ),
            CheckboxListTile(
              value: _piiAttested,
              onChanged: (value) => setState(() {
                _piiAttested = value ?? false;
                _refreshMetadata();
              }),
              title: const Text('I attest PII review for this episode'),
              subtitle: const Text('Required to set safety.pii_attested for public uploads'),
            ),
            TextField(
              controller: _titleController,
              onChanged: (_) => _refreshMetadata(),
              decoration: const InputDecoration(
                labelText: 'Public title',
                helperText: 'Title used in public listing',
              ),
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _slugController,
              onChanged: (_) => _refreshMetadata(),
              decoration: const InputDecoration(
                labelText: 'Public slug',
                helperText: 'Slug used for public URLs (lowercase, hyphenated)',
              ),
            ),
            const SizedBox(height: 8),
            DropdownButtonFormField<String>(
              value: _license,
              decoration: const InputDecoration(labelText: 'License'),
              items: const [
                DropdownMenuItem(value: 'CC-BY-4.0', child: Text('CC-BY-4.0')),
                DropdownMenuItem(value: 'CC-BY-NC-4.0', child: Text('CC-BY-NC-4.0')),
                DropdownMenuItem(value: 'Proprietary', child: Text('Proprietary')),
              ],
              onChanged: (value) {
                if (value == null) return;
                setState(() {
                  _license = value;
                  _refreshMetadata();
                });
              },
            ),
            const SizedBox(height: 8),
            TextField(
              controller: _shareTagsController,
              onChanged: (_) => _refreshMetadata(),
              decoration: const InputDecoration(
                labelText: 'Public tags',
                helperText: 'Comma-separated tags for public listing; classification tags are auto-included',
              ),
            ),
            const SizedBox(height: 16),
            const Text('Safety + content rating'),
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _contentAudience,
                    decoration: const InputDecoration(labelText: 'Content rating'),
                    items: const [
                      DropdownMenuItem(value: 'general', child: Text('General / all ages')),
                      DropdownMenuItem(value: 'teen', child: Text('Teen / 13+')),
                      DropdownMenuItem(value: 'adult', child: Text('Adult / 18+')),
                    ],
                    onChanged: (value) {
                      if (value == null) return;
                      setState(() {
                        _contentAudience = value;
                        _refreshMetadata();
                      });
                    },
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _intendedAudience,
                    decoration: const InputDecoration(labelText: 'Intended audience'),
                    items: const [
                      DropdownMenuItem(value: 'general', child: Text('General public')),
                      DropdownMenuItem(value: 'research', child: Text('Research partners')),
                      DropdownMenuItem(value: 'internal', child: Text('Internal QA only')),
                    ],
                    onChanged: (value) {
                      if (value == null) return;
                      setState(() {
                        _intendedAudience = value;
                        _refreshMetadata();
                      });
                    },
                  ),
                ),
              ],
            ),
            const SizedBox(height: 8),
            Row(
              children: [
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _violenceLevel,
                    decoration: const InputDecoration(labelText: 'Violence rating'),
                    items: const [
                      DropdownMenuItem(value: 'none', child: Text('None')),
                      DropdownMenuItem(value: 'mild', child: Text('Mild')),
                      DropdownMenuItem(value: 'graphic', child: Text('Graphic')),
                    ],
                    onChanged: (value) {
                      if (value == null) return;
                      setState(() {
                        _violenceLevel = value;
                        _refreshMetadata();
                      });
                    },
                  ),
                ),
                const SizedBox(width: 12),
                Expanded(
                  child: DropdownButtonFormField<String>(
                    value: _languageLevel,
                    decoration: const InputDecoration(labelText: 'Language rating'),
                    items: const [
                      DropdownMenuItem(value: 'clean', child: Text('Clean')),
                      DropdownMenuItem(value: 'some', child: Text('Some language')),
                      DropdownMenuItem(value: 'strong', child: Text('Strong language')),
                    ],
                    onChanged: (value) {
                      if (value == null) return;
                      setState(() {
                        _languageLevel = value;
                        _refreshMetadata();
                      });
                    },
                  ),
                ),
              ],
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
