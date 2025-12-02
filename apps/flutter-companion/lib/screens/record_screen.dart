import 'package:flutter/material.dart';

import '../models/rlds_models.dart';
import '../services/brain_client.dart';
import '../services/cloud_uploader.dart';
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
  final _notesController = TextEditingController();
  final _cloudUploader = CloudUploader();
  bool _uploading = false;
  String? _status;
  late String _controlRole;

  @override
  void initState() {
    super.initState();
    _controlRole = widget.initialControlRole;
    _recorder = _buildRecorder();
  }

  TaskRecorder _buildRecorder() {
    return TaskRecorder(
      metadata: EpisodeMetadata(
        xrMode: 'trainer',
        controlRole: _controlRole,
        environmentId: 'lab-mock',
      ),
    );
  }

  void _recordStep() {
    _recorder.addStep(
      EpisodeStep(
        observation: {
          'ui_context': {'notes': _notesController.text},
        },
        action: {
          'command': [],
          'source': 'flutter-companion',
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
            const SizedBox(height: 24),
            Text(_status ?? 'Pending'),
            const SizedBox(height: 12),
            const Text('Recording flows'),
            const Text('• Uses RLDS schema aligned with proto/rlds_episode.proto.'),
            const Text('• Upload broker expected to return signed storage URL.'),
          ],
        ),
      ),
    );
  }
}
