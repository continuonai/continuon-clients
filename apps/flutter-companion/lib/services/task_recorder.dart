import '../models/rlds_models.dart';

class TaskRecorder {
  TaskRecorder({required this.metadata});

  final EpisodeMetadata metadata;
  final List<EpisodeStep> _steps = [];

  void addStep(EpisodeStep step) {
    _steps.add(step);
  }

  EpisodeRecord buildRecord() => EpisodeRecord(metadata: metadata, steps: List.of(_steps));

  void reset() => _steps.clear();
}
