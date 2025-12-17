import '../models/rlds_models.dart';

class TaskRecorder {
  TaskRecorder({required this.metadata});

  EpisodeMetadata metadata;
  final List<EpisodeStep> _steps = [];

  void addStep(EpisodeStep step) {
    _steps.add(step);
  }

  EpisodeRecord buildRecord({EpisodeMetadata? metadataOverride}) =>
      EpisodeRecord(metadata: metadataOverride ?? metadata, steps: List.of(_steps));

  void updateMetadata(EpisodeMetadata value) {
    metadata = value;
  }

  void reset() => _steps.clear();
}
