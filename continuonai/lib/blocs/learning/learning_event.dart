import 'package:equatable/equatable.dart';

abstract class LearningEvent extends Equatable {
  const LearningEvent();

  @override
  List<Object?> get props => [];
}

/// Fetch current learning status
class FetchLearningStatus extends LearningEvent {
  const FetchLearningStatus();
}

/// Fetch detailed learning metrics
class FetchLearningMetrics extends LearningEvent {
  final int? limit;

  const FetchLearningMetrics({this.limit});

  @override
  List<Object?> get props => [limit];
}

/// Pause learning
class PauseLearning extends LearningEvent {
  const PauseLearning();
}

/// Resume learning
class ResumeLearning extends LearningEvent {
  const ResumeLearning();
}

/// Reset learning
class ResetLearning extends LearningEvent {
  const ResetLearning();
}

/// Start polling for metrics updates
class StartMetricsPolling extends LearningEvent {
  final Duration interval;

  const StartMetricsPolling({this.interval = const Duration(seconds: 5)});

  @override
  List<Object?> get props => [interval];
}

/// Stop polling
class StopMetricsPolling extends LearningEvent {
  const StopMetricsPolling();
}
