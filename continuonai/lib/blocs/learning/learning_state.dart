import 'package:equatable/equatable.dart';

import '../../models/learning_metrics.dart';

enum LearningStateStatus {
  initial,
  loading,
  success,
  failure,
}

class LearningState extends Equatable {
  final LearningStateStatus status;
  final bool isLearning;
  final bool isPaused;
  final LearningMetrics metrics;
  final String? error;
  final DateTime? lastUpdated;

  const LearningState({
    this.status = LearningStateStatus.initial,
    this.isLearning = false,
    this.isPaused = false,
    this.metrics = const LearningMetrics(),
    this.error,
    this.lastUpdated,
  });

  LearningState copyWith({
    LearningStateStatus? status,
    bool? isLearning,
    bool? isPaused,
    LearningMetrics? metrics,
    String? error,
    bool clearError = false,
    DateTime? lastUpdated,
  }) {
    return LearningState(
      status: status ?? this.status,
      isLearning: isLearning ?? this.isLearning,
      isPaused: isPaused ?? this.isPaused,
      metrics: metrics ?? this.metrics,
      error: clearError ? null : (error ?? this.error),
      lastUpdated: lastUpdated ?? this.lastUpdated,
    );
  }

  /// Convenience getters from metrics
  double get curiosity => metrics.curiosity;
  double get surprise => metrics.surprise;
  int get totalEpisodes => metrics.progress.totalEpisodes;
  int get totalSteps => metrics.progress.totalSteps;
  double get learningRate => metrics.progress.learningRate;
  List<MetricPoint> get lossHistory => metrics.lossHistory;

  @override
  List<Object?> get props => [
        status,
        isLearning,
        isPaused,
        metrics,
        error,
        lastUpdated,
      ];
}
