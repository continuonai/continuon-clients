import 'package:equatable/equatable.dart';

/// Single data point for metrics charting
class MetricPoint extends Equatable {
  final DateTime timestamp;
  final double value;
  final String? label;

  const MetricPoint({
    required this.timestamp,
    required this.value,
    this.label,
  });

  factory MetricPoint.fromJson(Map<String, dynamic> json) {
    return MetricPoint(
      timestamp: json['timestamp'] != null
          ? DateTime.tryParse(json['timestamp'].toString()) ?? DateTime.now()
          : DateTime.fromMillisecondsSinceEpoch(
              (json['epoch_ms'] as num?)?.toInt() ?? 0),
      value: (json['value'] as num?)?.toDouble() ?? 0.0,
      label: json['label'] as String?,
    );
  }

  @override
  List<Object?> get props => [timestamp, value, label];
}

/// Learning progress metrics from the slow loop
class LearningProgress extends Equatable {
  final int totalSteps;
  final int totalEpisodes;
  final int learningUpdates;
  final double avgParameterChange;
  final double learningRate;
  final bool isStable;

  const LearningProgress({
    this.totalSteps = 0,
    this.totalEpisodes = 0,
    this.learningUpdates = 0,
    this.avgParameterChange = 0.0,
    this.learningRate = 0.0,
    this.isStable = true,
  });

  factory LearningProgress.fromJson(Map<String, dynamic> json) {
    return LearningProgress(
      totalSteps: (json['total_steps'] as num?)?.toInt() ?? 0,
      totalEpisodes: (json['total_episodes'] as num?)?.toInt() ?? 0,
      learningUpdates: (json['learning_updates'] as num?)?.toInt() ?? 0,
      avgParameterChange:
          (json['avg_parameter_change'] as num?)?.toDouble() ?? 0.0,
      learningRate: (json['learning_rate'] as num?)?.toDouble() ?? 0.0,
      isStable: json['is_stable'] as bool? ?? true,
    );
  }

  LearningProgress copyWith({
    int? totalSteps,
    int? totalEpisodes,
    int? learningUpdates,
    double? avgParameterChange,
    double? learningRate,
    bool? isStable,
  }) {
    return LearningProgress(
      totalSteps: totalSteps ?? this.totalSteps,
      totalEpisodes: totalEpisodes ?? this.totalEpisodes,
      learningUpdates: learningUpdates ?? this.learningUpdates,
      avgParameterChange: avgParameterChange ?? this.avgParameterChange,
      learningRate: learningRate ?? this.learningRate,
      isStable: isStable ?? this.isStable,
    );
  }

  @override
  List<Object?> get props => [
        totalSteps,
        totalEpisodes,
        learningUpdates,
        avgParameterChange,
        learningRate,
        isStable,
      ];
}

/// Current status of the slow loop learning system
class SlowLoopStatus extends Equatable {
  final bool enabled;
  final bool running;
  final bool isPaused;
  final String? currentPhase;
  final DateTime? lastUpdateTime;

  const SlowLoopStatus({
    this.enabled = false,
    this.running = false,
    this.isPaused = false,
    this.currentPhase,
    this.lastUpdateTime,
  });

  factory SlowLoopStatus.fromJson(Map<String, dynamic> json) {
    return SlowLoopStatus(
      enabled: json['enabled'] as bool? ?? false,
      running: json['running'] as bool? ?? false,
      isPaused: json['is_paused'] as bool? ?? json['paused'] as bool? ?? false,
      currentPhase: json['current_phase'] as String? ?? json['phase'] as String?,
      lastUpdateTime: json['last_update_time'] != null
          ? DateTime.tryParse(json['last_update_time'].toString())
          : null,
    );
  }

  @override
  List<Object?> get props =>
      [enabled, running, isPaused, currentPhase, lastUpdateTime];
}

/// Comprehensive learning metrics including curiosity and surprise
class LearningMetrics extends Equatable {
  final SlowLoopStatus status;
  final LearningProgress progress;
  final double curiosity;
  final double surprise;
  final List<MetricPoint> lossHistory;
  final List<MetricPoint> curiosityHistory;
  final Map<String, dynamic>? wavecoreMetrics;

  const LearningMetrics({
    this.status = const SlowLoopStatus(),
    this.progress = const LearningProgress(),
    this.curiosity = 0.0,
    this.surprise = 0.0,
    this.lossHistory = const [],
    this.curiosityHistory = const [],
    this.wavecoreMetrics,
  });

  factory LearningMetrics.fromJson(Map<String, dynamic> json) {
    final statusJson = json['status'] as Map<String, dynamic>? ?? {};
    final progressJson = json['progress'] as Map<String, dynamic>? ?? {};

    final lossHistoryRaw = json['loss_history'] as List<dynamic>? ?? [];
    final curiosityHistoryRaw =
        json['curiosity_history'] as List<dynamic>? ?? [];

    return LearningMetrics(
      status: SlowLoopStatus.fromJson(statusJson),
      progress: LearningProgress.fromJson(progressJson),
      curiosity: (json['curiosity'] as num?)?.toDouble() ?? 0.0,
      surprise: (json['surprise'] as num?)?.toDouble() ?? 0.0,
      lossHistory: lossHistoryRaw
          .map((e) => MetricPoint.fromJson(e as Map<String, dynamic>))
          .toList(),
      curiosityHistory: curiosityHistoryRaw
          .map((e) => MetricPoint.fromJson(e as Map<String, dynamic>))
          .toList(),
      wavecoreMetrics: json['wavecore'] as Map<String, dynamic>?,
    );
  }

  LearningMetrics copyWith({
    SlowLoopStatus? status,
    LearningProgress? progress,
    double? curiosity,
    double? surprise,
    List<MetricPoint>? lossHistory,
    List<MetricPoint>? curiosityHistory,
    Map<String, dynamic>? wavecoreMetrics,
  }) {
    return LearningMetrics(
      status: status ?? this.status,
      progress: progress ?? this.progress,
      curiosity: curiosity ?? this.curiosity,
      surprise: surprise ?? this.surprise,
      lossHistory: lossHistory ?? this.lossHistory,
      curiosityHistory: curiosityHistory ?? this.curiosityHistory,
      wavecoreMetrics: wavecoreMetrics ?? this.wavecoreMetrics,
    );
  }

  @override
  List<Object?> get props => [
        status,
        progress,
        curiosity,
        surprise,
        lossHistory,
        curiosityHistory,
        wavecoreMetrics,
      ];
}

/// Training benchmark result
class TrainingBenchmark extends Equatable {
  final String id;
  final DateTime timestamp;
  final double score;
  final String? modelVersion;
  final Map<String, dynamic>? details;

  const TrainingBenchmark({
    required this.id,
    required this.timestamp,
    required this.score,
    this.modelVersion,
    this.details,
  });

  factory TrainingBenchmark.fromJson(Map<String, dynamic> json) {
    return TrainingBenchmark(
      id: json['id'] as String? ?? '',
      timestamp: json['timestamp'] != null
          ? DateTime.tryParse(json['timestamp'].toString()) ?? DateTime.now()
          : DateTime.now(),
      score: (json['score'] as num?)?.toDouble() ?? 0.0,
      modelVersion: json['model_version'] as String?,
      details: json['details'] as Map<String, dynamic>?,
    );
  }

  @override
  List<Object?> get props => [id, timestamp, score, modelVersion, details];
}
