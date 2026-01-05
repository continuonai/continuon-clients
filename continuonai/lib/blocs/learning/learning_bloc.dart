import 'dart:async';

import 'package:flutter_bloc/flutter_bloc.dart';

import '../../models/learning_metrics.dart';
import '../../services/brain_client.dart';
import 'learning_event.dart';
import 'learning_state.dart';

class LearningBloc extends Bloc<LearningEvent, LearningState> {
  final BrainClient brainClient;
  Timer? _pollingTimer;

  LearningBloc({required this.brainClient}) : super(const LearningState()) {
    on<FetchLearningStatus>(_onFetchStatus);
    on<FetchLearningMetrics>(_onFetchMetrics);
    on<PauseLearning>(_onPauseLearning);
    on<ResumeLearning>(_onResumeLearning);
    on<ResetLearning>(_onResetLearning);
    on<StartMetricsPolling>(_onStartPolling);
    on<StopMetricsPolling>(_onStopPolling);
  }

  Future<void> _onFetchStatus(
    FetchLearningStatus event,
    Emitter<LearningState> emit,
  ) async {
    emit(state.copyWith(status: LearningStateStatus.loading, clearError: true));

    try {
      final statusData = await brainClient.getLearningStatus();
      final progressData = await brainClient.getLearningProgress();

      final combinedData = <String, dynamic>{
        'status': statusData,
        'progress': progressData,
      };

      final metrics = LearningMetrics.fromJson(combinedData);

      emit(state.copyWith(
        status: LearningStateStatus.success,
        isLearning: metrics.status.running,
        isPaused: metrics.status.isPaused,
        metrics: metrics,
        lastUpdated: DateTime.now(),
      ));
    } catch (e) {
      emit(state.copyWith(
        status: LearningStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  Future<void> _onFetchMetrics(
    FetchLearningMetrics event,
    Emitter<LearningState> emit,
  ) async {
    emit(state.copyWith(status: LearningStateStatus.loading, clearError: true));

    try {
      final metricsData = await brainClient.getLearningMetrics(limit: event.limit);
      final statusData = await brainClient.getLearningStatus();
      final progressData = await brainClient.getLearningProgress();

      // Merge all data
      final combinedData = <String, dynamic>{
        ...metricsData,
        'status': statusData,
        'progress': progressData,
      };

      final metrics = LearningMetrics.fromJson(combinedData);

      emit(state.copyWith(
        status: LearningStateStatus.success,
        isLearning: metrics.status.running,
        isPaused: metrics.status.isPaused,
        metrics: metrics,
        lastUpdated: DateTime.now(),
      ));
    } catch (e) {
      emit(state.copyWith(
        status: LearningStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  Future<void> _onPauseLearning(
    PauseLearning event,
    Emitter<LearningState> emit,
  ) async {
    emit(state.copyWith(status: LearningStateStatus.loading, clearError: true));

    try {
      final success = await brainClient.pauseLearning();
      if (success) {
        emit(state.copyWith(
          status: LearningStateStatus.success,
          isPaused: true,
          isLearning: false,
        ));
      } else {
        emit(state.copyWith(
          status: LearningStateStatus.failure,
          error: 'Failed to pause learning',
        ));
      }
    } catch (e) {
      emit(state.copyWith(
        status: LearningStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  Future<void> _onResumeLearning(
    ResumeLearning event,
    Emitter<LearningState> emit,
  ) async {
    emit(state.copyWith(status: LearningStateStatus.loading, clearError: true));

    try {
      final success = await brainClient.resumeLearning();
      if (success) {
        emit(state.copyWith(
          status: LearningStateStatus.success,
          isPaused: false,
          isLearning: true,
        ));
      } else {
        emit(state.copyWith(
          status: LearningStateStatus.failure,
          error: 'Failed to resume learning',
        ));
      }
    } catch (e) {
      emit(state.copyWith(
        status: LearningStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  Future<void> _onResetLearning(
    ResetLearning event,
    Emitter<LearningState> emit,
  ) async {
    emit(state.copyWith(status: LearningStateStatus.loading, clearError: true));

    try {
      final success = await brainClient.resetLearning();
      if (success) {
        // Refresh status after reset
        add(const FetchLearningStatus());
      } else {
        emit(state.copyWith(
          status: LearningStateStatus.failure,
          error: 'Failed to reset learning',
        ));
      }
    } catch (e) {
      emit(state.copyWith(
        status: LearningStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  void _onStartPolling(
    StartMetricsPolling event,
    Emitter<LearningState> emit,
  ) {
    _pollingTimer?.cancel();
    _pollingTimer = Timer.periodic(event.interval, (_) {
      add(const FetchLearningMetrics());
    });
    // Fetch immediately
    add(const FetchLearningMetrics());
  }

  void _onStopPolling(
    StopMetricsPolling event,
    Emitter<LearningState> emit,
  ) {
    _pollingTimer?.cancel();
    _pollingTimer = null;
  }

  @override
  Future<void> close() {
    _pollingTimer?.cancel();
    return super.close();
  }
}
