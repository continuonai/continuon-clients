import 'dart:async';

import 'package:flutter_bloc/flutter_bloc.dart';

import '../../models/ota_update.dart';
import '../../services/brain_client.dart';
import 'ota_event.dart';
import 'ota_state.dart';

class OTABloc extends Bloc<OTAEvent, OTAState> {
  final BrainClient brainClient;
  Timer? _pollingTimer;

  OTABloc({required this.brainClient}) : super(const OTAState()) {
    on<CheckForUpdates>(_onCheckForUpdates);
    on<FetchOTAStatus>(_onFetchStatus);
    on<DownloadUpdate>(_onDownloadUpdate);
    on<ActivateUpdate>(_onActivateUpdate);
    on<RollbackUpdate>(_onRollbackUpdate);
    on<ClearOTAError>(_onClearError);
    on<StartOTAPolling>(_onStartPolling);
    on<StopOTAPolling>(_onStopPolling);
  }

  Future<void> _onCheckForUpdates(
    CheckForUpdates event,
    Emitter<OTAState> emit,
  ) async {
    emit(state.copyWith(status: OTAStateStatus.checking, clearError: true));

    try {
      final data = await brainClient.checkForUpdates();
      final result = UpdateCheckResult.fromJson(data);

      emit(state.copyWith(
        status: OTAStateStatus.success,
        updateAvailable: result.updateAvailable,
        availableUpdate: result.availableUpdate,
        lastChecked: DateTime.now(),
      ));
    } catch (e) {
      emit(state.copyWith(
        status: OTAStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  Future<void> _onFetchStatus(
    FetchOTAStatus event,
    Emitter<OTAState> emit,
  ) async {
    emit(state.copyWith(status: OTAStateStatus.loading, clearError: true));

    try {
      final data = await brainClient.getUpdateStatus();
      final updateStatus = UpdateStatus.fromJson(data);

      emit(state.copyWith(
        status: OTAStateStatus.success,
        updateStatus: updateStatus,
        downloadProgress: updateStatus.progressPercent,
      ));
    } catch (e) {
      emit(state.copyWith(
        status: OTAStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  Future<void> _onDownloadUpdate(
    DownloadUpdate event,
    Emitter<OTAState> emit,
  ) async {
    emit(state.copyWith(
      status: OTAStateStatus.downloading,
      downloadProgress: 0.0,
      clearError: true,
    ));

    try {
      final data = await brainClient.downloadUpdate(
        modelId: event.modelId,
        version: event.version,
      );

      if (data['success'] == true) {
        // Start polling for progress
        add(const StartOTAPolling());
        emit(state.copyWith(
          status: OTAStateStatus.success,
        ));
      } else {
        emit(state.copyWith(
          status: OTAStateStatus.failure,
          error: data['error'] as String? ?? 'Download failed',
        ));
      }
    } catch (e) {
      emit(state.copyWith(
        status: OTAStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  Future<void> _onActivateUpdate(
    ActivateUpdate event,
    Emitter<OTAState> emit,
  ) async {
    emit(state.copyWith(status: OTAStateStatus.activating, clearError: true));

    try {
      final data = await brainClient.activateUpdate(
        runHealthCheck: event.runHealthCheck,
        force: event.force,
      );

      if (data['success'] == true) {
        emit(state.copyWith(
          status: OTAStateStatus.success,
          updateAvailable: false,
          clearAvailableUpdate: true,
        ));
        // Refresh status after activation
        add(const FetchOTAStatus());
      } else {
        final rolledBack = data['rolled_back'] as bool? ?? false;
        emit(state.copyWith(
          status: OTAStateStatus.failure,
          error: rolledBack
              ? 'Activation failed, rolled back to previous version'
              : (data['error'] as String? ?? 'Activation failed'),
        ));
      }
    } catch (e) {
      emit(state.copyWith(
        status: OTAStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  Future<void> _onRollbackUpdate(
    RollbackUpdate event,
    Emitter<OTAState> emit,
  ) async {
    emit(state.copyWith(status: OTAStateStatus.rollingBack, clearError: true));

    try {
      final data = await brainClient.rollbackUpdate();

      if (data['success'] == true) {
        emit(state.copyWith(status: OTAStateStatus.success));
        // Refresh status after rollback
        add(const FetchOTAStatus());
      } else {
        emit(state.copyWith(
          status: OTAStateStatus.failure,
          error: data['error'] as String? ?? 'Rollback failed',
        ));
      }
    } catch (e) {
      emit(state.copyWith(
        status: OTAStateStatus.failure,
        error: e.toString(),
      ));
    }
  }

  void _onClearError(
    ClearOTAError event,
    Emitter<OTAState> emit,
  ) {
    emit(state.copyWith(clearError: true));
  }

  void _onStartPolling(
    StartOTAPolling event,
    Emitter<OTAState> emit,
  ) {
    _pollingTimer?.cancel();
    _pollingTimer = Timer.periodic(event.interval, (_) {
      add(const FetchOTAStatus());
    });
  }

  void _onStopPolling(
    StopOTAPolling event,
    Emitter<OTAState> emit,
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
