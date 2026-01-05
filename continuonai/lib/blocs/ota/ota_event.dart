import 'package:equatable/equatable.dart';

abstract class OTAEvent extends Equatable {
  const OTAEvent();

  @override
  List<Object?> get props => [];
}

/// Check for available updates
class CheckForUpdates extends OTAEvent {
  const CheckForUpdates();
}

/// Fetch current OTA status
class FetchOTAStatus extends OTAEvent {
  const FetchOTAStatus();
}

/// Download an available update
class DownloadUpdate extends OTAEvent {
  final String? modelId;
  final String? version;

  const DownloadUpdate({this.modelId, this.version});

  @override
  List<Object?> get props => [modelId, version];
}

/// Activate the downloaded update
class ActivateUpdate extends OTAEvent {
  final bool runHealthCheck;
  final bool force;

  const ActivateUpdate({
    this.runHealthCheck = true,
    this.force = false,
  });

  @override
  List<Object?> get props => [runHealthCheck, force];
}

/// Rollback to previous version
class RollbackUpdate extends OTAEvent {
  const RollbackUpdate();
}

/// Clear any error state
class ClearOTAError extends OTAEvent {
  const ClearOTAError();
}

/// Start polling for status updates during operations
class StartOTAPolling extends OTAEvent {
  final Duration interval;

  const StartOTAPolling({this.interval = const Duration(seconds: 2)});

  @override
  List<Object?> get props => [interval];
}

/// Stop polling
class StopOTAPolling extends OTAEvent {
  const StopOTAPolling();
}
