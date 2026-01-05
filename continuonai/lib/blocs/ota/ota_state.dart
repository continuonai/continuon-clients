import 'package:equatable/equatable.dart';

import '../../models/ota_update.dart';

enum OTAStateStatus {
  initial,
  loading,
  checking,
  downloading,
  activating,
  rollingBack,
  success,
  failure,
}

class OTAState extends Equatable {
  final OTAStateStatus status;
  final bool updateAvailable;
  final OTAUpdate? availableUpdate;
  final UpdateStatus updateStatus;
  final double downloadProgress;
  final String? error;
  final DateTime? lastChecked;

  const OTAState({
    this.status = OTAStateStatus.initial,
    this.updateAvailable = false,
    this.availableUpdate,
    this.updateStatus = const UpdateStatus(),
    this.downloadProgress = 0.0,
    this.error,
    this.lastChecked,
  });

  OTAState copyWith({
    OTAStateStatus? status,
    bool? updateAvailable,
    OTAUpdate? availableUpdate,
    bool clearAvailableUpdate = false,
    UpdateStatus? updateStatus,
    double? downloadProgress,
    String? error,
    bool clearError = false,
    DateTime? lastChecked,
  }) {
    return OTAState(
      status: status ?? this.status,
      updateAvailable: updateAvailable ?? this.updateAvailable,
      availableUpdate:
          clearAvailableUpdate ? null : (availableUpdate ?? this.availableUpdate),
      updateStatus: updateStatus ?? this.updateStatus,
      downloadProgress: downloadProgress ?? this.downloadProgress,
      error: clearError ? null : (error ?? this.error),
      lastChecked: lastChecked ?? this.lastChecked,
    );
  }

  /// Convenience getters
  String? get currentVersion => updateStatus.currentVersion;
  bool get rollbackAvailable => updateStatus.rollbackAvailable;
  bool get isProcessing => updateStatus.isProcessing ||
      status == OTAStateStatus.checking ||
      status == OTAStateStatus.downloading ||
      status == OTAStateStatus.activating ||
      status == OTAStateStatus.rollingBack;

  @override
  List<Object?> get props => [
        status,
        updateAvailable,
        availableUpdate,
        updateStatus,
        downloadProgress,
        error,
        lastChecked,
      ];
}
