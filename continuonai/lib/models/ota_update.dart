import 'package:equatable/equatable.dart';

/// OTA update system status
enum OTAStatus {
  idle,
  checking,
  downloading,
  verifying,
  extracting,
  activating,
  rollingBack,
  error;

  static OTAStatus fromString(String? value) {
    switch (value?.toLowerCase()) {
      case 'idle':
        return OTAStatus.idle;
      case 'checking':
        return OTAStatus.checking;
      case 'downloading':
        return OTAStatus.downloading;
      case 'verifying':
        return OTAStatus.verifying;
      case 'extracting':
        return OTAStatus.extracting;
      case 'activating':
        return OTAStatus.activating;
      case 'rolling_back':
      case 'rollingback':
        return OTAStatus.rollingBack;
      case 'error':
        return OTAStatus.error;
      default:
        return OTAStatus.idle;
    }
  }

  String get displayName {
    switch (this) {
      case OTAStatus.idle:
        return 'Ready';
      case OTAStatus.checking:
        return 'Checking for updates...';
      case OTAStatus.downloading:
        return 'Downloading...';
      case OTAStatus.verifying:
        return 'Verifying...';
      case OTAStatus.extracting:
        return 'Extracting...';
      case OTAStatus.activating:
        return 'Activating...';
      case OTAStatus.rollingBack:
        return 'Rolling back...';
      case OTAStatus.error:
        return 'Error';
    }
  }
}

/// Available update information
class OTAUpdate extends Equatable {
  final String modelId;
  final String version;
  final String? releaseNotes;
  final int? sizeBytes;
  final String? checksum;
  final DateTime? releaseDate;
  final String? priority;

  const OTAUpdate({
    required this.modelId,
    required this.version,
    this.releaseNotes,
    this.sizeBytes,
    this.checksum,
    this.releaseDate,
    this.priority,
  });

  factory OTAUpdate.fromJson(Map<String, dynamic> json) {
    return OTAUpdate(
      modelId: json['model_id'] as String? ?? json['id'] as String? ?? 'unknown',
      version: json['version'] as String? ?? '0.0.0',
      releaseNotes: json['release_notes'] as String?,
      sizeBytes: (json['size_bytes'] as num?)?.toInt(),
      checksum: json['checksum'] as String?,
      releaseDate: json['release_date'] != null
          ? DateTime.tryParse(json['release_date'].toString())
          : null,
      priority: json['priority'] as String?,
    );
  }

  String get formattedSize {
    if (sizeBytes == null) return 'Unknown size';
    final mb = sizeBytes! / (1024 * 1024);
    if (mb >= 1) {
      return '${mb.toStringAsFixed(1)} MB';
    }
    final kb = sizeBytes! / 1024;
    return '${kb.toStringAsFixed(0)} KB';
  }

  @override
  List<Object?> get props => [
        modelId,
        version,
        releaseNotes,
        sizeBytes,
        checksum,
        releaseDate,
        priority,
      ];
}

/// Installed model versions
class InstalledVersions extends Equatable {
  final String? current;
  final String? candidate;
  final String? rollback;

  const InstalledVersions({
    this.current,
    this.candidate,
    this.rollback,
  });

  factory InstalledVersions.fromJson(Map<String, dynamic> json) {
    return InstalledVersions(
      current: json['current'] as String?,
      candidate: json['candidate'] as String?,
      rollback: json['rollback'] as String?,
    );
  }

  @override
  List<Object?> get props => [current, candidate, rollback];
}

/// Full OTA update status
class UpdateStatus extends Equatable {
  final OTAStatus state;
  final String? currentVersion;
  final double progressPercent;
  final bool rollbackAvailable;
  final InstalledVersions installedVersions;
  final String? errorMessage;
  final DateTime? lastChecked;

  const UpdateStatus({
    this.state = OTAStatus.idle,
    this.currentVersion,
    this.progressPercent = 0.0,
    this.rollbackAvailable = false,
    this.installedVersions = const InstalledVersions(),
    this.errorMessage,
    this.lastChecked,
  });

  factory UpdateStatus.fromJson(Map<String, dynamic> json) {
    final statusJson = json['status'] as Map<String, dynamic>? ?? json;
    final versionsJson =
        json['installed_versions'] as Map<String, dynamic>? ?? {};

    return UpdateStatus(
      state: OTAStatus.fromString(statusJson['state'] as String?),
      currentVersion: statusJson['current_version'] as String?,
      progressPercent:
          (statusJson['progress_percent'] as num?)?.toDouble() ?? 0.0,
      rollbackAvailable: statusJson['rollback_available'] as bool? ?? false,
      installedVersions: InstalledVersions.fromJson(versionsJson),
      errorMessage: statusJson['error'] as String? ?? statusJson['error_message'] as String?,
      lastChecked: statusJson['last_checked'] != null
          ? DateTime.tryParse(statusJson['last_checked'].toString())
          : null,
    );
  }

  UpdateStatus copyWith({
    OTAStatus? state,
    String? currentVersion,
    double? progressPercent,
    bool? rollbackAvailable,
    InstalledVersions? installedVersions,
    String? errorMessage,
    DateTime? lastChecked,
  }) {
    return UpdateStatus(
      state: state ?? this.state,
      currentVersion: currentVersion ?? this.currentVersion,
      progressPercent: progressPercent ?? this.progressPercent,
      rollbackAvailable: rollbackAvailable ?? this.rollbackAvailable,
      installedVersions: installedVersions ?? this.installedVersions,
      errorMessage: errorMessage ?? this.errorMessage,
      lastChecked: lastChecked ?? this.lastChecked,
    );
  }

  bool get isProcessing =>
      state == OTAStatus.checking ||
      state == OTAStatus.downloading ||
      state == OTAStatus.verifying ||
      state == OTAStatus.extracting ||
      state == OTAStatus.activating ||
      state == OTAStatus.rollingBack;

  @override
  List<Object?> get props => [
        state,
        currentVersion,
        progressPercent,
        rollbackAvailable,
        installedVersions,
        errorMessage,
        lastChecked,
      ];
}

/// Result of update check
class UpdateCheckResult extends Equatable {
  final bool updateAvailable;
  final OTAUpdate? availableUpdate;
  final String? currentVersion;
  final String? message;

  const UpdateCheckResult({
    this.updateAvailable = false,
    this.availableUpdate,
    this.currentVersion,
    this.message,
  });

  factory UpdateCheckResult.fromJson(Map<String, dynamic> json) {
    final updateJson = json['update'] as Map<String, dynamic>?;

    return UpdateCheckResult(
      updateAvailable: json['update_available'] as bool? ?? false,
      availableUpdate:
          updateJson != null ? OTAUpdate.fromJson(updateJson) : null,
      currentVersion: json['current_version'] as String?,
      message: json['message'] as String?,
    );
  }

  @override
  List<Object?> get props =>
      [updateAvailable, availableUpdate, currentVersion, message];
}
