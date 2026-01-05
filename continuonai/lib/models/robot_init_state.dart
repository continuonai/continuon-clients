import 'package:equatable/equatable.dart';

/// Steps in the robot initialization wizard
enum RobotInitStep {
  welcome,
  claimOwnership,
  nameRobot,
  installSeed,
  complete;

  String get title {
    switch (this) {
      case RobotInitStep.welcome:
        return 'Welcome';
      case RobotInitStep.claimOwnership:
        return 'Claim Robot';
      case RobotInitStep.nameRobot:
        return 'Name';
      case RobotInitStep.installSeed:
        return 'Install AI';
      case RobotInitStep.complete:
        return 'Complete';
    }
  }

  String get description {
    switch (this) {
      case RobotInitStep.welcome:
        return 'Check network connectivity and robot status';
      case RobotInitStep.claimOwnership:
        return 'Register as the owner of this robot';
      case RobotInitStep.nameRobot:
        return 'Give your robot a name';
      case RobotInitStep.installSeed:
        return 'Install the base AI model';
      case RobotInitStep.complete:
        return 'Setup complete! Your robot is ready.';
    }
  }

  static RobotInitStep fromIndex(int index) {
    return RobotInitStep.values[index.clamp(0, RobotInitStep.values.length - 1)];
  }
}

/// Ownership status from the robot
class RobotOwnershipStatus extends Equatable {
  final bool owned;
  final bool subscriptionActive;
  final bool seedInstalled;
  final String? ownerId;
  final String? accountId;
  final String? accountType;

  const RobotOwnershipStatus({
    this.owned = false,
    this.subscriptionActive = false,
    this.seedInstalled = false,
    this.ownerId,
    this.accountId,
    this.accountType,
  });

  factory RobotOwnershipStatus.fromJson(Map<String, dynamic> json) {
    return RobotOwnershipStatus(
      owned: json['owned'] as bool? ?? false,
      subscriptionActive: json['subscription_active'] as bool? ?? false,
      seedInstalled: json['seed_installed'] as bool? ?? false,
      ownerId: json['owner_id'] as String?,
      accountId: json['account_id'] as String?,
      accountType: json['account_type'] as String?,
    );
  }

  @override
  List<Object?> get props => [
        owned,
        subscriptionActive,
        seedInstalled,
        ownerId,
        accountId,
        accountType,
      ];
}

/// Robot information from discovery/status
class RobotInfo extends Equatable {
  final String deviceId;
  final String name;
  final String? model;
  final String? firmwareVersion;
  final List<String> capabilities;
  final String? ipAddress;
  final int? port;

  const RobotInfo({
    required this.deviceId,
    required this.name,
    this.model,
    this.firmwareVersion,
    this.capabilities = const [],
    this.ipAddress,
    this.port,
  });

  factory RobotInfo.fromJson(Map<String, dynamic> json) {
    final capsList = json['capabilities'] as List<dynamic>? ?? [];

    return RobotInfo(
      deviceId: json['device_id'] as String? ??
          json['id'] as String? ??
          json['ruri'] as String? ??
          'unknown',
      name: json['name'] as String? ?? json['robot_name'] as String? ?? 'Robot',
      model: json['model'] as String?,
      firmwareVersion: json['firmware_version'] as String? ??
          json['version'] as String?,
      capabilities: capsList.map((e) => e.toString()).toList(),
      ipAddress: json['ip_address'] as String? ?? json['host'] as String?,
      port: (json['port'] as num?)?.toInt(),
    );
  }

  @override
  List<Object?> get props => [
        deviceId,
        name,
        model,
        firmwareVersion,
        capabilities,
        ipAddress,
        port,
      ];
}

/// Full state for the robot initialization wizard
class RobotInitState extends Equatable {
  final RobotInitStep currentStep;
  final bool isLoading;
  final String? error;
  final bool networkConnected;
  final RobotOwnershipStatus ownershipStatus;
  final RobotInfo? robotInfo;
  final String robotName;
  final bool claimSuccess;
  final bool seedInstallSuccess;
  final double seedInstallProgress;

  const RobotInitState({
    this.currentStep = RobotInitStep.welcome,
    this.isLoading = false,
    this.error,
    this.networkConnected = false,
    this.ownershipStatus = const RobotOwnershipStatus(),
    this.robotInfo,
    this.robotName = '',
    this.claimSuccess = false,
    this.seedInstallSuccess = false,
    this.seedInstallProgress = 0.0,
  });

  RobotInitState copyWith({
    RobotInitStep? currentStep,
    bool? isLoading,
    String? error,
    bool clearError = false,
    bool? networkConnected,
    RobotOwnershipStatus? ownershipStatus,
    RobotInfo? robotInfo,
    String? robotName,
    bool? claimSuccess,
    bool? seedInstallSuccess,
    double? seedInstallProgress,
  }) {
    return RobotInitState(
      currentStep: currentStep ?? this.currentStep,
      isLoading: isLoading ?? this.isLoading,
      error: clearError ? null : (error ?? this.error),
      networkConnected: networkConnected ?? this.networkConnected,
      ownershipStatus: ownershipStatus ?? this.ownershipStatus,
      robotInfo: robotInfo ?? this.robotInfo,
      robotName: robotName ?? this.robotName,
      claimSuccess: claimSuccess ?? this.claimSuccess,
      seedInstallSuccess: seedInstallSuccess ?? this.seedInstallSuccess,
      seedInstallProgress: seedInstallProgress ?? this.seedInstallProgress,
    );
  }

  /// Check if the current step can proceed to the next
  bool get canProceed {
    switch (currentStep) {
      case RobotInitStep.welcome:
        return networkConnected && !isLoading;
      case RobotInitStep.claimOwnership:
        return (ownershipStatus.owned || claimSuccess) && !isLoading;
      case RobotInitStep.nameRobot:
        return robotName.isNotEmpty && !isLoading;
      case RobotInitStep.installSeed:
        return (ownershipStatus.seedInstalled || seedInstallSuccess) &&
            !isLoading;
      case RobotInitStep.complete:
        return true;
    }
  }

  /// Check if the current step should be skipped
  bool get shouldSkipCurrentStep {
    switch (currentStep) {
      case RobotInitStep.claimOwnership:
        return ownershipStatus.owned;
      case RobotInitStep.installSeed:
        return ownershipStatus.seedInstalled;
      default:
        return false;
    }
  }

  @override
  List<Object?> get props => [
        currentStep,
        isLoading,
        error,
        networkConnected,
        ownershipStatus,
        robotInfo,
        robotName,
        claimSuccess,
        seedInstallSuccess,
        seedInstallProgress,
      ];
}
