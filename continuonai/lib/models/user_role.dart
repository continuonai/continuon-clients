/// User role hierarchy for RCAN protocol.
/// 
/// Roles are ordered by privilege level (5=highest, 1=lowest).
/// See docs/rcan-protocol.md for full specification.
enum UserRole {
  /// Level 5: Robot manufacturer/developer with full control
  creator(5, 'Creator', 'Full hardware/software control, model deployment'),
  
  /// Level 4: Robot owner with configuration and user management
  owner(4, 'Owner', 'Configuration, OTA, user management'),
  
  /// Level 3: Time-bound operational control via lease
  leasee(3, 'Leasee', 'Time-bound operational control'),
  
  /// Level 2: Authenticated user with operational access
  user(2, 'User', 'Operational control within allowed modes'),
  
  /// Level 1: Limited guest access
  guest(1, 'Guest', 'Chat, status queries, rate-limited'),
  
  /// Legacy role mappings
  developer(5, 'Developer', 'Alias for creator'),
  consumer(4, 'Consumer', 'Alias for owner'),
  fleet(4, 'Fleet', 'Fleet manager with multi-robot access'),
  
  /// Unknown/unauthenticated
  unknown(0, 'Unknown', 'Unauthenticated or invalid role');

  const UserRole(this.level, this.displayName, this.description);
  
  /// Privilege level (0-5, higher = more access)
  final int level;
  
  /// Human-readable display name
  final String displayName;
  
  /// Role description for UI
  final String description;
  
  /// Parse role from string (case-insensitive)
  static UserRole fromString(String role) {
    switch (role.toLowerCase()) {
      case 'creator':
        return UserRole.creator;
      case 'developer':
        return UserRole.developer;
      case 'owner':
        return UserRole.owner;
      case 'consumer':
        return UserRole.consumer;
      case 'leasee':
      case 'lessee':
        return UserRole.leasee;
      case 'user':
        return UserRole.user;
      case 'guest':
        return UserRole.guest;
      case 'fleet':
        return UserRole.fleet;
      default:
        return UserRole.unknown;
    }
  }
  
  /// Check if this role has at least the privileges of [other]
  bool hasPrivilegeOf(UserRole other) => level >= other.level;
  
  /// Check if this role can perform a specific capability
  bool canPerform(RobotCapability capability) {
    return _capabilityMatrix[this]?.contains(capability) ?? false;
  }
}

/// Robot capabilities that can be permission-gated
enum RobotCapability {
  viewStatus,
  chat,
  teleopControl,
  armControl,
  navigation,
  recordEpisodes,
  trainingContribute,
  installSkills,
  otaUpdates,
  safetyConfig,
  userManagement,
  modelDeployment,
  hardwareDiagnostics,
}

/// Capability matrix for each role
const Map<UserRole, Set<RobotCapability>> _capabilityMatrix = {
  UserRole.creator: {
    RobotCapability.viewStatus,
    RobotCapability.chat,
    RobotCapability.teleopControl,
    RobotCapability.armControl,
    RobotCapability.navigation,
    RobotCapability.recordEpisodes,
    RobotCapability.trainingContribute,
    RobotCapability.installSkills,
    RobotCapability.otaUpdates,
    RobotCapability.safetyConfig,
    RobotCapability.userManagement,
    RobotCapability.modelDeployment,
    RobotCapability.hardwareDiagnostics,
  },
  UserRole.owner: {
    RobotCapability.viewStatus,
    RobotCapability.chat,
    RobotCapability.teleopControl,
    RobotCapability.armControl,
    RobotCapability.navigation,
    RobotCapability.recordEpisodes,
    RobotCapability.trainingContribute,
    RobotCapability.installSkills,
    RobotCapability.otaUpdates,
    RobotCapability.userManagement,
    RobotCapability.hardwareDiagnostics,
  },
  UserRole.leasee: {
    RobotCapability.viewStatus,
    RobotCapability.chat,
    RobotCapability.teleopControl,
    RobotCapability.armControl,
    RobotCapability.navigation,
    RobotCapability.recordEpisodes,
  },
  UserRole.user: {
    RobotCapability.viewStatus,
    RobotCapability.chat,
    RobotCapability.teleopControl,
    RobotCapability.navigation,
  },
  UserRole.guest: {
    RobotCapability.viewStatus,
    RobotCapability.chat,
  },
  UserRole.developer: {
    RobotCapability.viewStatus,
    RobotCapability.chat,
    RobotCapability.teleopControl,
    RobotCapability.armControl,
    RobotCapability.navigation,
    RobotCapability.recordEpisodes,
    RobotCapability.trainingContribute,
    RobotCapability.installSkills,
    RobotCapability.otaUpdates,
    RobotCapability.safetyConfig,
    RobotCapability.userManagement,
    RobotCapability.modelDeployment,
    RobotCapability.hardwareDiagnostics,
  },
  UserRole.consumer: {
    RobotCapability.viewStatus,
    RobotCapability.chat,
    RobotCapability.teleopControl,
    RobotCapability.armControl,
    RobotCapability.navigation,
    RobotCapability.recordEpisodes,
    RobotCapability.trainingContribute,
    RobotCapability.installSkills,
    RobotCapability.otaUpdates,
    RobotCapability.userManagement,
    RobotCapability.hardwareDiagnostics,
  },
  UserRole.fleet: {
    RobotCapability.viewStatus,
    RobotCapability.chat,
    RobotCapability.teleopControl,
    RobotCapability.armControl,
    RobotCapability.navigation,
    RobotCapability.recordEpisodes,
    RobotCapability.trainingContribute,
    RobotCapability.installSkills,
    RobotCapability.otaUpdates,
    RobotCapability.userManagement,
    RobotCapability.hardwareDiagnostics,
  },
  UserRole.unknown: <RobotCapability>{},
};
