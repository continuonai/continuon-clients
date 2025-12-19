enum UserRole {
  creator,
  developer,
  consumer,
  lessee,
  fleet,
  unknown;

  static UserRole fromString(String role) {
    switch (role.toLowerCase()) {
      case 'creator':
        return UserRole.creator;
      case 'developer':
        return UserRole.developer;
      case 'consumer':
        return UserRole.consumer;
      case 'lessee':
        return UserRole.lessee;
      case 'fleet':
        return UserRole.fleet;
      default:
        return UserRole.unknown;
    }
  }
}
