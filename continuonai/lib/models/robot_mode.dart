enum RobotMode {
  manualControl('manual_control'),
  manualTraining('manual_training'),
  autonomous('autonomous'),
  sleepLearning('sleep_learning'),
  idle('idle'),
  emergencyStop('emergency_stop');

  final String value;
  const RobotMode(this.value);

  static RobotMode fromString(String value) {
    return RobotMode.values.firstWhere((e) => e.value == value, orElse: () => RobotMode.idle);
  }
}
