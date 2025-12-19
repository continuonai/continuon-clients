import 'package:equatable/equatable.dart';
import '../../models/robot_mode.dart';

abstract class RobotContextEvent extends Equatable {
  const RobotContextEvent();

  @override
  List<Object?> get props => [];
}

class RobotModeChanged extends RobotContextEvent {
  final RobotMode mode;
  const RobotModeChanged(this.mode);

  @override
  List<Object?> get props => [mode];
}

class RobotModeUpdateRequested extends RobotContextEvent {
  final RobotMode mode;
  const RobotModeUpdateRequested(this.mode);

  @override
  List<Object?> get props => [mode];
}
