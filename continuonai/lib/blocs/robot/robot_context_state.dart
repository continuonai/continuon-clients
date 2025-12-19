import 'package:equatable/equatable.dart';
import '../../models/robot_mode.dart';

class RobotContextState extends Equatable {
  final RobotMode currentMode;
  final bool isLoading;
  final String? error;

  const RobotContextState({
    this.currentMode = RobotMode.idle,
    this.isLoading = false,
    this.error,
  });

  RobotContextState copyWith({
    RobotMode? currentMode,
    bool? isLoading,
    String? error,
  }) {
    return RobotContextState(
      currentMode: currentMode ?? this.currentMode,
      isLoading: isLoading ?? this.isLoading,
      error: error,
    );
  }

  @override
  List<Object?> get props => [currentMode, isLoading, error];
}
