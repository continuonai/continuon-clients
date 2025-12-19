import 'package:flutter_bloc/flutter_bloc.dart';
import 'robot_context_event.dart';
import 'robot_context_state.dart';
import '../../services/brain_client.dart';

class RobotContextBloc extends Bloc<RobotContextEvent, RobotContextState> {
  final BrainClient brainClient;

  RobotContextBloc({required this.brainClient}) : super(const RobotContextState()) {
    on<RobotModeChanged>((event, emit) {
      emit(state.copyWith(currentMode: event.mode));
    });

    on<RobotModeUpdateRequested>((event, emit) async {
      emit(state.copyWith(isLoading: true));
      try {
        final success = await brainClient.setMode(event.mode.value);
        if (success) {
          emit(state.copyWith(currentMode: event.mode, isLoading: false));
        } else {
          emit(state.copyWith(isLoading: false, error: 'Failed to update mode'));
        }
      } catch (e) {
        emit(state.copyWith(isLoading: false, error: e.toString()));
      }
    });
  }
}
