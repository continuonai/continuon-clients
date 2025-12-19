import 'dart:async';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'brain_thought_event.dart';
import 'brain_thought_state.dart';
import '../../services/brain_client.dart';

class BrainThoughtBloc extends Bloc<BrainThoughtEvent, BrainThoughtState> {
  final BrainClient brainClient;
  StreamSubscription? _subscription;

  BrainThoughtBloc({required this.brainClient}) : super(const BrainThoughtState()) {
    on<ThoughtSubscriptionRequested>((event, emit) async {
      await _subscription?.cancel();
      _subscription = brainClient.subscribeToEvents().listen((thought) {
        add(ThoughtReceived(thought));
      });
    });

    on<ThoughtReceived>((event, emit) {
      final updatedThoughts = List<Map<String, dynamic>>.from(state.thoughts)
        ..insert(0, event.thought);
      if (updatedThoughts.length > 50) updatedThoughts.removeLast();
      emit(state.copyWith(thoughts: updatedThoughts));
    });
  }

  @override
  Future<void> close() {
    _subscription?.cancel();
    return super.close();
  }
}
