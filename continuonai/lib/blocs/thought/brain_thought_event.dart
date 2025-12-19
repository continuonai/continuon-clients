import 'package:equatable/equatable.dart';

abstract class BrainThoughtEvent extends Equatable {
  const BrainThoughtEvent();
  @override
  List<Object?> get props => [];
}

class ThoughtReceived extends BrainThoughtEvent {
  final Map<String, dynamic> thought;
  const ThoughtReceived(this.thought);
  @override
  List<Object?> get props => [thought];
}

class ThoughtSubscriptionRequested extends BrainThoughtEvent {}
