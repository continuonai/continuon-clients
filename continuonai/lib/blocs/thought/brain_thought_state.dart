import 'package:equatable/equatable.dart';

class BrainThoughtState extends Equatable {
  final List<Map<String, dynamic>> thoughts;
  const BrainThoughtState({this.thoughts = const []});

  BrainThoughtState copyWith({List<Map<String, dynamic>>? thoughts}) {
    return BrainThoughtState(thoughts: thoughts ?? this.thoughts);
  }

  @override
  List<Object?> get props => [thoughts];
}
