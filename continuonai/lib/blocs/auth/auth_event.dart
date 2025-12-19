import 'package:equatable/equatable.dart';
import '../../models/user_role.dart';

abstract class AuthEvent extends Equatable {
  const AuthEvent();

  @override
  List<Object?> get props => [];
}

class AuthUserChanged extends AuthEvent {
  final String? token;
  final UserRole role;
  final String? email;

  const AuthUserChanged({this.token, required this.role, this.email});

  @override
  List<Object?> get props => [token, role, email];
}

class AuthLogoutRequested extends AuthEvent {}
