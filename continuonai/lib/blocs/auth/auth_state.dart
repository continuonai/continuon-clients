import 'package:equatable/equatable.dart';
import '../../models/user_role.dart';

abstract class AuthState extends Equatable {
  const AuthState();
  
  @override
  List<Object?> get props => [];
}

class AuthInitial extends AuthState {}

class AuthAuthenticated extends AuthState {
  final String token;
  final UserRole role;
  final String email;

  const AuthAuthenticated({
    required this.token,
    required this.role,
    required this.email,
  });

  @override
  List<Object?> get props => [token, role, email];
}

class AuthUnauthenticated extends AuthState {}
