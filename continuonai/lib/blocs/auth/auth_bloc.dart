import 'package:flutter_bloc/flutter_bloc.dart';
import 'auth_event.dart';
import 'auth_state.dart';

class AuthBloc extends Bloc<AuthEvent, AuthState> {
  AuthBloc() : super(AuthInitial()) {
    on<AuthUserChanged>((event, emit) {
      if (event.token != null) {
        emit(AuthAuthenticated(
          token: event.token!,
          role: event.role,
          email: event.email ?? '',
        ));
      } else {
        emit(AuthUnauthenticated());
      }
    });

    on<AuthLogoutRequested>((event, emit) {
      emit(AuthUnauthenticated());
    });
  }
}
