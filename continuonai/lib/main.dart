import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'screens/connect_screen.dart';
import 'screens/control_screen.dart';
import 'screens/dashboard_screen.dart';
import 'screens/login_screen.dart';
import 'screens/manual_mode_screen.dart';
import 'screens/record_screen.dart';
import 'screens/robot_list_screen.dart';
import 'services/brain_client.dart';

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );
  runApp(const MyApp());
}

class MyApp extends StatelessWidget {
  const MyApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      debugShowCheckedModeBanner: false,
      title: 'Continuon AI',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.blue),
        useMaterial3: true,
      ),
      initialRoute: LoginScreen.routeName,
      routes: {
        LoginScreen.routeName: (context) => const LoginScreen(),
        RobotListScreen.routeName: (context) => const RobotListScreen(),
        ConnectScreen.routeName: (context) => ConnectScreen(brainClient: BrainClient()),
        DashboardScreen.routeName: (context) {
          return const Scaffold(body: Center(child: Text('Use navigation with arguments')));
        },
        ControlScreen.routeName: (context) {
           return const Scaffold(body: Center(child: Text('Use navigation with arguments')));
        },
        ManualModeScreen.routeName: (context) {
           return const Scaffold(body: Center(child: Text('Use navigation with arguments')));
        },
        RecordScreen.routeName: (context) {
           return const Scaffold(body: Center(child: Text('Use navigation with arguments')));
        },
      },
    );
  }
}
