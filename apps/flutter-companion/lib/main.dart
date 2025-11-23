import 'package:flutter/material.dart';
import 'screens/connect_screen.dart';
import 'screens/control_screen.dart';
import 'screens/record_screen.dart';
import 'services/brain_client.dart';

void main() {
  runApp(const FlutterCompanionApp());
}

class FlutterCompanionApp extends StatefulWidget {
  const FlutterCompanionApp({super.key});

  @override
  State<FlutterCompanionApp> createState() => _FlutterCompanionAppState();
}

class _FlutterCompanionAppState extends State<FlutterCompanionApp> {
  late final BrainClient _brainClient;

  @override
  void initState() {
    super.initState();
    _brainClient = BrainClient();
  }

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: 'ContinuonXR Companion',
      theme: ThemeData(
        colorScheme: ColorScheme.fromSeed(seedColor: Colors.indigo),
        useMaterial3: true,
      ),
      routes: {
        ConnectScreen.routeName: (_) => ConnectScreen(brainClient: _brainClient),
        ControlScreen.routeName: (_) => ControlScreen(brainClient: _brainClient),
        RecordScreen.routeName: (_) => RecordScreen(brainClient: _brainClient),
      },
      initialRoute: ConnectScreen.routeName,
    );
  }
}
