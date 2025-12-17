import 'package:flutter/material.dart';
import 'package:firebase_core/firebase_core.dart';
import 'firebase_options.dart';
import 'screens/connect_screen.dart';
import 'screens/control_screen.dart';
import 'screens/dashboard_screen.dart';
import 'screens/login_screen.dart';
import 'screens/public_episodes_screen.dart';
import 'screens/public_episode_detail_screen.dart';
import 'screens/manual_mode_screen.dart';
import 'screens/record_screen.dart';
import 'screens/robot_list_screen.dart';
import 'screens/pair_robot_screen.dart';
import 'screens/research_screen.dart';
import 'screens/youtube_import_screen.dart';
import 'services/brain_client.dart';
import 'services/gemma_runtime.dart';

import 'theme/continuon_theme.dart';

import 'screens/marketing_home.dart';

final GemmaAdapterHotReloader _gemmaAdapterHotReloader = GemmaAdapterHotReloader(
  manifestPath: '/opt/continuonos/brain/model/manifest.json',
);

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _gemmaAdapterHotReloader.initialize();
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
      theme: continuonLightTheme,
      darkTheme: continuonDarkTheme,
      themeMode: ThemeMode.system,
      initialRoute: MarketingHomeScreen.routeName,
      routes: {
        MarketingHomeScreen.routeName: (context) => const MarketingHomeScreen(),
        ResearchScreen.routeName: (context) => const ResearchScreen(),
        LoginScreen.routeName: (context) => const LoginScreen(),
        RobotListScreen.routeName: (context) => const RobotListScreen(),
        PairRobotScreen.routeName: (context) => const PairRobotScreen(),
        YoutubeImportScreen.routeName: (context) => const YoutubeImportScreen(),
        PublicEpisodesScreen.routeName: (context) =>
            const PublicEpisodesScreen(),
        PublicEpisodeDetailScreen.routeName: (context) {
          final args = ModalRoute.of(context)?.settings.arguments;
          if (args is String) {
            return PublicEpisodeDetailScreen(slug: args);
          }
          return const Scaffold(
            body: Center(child: Text('Episode slug missing')),
          );
        },
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
