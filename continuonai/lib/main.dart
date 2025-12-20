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
import 'screens/model_manager_screen.dart';
import 'screens/robot_list_screen.dart';
import 'screens/robot_portal_screen.dart';
import 'screens/pair_robot_screen.dart';
import 'screens/research_screen.dart';
import 'screens/youtube_import_screen.dart';
import 'services/brain_client.dart';
import 'services/gemma_runtime.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import 'blocs/auth/auth_bloc.dart';
import 'blocs/robot/robot_context_bloc.dart';
import 'blocs/thought/brain_thought_bloc.dart';
import 'blocs/thought/brain_thought_event.dart';

import 'theme/continuon_theme.dart';

import 'screens/marketing_home.dart';

final GemmaAdapterHotReloader _gemmaAdapterHotReloader =
    GemmaAdapterHotReloader(
  manifestPath: '/opt/continuonos/brain/model/manifest.json',
);

void main() async {
  WidgetsFlutterBinding.ensureInitialized();
  await _gemmaAdapterHotReloader.initialize();
  await Firebase.initializeApp(
    options: DefaultFirebaseOptions.currentPlatform,
  );

  final brainClient = BrainClient();
  await brainClient.loadAuthToken();

  runApp(
    MultiBlocProvider(
      providers: [
        BlocProvider(create: (context) => AuthBloc()),
        BlocProvider(
            create: (context) => RobotContextBloc(brainClient: brainClient)),
        BlocProvider(
            create: (context) => BrainThoughtBloc(brainClient: brainClient)
              ..add(ThoughtSubscriptionRequested())),
      ],
      child: MyApp(brainClient: brainClient),
    ),
  );
}

class MyApp extends StatelessWidget {
  final BrainClient brainClient;
  const MyApp({super.key, required this.brainClient});

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
        RobotPortalScreen.routeName: (context) {
          final args = ModalRoute.of(context)?.settings.arguments;
          if (args is Map<String, dynamic>) {
            return RobotPortalScreen(
              host: args['host'] as String,
              httpPort: args['httpPort'] as int? ?? 8080,
              robotName: args['robotName'] as String? ?? 'Robot',
            );
          }
          return const Scaffold(
            body: Center(child: Text('Robot portal: missing arguments')),
          );
        },
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
        ConnectScreen.routeName: (context) =>
            ConnectScreen(brainClient: brainClient),
        DashboardScreen.routeName: (context) =>
            DashboardScreen(brainClient: brainClient),
        ControlScreen.routeName: (context) =>
            ControlScreen(brainClient: brainClient),
        ManualModeScreen.routeName: (context) =>
            ManualModeScreen(brainClient: brainClient),
        RecordScreen.routeName: (context) =>
            RecordScreen(brainClient: brainClient),
        ModelManagerScreen.routeName: (context) =>
            ModelManagerScreen(brainClient: brainClient),
      },
    );
  }
}
