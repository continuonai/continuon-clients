import 'package:flutter/material.dart';

import '../widgets/marketing/hero_section.dart';
import '../widgets/marketing/problem_solution_section.dart';
import '../widgets/marketing/manifesto_section.dart';
import '../widgets/marketing/tech_stack_section.dart';
import '../widgets/marketing/cta_section.dart';

import 'package:firebase_auth/firebase_auth.dart';
import '../widgets/layout/continuon_drawer.dart';
import 'research_screen.dart';
import 'youtube_import_screen.dart';
import 'robot_list_screen.dart';
import 'login_screen.dart';

class MarketingHomeScreen extends StatefulWidget {
  const MarketingHomeScreen({super.key});

  static const routeName = '/';

  @override
  State<MarketingHomeScreen> createState() => _MarketingHomeScreenState();
}

class _MarketingHomeScreenState extends State<MarketingHomeScreen> {
  final GlobalKey<ScaffoldState> _scaffoldKey = GlobalKey<ScaffoldState>();

  @override
  Widget build(BuildContext context) {
    // Use the custom theme extension for gradients and accents
    // final brand = Theme.of(context).extension<ContinuonBrandExtension>()!;
    final user = FirebaseAuth.instance.currentUser;

    return Scaffold(
      key: _scaffoldKey,
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      endDrawer: const ContinuonDrawer(),
      body: CustomScrollView(
        slivers: [
          // 1. Sticky Navigation
          SliverAppBar(
            pinned: true,
            floating: true,
            backgroundColor: Theme.of(context).appBarTheme.backgroundColor,
            title: InkWell(
              onTap: () {}, // Already on home
              borderRadius: BorderRadius.circular(8),
              child: Padding(
                padding: const EdgeInsets.symmetric(vertical: 8.0),
                child: Image.asset(
                  'assets/branding/continuon_ai_logo_text_transparent.png',
                  height: 40,
                  fit: BoxFit.contain,
                ),
              ),
            ),
            actions: [
              if (MediaQuery.of(context).size.width > 800) ...[
                TextButton(
                  onPressed: () =>
                      Navigator.pushNamed(context, ResearchScreen.routeName),
                  style: _navStyle(context),
                  child: const Text('Research'),
                ),
                TextButton(
                  onPressed: () =>
                      Navigator.pushNamed(context, RobotListScreen.routeName),
                  style: _navStyle(context),
                  child: const Text('Robots'),
                ),
                TextButton(
                  onPressed: () => Navigator.pushNamed(
                      context, YoutubeImportScreen.routeName),
                  style: _navStyle(context),
                  child: const Text('Import'),
                ),
                const SizedBox(width: 16),
                if (user == null)
                  ElevatedButton(
                    onPressed: () =>
                        Navigator.pushNamed(context, LoginScreen.routeName),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 20, vertical: 12),
                    ),
                    child: const Text('Log in'),
                  )
                else
                  // Simple text for now or avatar if needed, but drawer handles profile
                  TextButton(
                    onPressed: () =>
                        Navigator.pushNamed(context, RobotListScreen.routeName),
                    child: Text(user.displayName ?? 'My Account'),
                  ),
                const SizedBox(width: 24),
              ] else ...[
                IconButton(
                  icon: const Icon(Icons.menu),
                  onPressed: () => _scaffoldKey.currentState?.openEndDrawer(),
                ),
                const SizedBox(width: 16),
              ],
            ],
          ),

          // 2. Hero Section
          const SliverToBoxAdapter(child: HeroSection()),

          // 3. Problem / Why Now
          const SliverToBoxAdapter(child: ProblemSolutionSection()),

          // 3.5 Manifesto & Design Principles
          const SliverToBoxAdapter(child: ManifestoSection()),

          // 4. The Continuon Stack & Architecture
          const SliverToBoxAdapter(child: TechStackSection()),

          // 6. Join the Mission
          const SliverToBoxAdapter(child: CallToActionSection()),

          // 7. Footer
          SliverToBoxAdapter(
            child: Container(
              color: Theme.of(context).colorScheme.onSurface,
              padding: const EdgeInsets.symmetric(vertical: 48, horizontal: 24),
              child: Column(
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.center,
                    children: [
                      Image.asset(
                        'assets/branding/continuon_ai_logo_text_transparent.png',
                        height: 24,
                        width: 24,
                      ),
                      const SizedBox(width: 12),
                      const Text(
                        'Continuon AI',
                        style: TextStyle(
                            color: Colors.white,
                            fontSize: 20,
                            fontWeight: FontWeight.bold),
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  const Text(
                    '© Continuon AI. Adaptive Intelligence for the Real World.',
                    style: TextStyle(color: Colors.white54),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }

  ButtonStyle _navStyle(BuildContext context) {
    return TextButton.styleFrom(
      foregroundColor: Theme.of(context).textTheme.bodyLarge?.color,
      textStyle: const TextStyle(
          fontWeight: FontWeight.w600, fontSize: 18, letterSpacing: 0.5),
      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
    );
  }
}
