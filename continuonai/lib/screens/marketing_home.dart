import 'package:flutter/material.dart';

import '../widgets/marketing/hero_section.dart';
import '../widgets/marketing/problem_solution_section.dart';
import '../widgets/marketing/manifesto_section.dart';
import '../widgets/marketing/tech_stack_section.dart';
import '../widgets/marketing/cta_section.dart';
import 'research_screen.dart';

class MarketingHomeScreen extends StatelessWidget {
  const MarketingHomeScreen({super.key});

  static const routeName = '/';

  @override
  Widget build(BuildContext context) {
    // Use the custom theme extension for gradients and accents
    // final brand = Theme.of(context).extension<ContinuonBrandExtension>()!;

    return Scaffold(
      backgroundColor: Theme.of(context).scaffoldBackgroundColor,
      body: CustomScrollView(
        slivers: [
          // 1. Sticky Navigation
          SliverAppBar(
            pinned: true,
            floating: true,
            backgroundColor: Theme.of(context).appBarTheme.backgroundColor,
            title: Row(
              children: [
                Image.asset(
                  'assets/branding/continuon_ai_logo_text_transparent.png',
                  height: 32,
                  width: 32,
                ),
                const SizedBox(width: 12),
                Text(
                  'Continuon AI',
                  style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
              ],
            ),
            actions: [
              Padding(
                padding: const EdgeInsets.only(right: 16.0),
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.pushNamed(context, '/login');
                  },
                  child: const Text('Sign up / Log in'),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(right: 8.0),
                child: TextButton(
                  onPressed: () {
                    Navigator.pushNamed(context, ResearchScreen.routeName);
                  },
                  child: const Text('Research'),
                ),
              ),
              Padding(
                padding: const EdgeInsets.only(right: 8.0),
                child: TextButton(
                  onPressed: () {
                    Navigator.pushNamed(context, '/episodes');
                  },
                  child: const Text('Public RLDS'),
                ),
              ),
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
}
