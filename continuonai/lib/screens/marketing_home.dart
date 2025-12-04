import 'package:flutter/material.dart';

import 'package:url_launcher/url_launcher.dart';
import '../theme/continuon_theme.dart';

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
              // Nav links removed for cleaner branding

              Padding(
                padding: const EdgeInsets.only(right: 16.0),
                child: ElevatedButton(
                  onPressed: () {
                    Navigator.pushNamed(context, '/login');
                  },
                  child: const Text('Get Access'),
                ),
              ),
            ],
          ),

          // 2. Hero Section
          SliverToBoxAdapter(
            child: Container(
              padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 64),
              child: Column(
                children: [
                  Text(
                    'Adaptive Intelligence\nfor the Real World',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.displayMedium?.copyWith(
                          height: 1.1,
                        ),
                  ),
                  const SizedBox(height: 24),
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 700),
                    child: Text(
                      'Continuon builds self-learning AI systems that run everywhere — on-device, on-robot, and in the cloud — enabling robots and assistive systems to learn continuously from real-world experience.',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                            color: Theme.of(context)
                                .textTheme
                                .bodyLarge
                                ?.color
                                ?.withValues(alpha: 0.8),
                          ),
                    ),
                  ),
                  const SizedBox(height: 48),
                  Wrap(
                    spacing: 16,
                    runSpacing: 16,
                    alignment: WrapAlignment.center,
                    children: [
                      ElevatedButton(
                        onPressed: () {
                          Navigator.pushNamed(context, '/login');
                        },
                        style: ElevatedButton.styleFrom(
                          padding: const EdgeInsets.symmetric(
                              horizontal: 32, vertical: 20),
                          textStyle: const TextStyle(fontSize: 18),
                        ),
                        child: const Text('Get Early Access'),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),

          // 3. Problem / Why Now
          SliverToBoxAdapter(
            child: Container(
              color: Theme.of(context)
                  .colorScheme
                  .surfaceContainerHighest
                  .withValues(alpha: 0.3),
              padding: const EdgeInsets.symmetric(vertical: 64, horizontal: 24),
              child: Column(
                children: [
                  Text(
                    'Robots don’t just need models.\nThey need memories.',
                    textAlign: TextAlign.center,
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                  const SizedBox(height: 48),
                  Wrap(
                    spacing: 24,
                    runSpacing: 24,
                    alignment: WrapAlignment.center,
                    children: [
                      _buildFeatureCard(
                        context,
                        title: 'Static models break',
                        description:
                            'Transformers trained once in the cloud can’t adapt on-device.',
                        icon: Icons.broken_image_outlined,
                      ),
                      _buildFeatureCard(
                        context,
                        title: 'Cloud is too slow',
                        description:
                            'Round-tripping every mistake to GPUs adds latency and cost.',
                        icon: Icons.cloud_off_outlined,
                      ),
                      _buildFeatureCard(
                        context,
                        title: 'Hardware is ready',
                        description:
                            'Pi 5, Jetson, and XR devices can host real brains — if the architecture is right.',
                        icon: Icons.memory_outlined,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),

          // 3.5 Manifesto & Design Principles
          SliverToBoxAdapter(
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 80, horizontal: 24),
              child: Column(
                children: [
                  // Manifesto
                  Text(
                    'The Continuon Manifesto',
                    style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                          color: ContinuonColors.primaryBlue,
                          fontWeight: FontWeight.bold,
                        ),
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 32),
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 800),
                    child: Text(
                      'We believe intelligence is not just about pattern matching—it is about adaptation. '
                      'True autonomy requires a system that can learn from its own mistakes in real-time, '
                      'without waiting for a cloud server update. \n\n'
                      'We are building the nervous system for the next generation of machines: '
                      'ones that grow with you, learn your preferences, and adapt to your world.',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                            fontSize: 20,
                            height: 1.6,
                          ),
                    ),
                  ),
                  const SizedBox(height: 80),

                  // Design Principles
                  Text(
                    'Design Principles',
                    style: Theme.of(context).textTheme.headlineMedium,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 48),
                  Wrap(
                    spacing: 32,
                    runSpacing: 32,
                    alignment: WrapAlignment.center,
                    children: [
                      _buildTechCard(
                        context,
                        'Local First',
                        'Intelligence should live where the action is. We prioritize on-device processing for speed, privacy, and reliability.',
                      ),
                      _buildTechCard(
                        context,
                        'Continuous Learning',
                        'Deployment is just the beginning. Our systems evolve with every interaction, getting smarter over time.',
                      ),
                      _buildTechCard(
                        context,
                        'Hybrid Architecture',
                        'We combine the best of symbolic logic, neural networks, and physical simulation to create robust, explainable AI.',
                      ),
                      _buildTechCard(
                        context,
                        'Human Centric',
                        'Technology serves humanity. We design for intuitive interaction, transparency, and user control.',
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),

          // 4. The Continuon Stack
          SliverToBoxAdapter(
            child: Padding(
              padding: const EdgeInsets.symmetric(vertical: 64, horizontal: 24),
              child: Column(
                children: [
                  Text(
                    'The Continuon Stack',
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                  const SizedBox(height: 48),
                  Wrap(
                    spacing: 24,
                    runSpacing: 24,
                    alignment: WrapAlignment.center,
                    children: [
                      _buildProductCard(
                        context,
                        title: 'Continuon AI',
                        subtitle: 'Platform',
                        description:
                            'The parent platform that coordinates robots, XR devices, and cloud training into one continuous learning loop.',
                        assetPath:
                            'assets/branding/continuon_ai_logo_text_transparent.png',
                        accentColor: ContinuonColors.primaryBlue,
                      ),
                      _buildProductCard(
                        context,
                        title: 'ContinuonBrain',
                        subtitle: 'On-Device World Model',
                        description:
                            'A wave–particle hybrid brain that runs on Pi, Jetson, phones, and XR devices. Linear-time, memory-aware, and capable of local learning.',
                        assetPath:
                            'assets/branding/continuon_brain_logo_text_transparent.png',
                        accentColor: ContinuonColors.cmsViolet,
                      ),
                      _buildProductCard(
                        context,
                        title: 'ContinuonCloud',
                        subtitle: 'Cloud Trainer & Skill Hub',
                        description:
                            'Where robots upload experience, get their skills validated, and receive new compressed brains as OTA updates.',
                        assetPath:
                            'assets/branding/continuon_cloud_logo_text_transparent.png',
                        accentColor: ContinuonColors.waveBlueStart,
                        isGradient: true,
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),

          // 5. Architecture Snapshot
          SliverToBoxAdapter(
            child: Container(
              color: Theme.of(context)
                  .colorScheme
                  .surfaceContainerHighest
                  .withValues(alpha: 0.3),
              padding: const EdgeInsets.symmetric(vertical: 64, horizontal: 24),
              child: Column(
                children: [
                  Text(
                    'Under the hood: HOPE + CMS',
                    style: Theme.of(context).textTheme.headlineMedium,
                  ),
                  const SizedBox(height: 16),
                  Text(
                    'Hybrid Object-centric Physical Emulation + Continuous Memory System',
                    style: Theme.of(context).textTheme.bodyLarge,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 48),
                  Wrap(
                    spacing: 24,
                    runSpacing: 24,
                    alignment: WrapAlignment.center,
                    children: [
                      _buildTechCard(context, 'HOPE Core',
                          'Hybrid dynamical system combining SSM-style wave dynamics with local nonlinear updates.'),
                      _buildTechCard(context, 'CMS Memory',
                          'Multi-level memory: fast state, episodic, semantic, and skills.'),
                      _buildTechCard(context, 'Nested Learning',
                          'Slow parameter updates via low-rank adapters for continual learning.'),
                    ],
                  ),
                ],
              ),
            ),
          ),

          // 6. Join the Mission
          SliverToBoxAdapter(
            child: Container(
              padding: const EdgeInsets.symmetric(vertical: 80, horizontal: 24),
              child: Column(
                children: [
                  Text(
                    'Join the Mission',
                    style: Theme.of(context).textTheme.headlineMedium,
                    textAlign: TextAlign.center,
                  ),
                  const SizedBox(height: 24),
                  ConstrainedBox(
                    constraints: const BoxConstraints(maxWidth: 600),
                    child: Text(
                      'We are looking for curious minds and technical experts to join us on this journey. '
                      'Whether you want to contribute to the codebase, test on your robot, or just follow our progress, we’d love to hear from you.',
                      textAlign: TextAlign.center,
                      style: Theme.of(context).textTheme.bodyLarge,
                    ),
                  ),
                  const SizedBox(height: 32),
                  ElevatedButton.icon(
                    onPressed: () async {
                      final Uri emailLaunchUri = Uri(
                        scheme: 'mailto',
                        path: 'craig@craigmerry.com',
                        query:
                            'subject=Continuon AI Interest&body=Hi Craig, I am interested in Continuon AI...',
                      );
                      if (await canLaunchUrl(emailLaunchUri)) {
                        await launchUrl(emailLaunchUri);
                      }
                    },
                    icon: const Icon(Icons.email_outlined),
                    label: const Text('Email Craig'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(
                          horizontal: 32, vertical: 20),
                      textStyle: const TextStyle(fontSize: 18),
                    ),
                  ),
                ],
              ),
            ),
          ),

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

  Widget _buildFeatureCard(BuildContext context,
      {required String title,
      required String description,
      required IconData icon}) {
    return Container(
      width: 300,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
        boxShadow: ContinuonTokens.lowShadow,
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Icon(icon, size: 32, color: ContinuonColors.primaryBlue),
          const SizedBox(height: 16),
          Text(title,
              style: Theme.of(context)
                  .textTheme
                  .titleLarge
                  ?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(description, style: Theme.of(context).textTheme.bodyMedium),
        ],
      ),
    );
  }

  Widget _buildProductCard(
    BuildContext context, {
    required String title,
    required String subtitle,
    required String description,
    required String assetPath,
    required Color accentColor,
    bool isGradient = false,
  }) {
    return Container(
      width: 350,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
        border: Border.all(color: accentColor.withValues(alpha: 0.3), width: 1),
        boxShadow: ContinuonTokens.midShadow,
      ),
      child: Column(
        children: [
          Image.asset(assetPath, height: 80, width: 80),
          const SizedBox(height: 24),
          Text(title,
              style: Theme.of(context)
                  .textTheme
                  .headlineSmall
                  ?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 4),
          Text(subtitle,
              style: Theme.of(context)
                  .textTheme
                  .labelLarge
                  ?.copyWith(color: accentColor)),
          const SizedBox(height: 16),
          Text(description,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium),
        ],
      ),
    );
  }

  Widget _buildTechCard(
      BuildContext context, String title, String description) {
    return Container(
      width: 280,
      padding: const EdgeInsets.all(24),
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
      ),
      child: Column(
        children: [
          Text(title,
              style: Theme.of(context)
                  .textTheme
                  .titleMedium
                  ?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(description,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall),
        ],
      ),
    );
  }
}
