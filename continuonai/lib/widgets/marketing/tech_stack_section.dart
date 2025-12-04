import 'package:flutter/material.dart';
import '../../theme/continuon_theme.dart';

class TechStackSection extends StatelessWidget {
  const TechStackSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        // The Continuon Stack
        Padding(
          padding: const EdgeInsets.symmetric(vertical: 80, horizontal: 24),
          child: Column(
            children: [
              Text(
                'The Continuon Stack',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 64),
              Wrap(
                spacing: 32,
                runSpacing: 32,
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

        // Architecture Snapshot
        Container(
          color: Theme.of(context)
              .colorScheme
              .surfaceContainerHighest
              .withValues(alpha: 0.3),
          padding: const EdgeInsets.symmetric(vertical: 80, horizontal: 24),
          child: Column(
            children: [
              Text(
                'Under the hood: HOPE + CMS',
                style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 16),
              Text(
                'Hybrid Object-centric Physical Emulation + Continuous Memory System',
                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                      color: Theme.of(context)
                          .textTheme
                          .bodyLarge
                          ?.color
                          ?.withValues(alpha: 0.7),
                    ),
                textAlign: TextAlign.center,
              ),
              const SizedBox(height: 64),
              Wrap(
                spacing: 32,
                runSpacing: 32,
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
      ],
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
      padding: const EdgeInsets.all(40),
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(ContinuonTokens.r16),
        border: Border.all(color: accentColor.withValues(alpha: 0.3), width: 1),
        boxShadow: ContinuonTokens.midShadow,
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            Theme.of(context).cardTheme.color!,
            Theme.of(context).cardTheme.color!.withValues(alpha: 0.8),
          ],
        ),
      ),
      child: Column(
        children: [
          Image.asset(assetPath, height: 100, width: 100),
          const SizedBox(height: 32),
          Text(title,
              style: Theme.of(context)
                  .textTheme
                  .headlineSmall
                  ?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 8),
          Text(subtitle,
              style: Theme.of(context)
                  .textTheme
                  .labelLarge
                  ?.copyWith(color: accentColor, fontWeight: FontWeight.bold)),
          const SizedBox(height: 24),
          Text(description,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    height: 1.5,
                  )),
        ],
      ),
    );
  }

  Widget _buildTechCard(
      BuildContext context, String title, String description) {
    return Container(
      width: 300,
      padding: const EdgeInsets.all(32),
      decoration: BoxDecoration(
        color: Theme.of(context).cardTheme.color,
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
        boxShadow: ContinuonTokens.lowShadow,
      ),
      child: Column(
        children: [
          Text(title,
              style: Theme.of(context)
                  .textTheme
                  .titleLarge
                  ?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 16),
          Text(description,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyMedium?.copyWith(
                    height: 1.4,
                  )),
        ],
      ),
    );
  }
}
