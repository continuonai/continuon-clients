import 'package:flutter/material.dart';
import '../../theme/continuon_theme.dart';

class ManifestoSection extends StatelessWidget {
  const ManifestoSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 100, horizontal: 24),
      child: Column(
        children: [
          // Manifesto
          Text(
            'The Continuon Manifesto',
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  color: ContinuonColors.primaryBlue,
                  fontWeight: FontWeight.bold,
                  letterSpacing: 1.2,
                ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 48),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 900),
            child: Text(
              'We believe intelligence is not just about pattern matching—it is about adaptation. '
              'True autonomy requires a system that can learn from its own mistakes in real-time, '
              'without waiting for a cloud server update. \n\n'
              'We are building the nervous system for the next generation of machines: '
              'ones that grow with you, learn your preferences, and adapt to your world.',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                    height: 1.6,
                    fontWeight: FontWeight.w300,
                  ),
            ),
          ),
          const SizedBox(height: 100),

          // Design Principles
          Text(
            'Design Principles',
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 64),
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
        border: Border.all(
          color: Theme.of(context).dividerColor.withValues(alpha: 0.1),
        ),
      ),
      child: Column(
        children: [
          Text(title,
              style: Theme.of(context)
                  .textTheme
                  .titleMedium
                  ?.copyWith(fontWeight: FontWeight.bold)),
          const SizedBox(height: 12),
          Text(description,
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodySmall?.copyWith(
                    height: 1.4,
                  )),
        ],
      ),
    );
  }
}
