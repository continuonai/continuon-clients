import 'package:flutter/material.dart';
import '../../screens/downloads_screen.dart';

class DownloadsSection extends StatelessWidget {
  const DownloadsSection({super.key});

  @override
  Widget build(BuildContext context) {
    return Container(
      padding: const EdgeInsets.symmetric(vertical: 80, horizontal: 24),
      child: Column(
        children: [
          Text(
            'Download ContinuonXR',
            style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                  fontWeight: FontWeight.bold,
                ),
            textAlign: TextAlign.center,
          ),
          const SizedBox(height: 16),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 600),
            child: Text(
              'On-device AI robot training running on Qualcomm NPU via Nexa SDK. No cloud required.',
              textAlign: TextAlign.center,
              style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                    fontSize: 18,
                    height: 1.5,
                  ),
            ),
          ),
          const SizedBox(height: 48),
          ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 800),
            child: _MiniAppCard(
              icon: Icons.smart_toy_outlined,
              title: 'ContinuonXR',
              subtitle: 'Android Robot Trainer',
              description:
                  'Train robots with on-device AI — camera VLM, voice commands, RLDS recording.',
            ),
          ),
          const SizedBox(height: 40),
          TextButton(
            onPressed: () =>
                Navigator.pushNamed(context, DownloadsScreen.routeName),
            child: Text(
              'See All Downloads →',
              style: TextStyle(
                fontSize: 16,
                fontWeight: FontWeight.w600,
                color: Theme.of(context).colorScheme.primary,
              ),
            ),
          ),
        ],
      ),
    );
  }
}

class _MiniAppCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String subtitle;
  final String description;

  const _MiniAppCard({
    required this.icon,
    required this.title,
    required this.subtitle,
    required this.description,
  });

  @override
  Widget build(BuildContext context) {
    return SizedBox(
      width: 360,
      child: Card(
        elevation: 2,
        child: Padding(
          padding: const EdgeInsets.all(24),
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Row(
                children: [
                  Icon(icon,
                      size: 28, color: Theme.of(context).colorScheme.primary),
                  const SizedBox(width: 12),
                  Column(
                    crossAxisAlignment: CrossAxisAlignment.start,
                    children: [
                      Text(title,
                          style: Theme.of(context)
                              .textTheme
                              .titleMedium
                              ?.copyWith(fontWeight: FontWeight.bold)),
                      Text(subtitle,
                          style: Theme.of(context).textTheme.bodySmall),
                    ],
                  ),
                ],
              ),
              const SizedBox(height: 12),
              Text(description,
                  style: Theme.of(context).textTheme.bodyMedium),
            ],
          ),
        ),
      ),
    );
  }
}
