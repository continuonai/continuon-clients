import 'package:flutter/material.dart';
import 'package:url_launcher/url_launcher.dart';
import '../widgets/layout/continuon_drawer.dart';

class DownloadsScreen extends StatelessWidget {
  const DownloadsScreen({super.key});

  static const routeName = '/downloads';

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      endDrawer: const ContinuonDrawer(),
      appBar: AppBar(
        title: const Text('Downloads'),
        leading: IconButton(
          icon: const Icon(Icons.arrow_back),
          onPressed: () => Navigator.pop(context),
        ),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 48),
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 800),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Download Our Apps',
                  style: Theme.of(context).textTheme.headlineMedium?.copyWith(
                        fontWeight: FontWeight.bold,
                      ),
                ),
                const SizedBox(height: 8),
                Text(
                  'On-device AI apps powered by Nexa SDK on Qualcomm NPU.',
                  style: Theme.of(context).textTheme.bodyLarge?.copyWith(
                        color: Theme.of(context)
                            .textTheme
                            .bodyLarge
                            ?.color
                            ?.withOpacity(0.7),
                      ),
                ),
                const SizedBox(height: 40),
                _AppCard(
                  icon: Icons.smart_toy_outlined,
                  title: 'ContinuonXR — Android Robot Trainer',
                  description:
                      'On-device AI robot training via Nexa SDK on Qualcomm NPU. '
                      'Train robots in the real world using your Android device as the brain.',
                  features: const [
                    'Camera-based VLM scene understanding',
                    'Voice commands via on-device ASR',
                    'RLDS training data recording',
                    'BLE glove support for teleoperation',
                    'XR spatial input',
                  ],
                  downloadUrl:
                      'https://github.com/continuonai/ContinuonXR/releases/latest/download/app-release.apk',
                ),
                const SizedBox(height: 24),
                _AppCard(
                  icon: Icons.closed_caption_outlined,
                  title: 'LiveCaptionsXR — AI Accessibility Captions',
                  description:
                      'Real-time spatially-aware closed captioning powered by Nexa SDK NPU. '
                      'Privacy-first, fully on-device speech recognition.',
                  features: const [
                    'Spatial AR captions anchored in 3D',
                    'On-device privacy-first processing',
                    'Hybrid localization support',
                    'NPU-accelerated ASR inference',
                  ],
                  downloadUrl:
                      'https://github.com/craigm26/LiveCaptionsXR/releases/latest/download/app-release.apk',
                ),
                const SizedBox(height: 40),
                Card(
                  child: Padding(
                    padding: const EdgeInsets.all(20),
                    child: Row(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Icon(Icons.info_outline,
                            color: Theme.of(context).colorScheme.primary),
                        const SizedBox(width: 16),
                        Expanded(
                          child: Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'Sideloading Instructions',
                                style: Theme.of(context)
                                    .textTheme
                                    .titleSmall
                                    ?.copyWith(fontWeight: FontWeight.bold),
                              ),
                              const SizedBox(height: 8),
                              Text(
                                'These APKs are installed via sideloading. '
                                'On your Android device, go to Settings → Security → '
                                'enable "Install from unknown sources" for your browser. '
                                'Then download and open the APK to install.',
                                style: Theme.of(context).textTheme.bodyMedium,
                              ),
                            ],
                          ),
                        ),
                      ],
                    ),
                  ),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }
}

class _AppCard extends StatelessWidget {
  final IconData icon;
  final String title;
  final String description;
  final List<String> features;
  final String downloadUrl;

  const _AppCard({
    required this.icon,
    required this.title,
    required this.description,
    required this.features,
    required this.downloadUrl,
  });

  @override
  Widget build(BuildContext context) {
    return Card(
      elevation: 2,
      child: Padding(
        padding: const EdgeInsets.all(24),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Row(
              children: [
                Icon(icon, size: 32, color: Theme.of(context).colorScheme.primary),
                const SizedBox(width: 12),
                Expanded(
                  child: Text(
                    title,
                    style: Theme.of(context).textTheme.titleLarge?.copyWith(
                          fontWeight: FontWeight.bold,
                        ),
                  ),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Text(description, style: Theme.of(context).textTheme.bodyLarge),
            const SizedBox(height: 16),
            Wrap(
              spacing: 8,
              runSpacing: 8,
              children: features
                  .map((f) => Chip(
                        label: Text(f, style: const TextStyle(fontSize: 12)),
                        visualDensity: VisualDensity.compact,
                      ))
                  .toList(),
            ),
            const SizedBox(height: 20),
            ElevatedButton.icon(
              onPressed: () async {
                final uri = Uri.parse(downloadUrl);
                if (await canLaunchUrl(uri)) {
                  await launchUrl(uri, mode: LaunchMode.externalApplication);
                }
              },
              icon: const Icon(Icons.download),
              label: const Text('Download APK'),
              style: ElevatedButton.styleFrom(
                padding: const EdgeInsets.symmetric(horizontal: 24, vertical: 14),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
