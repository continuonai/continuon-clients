import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
import '../theme/continuon_theme.dart';
import '../widgets/layout/continuon_card.dart';

class WebDiscoveryNotice extends StatelessWidget {
  const WebDiscoveryNotice({super.key});

  @override
  Widget build(BuildContext context) {
    if (!kIsWeb) return const SizedBox.shrink();

    return ContinuonCard(
      margin: const EdgeInsets.symmetric(horizontal: 24, vertical: 12),
      backgroundColor: Colors.amber.withOpacity(0.1),
      border: Border.all(color: Colors.amber.shade700, width: 1),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Icon(Icons.warning_amber_rounded, color: Colors.amber.shade800),
              const SizedBox(width: 12),
              Expanded(
                child: Text(
                  'Web Browser Discovery Limitations',
                  style: Theme.of(context).textTheme.titleSmall?.copyWith(
                        fontWeight: FontWeight.bold,
                        color: Colors.amber.shade900,
                      ),
                ),
              ),
            ],
          ),
          const SizedBox(height: 12),
          Text(
            'Browsers restrict automatic local network discovery (mDNS) for security. If your robot does not appear below:',
            style: Theme.of(context).textTheme.bodySmall,
          ),
          const SizedBox(height: 8),
          _bulletPoint(context, 'Use the "Manual Connect" section to enter your robot\'s IP.'),
          _bulletPoint(context, 'Ensure the robot API server is running on the Pi.'),
          _bulletPoint(context, 'Run "python scripts/tunnel_robot.py" if connecting from a remote network.'),
          const SizedBox(height: 12),
          Text(
            'Pro-tip: Connecting to "localhost" usually works if you are running the Brain Simulator on this machine.',
            style: Theme.of(context).textTheme.bodySmall?.copyWith(
                  fontStyle: FontStyle.italic,
                  color: Colors.grey.shade700,
                ),
          ),
        ],
      ),
    );
  }

  Widget _bulletPoint(BuildContext context, String text) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 4, left: 8),
      child: Row(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          const Text('• ', style: TextStyle(fontWeight: FontWeight.bold)),
          Expanded(
            child: Text(
              text,
              style: Theme.of(context).textTheme.bodySmall,
            ),
          ),
        ],
      ),
    );
  }
}
