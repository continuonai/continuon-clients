import 'package:flutter/material.dart';
import 'package:firebase_auth/firebase_auth.dart';

import '../../screens/marketing_home.dart';
import '../../screens/research_screen.dart';
import '../../screens/youtube_import_screen.dart';
import '../../screens/robot_list_screen.dart';
import '../../screens/login_screen.dart';
import '../../services/brain_client.dart';
import '../../screens/dashboard_screen.dart';

class ContinuonDrawer extends StatefulWidget {
  const ContinuonDrawer({super.key});

  @override
  State<ContinuonDrawer> createState() => _ContinuonDrawerState();
}

class _ContinuonDrawerState extends State<ContinuonDrawer> {
  final TextEditingController _quickConnectController = TextEditingController();
  bool _isConnecting = false;

  @override
  void dispose() {
    _quickConnectController.dispose();
    super.dispose();
  }

  Future<void> _handleQuickConnect() async {
    final host = _quickConnectController.text.trim();
    if (host.isEmpty) return;

    setState(() => _isConnecting = true);

    try {
      final client = BrainClient();
      // Basic ping test before navigating
      await client.connect(
        host: host,
        port: 50051,
        httpPort: 8080,
        useTls: false,
      );

      if (mounted) {
        Navigator.pop(context); // Close drawer
        Navigator.push(
          context,
          MaterialPageRoute(
            builder: (context) => DashboardScreen(brainClient: client),
          ),
        );
      }
    } catch (e) {
      if (mounted) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Quick connect failed: $e')),
        );
      }
    } finally {
      if (mounted) setState(() => _isConnecting = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;

    return Drawer(
      child: Column(
        children: [
          DrawerHeader(
            decoration: BoxDecoration(
              color: Theme.of(context).colorScheme.surface,
              border: Border(
                  bottom: BorderSide(
                      color: Theme.of(context).dividerColor.withOpacity(0.1))),
            ),
            child: Center(
              child: Column(
                mainAxisSize: MainAxisSize.min,
                children: [
                  Image.asset(
                    'assets/branding/continuon_ai_logo_text_transparent.png',
                    height: 48,
                    fit: BoxFit.contain,
                  ),
                  const SizedBox(height: 12),
                  Text(
                    'Continuon AI',
                    style: Theme.of(context)
                        .textTheme
                        .titleMedium
                        ?.copyWith(fontWeight: FontWeight.bold),
                  ),
                ],
              ),
            ),
          ),
          Expanded(
            child: ListView(
              padding: EdgeInsets.zero,
              children: [
                _DrawerItem(
                  icon: Icons.science_outlined,
                  label: 'Research',
                  onTap: () =>
                      Navigator.pushNamed(context, ResearchScreen.routeName),
                ),
                _DrawerItem(
                  icon: Icons.smart_toy_outlined,
                  label: 'Robots',
                  onTap: () =>
                      Navigator.pushNamed(context, RobotListScreen.routeName),
                ),
                _DrawerItem(
                  icon: Icons.cloud_upload_outlined,
                  label: 'Import',
                  onTap: () => Navigator.pushNamed(
                      context, YoutubeImportScreen.routeName),
                ),
                const Divider(),
                if (user != null) ...[
                  ListTile(
                    leading: CircleAvatar(
                      radius: 12,
                      backgroundImage: user.photoURL != null
                          ? NetworkImage(user.photoURL!)
                          : null,
                      child: user.photoURL == null
                          ? Text(
                              (user.displayName ?? user.email ?? 'U')[0]
                                  .toUpperCase(),
                              style: const TextStyle(fontSize: 10),
                            )
                          : null,
                    ),
                    title: Text(user.displayName ?? user.email ?? 'User'),
                  ),
                  _DrawerItem(
                    icon: Icons.logout,
                    label: 'Sign Out',
                    onTap: () async {
                      await FirebaseAuth.instance.signOut();
                      if (context.mounted) {
                        Navigator.pushNamedAndRemoveUntil(context,
                            MarketingHomeScreen.routeName, (route) => false);
                      }
                    },
                  ),
                ] else
                  _DrawerItem(
                    icon: Icons.login,
                    label: 'Log In',
                    onTap: () =>
                        Navigator.pushNamed(context, LoginScreen.routeName),
                  ),
              ],
            ),
          ),
          const Divider(),
          Padding(
            padding: const EdgeInsets.all(16.0),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Quick Connect',
                  style: Theme.of(context).textTheme.labelLarge?.copyWith(color: Colors.grey),
                ),
                const SizedBox(height: 8),
                Row(
                  children: [
                    Expanded(
                      child: TextField(
                        controller: _quickConnectController,
                        decoration: const InputDecoration(
                          hintText: 'Robot IP or host',
                          isDense: true,
                          contentPadding: EdgeInsets.symmetric(horizontal: 12, vertical: 12),
                          border: OutlineInputBorder(),
                        ),
                        style: const TextStyle(fontSize: 14),
                        onSubmitted: (_) => _handleQuickConnect(),
                      ),
                    ),
                    const SizedBox(width: 8),
                    if (_isConnecting)
                      const SizedBox(width: 24, height: 24, child: CircularProgressIndicator(strokeWidth: 2))
                    else
                      IconButton(
                        onPressed: _handleQuickConnect,
                        icon: const Icon(Icons.send_rounded, size: 20),
                        color: Theme.of(context).primaryColor,
                      ),
                  ],
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }
}

class _DrawerItem extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _DrawerItem(
      {required this.icon, required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return ListTile(
      leading: Icon(icon, color: Theme.of(context).iconTheme.color),
      title: Text(label, style: const TextStyle(fontSize: 16)),
      onTap: () {
        Navigator.pop(context); // Close drawer
        onTap();
      },
    );
  }
}
