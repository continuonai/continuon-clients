import 'package:flutter/material.dart';
import 'package:flutter_svg/flutter_svg.dart';
import 'package:firebase_auth/firebase_auth.dart';
import '../../screens/marketing_home.dart';
import '../../screens/research_screen.dart';
import '../../screens/youtube_import_screen.dart';
import '../../screens/robot_list_screen.dart';
import '../../screens/login_screen.dart';

class ContinuonAppBar extends StatelessWidget implements PreferredSizeWidget {
  final List<Widget>? actions;

  const ContinuonAppBar({super.key, this.actions});

  @override
  Size get preferredSize =>
      const Size.fromHeight(72.0); // Taller for larger logo

  @override
  Widget build(BuildContext context) {
    final user = FirebaseAuth.instance.currentUser;
    final isDark = Theme.of(context).brightness == Brightness.dark;

    return Container(
      decoration: BoxDecoration(
        color: Theme.of(context).appBarTheme.backgroundColor?.withOpacity(0.95),
        border: Border(
          bottom: BorderSide(
            color: isDark
                ? Colors.white.withOpacity(0.1)
                : Colors.black.withOpacity(0.05),
            width: 1,
          ),
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withOpacity(0.05),
            blurRadius: 10,
            offset: const Offset(0, 4),
          ),
        ],
      ),
      child: SafeArea(
        child: Padding(
          padding: const EdgeInsets.symmetric(horizontal: 24.0),
          child: Row(
            children: [
              // 1. Large Logo (Left) - Clickable to Home
              InkWell(
                onTap: () => Navigator.of(context).pushNamedAndRemoveUntil(
                  MarketingHomeScreen.routeName,
                  (route) => false,
                ),
                borderRadius: BorderRadius.circular(8),
                child: Padding(
                  padding: const EdgeInsets.symmetric(vertical: 8.0),
                  child: SvgPicture.asset(
                    'assets/branding/continuon_ai_logo.svg',
                    height: 40, // Larger than 32, logic only (no text)
                    fit: BoxFit.contain,
                  ),
                ),
              ),

              const Spacer(),

              // 2. Navigation Links (Desktop/Tablet)
              // In a real responsive app, we'd hide these on mobile, but for now we keep them or wrap in viewport check.
              // Assuming desktop/tablet focus for "browser experience"
              if (MediaQuery.of(context).size.width > 800) ...[
                _NavButton(
                  label: 'Research',
                  onTap: () =>
                      Navigator.pushNamed(context, ResearchScreen.routeName),
                ),
                const SizedBox(width: 16),
                _NavButton(
                  label: 'Robots',
                  onTap: () {
                    // Check access logic if needed, or just go there. Robot screen handles gating.
                    Navigator.pushNamed(context, RobotListScreen.routeName);
                  },
                ),
                const SizedBox(width: 16),
                _NavButton(
                  label: 'Import',
                  onTap: () => Navigator.pushNamed(
                      context, YoutubeImportScreen.routeName),
                ),
                const SizedBox(width: 24),
              ],

              // 2.5 Screen Specific Actions
              if (actions != null) ...[
                ...actions!,
                const SizedBox(width: 16),
              ],

              // 3. Right Side Actions (User Profile / Auth)
              if (user != null)
                _UserBadge(user: user)
              else
                ElevatedButton(
                  onPressed: () =>
                      Navigator.pushNamed(context, LoginScreen.routeName),
                  style: ElevatedButton.styleFrom(
                    padding: const EdgeInsets.symmetric(
                        horizontal: 20, vertical: 12),
                  ),
                  child: const Text('Log in'),
                ),
            ],
          ),
        ),
      ),
    );
  }
}

class _NavButton extends StatelessWidget {
  final String label;
  final VoidCallback onTap;

  const _NavButton({required this.label, required this.onTap});

  @override
  Widget build(BuildContext context) {
    return TextButton(
      onPressed: onTap,
      style: TextButton.styleFrom(
        foregroundColor: Theme.of(context).textTheme.bodyLarge?.color,
        textStyle: const TextStyle(
            fontWeight: FontWeight.w600, fontSize: 18, letterSpacing: 0.5),
        padding: const EdgeInsets.symmetric(horizontal: 20, vertical: 16),
      ),
      child: Text(label),
    );
  }
}

class _UserBadge extends StatelessWidget {
  final User user;

  const _UserBadge({required this.user});

  @override
  Widget build(BuildContext context) {
    return PopupMenuButton<String>(
      offset: const Offset(0, 48),
      tooltip: 'User menu',
      itemBuilder: (context) => [
        const PopupMenuItem(
          value: 'robots',
          child: Row(
            children: [
              Icon(Icons.smart_toy, size: 20),
              SizedBox(width: 12),
              Text('My Robots'),
            ],
          ),
        ),
        const PopupMenuDivider(),
        const PopupMenuItem(
          value: 'logout',
          child: Row(
            children: [
              Icon(Icons.logout, size: 20),
              SizedBox(width: 12),
              Text('Sign out'),
            ],
          ),
        ),
      ],
      onSelected: (value) async {
        if (value == 'logout') {
          await FirebaseAuth.instance.signOut();
          if (context.mounted) {
            Navigator.of(context).pushNamedAndRemoveUntil(
                MarketingHomeScreen.routeName, (route) => false);
          }
        } else if (value == 'robots') {
          Navigator.pushNamed(context, RobotListScreen.routeName);
        }
      },
      child: CircleAvatar(
        radius: 20,
        backgroundColor: Theme.of(context).primaryColor,
        backgroundImage:
            user.photoURL != null ? NetworkImage(user.photoURL!) : null,
        child: user.photoURL == null
            ? Text(
                (user.displayName ?? user.email ?? 'U')
                    .substring(0, 1)
                    .toUpperCase(),
                style: const TextStyle(
                    color: Colors.white, fontWeight: FontWeight.bold),
              )
            : null,
      ),
    );
  }
}
