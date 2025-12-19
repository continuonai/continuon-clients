import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../blocs/auth/auth_bloc.dart';
import '../blocs/auth/auth_state.dart';
import '../blocs/robot/robot_context_bloc.dart';
import '../blocs/robot/robot_context_event.dart';
import '../blocs/robot/robot_context_state.dart';
import '../blocs/thought/brain_thought_bloc.dart';
import '../blocs/thought/brain_thought_state.dart';
import '../models/user_role.dart';
import '../models/robot_mode.dart';
import '../screens/model_manager_screen.dart';

class CreatorDashboard extends StatelessWidget {
  const CreatorDashboard({super.key});

  @override
  Widget build(BuildContext context) {
    return BlocBuilder<AuthBloc, AuthState>(
      builder: (context, authState) {
        if (authState is! AuthAuthenticated || authState.role != UserRole.creator) {
          return const SizedBox.shrink();
        }

        return Card(
          elevation: 4,
          shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
          margin: const EdgeInsets.all(16),
          child: Container(
            decoration: BoxDecoration(
              borderRadius: BorderRadius.circular(16),
              gradient: LinearGradient(
                colors: [
                  Theme.of(context).colorScheme.surface,
                  Theme.of(context).colorScheme.surfaceContainerHighest.withOpacity(0.5),
                ],
                begin: Alignment.topLeft,
                end: Alignment.bottomRight,
              ),
            ),
            child: Padding(
              padding: const EdgeInsets.all(20.0),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    mainAxisAlignment: MainAxisAlignment.spaceBetween,
                    children: [
                      Row(
                        children: [
                          Container(
                            padding: const EdgeInsets.all(8),
                            decoration: BoxDecoration(
                              color: Colors.red.withOpacity(0.1),
                              shape: BoxShape.circle,
                            ),
                            child: const Icon(Icons.bolt, color: Colors.red),
                          ),
                          const SizedBox(width: 12),
                          Column(
                            crossAxisAlignment: CrossAxisAlignment.start,
                            children: [
                              Text(
                                'CREATOR STUDIO',
                                style: Theme.of(context).textTheme.labelSmall?.copyWith(
                                  letterSpacing: 1.2,
                                  fontWeight: FontWeight.bold,
                                  color: Colors.red,
                                ),
                              ),
                              Text(
                                'Brain Orchestration',
                                style: Theme.of(context).textTheme.titleLarge?.copyWith(
                                  fontWeight: FontWeight.w900,
                                ),
                              ),
                            ],
                          ),
                        ],
                      ),
                      const Badge(
                        label: Text('LIVE'),
                        backgroundColor: Colors.red,
                      ),
                    ],
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'Context Control',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                  ),
                  const SizedBox(height: 12),
                  BlocBuilder<RobotContextBloc, RobotContextState>(
                    builder: (context, state) {
                      return Wrap(
                        spacing: 8,
                        runSpacing: 8,
                        children: RobotMode.values.map((mode) {
                          final isSelected = state.currentMode == mode;
                          return AnimatedContainer(
                            duration: const Duration(milliseconds: 200),
                            child: FilterChip(
                              showCheckmark: false,
                              avatar: isSelected ? const Icon(Icons.check, size: 16) : null,
                              label: Text(mode.value.toUpperCase().replaceAll('_', ' ')),
                              selected: isSelected,
                              onSelected: (selected) {
                                if (selected) {
                                  context.read<RobotContextBloc>().add(RobotModeUpdateRequested(mode));
                                }
                              },
                              selectedColor: Colors.red.withOpacity(0.2),
                              labelStyle: TextStyle(
                                fontSize: 10,
                                fontWeight: isSelected ? FontWeight.bold : FontWeight.normal,
                                color: isSelected ? Colors.red : null,
                              ),
                            ),
                          );
                        }).toList(),
                      );
                    },
                  ),
                  const SizedBox(height: 24),
                  Text(
                    'Cognition Feed',
                    style: Theme.of(context).textTheme.titleSmall?.copyWith(
                      fontWeight: FontWeight.bold,
                      color: Theme.of(context).colorScheme.primary,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Container(
                    height: 120,
                    decoration: BoxDecoration(
                      color: Colors.black.withOpacity(0.05),
                      borderRadius: BorderRadius.circular(12),
                    ),
                    child: BlocBuilder<BrainThoughtBloc, BrainThoughtState>(
                      builder: (context, state) {
                        if (state.thoughts.isEmpty) {
                          return const Center(
                            child: Text(
                              'Awaiting brain signals...',
                              style: TextStyle(fontSize: 10, fontStyle: FontStyle.italic),
                            ),
                          );
                        }
                        return ListView.builder(
                          padding: const EdgeInsets.all(8),
                          itemCount: state.thoughts.length,
                          itemBuilder: (context, index) {
                            final t = state.thoughts[index];
                            return Padding(
                              padding: const EdgeInsets.only(bottom: 4.0),
                              child: Text(
                                '>> ${t['message'] ?? t['status']?['mode'] ?? t.toString()}',
                                style: const TextStyle(
                                  fontFamily: 'Courier',
                                  fontSize: 10,
                                  color: Colors.green,
                                ),
                              ),
                            );
                          },
                        );
                      },
                    ),
                  ),
                  const SizedBox(height: 24),
                  Row(
                    children: [
                      Expanded(
                        child: _StudioActionCard(
                          icon: Icons.upload_file,
                          label: 'Upload Seed',
                          onTap: () {},
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _StudioActionCard(
                          icon: Icons.psychology,
                          label: 'Model Manager',
                          onTap: () {
                            Navigator.pushNamed(context, ModelManagerScreen.routeName);
                          },
                        ),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: _StudioActionCard(
                          icon: Icons.auto_graph,
                          label: 'HOPE Training',
                          onTap: () {
                             context.read<RobotContextBloc>().add(const RobotModeUpdateRequested(RobotMode.autonomous));
                          },
                        ),
                      ),
                    ],
                  ),
                ],
              ),
            ),
          ),
        );
      },
    );
  }
}

class _StudioActionCard extends StatelessWidget {
  final IconData icon;
  final String label;
  final VoidCallback onTap;

  const _StudioActionCard({
    required this.icon,
    required this.label,
    required this.onTap,
  });

  @override
  Widget build(BuildContext context) {
    return InkWell(
      onTap: onTap,
      borderRadius: BorderRadius.circular(12),
      child: Container(
        padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 12),
        decoration: BoxDecoration(
          border: Border.all(color: Theme.of(context).dividerColor),
          borderRadius: BorderRadius.circular(12),
        ),
        child: Column(
          children: [
            Icon(icon, size: 28),
            const SizedBox(height: 8),
            Text(
              label,
              style: const TextStyle(fontSize: 12, fontWeight: FontWeight.bold),
            ),
          ],
        ),
      ),
    );
  }
}
