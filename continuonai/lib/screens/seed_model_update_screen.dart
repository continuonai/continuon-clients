import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../blocs/ota/ota_bloc.dart';
import '../blocs/ota/ota_event.dart';
import '../blocs/ota/ota_state.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';

/// OTA Seed Model Update Screen
/// Allows users to check, download, activate, and rollback model updates
class SeedModelUpdateScreen extends StatefulWidget {
  static const routeName = '/seed-model-update';

  final BrainClient brainClient;

  const SeedModelUpdateScreen({
    super.key,
    required this.brainClient,
  });

  @override
  State<SeedModelUpdateScreen> createState() => _SeedModelUpdateScreenState();
}

class _SeedModelUpdateScreenState extends State<SeedModelUpdateScreen> {
  bool _runHealthCheck = true;

  @override
  void initState() {
    super.initState();
    // Fetch initial status
    WidgetsBinding.instance.addPostFrameCallback((_) {
      context.read<OTABloc>().add(const FetchOTAStatus());
    });
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Seed Model Updates'),
        actions: [
          BlocBuilder<OTABloc, OTAState>(
            builder: (context, state) {
              return IconButton(
                icon: state.status == OTAStateStatus.checking
                    ? SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: theme.colorScheme.onSurface,
                        ),
                      )
                    : const Icon(Icons.refresh),
                onPressed: state.isProcessing
                    ? null
                    : () => context.read<OTABloc>().add(const CheckForUpdates()),
                tooltip: 'Check for updates',
              );
            },
          ),
        ],
      ),
      body: BlocConsumer<OTABloc, OTAState>(
        listener: (context, state) {
          if (state.error != null) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(state.error!),
                backgroundColor: theme.colorScheme.error,
                action: SnackBarAction(
                  label: 'Dismiss',
                  textColor: theme.colorScheme.onError,
                  onPressed: () =>
                      context.read<OTABloc>().add(const ClearOTAError()),
                ),
              ),
            );
          }
        },
        builder: (context, state) {
          return RefreshIndicator(
            onRefresh: () async {
              context.read<OTABloc>().add(const CheckForUpdates());
              await Future.delayed(const Duration(seconds: 1));
            },
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(ContinuonTokens.s16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Current Version Card
                  _buildCurrentVersionCard(context, state, isDark),
                  const SizedBox(height: ContinuonTokens.s16),

                  // Update Available Card (if applicable)
                  if (state.updateAvailable && state.availableUpdate != null)
                    _buildUpdateAvailableCard(context, state, isDark),

                  // No Update Card
                  if (!state.updateAvailable &&
                      state.status != OTAStateStatus.checking &&
                      state.status != OTAStateStatus.initial)
                    _buildNoUpdateCard(context, isDark),

                  // Download Progress Card
                  if (state.status == OTAStateStatus.downloading)
                    _buildProgressCard(
                      context,
                      isDark,
                      title: 'Downloading Update',
                      progress: state.downloadProgress,
                      statusText: state.updateStatus.state.displayName,
                    ),

                  // Activation Progress Card
                  if (state.status == OTAStateStatus.activating)
                    _buildProgressCard(
                      context,
                      isDark,
                      title: 'Activating Update',
                      progress: state.updateStatus.progressPercent / 100,
                      statusText: state.updateStatus.state.displayName,
                      isIndeterminate: true,
                    ),

                  // Rollback Progress Card
                  if (state.status == OTAStateStatus.rollingBack)
                    _buildProgressCard(
                      context,
                      isDark,
                      title: 'Rolling Back',
                      progress: 0,
                      statusText: 'Reverting to previous version...',
                      isIndeterminate: true,
                    ),

                  const SizedBox(height: ContinuonTokens.s24),

                  // Rollback Option
                  if (state.rollbackAvailable &&
                      !state.isProcessing &&
                      state.updateStatus.installedVersions.rollback != null)
                    _buildRollbackCard(context, state, isDark),

                  const SizedBox(height: ContinuonTokens.s24),

                  // Version History
                  _buildVersionHistoryCard(context, state, isDark),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildCurrentVersionCard(
    BuildContext context,
    OTAState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final currentVersion = state.currentVersion ?? 'Unknown';

    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : ContinuonColors.brandWhite,
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 48,
                height: 48,
                decoration: BoxDecoration(
                  color: ContinuonColors.primaryBlue.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(ContinuonTokens.r12),
                ),
                child: const Icon(
                  Icons.memory,
                  color: ContinuonColors.primaryBlue,
                  size: 24,
                ),
              ),
              const SizedBox(width: ContinuonTokens.s12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Current Seed Model',
                      style: theme.textTheme.bodyMedium?.copyWith(
                        color: isDark
                            ? ContinuonColors.gray400
                            : ContinuonColors.gray500,
                      ),
                    ),
                    const SizedBox(height: ContinuonTokens.s4),
                    Text(
                      'v$currentVersion',
                      style: theme.textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                        color: isDark
                            ? ContinuonColors.brandWhite
                            : ContinuonColors.brandBlack,
                      ),
                    ),
                  ],
                ),
              ),
              _buildStatusBadge(context, state, isDark),
            ],
          ),
          if (state.lastChecked != null) ...[
            const SizedBox(height: ContinuonTokens.s12),
            Text(
              'Last checked: ${_formatDateTime(state.lastChecked!)}',
              style: theme.textTheme.bodySmall?.copyWith(
                color:
                    isDark ? ContinuonColors.gray500 : ContinuonColors.gray400,
              ),
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildStatusBadge(
    BuildContext context,
    OTAState state,
    bool isDark,
  ) {
    Color bgColor;
    Color textColor;
    String text;
    IconData icon;

    if (state.updateAvailable) {
      bgColor = ContinuonColors.particleOrange.withValues(alpha: 0.15);
      textColor = ContinuonColors.particleOrange;
      text = 'Update Available';
      icon = Icons.arrow_upward;
    } else if (state.isProcessing) {
      bgColor = ContinuonColors.primaryBlue.withValues(alpha: 0.15);
      textColor = ContinuonColors.primaryBlue;
      text = 'Updating';
      icon = Icons.sync;
    } else {
      bgColor = Colors.green.withValues(alpha: 0.15);
      textColor = Colors.green;
      text = 'Up to date';
      icon = Icons.check_circle;
    }

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: ContinuonTokens.s12,
        vertical: ContinuonTokens.s8,
      ),
      decoration: BoxDecoration(
        color: bgColor,
        borderRadius: BorderRadius.circular(ContinuonTokens.rFull),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(icon, size: 14, color: textColor),
          const SizedBox(width: ContinuonTokens.s4),
          Text(
            text,
            style: TextStyle(
              color: textColor,
              fontSize: 12,
              fontWeight: FontWeight.w600,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildUpdateAvailableCard(
    BuildContext context,
    OTAState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final update = state.availableUpdate!;

    return Container(
      margin: const EdgeInsets.only(bottom: ContinuonTokens.s16),
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            ContinuonColors.primaryBlue.withValues(alpha: 0.1),
            ContinuonColors.cmsViolet.withValues(alpha: 0.1),
          ],
        ),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: ContinuonColors.primaryBlue.withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: ContinuonColors.primaryBlue,
                  borderRadius: BorderRadius.circular(ContinuonTokens.r8),
                ),
                child: const Icon(
                  Icons.system_update,
                  color: ContinuonColors.brandWhite,
                  size: 20,
                ),
              ),
              const SizedBox(width: ContinuonTokens.s12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'New Version Available',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                        color: isDark
                            ? ContinuonColors.brandWhite
                            : ContinuonColors.brandBlack,
                      ),
                    ),
                    Text(
                      'v${update.version}',
                      style: theme.textTheme.bodyLarge?.copyWith(
                        color: ContinuonColors.primaryBlue,
                        fontWeight: FontWeight.w500,
                      ),
                    ),
                  ],
                ),
              ),
              if (update.priority == 'critical')
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: ContinuonTokens.s8,
                    vertical: ContinuonTokens.s4,
                  ),
                  decoration: BoxDecoration(
                    color: theme.colorScheme.error,
                    borderRadius: BorderRadius.circular(ContinuonTokens.r4),
                  ),
                  child: Text(
                    'Critical',
                    style: TextStyle(
                      color: theme.colorScheme.onError,
                      fontSize: 10,
                      fontWeight: FontWeight.w600,
                    ),
                  ),
                ),
            ],
          ),
          if (update.releaseNotes != null) ...[
            const SizedBox(height: ContinuonTokens.s12),
            Container(
              padding: const EdgeInsets.all(ContinuonTokens.s12),
              decoration: BoxDecoration(
                color: isDark
                    ? ContinuonColors.brandBlack.withValues(alpha: 0.3)
                    : ContinuonColors.brandWhite.withValues(alpha: 0.7),
                borderRadius: BorderRadius.circular(ContinuonTokens.r8),
              ),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Text(
                    'Release Notes',
                    style: theme.textTheme.labelSmall?.copyWith(
                      color: isDark
                          ? ContinuonColors.gray400
                          : ContinuonColors.gray500,
                    ),
                  ),
                  const SizedBox(height: ContinuonTokens.s4),
                  Text(
                    update.releaseNotes!,
                    style: theme.textTheme.bodySmall?.copyWith(
                      color: isDark
                          ? ContinuonColors.gray200
                          : ContinuonColors.gray700,
                    ),
                  ),
                ],
              ),
            ),
          ],
          const SizedBox(height: ContinuonTokens.s12),
          Row(
            children: [
              if (update.sizeBytes != null)
                _buildInfoChip(
                  Icons.storage,
                  update.formattedSize,
                  isDark,
                ),
              if (update.releaseDate != null) ...[
                const SizedBox(width: ContinuonTokens.s8),
                _buildInfoChip(
                  Icons.calendar_today,
                  _formatDate(update.releaseDate!),
                  isDark,
                ),
              ],
            ],
          ),
          const SizedBox(height: ContinuonTokens.s16),

          // Health Check Toggle
          Row(
            children: [
              Checkbox(
                value: _runHealthCheck,
                onChanged: (value) {
                  setState(() => _runHealthCheck = value ?? true);
                },
                activeColor: ContinuonColors.primaryBlue,
              ),
              Expanded(
                child: GestureDetector(
                  onTap: () {
                    setState(() => _runHealthCheck = !_runHealthCheck);
                  },
                  child: Text(
                    'Run health check after activation',
                    style: theme.textTheme.bodyMedium?.copyWith(
                      color: isDark
                          ? ContinuonColors.gray200
                          : ContinuonColors.gray700,
                    ),
                  ),
                ),
              ),
            ],
          ),
          const SizedBox(height: ContinuonTokens.s12),

          // Action Buttons
          Row(
            children: [
              Expanded(
                child: ElevatedButton.icon(
                  onPressed: state.isProcessing
                      ? null
                      : () {
                          context.read<OTABloc>().add(DownloadUpdate(
                                modelId: update.modelId,
                                version: update.version,
                              ));
                        },
                  icon: const Icon(Icons.download),
                  label: const Text('Download & Install'),
                  style: ElevatedButton.styleFrom(
                    backgroundColor: ContinuonColors.primaryBlue,
                    foregroundColor: ContinuonColors.brandWhite,
                    padding: const EdgeInsets.symmetric(
                      vertical: ContinuonTokens.s12,
                    ),
                  ),
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildNoUpdateCard(BuildContext context, bool isDark) {
    final theme = Theme.of(context);

    return Container(
      margin: const EdgeInsets.only(bottom: ContinuonTokens.s16),
      padding: const EdgeInsets.all(ContinuonTokens.s24),
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : ContinuonColors.brandWhite,
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: Colors.green.withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        children: [
          Container(
            width: 64,
            height: 64,
            decoration: BoxDecoration(
              color: Colors.green.withValues(alpha: 0.1),
              shape: BoxShape.circle,
            ),
            child: const Icon(
              Icons.check_circle_outline,
              color: Colors.green,
              size: 32,
            ),
          ),
          const SizedBox(height: ContinuonTokens.s16),
          Text(
            'Your seed model is up to date',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: isDark
                  ? ContinuonColors.brandWhite
                  : ContinuonColors.brandBlack,
            ),
          ),
          const SizedBox(height: ContinuonTokens.s8),
          Text(
            'Check back later for new updates',
            style: theme.textTheme.bodyMedium?.copyWith(
              color:
                  isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProgressCard(
    BuildContext context,
    bool isDark, {
    required String title,
    required double progress,
    required String statusText,
    bool isIndeterminate = false,
  }) {
    final theme = Theme.of(context);

    return Container(
      margin: const EdgeInsets.only(bottom: ContinuonTokens.s16),
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : ContinuonColors.brandWhite,
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: ContinuonColors.primaryBlue.withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              SizedBox(
                width: 24,
                height: 24,
                child: CircularProgressIndicator(
                  strokeWidth: 2.5,
                  value: isIndeterminate ? null : progress,
                  color: ContinuonColors.primaryBlue,
                ),
              ),
              const SizedBox(width: ContinuonTokens.s12),
              Expanded(
                child: Text(
                  title,
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: isDark
                        ? ContinuonColors.brandWhite
                        : ContinuonColors.brandBlack,
                  ),
                ),
              ),
              if (!isIndeterminate)
                Text(
                  '${(progress * 100).toInt()}%',
                  style: theme.textTheme.titleMedium?.copyWith(
                    fontWeight: FontWeight.w600,
                    color: ContinuonColors.primaryBlue,
                  ),
                ),
            ],
          ),
          const SizedBox(height: ContinuonTokens.s12),
          if (!isIndeterminate)
            ClipRRect(
              borderRadius: BorderRadius.circular(ContinuonTokens.r4),
              child: LinearProgressIndicator(
                value: progress,
                minHeight: 8,
                backgroundColor:
                    isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
                valueColor: const AlwaysStoppedAnimation<Color>(
                  ContinuonColors.primaryBlue,
                ),
              ),
            )
          else
            ClipRRect(
              borderRadius: BorderRadius.circular(ContinuonTokens.r4),
              child: LinearProgressIndicator(
                minHeight: 8,
                backgroundColor:
                    isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
                valueColor: const AlwaysStoppedAnimation<Color>(
                  ContinuonColors.primaryBlue,
                ),
              ),
            ),
          const SizedBox(height: ContinuonTokens.s8),
          Text(
            statusText,
            style: theme.textTheme.bodySmall?.copyWith(
              color:
                  isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildRollbackCard(
    BuildContext context,
    OTAState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final rollbackVersion = state.updateStatus.installedVersions.rollback;

    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : ContinuonColors.brandWhite,
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: ContinuonColors.particleOrange.withValues(alpha: 0.3),
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 40,
                height: 40,
                decoration: BoxDecoration(
                  color: ContinuonColors.particleOrange.withValues(alpha: 0.1),
                  borderRadius: BorderRadius.circular(ContinuonTokens.r8),
                ),
                child: const Icon(
                  Icons.history,
                  color: ContinuonColors.particleOrange,
                  size: 20,
                ),
              ),
              const SizedBox(width: ContinuonTokens.s12),
              Expanded(
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Text(
                      'Rollback Available',
                      style: theme.textTheme.titleMedium?.copyWith(
                        fontWeight: FontWeight.w600,
                        color: isDark
                            ? ContinuonColors.brandWhite
                            : ContinuonColors.brandBlack,
                      ),
                    ),
                    Text(
                      'Revert to v$rollbackVersion',
                      style: theme.textTheme.bodySmall?.copyWith(
                        color: isDark
                            ? ContinuonColors.gray400
                            : ContinuonColors.gray500,
                      ),
                    ),
                  ],
                ),
              ),
            ],
          ),
          const SizedBox(height: ContinuonTokens.s12),
          Text(
            'If you experience issues with the current version, you can roll back to the previous working version.',
            style: theme.textTheme.bodySmall?.copyWith(
              color:
                  isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
          const SizedBox(height: ContinuonTokens.s12),
          OutlinedButton.icon(
            onPressed: () => _showRollbackConfirmation(context),
            icon: const Icon(Icons.restore),
            label: const Text('Roll Back'),
            style: OutlinedButton.styleFrom(
              foregroundColor: ContinuonColors.particleOrange,
              side: const BorderSide(color: ContinuonColors.particleOrange),
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildVersionHistoryCard(
    BuildContext context,
    OTAState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final versions = state.updateStatus.installedVersions;

    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : ContinuonColors.brandWhite,
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
        ),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'Version History',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: isDark
                  ? ContinuonColors.brandWhite
                  : ContinuonColors.brandBlack,
            ),
          ),
          const SizedBox(height: ContinuonTokens.s16),
          _buildVersionRow(
            context,
            isDark,
            label: 'Current',
            version: versions.current ?? 'Unknown',
            isActive: true,
          ),
          if (versions.candidate != null) ...[
            const SizedBox(height: ContinuonTokens.s8),
            _buildVersionRow(
              context,
              isDark,
              label: 'Candidate',
              version: versions.candidate!,
              isPending: true,
            ),
          ],
          if (versions.rollback != null) ...[
            const SizedBox(height: ContinuonTokens.s8),
            _buildVersionRow(
              context,
              isDark,
              label: 'Rollback',
              version: versions.rollback!,
            ),
          ],
        ],
      ),
    );
  }

  Widget _buildVersionRow(
    BuildContext context,
    bool isDark, {
    required String label,
    required String version,
    bool isActive = false,
    bool isPending = false,
  }) {
    final theme = Theme.of(context);

    Color dotColor;
    if (isActive) {
      dotColor = Colors.green;
    } else if (isPending) {
      dotColor = ContinuonColors.particleOrange;
    } else {
      dotColor = ContinuonColors.gray500;
    }

    return Row(
      children: [
        Container(
          width: 8,
          height: 8,
          decoration: BoxDecoration(
            color: dotColor,
            shape: BoxShape.circle,
          ),
        ),
        const SizedBox(width: ContinuonTokens.s12),
        Expanded(
          child: Text(
            label,
            style: theme.textTheme.bodyMedium?.copyWith(
              color:
                  isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
        ),
        Text(
          'v$version',
          style: theme.textTheme.bodyMedium?.copyWith(
            fontWeight: isActive ? FontWeight.w600 : FontWeight.w400,
            color: isActive
                ? (isDark
                    ? ContinuonColors.brandWhite
                    : ContinuonColors.brandBlack)
                : (isDark
                    ? ContinuonColors.gray400
                    : ContinuonColors.gray500),
          ),
        ),
      ],
    );
  }

  Widget _buildInfoChip(IconData icon, String text, bool isDark) {
    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: ContinuonTokens.s8,
        vertical: ContinuonTokens.s4,
      ),
      decoration: BoxDecoration(
        color: isDark
            ? ContinuonColors.gray800
            : ContinuonColors.gray200.withValues(alpha: 0.5),
        borderRadius: BorderRadius.circular(ContinuonTokens.rFull),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Icon(
            icon,
            size: 12,
            color:
                isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
          ),
          const SizedBox(width: ContinuonTokens.s4),
          Text(
            text,
            style: TextStyle(
              fontSize: 11,
              color:
                  isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
        ],
      ),
    );
  }

  void _showRollbackConfirmation(BuildContext context) {
    final theme = Theme.of(context);

    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Confirm Rollback'),
        content: const Text(
          'This will revert your seed model to the previous version. '
          'Any learned behaviors from the current version may be affected.\n\n'
          'Are you sure you want to continue?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.of(dialogContext).pop();
              context.read<OTABloc>().add(const RollbackUpdate());
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: ContinuonColors.particleOrange,
              foregroundColor: theme.colorScheme.onError,
            ),
            child: const Text('Roll Back'),
          ),
        ],
      ),
    );
  }

  String _formatDateTime(DateTime dateTime) {
    final now = DateTime.now();
    final difference = now.difference(dateTime);

    if (difference.inMinutes < 1) {
      return 'Just now';
    } else if (difference.inHours < 1) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inDays < 1) {
      return '${difference.inHours}h ago';
    } else {
      return '${dateTime.day}/${dateTime.month}/${dateTime.year}';
    }
  }

  String _formatDate(DateTime date) {
    final months = [
      'Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun',
      'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec',
    ];
    return '${months[date.month - 1]} ${date.day}, ${date.year}';
  }
}
