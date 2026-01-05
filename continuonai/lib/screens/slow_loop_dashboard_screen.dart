import 'package:fl_chart/fl_chart.dart';
import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';

import '../blocs/learning/learning_bloc.dart';
import '../blocs/learning/learning_event.dart';
import '../blocs/learning/learning_state.dart';
import '../models/learning_metrics.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';

/// Slow Loop Learning Dashboard
/// Full dashboard with metrics, charts, controls, and learning status
class SlowLoopDashboardScreen extends StatefulWidget {
  static const routeName = '/slow-loop-dashboard';

  final BrainClient brainClient;

  const SlowLoopDashboardScreen({
    super.key,
    required this.brainClient,
  });

  @override
  State<SlowLoopDashboardScreen> createState() =>
      _SlowLoopDashboardScreenState();
}

class _SlowLoopDashboardScreenState extends State<SlowLoopDashboardScreen> {
  @override
  void initState() {
    super.initState();
    WidgetsBinding.instance.addPostFrameCallback((_) {
      final bloc = context.read<LearningBloc>();
      bloc.add(const FetchLearningMetrics());
      bloc.add(const StartMetricsPolling());
    });
  }

  @override
  void dispose() {
    // Stop polling when leaving screen
    context.read<LearningBloc>().add(const StopMetricsPolling());
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Scaffold(
      appBar: AppBar(
        title: const Text('Slow Loop Learning'),
        actions: [
          BlocBuilder<LearningBloc, LearningState>(
            builder: (context, state) {
              return IconButton(
                icon: state.status == LearningStateStatus.loading
                    ? SizedBox(
                        width: 20,
                        height: 20,
                        child: CircularProgressIndicator(
                          strokeWidth: 2,
                          color: theme.colorScheme.onSurface,
                        ),
                      )
                    : const Icon(Icons.refresh),
                onPressed: () =>
                    context.read<LearningBloc>().add(const FetchLearningMetrics()),
                tooltip: 'Refresh metrics',
              );
            },
          ),
        ],
      ),
      body: BlocConsumer<LearningBloc, LearningState>(
        listener: (context, state) {
          if (state.error != null) {
            ScaffoldMessenger.of(context).showSnackBar(
              SnackBar(
                content: Text(state.error!),
                backgroundColor: theme.colorScheme.error,
              ),
            );
          }
        },
        builder: (context, state) {
          return RefreshIndicator(
            onRefresh: () async {
              context.read<LearningBloc>().add(const FetchLearningMetrics());
              await Future.delayed(const Duration(milliseconds: 500));
            },
            child: SingleChildScrollView(
              physics: const AlwaysScrollableScrollPhysics(),
              padding: const EdgeInsets.all(ContinuonTokens.s16),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.stretch,
                children: [
                  // Status Header
                  _buildStatusHeader(context, state, isDark),
                  const SizedBox(height: ContinuonTokens.s16),

                  // Episode Counter
                  _buildEpisodeCounter(context, state, isDark),
                  const SizedBox(height: ContinuonTokens.s16),

                  // Curiosity & Surprise Gauges
                  _buildMetricGauges(context, state, isDark),
                  const SizedBox(height: ContinuonTokens.s16),

                  // Loss Curve Chart
                  _buildLossCurveCard(context, state, isDark),
                  const SizedBox(height: ContinuonTokens.s16),

                  // Learning Rate Display
                  _buildLearningRateCard(context, state, isDark),
                  const SizedBox(height: ContinuonTokens.s16),

                  // Control Buttons
                  _buildControlButtons(context, state, isDark),
                  const SizedBox(height: ContinuonTokens.s24),

                  // Quick Stats
                  _buildQuickStats(context, state, isDark),
                ],
              ),
            ),
          );
        },
      ),
    );
  }

  Widget _buildStatusHeader(
    BuildContext context,
    LearningState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final status = state.metrics.status;

    Color statusColor;
    String statusText;
    IconData statusIcon;

    if (!status.enabled) {
      statusColor = ContinuonColors.gray500;
      statusText = 'Disabled';
      statusIcon = Icons.block;
    } else if (state.isPaused || status.isPaused) {
      statusColor = ContinuonColors.particleOrange;
      statusText = 'Paused';
      statusIcon = Icons.pause_circle;
    } else if (status.running || state.isLearning) {
      statusColor = Colors.green;
      statusText = 'Learning';
      statusIcon = Icons.play_circle;
    } else {
      statusColor = ContinuonColors.gray500;
      statusText = 'Idle';
      statusIcon = Icons.circle_outlined;
    }

    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        gradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            statusColor.withValues(alpha: 0.15),
            statusColor.withValues(alpha: 0.05),
          ],
        ),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: statusColor.withValues(alpha: 0.3),
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 56,
            height: 56,
            decoration: BoxDecoration(
              color: statusColor.withValues(alpha: 0.2),
              borderRadius: BorderRadius.circular(ContinuonTokens.r12),
            ),
            child: Icon(
              statusIcon,
              color: statusColor,
              size: 28,
            ),
          ),
          const SizedBox(width: ContinuonTokens.s16),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Slow Loop Status',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: isDark
                        ? ContinuonColors.gray400
                        : ContinuonColors.gray500,
                  ),
                ),
                const SizedBox(height: ContinuonTokens.s4),
                Row(
                  children: [
                    Text(
                      statusText,
                      style: theme.textTheme.headlineSmall?.copyWith(
                        fontWeight: FontWeight.w600,
                        color: statusColor,
                      ),
                    ),
                    if (status.currentPhase != null) ...[
                      const SizedBox(width: ContinuonTokens.s8),
                      Container(
                        padding: const EdgeInsets.symmetric(
                          horizontal: ContinuonTokens.s8,
                          vertical: ContinuonTokens.s4,
                        ),
                        decoration: BoxDecoration(
                          color: isDark
                              ? ContinuonColors.gray800
                              : ContinuonColors.gray200,
                          borderRadius:
                              BorderRadius.circular(ContinuonTokens.rFull),
                        ),
                        child: Text(
                          status.currentPhase!,
                          style: TextStyle(
                            fontSize: 10,
                            fontWeight: FontWeight.w500,
                            color: isDark
                                ? ContinuonColors.gray400
                                : ContinuonColors.gray500,
                          ),
                        ),
                      ),
                    ],
                  ],
                ),
              ],
            ),
          ),
          _buildPulsingIndicator(status.running || state.isLearning, statusColor),
        ],
      ),
    );
  }

  Widget _buildPulsingIndicator(bool isActive, Color color) {
    if (!isActive) {
      return Container(
        width: 12,
        height: 12,
        decoration: BoxDecoration(
          color: color.withValues(alpha: 0.3),
          shape: BoxShape.circle,
        ),
      );
    }

    return TweenAnimationBuilder<double>(
      tween: Tween(begin: 0.5, end: 1.0),
      duration: const Duration(milliseconds: 1000),
      builder: (context, value, child) {
        return Container(
          width: 12,
          height: 12,
          decoration: BoxDecoration(
            color: color.withValues(alpha: value),
            shape: BoxShape.circle,
            boxShadow: [
              BoxShadow(
                color: color.withValues(alpha: value * 0.5),
                blurRadius: 8,
                spreadRadius: 2,
              ),
            ],
          ),
        );
      },
      onEnd: () {
        // Rebuild to restart animation
        if (mounted) setState(() {});
      },
    );
  }

  Widget _buildEpisodeCounter(
    BuildContext context,
    LearningState state,
    bool isDark,
  ) {
    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : ContinuonColors.brandWhite,
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
        ),
      ),
      child: Row(
        children: [
          Expanded(
            child: _buildCounterItem(
              context,
              isDark,
              icon: Icons.replay,
              label: 'Episodes',
              value: _formatNumber(state.totalEpisodes),
              color: ContinuonColors.primaryBlue,
            ),
          ),
          Container(
            width: 1,
            height: 48,
            color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
          ),
          Expanded(
            child: _buildCounterItem(
              context,
              isDark,
              icon: Icons.timeline,
              label: 'Total Steps',
              value: _formatNumber(state.totalSteps),
              color: ContinuonColors.cmsViolet,
            ),
          ),
          Container(
            width: 1,
            height: 48,
            color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
          ),
          Expanded(
            child: _buildCounterItem(
              context,
              isDark,
              icon: Icons.sync,
              label: 'Updates',
              value: _formatNumber(state.metrics.progress.learningUpdates),
              color: Colors.green,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildCounterItem(
    BuildContext context,
    bool isDark, {
    required IconData icon,
    required String label,
    required String value,
    required Color color,
  }) {
    final theme = Theme.of(context);

    return Column(
      children: [
        Icon(icon, color: color, size: 20),
        const SizedBox(height: ContinuonTokens.s8),
        Text(
          value,
          style: theme.textTheme.headlineSmall?.copyWith(
            fontWeight: FontWeight.w700,
            color: isDark ? ContinuonColors.brandWhite : ContinuonColors.brandBlack,
          ),
        ),
        const SizedBox(height: ContinuonTokens.s4),
        Text(
          label,
          style: theme.textTheme.bodySmall?.copyWith(
            color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
          ),
        ),
      ],
    );
  }

  Widget _buildMetricGauges(
    BuildContext context,
    LearningState state,
    bool isDark,
  ) {
    return Row(
      children: [
        Expanded(
          child: _buildGaugeCard(
            context,
            isDark,
            label: 'Curiosity',
            value: state.curiosity,
            color: ContinuonColors.primaryBlue,
            icon: Icons.explore,
          ),
        ),
        const SizedBox(width: ContinuonTokens.s12),
        Expanded(
          child: _buildGaugeCard(
            context,
            isDark,
            label: 'Surprise',
            value: state.surprise,
            color: ContinuonColors.particleOrange,
            icon: Icons.lightbulb,
          ),
        ),
      ],
    );
  }

  Widget _buildGaugeCard(
    BuildContext context,
    bool isDark, {
    required String label,
    required double value,
    required Color color,
    required IconData icon,
  }) {
    final theme = Theme.of(context);
    final clampedValue = value.clamp(0.0, 1.0);

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
              Icon(icon, color: color, size: 18),
              const SizedBox(width: ContinuonTokens.s8),
              Text(
                label,
                style: theme.textTheme.bodyMedium?.copyWith(
                  fontWeight: FontWeight.w500,
                  color: isDark
                      ? ContinuonColors.gray200
                      : ContinuonColors.gray700,
                ),
              ),
            ],
          ),
          const SizedBox(height: ContinuonTokens.s12),
          Row(
            children: [
              Expanded(
                child: ClipRRect(
                  borderRadius: BorderRadius.circular(ContinuonTokens.r4),
                  child: LinearProgressIndicator(
                    value: clampedValue,
                    minHeight: 8,
                    backgroundColor:
                        isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
                    valueColor: AlwaysStoppedAnimation<Color>(color),
                  ),
                ),
              ),
              const SizedBox(width: ContinuonTokens.s12),
              Text(
                '${(clampedValue * 100).toInt()}%',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: color,
                ),
              ),
            ],
          ),
        ],
      ),
    );
  }

  Widget _buildLossCurveCard(
    BuildContext context,
    LearningState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final lossHistory = state.lossHistory;

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
            mainAxisAlignment: MainAxisAlignment.spaceBetween,
            children: [
              Text(
                'Loss Curve',
                style: theme.textTheme.titleMedium?.copyWith(
                  fontWeight: FontWeight.w600,
                  color: isDark
                      ? ContinuonColors.brandWhite
                      : ContinuonColors.brandBlack,
                ),
              ),
              if (lossHistory.isNotEmpty)
                Container(
                  padding: const EdgeInsets.symmetric(
                    horizontal: ContinuonTokens.s8,
                    vertical: ContinuonTokens.s4,
                  ),
                  decoration: BoxDecoration(
                    color: isDark
                        ? ContinuonColors.gray800
                        : ContinuonColors.gray200,
                    borderRadius: BorderRadius.circular(ContinuonTokens.rFull),
                  ),
                  child: Text(
                    'Latest: ${lossHistory.last.value.toStringAsFixed(4)}',
                    style: TextStyle(
                      fontSize: 11,
                      fontWeight: FontWeight.w500,
                      color: isDark
                          ? ContinuonColors.gray400
                          : ContinuonColors.gray500,
                    ),
                  ),
                ),
            ],
          ),
          const SizedBox(height: ContinuonTokens.s16),
          SizedBox(
            height: 200,
            child: lossHistory.isEmpty
                ? _buildEmptyChartPlaceholder(context, isDark)
                : _buildLossChart(context, lossHistory, isDark),
          ),
        ],
      ),
    );
  }

  Widget _buildEmptyChartPlaceholder(BuildContext context, bool isDark) {
    final theme = Theme.of(context);

    return Center(
      child: Column(
        mainAxisAlignment: MainAxisAlignment.center,
        children: [
          Icon(
            Icons.show_chart,
            size: 48,
            color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
          ),
          const SizedBox(height: ContinuonTokens.s12),
          Text(
            'No loss data available yet',
            style: theme.textTheme.bodyMedium?.copyWith(
              color: isDark ? ContinuonColors.gray500 : ContinuonColors.gray400,
            ),
          ),
          Text(
            'Data will appear as the model learns',
            style: theme.textTheme.bodySmall?.copyWith(
              color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildLossChart(
    BuildContext context,
    List<MetricPoint> lossHistory,
    bool isDark,
  ) {
    final spots = lossHistory.asMap().entries.map((entry) {
      return FlSpot(entry.key.toDouble(), entry.value.value);
    }).toList();

    final maxY = lossHistory.map((p) => p.value).reduce((a, b) => a > b ? a : b);
    final minY = lossHistory.map((p) => p.value).reduce((a, b) => a < b ? a : b);
    final yPadding = (maxY - minY) * 0.1;

    return LineChart(
      LineChartData(
        gridData: FlGridData(
          show: true,
          drawVerticalLine: false,
          horizontalInterval: (maxY - minY) / 4,
          getDrawingHorizontalLine: (value) {
            return FlLine(
              color: isDark ? ContinuonColors.gray800 : ContinuonColors.gray200,
              strokeWidth: 1,
            );
          },
        ),
        titlesData: FlTitlesData(
          leftTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 40,
              getTitlesWidget: (value, meta) {
                return Text(
                  value.toStringAsFixed(3),
                  style: TextStyle(
                    color: isDark
                        ? ContinuonColors.gray500
                        : ContinuonColors.gray400,
                    fontSize: 10,
                  ),
                );
              },
            ),
          ),
          bottomTitles: AxisTitles(
            sideTitles: SideTitles(
              showTitles: true,
              reservedSize: 22,
              interval: (spots.length / 5).ceilToDouble(),
              getTitlesWidget: (value, meta) {
                return Text(
                  value.toInt().toString(),
                  style: TextStyle(
                    color: isDark
                        ? ContinuonColors.gray500
                        : ContinuonColors.gray400,
                    fontSize: 10,
                  ),
                );
              },
            ),
          ),
          topTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
          rightTitles: const AxisTitles(
            sideTitles: SideTitles(showTitles: false),
          ),
        ),
        borderData: FlBorderData(show: false),
        minY: minY - yPadding,
        maxY: maxY + yPadding,
        lineBarsData: [
          LineChartBarData(
            spots: spots,
            isCurved: true,
            curveSmoothness: 0.2,
            color: ContinuonColors.primaryBlue,
            barWidth: 2,
            isStrokeCapRound: true,
            dotData: const FlDotData(show: false),
            belowBarData: BarAreaData(
              show: true,
              color: ContinuonColors.primaryBlue.withValues(alpha: 0.1),
            ),
          ),
        ],
        lineTouchData: LineTouchData(
          touchTooltipData: LineTouchTooltipData(
            getTooltipColor: (spot) => isDark
                ? ContinuonColors.gray800
                : ContinuonColors.brandWhite,
            getTooltipItems: (touchedSpots) {
              return touchedSpots.map((spot) {
                return LineTooltipItem(
                  'Step ${spot.x.toInt()}\nLoss: ${spot.y.toStringAsFixed(4)}',
                  TextStyle(
                    color: isDark
                        ? ContinuonColors.brandWhite
                        : ContinuonColors.brandBlack,
                    fontSize: 12,
                  ),
                );
              }).toList();
            },
          ),
        ),
      ),
    );
  }

  Widget _buildLearningRateCard(
    BuildContext context,
    LearningState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final lr = state.learningRate;

    return Container(
      padding: const EdgeInsets.all(ContinuonTokens.s16),
      decoration: BoxDecoration(
        color: isDark ? ContinuonColors.gray900 : ContinuonColors.brandWhite,
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
          color: isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
        ),
      ),
      child: Row(
        children: [
          Container(
            width: 40,
            height: 40,
            decoration: BoxDecoration(
              color: ContinuonColors.cmsViolet.withValues(alpha: 0.1),
              borderRadius: BorderRadius.circular(ContinuonTokens.r8),
            ),
            child: const Icon(
              Icons.speed,
              color: ContinuonColors.cmsViolet,
              size: 20,
            ),
          ),
          const SizedBox(width: ContinuonTokens.s12),
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(
                  'Learning Rate',
                  style: theme.textTheme.bodyMedium?.copyWith(
                    color: isDark
                        ? ContinuonColors.gray400
                        : ContinuonColors.gray500,
                  ),
                ),
                const SizedBox(height: ContinuonTokens.s4),
                Text(
                  lr > 0 ? lr.toStringAsExponential(2) : 'N/A',
                  style: theme.textTheme.titleLarge?.copyWith(
                    fontWeight: FontWeight.w600,
                    fontFamily: 'monospace',
                    color: isDark
                        ? ContinuonColors.brandWhite
                        : ContinuonColors.brandBlack,
                  ),
                ),
              ],
            ),
          ),
          Container(
            padding: const EdgeInsets.symmetric(
              horizontal: ContinuonTokens.s8,
              vertical: ContinuonTokens.s4,
            ),
            decoration: BoxDecoration(
              color: state.metrics.progress.isStable
                  ? Colors.green.withValues(alpha: 0.15)
                  : ContinuonColors.particleOrange.withValues(alpha: 0.15),
              borderRadius: BorderRadius.circular(ContinuonTokens.rFull),
            ),
            child: Row(
              mainAxisSize: MainAxisSize.min,
              children: [
                Icon(
                  state.metrics.progress.isStable
                      ? Icons.check_circle
                      : Icons.warning,
                  size: 12,
                  color: state.metrics.progress.isStable
                      ? Colors.green
                      : ContinuonColors.particleOrange,
                ),
                const SizedBox(width: ContinuonTokens.s4),
                Text(
                  state.metrics.progress.isStable ? 'Stable' : 'Unstable',
                  style: TextStyle(
                    fontSize: 11,
                    fontWeight: FontWeight.w600,
                    color: state.metrics.progress.isStable
                        ? Colors.green
                        : ContinuonColors.particleOrange,
                  ),
                ),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildControlButtons(
    BuildContext context,
    LearningState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final isActive = state.isLearning && !state.isPaused;

    return Row(
      children: [
        Expanded(
          child: ElevatedButton.icon(
            onPressed: () {
              if (isActive) {
                context.read<LearningBloc>().add(const PauseLearning());
              } else {
                context.read<LearningBloc>().add(const ResumeLearning());
              }
            },
            icon: Icon(isActive ? Icons.pause : Icons.play_arrow),
            label: Text(isActive ? 'Pause Learning' : 'Resume Learning'),
            style: ElevatedButton.styleFrom(
              backgroundColor: isActive
                  ? ContinuonColors.particleOrange
                  : ContinuonColors.primaryBlue,
              foregroundColor: ContinuonColors.brandWhite,
              padding: const EdgeInsets.symmetric(
                vertical: ContinuonTokens.s12,
              ),
            ),
          ),
        ),
        const SizedBox(width: ContinuonTokens.s12),
        OutlinedButton.icon(
          onPressed: () => _showResetConfirmation(context),
          icon: const Icon(Icons.restart_alt),
          label: const Text('Reset'),
          style: OutlinedButton.styleFrom(
            foregroundColor: theme.colorScheme.error,
            side: BorderSide(color: theme.colorScheme.error),
            padding: const EdgeInsets.symmetric(
              vertical: ContinuonTokens.s12,
              horizontal: ContinuonTokens.s16,
            ),
          ),
        ),
      ],
    );
  }

  Widget _buildQuickStats(
    BuildContext context,
    LearningState state,
    bool isDark,
  ) {
    final theme = Theme.of(context);
    final progress = state.metrics.progress;

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
            'Quick Stats',
            style: theme.textTheme.titleMedium?.copyWith(
              fontWeight: FontWeight.w600,
              color: isDark
                  ? ContinuonColors.brandWhite
                  : ContinuonColors.brandBlack,
            ),
          ),
          const SizedBox(height: ContinuonTokens.s12),
          _buildStatRow(
            context,
            isDark,
            label: 'Avg Parameter Change',
            value: progress.avgParameterChange.toStringAsExponential(2),
          ),
          _buildStatRow(
            context,
            isDark,
            label: 'Last Update',
            value: state.lastUpdated != null
                ? _formatTimeAgo(state.lastUpdated!)
                : 'N/A',
          ),
          if (state.metrics.wavecoreMetrics != null) ...[
            const SizedBox(height: ContinuonTokens.s8),
            const Divider(),
            const SizedBox(height: ContinuonTokens.s8),
            Text(
              'WaveCore Metrics',
              style: theme.textTheme.labelMedium?.copyWith(
                fontWeight: FontWeight.w600,
                color: ContinuonColors.cmsViolet,
              ),
            ),
            const SizedBox(height: ContinuonTokens.s8),
            ...state.metrics.wavecoreMetrics!.entries.take(4).map((entry) {
              return _buildStatRow(
                context,
                isDark,
                label: _formatLabel(entry.key),
                value: _formatMetricValue(entry.value),
              );
            }),
          ],
        ],
      ),
    );
  }

  Widget _buildStatRow(
    BuildContext context,
    bool isDark, {
    required String label,
    required String value,
  }) {
    final theme = Theme.of(context);

    return Padding(
      padding: const EdgeInsets.symmetric(vertical: ContinuonTokens.s4),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: theme.textTheme.bodyMedium?.copyWith(
              color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            ),
          ),
          Text(
            value,
            style: theme.textTheme.bodyMedium?.copyWith(
              fontWeight: FontWeight.w500,
              fontFamily: 'monospace',
              color: isDark
                  ? ContinuonColors.brandWhite
                  : ContinuonColors.brandBlack,
            ),
          ),
        ],
      ),
    );
  }

  void _showResetConfirmation(BuildContext context) {
    showDialog(
      context: context,
      builder: (dialogContext) => AlertDialog(
        title: const Text('Reset Learning'),
        content: const Text(
          'This will reset all learning progress and start fresh. '
          'Episode counts and metrics will be cleared.\n\n'
          'This action cannot be undone. Are you sure?',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.of(dialogContext).pop(),
            child: const Text('Cancel'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.of(dialogContext).pop();
              context.read<LearningBloc>().add(const ResetLearning());
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: Theme.of(context).colorScheme.error,
              foregroundColor: Theme.of(context).colorScheme.onError,
            ),
            child: const Text('Reset'),
          ),
        ],
      ),
    );
  }

  String _formatNumber(int number) {
    if (number >= 1000000) {
      return '${(number / 1000000).toStringAsFixed(1)}M';
    } else if (number >= 1000) {
      return '${(number / 1000).toStringAsFixed(1)}K';
    }
    return number.toString();
  }

  String _formatTimeAgo(DateTime time) {
    final difference = DateTime.now().difference(time);

    if (difference.inSeconds < 60) {
      return '${difference.inSeconds}s ago';
    } else if (difference.inMinutes < 60) {
      return '${difference.inMinutes}m ago';
    } else if (difference.inHours < 24) {
      return '${difference.inHours}h ago';
    } else {
      return '${difference.inDays}d ago';
    }
  }

  String _formatLabel(String key) {
    return key
        .replaceAll('_', ' ')
        .split(' ')
        .map((word) => word.isNotEmpty
            ? '${word[0].toUpperCase()}${word.substring(1)}'
            : '')
        .join(' ');
  }

  String _formatMetricValue(dynamic value) {
    if (value is double) {
      if (value.abs() < 0.001) {
        return value.toStringAsExponential(2);
      }
      return value.toStringAsFixed(4);
    } else if (value is int) {
      return _formatNumber(value);
    }
    return value.toString();
  }
}
