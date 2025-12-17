import 'dart:convert';

import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;

import '../theme/continuon_theme.dart';

/// Research page: "science + evidence" widgets that visualize HOPE/CMS/WaveCore/SSM
/// and symbolic search claims from the repo docs.
///
/// This is intentionally lightweight and self-contained so it can ship in the
/// Flutter app (web/iOS/Android/desktop) without introducing new chart deps.
import '../widgets/layout/continuon_layout.dart';
import '../widgets/layout/continuon_card.dart';

class ResearchScreen extends StatelessWidget {
  const ResearchScreen({super.key});

  static const routeName = '/research';

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final brand = theme.extension<ContinuonBrandExtension>();

    return ContinuonLayout(
      // No specific actions for Research screen, generic nav is fine
      body: Container(
        decoration:
            brand != null ? BoxDecoration(gradient: brand.waveGradient) : null,
        child: ListView(
          padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 16),
          children: const [
            _ResearchHero(),
            SizedBox(height: 16),
            _LiveRuntimeEvidenceSection(),
            SizedBox(height: 16),
            _HypothesisEvidenceSection(),
            SizedBox(height: 16),
            _HopeCmsTimescalesSection(),
            SizedBox(height: 16),
            _WaveCoreSection(),
            SizedBox(height: 16),
            _SsmMambaSection(),
            SizedBox(height: 16),
            _SymbolicSearchSection(),
            SizedBox(height: 24),
          ],
        ),
      ),
    );
  }
}

class _ResearchHero extends StatelessWidget {
  const _ResearchHero();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return _ResearchCard(
      title: 'What we’re trying to prove',
      subtitle:
          'Measurements + visualizations tied to HOPE/CMS/WaveCore/SSM + symbolic search.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'This page is a living “evidence dashboard”: each widget maps a claim from the READMEs/AGENTS docs '
            'to a measurable signal (latency, eval lift, retention/compaction, search efficiency).',
            style: theme.textTheme.bodyMedium?.copyWith(height: 1.4),
          ),
          const SizedBox(height: 12),
          Wrap(
            spacing: 8,
            runSpacing: 8,
            children: const [
              _KpiPill(label: 'Fast loop latency', value: '≤100ms'),
              _KpiPill(label: 'RLDS validity', value: '≥95%'),
              _KpiPill(label: 'HOPE eval lift', value: '↑ over time'),
              _KpiPill(label: 'Search', value: 'plan ≠ recall'),
            ],
          ),
        ],
      ),
    );
  }
}

class _HypothesisEvidenceSection extends StatelessWidget {
  const _HypothesisEvidenceSection();

  @override
  Widget build(BuildContext context) {
    return _ResearchCard(
      title: 'Claims → Measurements → Evidence',
      subtitle:
          'Ammaba-style cards: each claim has an observable metric and a visual.',
      child: Column(
        children: const [
          _HypothesisCard(
            claim:
                'HOPE/CMS avoids catastrophic forgetting via nested fast/mid/slow updates.',
            metric: 'HOPE eval score stays stable while new skills are added',
            status: _EvidenceStatus.partial,
            series: [0.46, 0.51, 0.58, 0.61, 0.64, 0.66, 0.69],
            note:
                'Wire to `/api/training/eval_summary` once available in the connected runtime; this is sample data.',
          ),
          SizedBox(height: 12),
          _HypothesisCard(
            claim:
                'Edge runtime keeps reflex/teleop in-budget (50–100ms control ticks).',
            metric: 'p95 tick latency under threshold; safe-stop on overruns',
            status: _EvidenceStatus.supported,
            series: [42, 55, 61, 73, 88, 79, 64, 71, 58, 62],
            note: 'Sample p95 latency series (ms); threshold line is 100ms.',
            threshold: 100,
          ),
          SizedBox(height: 12),
          _HypothesisCard(
            claim:
                'Symbolic search invents plans using the world model (not pure imitation).',
            metric:
                'Search expands candidates then converges on a plan with higher goal score',
            status: _EvidenceStatus.partial,
            series: [0.10, 0.18, 0.33, 0.41, 0.55, 0.62, 0.71],
            note: 'Sample “best plan score by expansion step”.',
          ),
        ],
      ),
    );
  }
}

class _HopeCmsTimescalesSection extends StatelessWidget {
  const _HopeCmsTimescalesSection();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return _ResearchCard(
      title: 'HOPE + CMS: multi-timescale recurrence',
      subtitle:
          'Fast / Mid / Slow loops with distinct responsibilities + measurable budgets.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'CMS is the operational contract: fast loop for reflexes, mid loop for sequencing/world-model updates, '
            'slow loop for longer-horizon training and consolidation.',
            style: theme.textTheme.bodyMedium?.copyWith(height: 1.4),
          ),
          const SizedBox(height: 16),
          const _TimescaleBars(),
        ],
      ),
    );
  }
}

class _WaveCoreSection extends StatelessWidget {
  const _WaveCoreSection();

  @override
  Widget build(BuildContext context) {
    return _ResearchCard(
      title: 'WaveCore loops (fast/mid/slow)',
      subtitle: 'Training progression view (loss + sparsity + eval lift).',
      child: Column(
        children: const [
          _MiniMetricRow(
            label: 'Fast loop: safety/reflex overrides',
            unit: 'risk score',
            values: [0.62, 0.51, 0.44, 0.40, 0.37, 0.33],
            goodDirection: _GoodDirection.down,
          ),
          SizedBox(height: 12),
          _MiniMetricRow(
            label: 'Mid loop: adapter learning (bounded)',
            unit: 'loss',
            values: [1.20, 0.98, 0.83, 0.71, 0.66, 0.61],
            goodDirection: _GoodDirection.down,
          ),
          SizedBox(height: 12),
          _MiniMetricRow(
            label: 'Slow loop: consolidation (cloud)',
            unit: 'eval',
            values: [0.42, 0.48, 0.55, 0.57, 0.61, 0.63],
            goodDirection: _GoodDirection.up,
          ),
        ],
      ),
    );
  }
}

class _SsmMambaSection extends StatelessWidget {
  const _SsmMambaSection();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return _ResearchCard(
      title: 'SSM / “Mamba-style” wave dynamics',
      subtitle: 'Why linear-time state evolution matters for on-device memory.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'The claim: the wave path maintains compressed global state with linear-time recurrence, '
            'while the particle path handles local nonlinear updates. We visualize state evolution and '
            'a simple compute proxy vs sequence length.',
            style: theme.textTheme.bodyMedium?.copyWith(height: 1.4),
          ),
          const SizedBox(height: 16),
          const _TwoPanelCharts(),
        ],
      ),
    );
  }
}

class _SymbolicSearchSection extends StatelessWidget {
  const _SymbolicSearchSection();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return _ResearchCard(
      title: 'Symbolic search (planning) demo',
      subtitle: 'A small tree showing exploration → scoring → chosen plan.',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            'This is the “search” half of the hypothesis: the system proposes multiple futures, '
            'scores them against a goal, and commits the best plan.',
            style: theme.textTheme.bodyMedium?.copyWith(height: 1.4),
          ),
          const SizedBox(height: 16),
          const _SearchTreeDemo(),
        ],
      ),
    );
  }
}

class _ResearchCard extends StatelessWidget {
  const _ResearchCard({
    required this.title,
    required this.subtitle,
    required this.child,
  });

  final String title;
  final String subtitle;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final onCard = theme.colorScheme.onSurface;
    return ContinuonCard(
      padding: const EdgeInsets.all(20),
      backgroundColor: theme.cardColor.withOpacity(0.92),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(
            title,
            style: theme.textTheme.titleLarge?.copyWith(
              fontWeight: FontWeight.w700,
              color: onCard,
            ),
          ),
          const SizedBox(height: 6),
          Text(
            subtitle,
            style: theme.textTheme.bodySmall?.copyWith(
              height: 1.3,
              color: onCard.withValues(alpha: 0.75),
            ),
          ),
          const SizedBox(height: 16),
          child,
        ],
      ),
    );
  }
}

class _KpiPill extends StatelessWidget {
  const _KpiPill({required this.label, required this.value});

  final String label;
  final String value;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
      decoration: BoxDecoration(
        color: theme.colorScheme.primary.withValues(alpha: 0.08),
        borderRadius: BorderRadius.circular(999),
        border: Border.all(
            color: theme.colorScheme.primary.withValues(alpha: 0.22)),
      ),
      child: Row(
        mainAxisSize: MainAxisSize.min,
        children: [
          Text(label, style: theme.textTheme.bodySmall),
          const SizedBox(width: 8),
          Text(
            value,
            style: theme.textTheme.bodySmall?.copyWith(
              fontWeight: FontWeight.w700,
              color: theme.colorScheme.primary,
            ),
          ),
        ],
      ),
    );
  }
}

enum _EvidenceStatus { supported, partial, missing }

class _HypothesisCard extends StatelessWidget {
  const _HypothesisCard({
    required this.claim,
    required this.metric,
    required this.status,
    required this.series,
    required this.note,
    this.threshold,
  });

  final String claim;
  final String metric;
  final _EvidenceStatus status;
  final List<double> series;
  final String note;
  final double? threshold;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final statusColor = switch (status) {
      _EvidenceStatus.supported => Colors.green.shade600,
      _EvidenceStatus.partial => Colors.orange.shade700,
      _EvidenceStatus.missing => Colors.red.shade700,
    };
    final statusLabel = switch (status) {
      _EvidenceStatus.supported => 'supported',
      _EvidenceStatus.partial => 'partial',
      _EvidenceStatus.missing => 'missing',
    };

    return Container(
      padding: const EdgeInsets.all(14),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(color: statusColor.withValues(alpha: 0.35)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                padding:
                    const EdgeInsets.symmetric(horizontal: 10, vertical: 6),
                decoration: BoxDecoration(
                  color: statusColor.withValues(alpha: 0.12),
                  borderRadius: BorderRadius.circular(999),
                  border:
                      Border.all(color: statusColor.withValues(alpha: 0.35)),
                ),
                child: Text(
                  statusLabel,
                  style: theme.textTheme.bodySmall?.copyWith(
                    fontWeight: FontWeight.w700,
                    color: statusColor,
                  ),
                ),
              ),
              const SizedBox(width: 10),
              Expanded(
                child: Text(
                  claim,
                  style: theme.textTheme.bodyMedium
                      ?.copyWith(fontWeight: FontWeight.w600),
                ),
              ),
            ],
          ),
          const SizedBox(height: 10),
          Text(
            'Metric: $metric',
            style: theme.textTheme.bodySmall?.copyWith(height: 1.3),
          ),
          const SizedBox(height: 10),
          SizedBox(
            height: 56,
            child: _Sparkline(
              values: series,
              lineColor: theme.colorScheme.primary,
              threshold: threshold,
              thresholdColor: Colors.red.shade400,
            ),
          ),
          const SizedBox(height: 10),
          Text(
            note,
            style: theme.textTheme.bodySmall?.copyWith(
              color: theme.colorScheme.onSurface.withValues(alpha: 0.7),
              height: 1.3,
            ),
          ),
        ],
      ),
    );
  }
}

class _Sparkline extends StatelessWidget {
  const _Sparkline({
    required this.values,
    required this.lineColor,
    this.threshold,
    this.thresholdColor,
  });

  final List<double> values;
  final Color lineColor;
  final double? threshold;
  final Color? thresholdColor;

  @override
  Widget build(BuildContext context) {
    return CustomPaint(
      painter: _SparklinePainter(
        values: values,
        lineColor: lineColor,
        threshold: threshold,
        thresholdColor: thresholdColor ?? Colors.red,
      ),
      child: const SizedBox.expand(),
    );
  }
}

class _SparklinePainter extends CustomPainter {
  _SparklinePainter({
    required this.values,
    required this.lineColor,
    required this.thresholdColor,
    this.threshold,
  });

  final List<double> values;
  final Color lineColor;
  final double? threshold;
  final Color thresholdColor;

  @override
  void paint(Canvas canvas, Size size) {
    if (values.length < 2) return;
    final minV = values.reduce((a, b) => a < b ? a : b);
    final maxV = values.reduce((a, b) => a > b ? a : b);
    final range = (maxV - minV).abs() < 1e-9 ? 1.0 : (maxV - minV);

    final gridPaint = Paint()
      ..color = Colors.black.withValues(alpha: 0.06)
      ..strokeWidth = 1;
    canvas.drawLine(Offset(0, size.height - 1),
        Offset(size.width, size.height - 1), gridPaint);

    if (threshold != null) {
      final tNorm = ((threshold! - minV) / range).clamp(0.0, 1.0);
      final y = size.height - (tNorm * size.height);
      final tPaint = Paint()
        ..color = thresholdColor.withValues(alpha: 0.8)
        ..strokeWidth = 1.5;
      canvas.drawLine(Offset(0, y), Offset(size.width, y), tPaint);
    }

    final paint = Paint()
      ..color = lineColor
      ..strokeWidth = 2.0
      ..style = PaintingStyle.stroke;

    final path = Path();
    for (var i = 0; i < values.length; i++) {
      final x = (i / (values.length - 1)) * size.width;
      final vNorm = ((values[i] - minV) / range).clamp(0.0, 1.0);
      final y = size.height - (vNorm * size.height);
      if (i == 0) {
        path.moveTo(x, y);
      } else {
        path.lineTo(x, y);
      }
    }
    canvas.drawPath(path, paint);
  }

  @override
  bool shouldRepaint(covariant _SparklinePainter oldDelegate) {
    return oldDelegate.values != values ||
        oldDelegate.lineColor != lineColor ||
        oldDelegate.threshold != threshold ||
        oldDelegate.thresholdColor != thresholdColor;
  }
}

class _TimescaleBars extends StatelessWidget {
  const _TimescaleBars();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Column(
      children: [
        _TimescaleBar(
          title: 'Fast',
          subtitle: '50–100ms: reflexes, safety, teleop mirroring',
          color: theme.colorScheme.tertiary,
          fill: 0.82,
          kpi: 'p95 tick 88ms',
        ),
        const SizedBox(height: 10),
        _TimescaleBar(
          title: 'Mid',
          subtitle: '0.5–10s: sequencing, intent, short-horizon world model',
          color: theme.colorScheme.secondary,
          fill: 0.56,
          kpi: 'adapter loss ↓',
        ),
        const SizedBox(height: 10),
        _TimescaleBar(
          title: 'Slow',
          subtitle:
              'minutes–hours: consolidation, global alignment, distillation',
          color: theme.colorScheme.primary,
          fill: 0.34,
          kpi: 'eval ↑',
        ),
      ],
    );
  }
}

class _TimescaleBar extends StatelessWidget {
  const _TimescaleBar({
    required this.title,
    required this.subtitle,
    required this.color,
    required this.fill,
    required this.kpi,
  });

  final String title;
  final String subtitle;
  final Color color;
  final double fill;
  final String kpi;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final clamped = fill.clamp(0.0, 1.0);
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(color: color.withValues(alpha: 0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Text(title,
                  style: theme.textTheme.titleMedium
                      ?.copyWith(fontWeight: FontWeight.w800)),
              const SizedBox(width: 10),
              Text(kpi,
                  style: theme.textTheme.bodySmall
                      ?.copyWith(color: color, fontWeight: FontWeight.w700)),
            ],
          ),
          const SizedBox(height: 6),
          Text(subtitle,
              style: theme.textTheme.bodySmall?.copyWith(height: 1.3)),
          const SizedBox(height: 10),
          ClipRRect(
            borderRadius: BorderRadius.circular(999),
            child: SizedBox(
              height: 10,
              child: Stack(
                children: [
                  Positioned.fill(
                    child: Container(
                        color: theme.colorScheme.onSurface
                            .withValues(alpha: 0.08)),
                  ),
                  Positioned.fill(
                    child: FractionallySizedBox(
                      alignment: Alignment.centerLeft,
                      widthFactor: clamped,
                      child: Container(color: color.withValues(alpha: 0.85)),
                    ),
                  ),
                ],
              ),
            ),
          ),
        ],
      ),
    );
  }
}

enum _GoodDirection { up, down }

class _MiniMetricRow extends StatelessWidget {
  const _MiniMetricRow({
    required this.label,
    required this.unit,
    required this.values,
    required this.goodDirection,
  });

  final String label;
  final String unit;
  final List<double> values;
  final _GoodDirection goodDirection;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final last = values.isNotEmpty ? values.last : 0.0;
    final delta =
        values.length >= 2 ? (values.last - values[values.length - 2]) : 0.0;
    final good = switch (goodDirection) {
      _GoodDirection.up => delta >= 0,
      _GoodDirection.down => delta <= 0,
    };
    final deltaColor = good ? Colors.green.shade700 : Colors.red.shade700;

    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
            color: theme.colorScheme.outline.withValues(alpha: 0.25)),
      ),
      child: Row(
        children: [
          Expanded(
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text(label,
                    style: theme.textTheme.bodyMedium
                        ?.copyWith(fontWeight: FontWeight.w600)),
                const SizedBox(height: 4),
                Text(
                  '${last.toStringAsFixed(2)} $unit  (${delta >= 0 ? '+' : ''}${delta.toStringAsFixed(2)})',
                  style: theme.textTheme.bodySmall?.copyWith(
                      color: deltaColor, fontWeight: FontWeight.w700),
                ),
              ],
            ),
          ),
          const SizedBox(width: 12),
          SizedBox(
            width: 140,
            height: 42,
            child: _Sparkline(
                values: values, lineColor: theme.colorScheme.primary),
          ),
        ],
      ),
    );
  }
}

class _TwoPanelCharts extends StatelessWidget {
  const _TwoPanelCharts();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return LayoutBuilder(
      builder: (context, constraints) {
        final isNarrow = constraints.maxWidth < 720;
        final panels = [
          _MiniPanel(
            title: 'SSM state norm (sample)',
            subtitle: 'recurrence keeps bounded memory',
            child: _Sparkline(
              values: const [
                0.12,
                0.18,
                0.31,
                0.45,
                0.40,
                0.36,
                0.39,
                0.33,
                0.29
              ],
              lineColor: theme.colorScheme.secondary,
            ),
          ),
          _MiniPanel(
            title: 'Compute proxy vs sequence length',
            subtitle: 'attention ~ n², SSM ~ n (illustrative)',
            child: _Sparkline(
              values: const [
                0.06,
                0.08,
                0.11,
                0.14,
                0.18,
                0.23,
                0.29,
                0.36,
                0.44
              ],
              lineColor: theme.colorScheme.primary,
            ),
          ),
        ];
        if (isNarrow) {
          return Column(
            children: [
              panels[0],
              const SizedBox(height: 12),
              panels[1],
            ],
          );
        }
        return Row(
          children: [
            Expanded(child: panels[0]),
            const SizedBox(width: 12),
            Expanded(child: panels[1]),
          ],
        );
      },
    );
  }
}

class _MiniPanel extends StatelessWidget {
  const _MiniPanel(
      {required this.title, required this.subtitle, required this.child});

  final String title;
  final String subtitle;
  final Widget child;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
            color: theme.colorScheme.outline.withValues(alpha: 0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Text(title,
              style: theme.textTheme.bodyMedium
                  ?.copyWith(fontWeight: FontWeight.w700)),
          const SizedBox(height: 2),
          Text(subtitle,
              style: theme.textTheme.bodySmall?.copyWith(height: 1.3)),
          const SizedBox(height: 10),
          SizedBox(height: 52, child: child),
        ],
      ),
    );
  }
}

class _SearchTreeDemo extends StatelessWidget {
  const _SearchTreeDemo();

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(
        color: theme.colorScheme.surface.withValues(alpha: 0.9),
        borderRadius: BorderRadius.circular(ContinuonTokens.r12),
        border: Border.all(
            color: theme.colorScheme.outline.withValues(alpha: 0.25)),
      ),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: const [
          _TreeNode(
            label: 'Root: state s0',
            score: 0.12,
            best: false,
            children: [
              _TreeNode(
                label: 'push_left',
                score: 0.41,
                best: false,
                children: [
                  _TreeNode(
                      label: 'push_left + lift', score: 0.55, best: false),
                  _TreeNode(
                      label: 'push_left + pull', score: 0.48, best: false),
                ],
              ),
              _TreeNode(
                label: 'push_right',
                score: 0.62,
                best: true,
                children: [
                  _TreeNode(
                      label: 'push_right + lift', score: 0.71, best: true),
                  _TreeNode(
                      label: 'push_right + pull', score: 0.60, best: false),
                ],
              ),
              _TreeNode(label: 'do_nothing', score: 0.10, best: false),
            ],
          ),
          SizedBox(height: 10),
          Text(
            'Best plan path highlighted: push_right → push_right + lift',
            style: TextStyle(fontWeight: FontWeight.w600),
          ),
        ],
      ),
    );
  }
}

class _TreeNode extends StatelessWidget {
  const _TreeNode({
    required this.label,
    required this.score,
    required this.best,
    this.children = const [],
  });

  final String label;
  final double score;
  final bool best;
  final List<_TreeNode> children;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final color = best
        ? Colors.green.shade700
        : theme.colorScheme.onSurface.withValues(alpha: 0.85);
    return Padding(
      padding: const EdgeInsets.only(bottom: 6),
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              Container(
                width: 10,
                height: 10,
                decoration: BoxDecoration(
                  color: best
                      ? Colors.green.shade600
                      : theme.colorScheme.primary.withValues(alpha: 0.35),
                  shape: BoxShape.circle,
                ),
              ),
              const SizedBox(width: 8),
              Expanded(
                child: Text(
                  '$label  (score ${score.toStringAsFixed(2)})',
                  style: theme.textTheme.bodySmall?.copyWith(
                      color: color, fontWeight: best ? FontWeight.w800 : null),
                ),
              ),
            ],
          ),
          if (children.isNotEmpty)
            Padding(
              padding: const EdgeInsets.only(left: 18, top: 6),
              child: Column(children: children),
            ),
        ],
      ),
    );
  }
}

class _LiveRuntimeEvidenceSection extends StatefulWidget {
  const _LiveRuntimeEvidenceSection();

  @override
  State<_LiveRuntimeEvidenceSection> createState() =>
      _LiveRuntimeEvidenceSectionState();
}

class _LiveRuntimeEvidenceSectionState
    extends State<_LiveRuntimeEvidenceSection> {
  final _host = TextEditingController(text: 'localhost');
  final _port = TextEditingController(text: '8080');

  bool _loading = false;
  String? _error;
  Map<String, dynamic>? _metrics;
  Map<String, dynamic>? _evalSummary;

  @override
  void dispose() {
    _host.dispose();
    _port.dispose();
    super.dispose();
  }

  Uri _uri(String path, [Map<String, String>? query]) {
    final host = _host.text.trim();
    final port = int.tryParse(_port.text.trim()) ?? 8080;
    return Uri.http('$host:$port', path, query);
  }

  Future<void> _load() async {
    setState(() {
      _loading = true;
      _error = null;
    });
    try {
      final metricsRes =
          await http.get(_uri('/api/training/metrics', {'limit': '140'}));
      final evalRes =
          await http.get(_uri('/api/training/eval_summary', {'limit': '6'}));
      if (!mounted) return;
      if (metricsRes.statusCode != 200) {
        throw StateError('metrics HTTP ${metricsRes.statusCode}');
      }
      if (evalRes.statusCode != 200) {
        throw StateError('eval_summary HTTP ${evalRes.statusCode}');
      }
      final metrics = _tryJsonMap(metricsRes.body) ?? <String, dynamic>{};
      final evalSummary = _tryJsonMap(evalRes.body) ?? <String, dynamic>{};
      setState(() {
        _metrics = metrics;
        _evalSummary = evalSummary;
      });
    } catch (e) {
      setState(() => _error = e.toString());
    } finally {
      if (mounted) setState(() => _loading = false);
    }
  }

  static Map<String, dynamic>? _tryJsonMap(String body) {
    try {
      final decoded = jsonDecode(body);
      return decoded is Map ? Map<String, dynamic>.from(decoded) : null;
    } catch (_) {
      return null;
    }
  }

  static List<double> _extractSeries(
    Map<String, dynamic>? root, {
    required List<String> path,
    required String yKey,
  }) {
    dynamic cur = root;
    for (final p in path) {
      if (cur is Map) {
        cur = cur[p];
      } else {
        return const [];
      }
    }
    if (cur is! List) return const [];
    final out = <double>[];
    for (final item in cur) {
      if (item is Map && item[yKey] != null) {
        final v = item[yKey];
        if (v is num) out.add(v.toDouble());
      }
    }
    return out;
  }

  static double? _extractLatestRate(
      Map<String, dynamic>? evalSummary, String key) {
    if (evalSummary == null) return null;
    final section = evalSummary[key];
    if (section is! Map) return null;
    final latest = section['latest'];
    if (latest is! Map) return null;
    final sr = latest['success_rate'];
    return sr is num ? sr.toDouble() : null;
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);

    final waveFast = _extractSeries(_metrics,
        path: const ['wavecore', 'fast', 'points'], yKey: 'loss');
    final waveMid = _extractSeries(_metrics,
        path: const ['wavecore', 'mid', 'points'], yKey: 'loss');
    final waveSlow = _extractSeries(_metrics,
        path: const ['wavecore', 'slow', 'points'], yKey: 'loss');
    final toolAcc = _extractSeries(_metrics,
        path: const ['tool_router', 'acc', 'points'], yKey: 'acc');
    final toolTop1 = _extractSeries(_metrics,
        path: const ['tool_router_eval', 'top1', 'points'], yKey: 'top1');
    final toolTop5 = _extractSeries(_metrics,
        path: const ['tool_router_eval', 'top5', 'points'], yKey: 'top5');

    final hopeEvalSuccess = _extractLatestRate(_evalSummary, 'hope_eval');

    return _ResearchCard(
      title: 'Live evidence (from connected runtime)',
      subtitle:
          'Optional: fetch `/api/training/metrics` + `/api/training/eval_summary` from a Continuon Brain runtime (LAN/local).',
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Wrap(
            spacing: 12,
            runSpacing: 10,
            crossAxisAlignment: WrapCrossAlignment.center,
            children: [
              SizedBox(
                width: 220,
                child: TextField(
                  controller: _host,
                  decoration: const InputDecoration(
                      labelText: 'host (e.g. 192.168.1.42)'),
                ),
              ),
              SizedBox(
                width: 120,
                child: TextField(
                  controller: _port,
                  decoration: const InputDecoration(labelText: 'port'),
                  keyboardType: TextInputType.number,
                ),
              ),
              ElevatedButton.icon(
                onPressed: _loading ? null : _load,
                icon: _loading
                    ? const SizedBox(
                        width: 16,
                        height: 16,
                        child: CircularProgressIndicator(strokeWidth: 2))
                    : const Icon(Icons.cloud_download),
                label: Text(_loading ? 'Loading' : 'Load live metrics'),
              ),
              if (hopeEvalSuccess != null)
                _KpiPill(
                  label: 'HOPE eval success',
                  value: '${(hopeEvalSuccess * 100).toStringAsFixed(1)}%',
                ),
            ],
          ),
          if (_error != null) ...[
            const SizedBox(height: 10),
            Text(_error!,
                style: theme.textTheme.bodySmall
                    ?.copyWith(color: theme.colorScheme.error)),
          ],
          const SizedBox(height: 14),
          LayoutBuilder(
            builder: (context, constraints) {
              final isNarrow = constraints.maxWidth < 820;
              final panels = [
                _MiniPanel(
                  title: 'WaveCore fast loss (live)',
                  subtitle: waveFast.isEmpty
                      ? 'no data yet'
                      : 'last ${waveFast.last.toStringAsFixed(3)}',
                  child: _Sparkline(
                      values: waveFast, lineColor: theme.colorScheme.tertiary),
                ),
                _MiniPanel(
                  title: 'WaveCore mid loss (live)',
                  subtitle: waveMid.isEmpty
                      ? 'no data yet'
                      : 'last ${waveMid.last.toStringAsFixed(3)}',
                  child: _Sparkline(
                      values: waveMid, lineColor: theme.colorScheme.secondary),
                ),
                _MiniPanel(
                  title: 'WaveCore slow loss (live)',
                  subtitle: waveSlow.isEmpty
                      ? 'no data yet'
                      : 'last ${waveSlow.last.toStringAsFixed(3)}',
                  child: _Sparkline(
                      values: waveSlow, lineColor: theme.colorScheme.primary),
                ),
                _MiniPanel(
                  title: 'Tool router acc (live)',
                  subtitle: toolAcc.isEmpty
                      ? 'no data yet'
                      : 'last ${(toolAcc.last * 100).toStringAsFixed(1)}%',
                  child: _Sparkline(
                      values: toolAcc, lineColor: theme.colorScheme.primary),
                ),
                _MiniPanel(
                  title: 'Tool router eval top1 (live)',
                  subtitle: toolTop1.isEmpty
                      ? 'no data yet'
                      : 'last ${(toolTop1.last * 100).toStringAsFixed(1)}%',
                  child: _Sparkline(
                      values: toolTop1, lineColor: theme.colorScheme.secondary),
                ),
                _MiniPanel(
                  title: 'Tool router eval top5 (live)',
                  subtitle: toolTop5.isEmpty
                      ? 'no data yet'
                      : 'last ${(toolTop5.last * 100).toStringAsFixed(1)}%',
                  child: _Sparkline(
                      values: toolTop5, lineColor: theme.colorScheme.tertiary),
                ),
              ];

              if (isNarrow) {
                return Column(
                  children: [
                    for (var i = 0; i < panels.length; i++) ...[
                      panels[i],
                      if (i != panels.length - 1) const SizedBox(height: 12),
                    ],
                  ],
                );
              }

              return Wrap(
                spacing: 12,
                runSpacing: 12,
                children: panels
                    .map(
                      (p) => SizedBox(
                          width: (constraints.maxWidth - 24) / 3, child: p),
                    )
                    .toList(),
              );
            },
          ),
        ],
      ),
    );
  }
}
