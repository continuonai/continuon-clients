import 'package:flutter/material.dart';

import '../../models/robot_init_state.dart';
import '../../theme/continuon_theme.dart';

/// Horizontal stepper widget for wizard progress indication
class WizardStepIndicator extends StatelessWidget {
  final RobotInitStep currentStep;
  final List<RobotInitStep> steps;

  const WizardStepIndicator({
    super.key,
    required this.currentStep,
    this.steps = RobotInitStep.values,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Container(
      padding: const EdgeInsets.symmetric(
        horizontal: ContinuonTokens.s16,
        vertical: ContinuonTokens.s12,
      ),
      child: Row(
        children: List.generate(steps.length * 2 - 1, (index) {
          if (index.isOdd) {
            // Connector line
            final stepIndex = index ~/ 2;
            final isCompleted = steps[stepIndex].index < currentStep.index;
            return Expanded(
              child: Container(
                height: 2,
                color: isCompleted
                    ? ContinuonColors.primaryBlue
                    : (isDark ? ContinuonColors.gray700 : ContinuonColors.gray200),
              ),
            );
          } else {
            // Step indicator
            final stepIndex = index ~/ 2;
            final step = steps[stepIndex];
            final isCompleted = step.index < currentStep.index;
            final isCurrent = step == currentStep;

            return _StepCircle(
              stepNumber: step.index + 1,
              label: step.title,
              isCompleted: isCompleted,
              isCurrent: isCurrent,
              isDark: isDark,
            );
          }
        }),
      ),
    );
  }
}

class _StepCircle extends StatelessWidget {
  final int stepNumber;
  final String label;
  final bool isCompleted;
  final bool isCurrent;
  final bool isDark;

  const _StepCircle({
    required this.stepNumber,
    required this.label,
    required this.isCompleted,
    required this.isCurrent,
    required this.isDark,
  });

  @override
  Widget build(BuildContext context) {
    final circleColor = isCompleted || isCurrent
        ? ContinuonColors.primaryBlue
        : (isDark ? ContinuonColors.gray700 : ContinuonColors.gray200);

    final textColor = isCompleted || isCurrent
        ? ContinuonColors.brandWhite
        : (isDark ? ContinuonColors.gray500 : ContinuonColors.gray500);

    final labelColor = isCurrent
        ? ContinuonColors.primaryBlue
        : (isDark ? ContinuonColors.gray400 : ContinuonColors.gray500);

    return Column(
      mainAxisSize: MainAxisSize.min,
      children: [
        AnimatedContainer(
          duration: const Duration(milliseconds: 200),
          width: isCurrent ? 36 : 28,
          height: isCurrent ? 36 : 28,
          decoration: BoxDecoration(
            color: circleColor,
            shape: BoxShape.circle,
            boxShadow: isCurrent ? ContinuonTokens.glowShadow : null,
          ),
          child: Center(
            child: isCompleted
                ? Icon(
                    Icons.check,
                    size: isCurrent ? 20 : 16,
                    color: textColor,
                  )
                : Text(
                    stepNumber.toString(),
                    style: TextStyle(
                      color: textColor,
                      fontWeight: isCurrent ? FontWeight.w600 : FontWeight.w500,
                      fontSize: isCurrent ? 14 : 12,
                    ),
                  ),
          ),
        ),
        const SizedBox(height: ContinuonTokens.s4),
        Text(
          label,
          style: TextStyle(
            color: labelColor,
            fontSize: 10,
            fontWeight: isCurrent ? FontWeight.w600 : FontWeight.w400,
          ),
          textAlign: TextAlign.center,
        ),
      ],
    );
  }
}

/// Compact version for space-constrained layouts
class WizardStepIndicatorCompact extends StatelessWidget {
  final RobotInitStep currentStep;
  final int totalSteps;

  const WizardStepIndicatorCompact({
    super.key,
    required this.currentStep,
    this.totalSteps = 5,
  });

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final isDark = theme.brightness == Brightness.dark;

    return Row(
      mainAxisAlignment: MainAxisAlignment.center,
      children: [
        Text(
          'Step ${currentStep.index + 1} of $totalSteps',
          style: TextStyle(
            color: isDark ? ContinuonColors.gray400 : ContinuonColors.gray500,
            fontSize: 12,
            fontWeight: FontWeight.w500,
          ),
        ),
        const SizedBox(width: ContinuonTokens.s12),
        SizedBox(
          width: 100,
          child: LinearProgressIndicator(
            value: (currentStep.index + 1) / totalSteps,
            backgroundColor:
                isDark ? ContinuonColors.gray700 : ContinuonColors.gray200,
            valueColor:
                const AlwaysStoppedAnimation<Color>(ContinuonColors.primaryBlue),
            borderRadius: BorderRadius.circular(ContinuonTokens.r4),
          ),
        ),
      ],
    );
  }
}
