import 'package:flutter/material.dart';

class AppColors {
  static const background = Color(0xFFF5F5F7);
  static const cardBackground = Colors.white;
  static const primaryBlue = Color(0xFF007AFF);
  static const successGreen = Color(0xFF34C759);
  static const dangerRed = Color(0xFFFF3B30);
  static const warningOrange = Color(0xFFFF9500);
  static const purple = Color(0xFFAF52DE);
  static const textPrimary = Color(0xFF1D1D1F);
  static const textSecondary = Color(0xFF86868B);

  // Dark mode colors
  static const darkBackground = Color(0xFF000000);
  static const darkPanel = Color(0xFF1D1D1F);
  static const darkSurface = Color(0xFF2A2A2C);
  static const darkTextPrimary = Colors.white;
  static const darkTextSecondary = Color(0xFF86868B);
}

class AppTextStyles {
  static const header = TextStyle(
    color: AppColors.textPrimary,
    fontSize: 28,
    fontWeight: FontWeight.bold,
    fontFamily: '.SF Pro Display', // iOS system font fallback
  );

  static const sectionHeader = TextStyle(
    color: AppColors.textPrimary,
    fontSize: 20,
    fontWeight: FontWeight.bold,
  );

  static const label = TextStyle(
    color: AppColors.textSecondary,
    fontSize: 14,
  );

  static const value = TextStyle(
    color: AppColors.textPrimary,
    fontSize: 16,
    fontWeight: FontWeight.w600,
  );

  static const darkLabel = TextStyle(
    color: AppColors.darkTextSecondary,
    fontSize: 11,
  );

  static const darkValue = TextStyle(
    color: AppColors.darkTextPrimary,
    fontSize: 16,
    fontWeight: FontWeight.w600,
  );
}

class AppDecorations {
  static final card = BoxDecoration(
    color: AppColors.cardBackground,
    borderRadius: BorderRadius.circular(12),
    boxShadow: [
      BoxShadow(
        color: Colors.black.withValues(alpha: 0.1),
        blurRadius: 8,
        offset: const Offset(0, 2),
      ),
    ],
  );

  static final statusContainer = BoxDecoration(
    color: AppColors.background,
    borderRadius: BorderRadius.circular(8),
  );
}
