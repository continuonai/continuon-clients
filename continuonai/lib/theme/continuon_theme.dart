import 'package:flutter/material.dart';

/// Continuon brand color tokens
class ContinuonColors {
  // Core brand
  static const primaryBlue = Color(0xFF0A84FF); // Continuon Blue
  static const brandBlack = Color(0xFF0A0A0C);
  static const brandWhite = Color(0xFFFFFFFF);
  static const black = Color(0xFF000000);

  // Secondary / identity
  static const waveBlueStart = Color(0xFF3F8CFF);
  static const waveBlueEnd = Color(0xFF67D3FF);
  static const particleOrange = Color(0xFFFF8C42);
  static const cmsViolet = Color(0xFF8A52FF);

  // Greys
  static const gray900 = Color(0xFF222228);
  static const gray800 = Color(0xFF2C2C35);
  static const gray700 = Color(0xFF3C3C45);
  static const gray500 = Color(0xFF8E8E93);
  static const gray400 = Color(0xFFAAAAAA);
  static const gray200 = Color(0xFFE5E8EC);

  // Product accents
  static const brainAccent = cmsViolet;
  static const cloudAccentStart = waveBlueStart;
  static const cloudAccentEnd = waveBlueEnd;
}

/// Simple design tokens for spacing / radius / shadow
class ContinuonTokens {
  // Spacing
  static const s4 = 4.0;
  static const s8 = 8.0;
  static const s12 = 12.0;
  static const s16 = 16.0;
  static const s24 = 24.0;
  static const s32 = 32.0;
  static const s48 = 48.0;

  // Radius
  static const r4 = 4.0;
  static const r8 = 8.0;
  static const r16 = 16.0;
  static const rFull = 999.0;

  // Shadows
  static const lowShadow = [
    BoxShadow(
      color: Colors.black26,
      blurRadius: 8,
      offset: Offset(0, 4),
    ),
  ];

  static const midShadow = [
    BoxShadow(
      color: Colors.black38,
      blurRadius: 16,
      offset: Offset(0, 8),
    ),
  ];
}

/// Gradient + product accents as a ThemeExtension
class ContinuonBrandExtension extends ThemeExtension<ContinuonBrandExtension> {
  final Gradient waveGradient;
  final Color brainAccent;
  final Gradient cloudGradient;

  const ContinuonBrandExtension({
    required this.waveGradient,
    required this.brainAccent,
    required this.cloudGradient,
  });

  @override
  ContinuonBrandExtension copyWith({
    Gradient? waveGradient,
    Color? brainAccent,
    Gradient? cloudGradient,
  }) {
    return ContinuonBrandExtension(
      waveGradient: waveGradient ?? this.waveGradient,
      brainAccent: brainAccent ?? this.brainAccent,
      cloudGradient: cloudGradient ?? this.cloudGradient,
    );
  }

  @override
  ThemeExtension<ContinuonBrandExtension> lerp(
    ThemeExtension<ContinuonBrandExtension>? other,
    double t,
  ) {
    if (other is! ContinuonBrandExtension) return this;

    return ContinuonBrandExtension(
      waveGradient: LinearGradient(
        colors: [
          Color.lerp(
            (waveGradient as LinearGradient).colors.first,
            (other.waveGradient as LinearGradient).colors.first,
            t,
          )!,
          Color.lerp(
            (waveGradient as LinearGradient).colors.last,
            (other.waveGradient as LinearGradient).colors.last,
            t,
          )!,
        ],
      ),
      brainAccent: Color.lerp(brainAccent, other.brainAccent, t)!,
      cloudGradient: LinearGradient(
        colors: [
          Color.lerp(
            (cloudGradient as LinearGradient).colors.first,
            (other.cloudGradient as LinearGradient).colors.first,
            t,
          )!,
          Color.lerp(
            (cloudGradient as LinearGradient).colors.last,
            (other.cloudGradient as LinearGradient).colors.last,
            t,
          )!,
        ],
      ),
    );
  }

  static ContinuonBrandExtension get light => const ContinuonBrandExtension(
        waveGradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            ContinuonColors.waveBlueStart,
            ContinuonColors.waveBlueEnd,
          ],
        ),
        brainAccent: ContinuonColors.brainAccent,
        cloudGradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            ContinuonColors.cloudAccentStart,
            ContinuonColors.cloudAccentEnd,
          ],
        ),
      );

  static ContinuonBrandExtension get dark => const ContinuonBrandExtension(
        waveGradient: LinearGradient(
          begin: Alignment.topLeft,
          end: Alignment.bottomRight,
          colors: [
            ContinuonColors.waveBlueStart,
            ContinuonColors.waveBlueEnd,
          ],
        ),
        brainAccent: ContinuonColors.brainAccent,
        cloudGradient: LinearGradient(
          begin: Alignment.topCenter,
          end: Alignment.bottomCenter,
          colors: [
            ContinuonColors.cloudAccentStart,
            ContinuonColors.cloudAccentEnd,
          ],
        ),
      );
}

/// Typography helpers
class ContinuonTypography {
  static TextTheme buildLightTextTheme([TextTheme? base]) {
    final b = base ?? ThemeData.light().textTheme;

    return b.copyWith(
      displayLarge: b.displayLarge?.copyWith(
        fontSize: 72,
        fontWeight: FontWeight.w600,
        letterSpacing: -1.0,
      ),
      displayMedium: b.displayMedium?.copyWith(
        fontSize: 56,
        fontWeight: FontWeight.w600,
        letterSpacing: -0.5,
      ),
      headlineMedium: b.headlineMedium?.copyWith(
        fontSize: 32,
        fontWeight: FontWeight.w600,
      ),
      headlineSmall: b.headlineSmall?.copyWith(
        fontSize: 24,
        fontWeight: FontWeight.w500,
      ),
      bodyLarge: b.bodyLarge?.copyWith(
        fontSize: 18,
        fontWeight: FontWeight.w400,
      ),
      bodyMedium: b.bodyMedium?.copyWith(
        fontSize: 16,
        fontWeight: FontWeight.w400,
      ),
      bodySmall: b.bodySmall?.copyWith(
        fontSize: 13,
        fontWeight: FontWeight.w400,
      ),
      labelLarge: b.labelLarge?.copyWith(
        fontSize: 14,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.2,
      ),
    );
  }

  static TextTheme buildDarkTextTheme([TextTheme? base]) {
    final b = base ?? ThemeData.dark().textTheme;

    return b.copyWith(
      displayLarge: b.displayLarge?.copyWith(
        fontSize: 72,
        fontWeight: FontWeight.w600,
        letterSpacing: -1.0,
        color: ContinuonColors.brandWhite,
      ),
      displayMedium: b.displayMedium?.copyWith(
        fontSize: 56,
        fontWeight: FontWeight.w600,
        letterSpacing: -0.5,
        color: ContinuonColors.brandWhite,
      ),
      headlineMedium: b.headlineMedium?.copyWith(
        fontSize: 32,
        fontWeight: FontWeight.w600,
        color: ContinuonColors.brandWhite,
      ),
      headlineSmall: b.headlineSmall?.copyWith(
        fontSize: 24,
        fontWeight: FontWeight.w500,
        color: ContinuonColors.brandWhite,
      ),
      bodyLarge: b.bodyLarge?.copyWith(
        fontSize: 18,
        fontWeight: FontWeight.w400,
        color: ContinuonColors.gray200,
      ),
      bodyMedium: b.bodyMedium?.copyWith(
        fontSize: 16,
        fontWeight: FontWeight.w400,
        color: ContinuonColors.gray200,
      ),
      bodySmall: b.bodySmall?.copyWith(
        fontSize: 13,
        fontWeight: FontWeight.w400,
        color: ContinuonColors.gray200,
      ),
      labelLarge: b.labelLarge?.copyWith(
        fontSize: 14,
        fontWeight: FontWeight.w600,
        letterSpacing: 0.2,
        color: ContinuonColors.brandWhite,
      ),
    );
  }

  static TextStyle soraAccentHeading({
    Color color = ContinuonColors.primaryBlue,
  }) {
    return TextStyle(
      fontSize: 32,
      fontWeight: FontWeight.w600,
      letterSpacing: 0.2,
      color: color,
      fontFamily: 'Sora',
    );
  }
}

/// Public ThemeData instances

final ThemeData continuonLightTheme = ThemeData(
  useMaterial3: true,
  colorScheme: _continuonLightScheme,
  scaffoldBackgroundColor: _continuonLightScheme.surface,
  canvasColor: _continuonLightScheme.surface,
  textTheme: ContinuonTypography.buildLightTextTheme(),
  appBarTheme: const AppBarTheme(
    backgroundColor: ContinuonColors.brandWhite,
    foregroundColor: ContinuonColors.brandBlack,
    elevation: 0,
    centerTitle: false,
  ),
  cardTheme: CardThemeData(
    color: ContinuonColors.brandWhite,
    elevation: 2,
    shadowColor: Colors.black12,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(ContinuonTokens.r8),
    ),
  ),
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      backgroundColor: ContinuonColors.primaryBlue,
      foregroundColor: ContinuonColors.brandWhite,
      padding: const EdgeInsets.symmetric(
        horizontal: ContinuonTokens.s16,
        vertical: ContinuonTokens.s12,
      ),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
      ),
    ),
  ),
  extensions: <ThemeExtension<dynamic>>[
    ContinuonBrandExtension.light,
  ],
);

final ThemeData continuonDarkTheme = ThemeData(
  useMaterial3: true,
  colorScheme: _continuonDarkScheme,
  scaffoldBackgroundColor: _continuonDarkScheme.surface,
  canvasColor: _continuonDarkScheme.surface,
  textTheme: ContinuonTypography.buildDarkTextTheme(),
  appBarTheme: const AppBarTheme(
    backgroundColor: ContinuonColors.brandBlack,
    foregroundColor: ContinuonColors.gray200,
    elevation: 0,
    centerTitle: false,
  ),
  cardTheme: CardThemeData(
    color: ContinuonColors.gray900,
    elevation: 2,
    shadowColor: Colors.black45,
    shape: RoundedRectangleBorder(
      borderRadius: BorderRadius.circular(ContinuonTokens.r8),
    ),
  ),
  elevatedButtonTheme: ElevatedButtonThemeData(
    style: ElevatedButton.styleFrom(
      backgroundColor: ContinuonColors.primaryBlue,
      foregroundColor: ContinuonColors.brandWhite,
      padding: const EdgeInsets.symmetric(
        horizontal: ContinuonTokens.s16,
        vertical: ContinuonTokens.s12,
      ),
      shape: RoundedRectangleBorder(
        borderRadius: BorderRadius.circular(ContinuonTokens.r8),
      ),
    ),
  ),
  extensions: <ThemeExtension<dynamic>>[
    ContinuonBrandExtension.dark,
  ],
);

/// Light color scheme based on Continuon brand
final ColorScheme _continuonLightScheme = ColorScheme(
  brightness: Brightness.light,
  primary: ContinuonColors.primaryBlue,
  onPrimary: ContinuonColors.brandWhite,
  primaryContainer: ContinuonColors.waveBlueStart,
  onPrimaryContainer: ContinuonColors.brandWhite,
  secondary: ContinuonColors.cmsViolet,
  onSecondary: ContinuonColors.brandWhite,
  secondaryContainer: ContinuonColors.cmsViolet.withValues(alpha: 0.15),
  onSecondaryContainer: ContinuonColors.cmsViolet,
  tertiary: ContinuonColors.particleOrange,
  onTertiary: ContinuonColors.brandBlack,
  tertiaryContainer: ContinuonColors.particleOrange.withValues(alpha: 0.15),
  onTertiaryContainer: ContinuonColors.particleOrange,
  error: Colors.red.shade600,
  onError: ContinuonColors.brandWhite,
  errorContainer: Colors.red.shade50,
  onErrorContainer: Colors.red.shade900,
  surface: ContinuonColors.brandWhite,
  onSurface: ContinuonColors.gray900,
  surfaceContainerHighest: ContinuonColors.gray200,
  onSurfaceVariant: ContinuonColors.gray700,
  outline: ContinuonColors.gray200,
  shadow: Colors.black26,
  inverseSurface: ContinuonColors.gray900,
  onInverseSurface: ContinuonColors.gray200,
  inversePrimary: ContinuonColors.waveBlueEnd,
);

/// Dark color scheme based on Continuon brand
final ColorScheme _continuonDarkScheme = ColorScheme(
  brightness: Brightness.dark,
  primary: ContinuonColors.primaryBlue,
  onPrimary: ContinuonColors.brandWhite,
  primaryContainer: ContinuonColors.waveBlueStart,
  onPrimaryContainer: ContinuonColors.brandWhite,
  secondary: ContinuonColors.cmsViolet,
  onSecondary: ContinuonColors.brandWhite,
  secondaryContainer: ContinuonColors.cmsViolet.withValues(alpha: 0.3),
  onSecondaryContainer: ContinuonColors.brandWhite,
  tertiary: ContinuonColors.particleOrange,
  onTertiary: ContinuonColors.brandBlack,
  tertiaryContainer: ContinuonColors.particleOrange.withValues(alpha: 0.3),
  onTertiaryContainer: ContinuonColors.brandBlack,
  error: Colors.red.shade300,
  onError: ContinuonColors.brandBlack,
  errorContainer: Colors.red.shade900,
  onErrorContainer: ContinuonColors.brandWhite,
  surface: ContinuonColors.brandBlack,
  onSurface: ContinuonColors.gray200,
  surfaceContainerHighest: ContinuonColors.gray700,
  onSurfaceVariant: ContinuonColors.gray200,
  outline: ContinuonColors.gray700,
  shadow: Colors.black,
  inverseSurface: ContinuonColors.brandWhite,
  onInverseSurface: ContinuonColors.gray900,
  inversePrimary: ContinuonColors.waveBlueEnd,
);
