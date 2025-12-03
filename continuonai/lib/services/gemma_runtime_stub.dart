import 'package:flutter/foundation.dart';

typedef GemmaManifestReload = Future<void> Function(
  GemmaRuntimeManifest manifest,
  String reason,
);

class GemmaRuntimeManifest {
  GemmaRuntimeManifest({
    required this.baseModelPath,
    required this.adapterPath,
    this.safetyHeadPath,
  });

  final String baseModelPath;
  final String adapterPath;
  final String? safetyHeadPath;

  static Future<GemmaRuntimeManifest?> load(String manifestPath) async => null;

  Map<String, Object?> toJson() => {
        'base_model': baseModelPath,
        'adapter': adapterPath,
        if (safetyHeadPath != null) 'safety_head': safetyHeadPath,
      };
}

class GemmaAdapterHotReloader {
  GemmaAdapterHotReloader({
    this.manifestPath = '/opt/continuonos/brain/model/manifest.json',
    this.debounceDuration = const Duration(milliseconds: 200),
    GemmaManifestReload? onReload,
  });

  final String manifestPath;
  final Duration debounceDuration;

  Future<void> initialize() async {
    debugPrint('Gemma adapter hot-reload disabled on web platform.');
  }

  Future<void> dispose() async {}

  GemmaRuntimeManifest? get cachedManifest => null;
}

Future<void> notifyAdapterPromotion(
  GemmaRuntimeManifest manifest,
  String reason,
) async {
  debugPrint('Adapter reload signal ignored on web (${manifest.adapterPath}).');
}
