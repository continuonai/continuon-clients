import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:flutter/foundation.dart';
import 'package:flutter/services.dart';

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

  factory GemmaRuntimeManifest.fromJson(Map<String, dynamic> json) {
    final paths = json['paths'] as Map<String, dynamic>? ?? <String, dynamic>{};
    final adapterPath = paths['adapter']?.toString() ?? '';
    return GemmaRuntimeManifest(
      baseModelPath: paths['base_model']?.toString() ?? '',
      adapterPath: adapterPath,
      safetyHeadPath: paths['safety_head']?.toString(),
    );
  }

  static Future<GemmaRuntimeManifest?> load(String manifestPath) async {
    final manifestFile = File(manifestPath);
    if (!await manifestFile.exists()) {
      debugPrint('Gemma manifest not found at $manifestPath');
      return null;
    }
    try {
      final payload = jsonDecode(await manifestFile.readAsString()) as Map<String, dynamic>;
      return GemmaRuntimeManifest.fromJson(payload);
    } catch (error) {
      debugPrint('Failed to parse manifest $manifestPath: $error');
      return null;
    }
  }

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
  }) : _onReload = onReload ?? notifyAdapterPromotion;

  final String manifestPath;
  final Duration debounceDuration;
  final GemmaManifestReload _onReload;

  StreamSubscription<FileSystemEvent>? _manifestWatcher;
  StreamSubscription<FileSystemEvent>? _adapterWatcher;
  StreamSubscription<ProcessSignal>? _signalSubscription;
  Timer? _debounce;
  GemmaRuntimeManifest? _cachedManifest;

  Future<void> initialize() async {
    await _reload(reason: 'initial');
    _watchManifest();
    _listenForSignals();
  }

  Future<void> dispose() async {
    await _manifestWatcher?.cancel();
    await _adapterWatcher?.cancel();
    await _signalSubscription?.cancel();
    _debounce?.cancel();
  }

  void _watchManifest() {
    final file = File(manifestPath);
    final directory = file.parent;
    if (!directory.existsSync()) {
      debugPrint('Manifest directory missing: ${directory.path}');
      return;
    }
    _manifestWatcher = directory.watch(events: FileSystemEvent.modify | FileSystemEvent.create).listen((event) {
      if (event.path == file.path) {
        _reload(reason: 'manifest_changed');
      }
    });
  }

  void _watchAdapter(String adapterPath) {
    if (adapterPath.isEmpty) {
      return;
    }
    _adapterWatcher?.cancel();
    final adapterFile = File(adapterPath);
    final adapterDir = adapterFile.parent;
    if (!adapterDir.existsSync()) {
      debugPrint('Adapter directory missing: ${adapterDir.path}');
      return;
    }
    _adapterWatcher = adapterDir
        .watch(events: FileSystemEvent.modify | FileSystemEvent.create | FileSystemEvent.move)
        .listen((event) {
      if (event.path == adapterFile.path) {
        _reload(reason: 'adapter_changed');
      }
    });
  }

  void _listenForSignals() {
    if (!(Platform.isLinux || Platform.isMacOS)) {
      return;
    }
    _signalSubscription = ProcessSignal.sigusr1.watch().listen((_) {
      _reload(reason: 'signal');
    });
  }

  Future<void> _reload({required String reason}) async {
    _debounce?.cancel();
    _debounce = Timer(debounceDuration, () async {
      final manifest = await GemmaRuntimeManifest.load(manifestPath);
      if (manifest == null) {
        return;
      }
      _cachedManifest = manifest;
      _watchAdapter(manifest.adapterPath);
      try {
        await _onReload(manifest, reason);
      } catch (error) {
        debugPrint('Adapter reload failed ($reason): $error');
      }
    });
  }

  GemmaRuntimeManifest? get cachedManifest => _cachedManifest;
}

Future<void> notifyAdapterPromotion(
  GemmaRuntimeManifest manifest,
  String reason,
) async {
  const channel = MethodChannel('continuonai/gemma_runtime');
  try {
    await channel.invokeMethod('reloadAdapters', {
      'reason': reason,
      ...manifest.toJson(),
    });
  } catch (error) {
    debugPrint('Adapter reload channel unavailable: $error');
  }
}
