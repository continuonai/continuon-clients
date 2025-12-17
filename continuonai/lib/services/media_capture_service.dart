import 'dart:io';

import 'package:flutter/services.dart';
import 'package:path_provider/path_provider.dart';

import '../models/rlds_models.dart';

class CapturedAudio {
  CapturedAudio({
    required this.path,
    required this.frameId,
    required this.timestampMs,
    required this.sampleRateHz,
    required this.numChannels,
    required this.sensor,
  });

  final String path;
  final String frameId;
  final int timestampMs;
  final int sampleRateHz;
  final int numChannels;
  final String sensor;
}

class CapturedVideo {
  CapturedVideo({
    required this.path,
    required this.frameId,
    required this.timestampMs,
    required this.mount,
    required this.sensor,
    this.depthPath,
  });

  final String path;
  final String frameId;
  final int timestampMs;
  final String mount;
  final String sensor;
  final String? depthPath;
}

class MediaCaptureResult {
  const MediaCaptureResult({this.deviceMic, this.robotMic, this.egocentric});

  final CapturedAudio? deviceMic;
  final CapturedAudio? robotMic;
  final CapturedVideo? egocentric;

  List<EpisodeAsset> toAssets() {
    final assets = <EpisodeAsset>[];
    if (deviceMic != null) {
      assets.add(
        EpisodeAsset(
          localUri: deviceMic!.path,
          sensor: deviceMic!.sensor,
          frameId: deviceMic!.frameId,
          capturedAtMillis: deviceMic!.timestampMs,
          mimeType: 'audio/wav',
        ),
      );
    }
    if (robotMic != null) {
      assets.add(
        EpisodeAsset(
          localUri: robotMic!.path,
          sensor: robotMic!.sensor,
          frameId: robotMic!.frameId,
          capturedAtMillis: robotMic!.timestampMs,
          mimeType: 'audio/wav',
        ),
      );
    }
    if (egocentric != null) {
      assets.add(
        EpisodeAsset(
          localUri: egocentric!.path,
          sensor: egocentric!.sensor,
          frameId: egocentric!.frameId,
          capturedAtMillis: egocentric!.timestampMs,
          mount: egocentric!.mount,
          mimeType: 'video/mp4',
        ),
      );
      if (egocentric!.depthPath != null) {
        assets.add(
          EpisodeAsset(
            localUri: egocentric!.depthPath!,
            sensor: '${egocentric!.sensor}_depth',
            frameId: egocentric!.frameId,
            capturedAtMillis: egocentric!.timestampMs,
            mount: egocentric!.mount,
            mimeType: 'application/octet-stream',
          ),
        );
      }
    }
    return assets;
  }
}

class MediaCaptureService {
  MediaCaptureService({MethodChannel? channel})
      : _channel = channel ?? const MethodChannel('continuonai/media_capture');

  final MethodChannel _channel;

  Future<MediaCaptureResult> capture({
    required bool includeDeviceMic,
    required bool includeRobotMic,
    required bool includeEgocentric,
    String egocentricMount = 'chest',
  }) async {
    CapturedAudio? deviceMic;
    CapturedAudio? robotMic;
    CapturedVideo? egocentric;

    if (includeDeviceMic) {
      deviceMic = await _captureAudio(sensor: 'device_mic');
    }
    if (includeRobotMic) {
      robotMic = await _captureAudio(sensor: 'robot_mic');
    }
    if (includeEgocentric) {
      egocentric = await _captureVideo(sensor: 'egocentric', mount: egocentricMount);
    }

    return MediaCaptureResult(
      deviceMic: deviceMic,
      robotMic: robotMic,
      egocentric: egocentric,
    );
  }

  Future<CapturedAudio> _captureAudio({required String sensor}) async {
    final now = DateTime.now();
    final timestampMs = now.millisecondsSinceEpoch;
    final fallback = await _buildStubFile(sensor: sensor, extension: 'wav');
    try {
      final response = await _channel.invokeMapMethod<String, dynamic>('captureAudio', {
        'sensor': sensor,
        'timestamp_ms': timestampMs,
      });
      final path = response?['path'] as String? ?? fallback.path;
      final sampleRate = response?['sample_rate_hz'] as int? ?? 16000;
      final numChannels = response?['num_channels'] as int? ?? 1;
      final frameId = response?['frame_id'] as String? ?? timestampMs.toString();
      return CapturedAudio(
        path: path,
        frameId: frameId,
        timestampMs: timestampMs,
        sampleRateHz: sampleRate,
        numChannels: numChannels,
        sensor: sensor,
      );
    } on PlatformException catch (_) {
      return CapturedAudio(
        path: fallback.path,
        frameId: timestampMs.toString(),
        timestampMs: timestampMs,
        sampleRateHz: 16000,
        numChannels: 1,
        sensor: sensor,
      );
    }
  }

  Future<CapturedVideo> _captureVideo({required String sensor, required String mount}) async {
    final now = DateTime.now();
    final timestampMs = now.millisecondsSinceEpoch;
    final fallback = await _buildStubFile(sensor: sensor, extension: 'mp4');
    try {
      final response = await _channel.invokeMapMethod<String, dynamic>('captureVideoFrame', {
        'sensor': sensor,
        'mount': mount,
        'timestamp_ms': timestampMs,
      });
      final path = response?['path'] as String? ?? fallback.path;
      final depthPath = response?['depth_path'] as String?;
      final frameId = response?['frame_id'] as String? ?? timestampMs.toString();
      return CapturedVideo(
        path: path,
        frameId: frameId,
        timestampMs: timestampMs,
        mount: mount,
        sensor: sensor,
        depthPath: depthPath,
      );
    } on PlatformException catch (_) {
      return CapturedVideo(
        path: fallback.path,
        frameId: timestampMs.toString(),
        timestampMs: timestampMs,
        mount: mount,
        sensor: sensor,
        depthPath: null,
      );
    }
  }

  Future<File> _buildStubFile({required String sensor, required String extension}) async {
    final directory = await getTemporaryDirectory();
    final file = File('${directory.path}/$sensor-${DateTime.now().millisecondsSinceEpoch}.$extension');
    if (!await file.exists()) {
      await file.create(recursive: true);
      await file.writeAsBytes(const []);
    }
    return file;
  }
}
