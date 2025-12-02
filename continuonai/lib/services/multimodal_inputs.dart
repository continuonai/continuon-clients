import '../models/teleop_models.dart';

class AudioSample {
  const AudioSample({
    required this.uri,
    required this.sampleRateHz,
    required this.numChannels,
    required this.frameId,
    this.source,
  });

  final String uri;
  final int sampleRateHz;
  final int numChannels;
  final String frameId;
  final String? source;

  Map<String, dynamic> toJson() => {
        'uri': uri,
        'sample_rate_hz': sampleRateHz,
        'num_channels': numChannels,
        'frame_id': frameId,
        if (source != null) 'source': source,
      };
}

class EgocentricFrame {
  const EgocentricFrame({
    required this.uri,
    required this.frameId,
    this.depthUri,
  });

  final String uri;
  final String frameId;
  final String? depthUri;

  Map<String, dynamic> toVideoJson() => {
        'uri': uri,
        'frame_id': frameId,
      };

  Map<String, dynamic>? toDepthJson() =>
      depthUri != null ? {'uri': depthUri, 'frame_id': frameId} : null;
}

class GazeSample {
  const GazeSample({
    required this.origin,
    required this.direction,
    this.confidence,
    this.targetId,
  });

  final Vector3 origin;
  final Vector3 direction;
  final double? confidence;
  final String? targetId;

  Map<String, dynamic> toJson() => {
        'origin': origin.toJson(),
        'direction': direction.toJson(),
        if (confidence != null) 'confidence': confidence,
        if (targetId != null) 'target_id': targetId,
      };
}

class MultimodalInputs {
  AudioSample? _deviceMic;
  AudioSample? _robotMic;
  EgocentricFrame? _egocentricFrame;
  GazeSample? _gaze;

  void updateDeviceMic({
    required String uri,
    required String frameId,
    int sampleRateHz = 16000,
    int numChannels = 1,
  }) {
    _deviceMic = AudioSample(
      uri: uri,
      sampleRateHz: sampleRateHz,
      numChannels: numChannels,
      frameId: frameId,
      source: 'device_mic',
    );
  }

  void updateRobotMic({
    required String uri,
    required String frameId,
    int sampleRateHz = 16000,
    int numChannels = 1,
  }) {
    _robotMic = AudioSample(
      uri: uri,
      sampleRateHz: sampleRateHz,
      numChannels: numChannels,
      frameId: frameId,
      source: 'robot_mic',
    );
  }

  void clearAudio() {
    _deviceMic = null;
    _robotMic = null;
  }

  void updateEgocentricFrame({
    required String uri,
    required String frameId,
    String? depthUri,
  }) {
    _egocentricFrame = EgocentricFrame(uri: uri, frameId: frameId, depthUri: depthUri);
  }

  void clearEgocentricFrame() {
    _egocentricFrame = null;
  }

  void updateGaze({
    required Vector3 origin,
    required Vector3 direction,
    double? confidence,
    String? targetId,
  }) {
    _gaze = GazeSample(
      origin: origin,
      direction: direction,
      confidence: confidence,
      targetId: targetId,
    );
  }

  void clearGaze() {
    _gaze = null;
  }

  Map<String, dynamic> buildObservation({Map<String, dynamic>? uiContext}) {
    final observation = <String, dynamic>{};

    final audio = <String, dynamic>{};
    if (_deviceMic != null) {
      audio['device_mic'] = _deviceMic!.toJson();
    }
    if (_robotMic != null) {
      audio['robot_mic'] = _robotMic!.toJson();
    }
    if (audio.isNotEmpty) {
      observation['audio'] = audio;
    }

    if (_egocentricFrame != null) {
      observation['egocentric_video'] = _egocentricFrame!.toVideoJson();
      final depth = _egocentricFrame!.toDepthJson();
      if (depth != null) {
        observation['egocentric_depth'] = depth;
      }
    }

    if (_gaze != null) {
      observation['gaze'] = _gaze!.toJson();
    }

    if (uiContext != null && uiContext.isNotEmpty) {
      observation['ui_context'] = uiContext;
    }

    return observation;
  }
}
