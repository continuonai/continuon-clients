class EpisodeMetadata {
  EpisodeMetadata({
    required this.xrMode,
    required this.controlRole,
    required this.environmentId,
    this.tags = const [],
    this.softwareVersions,
  });

  final String xrMode;
  final String controlRole;
  final String environmentId;
  final List<String> tags;
  final SoftwareVersions? softwareVersions;

  Map<String, dynamic> toJson() => {
        'xr_mode': xrMode,
        'control_role': controlRole,
        'environment_id': environmentId,
        'tags': tags,
        if (softwareVersions != null) 'software': softwareVersions!.toJson(),
      };
}

class EpisodeAsset {
  EpisodeAsset({
    required this.localUri,
    required this.sensor,
    required this.frameId,
    required this.capturedAtMillis,
    this.mount,
    this.remoteUri,
    this.mimeType,
    this.status = 'queued',
  });

  final String localUri;
  final String sensor;
  final String frameId;
  final int capturedAtMillis;
  final String? mount;
  final String? remoteUri;
  final String? mimeType;
  final String status;

  Map<String, dynamic> toJson() => {
        'local_uri': localUri,
        'sensor': sensor,
        'frame_id': frameId,
        'captured_at_ms': capturedAtMillis,
        if (mount != null) 'mount': mount,
        if (remoteUri != null) 'remote_uri': remoteUri,
        if (mimeType != null) 'mime_type': mimeType,
        'status': status,
      };

  EpisodeAsset withRemote({required String uri}) => EpisodeAsset(
        localUri: localUri,
        sensor: sensor,
        frameId: frameId,
        capturedAtMillis: capturedAtMillis,
        mount: mount,
        mimeType: mimeType,
        remoteUri: uri,
        status: 'transferred',
      );
}

class SoftwareVersions {
  const SoftwareVersions({
    this.xrApp,
    this.continuonBrainOs,
    this.gloveFirmware,
  });

  final String? xrApp;
  final String? continuonBrainOs;
  final String? gloveFirmware;

  Map<String, dynamic> toJson() => {
        if (xrApp != null) 'xr_app': xrApp,
        if (continuonBrainOs != null) 'continuonbrain_os': continuonBrainOs,
        if (gloveFirmware != null) 'glove_firmware': gloveFirmware,
      };
}

class EpisodeStep {
  EpisodeStep({
    required this.observation,
    required this.action,
    this.isTerminal = false,
    this.stepMetadata = const {},
  });

  final Map<String, dynamic> observation;
  final Map<String, dynamic> action;
  final bool isTerminal;
  final Map<String, String> stepMetadata;

  Map<String, dynamic> toJson() => {
        'observation': observation,
        'action': action,
        'is_terminal': isTerminal,
        'step_metadata': stepMetadata,
      };
}

class EpisodeRecord {
  EpisodeRecord({
    required this.metadata,
    this.steps = const [],
    this.assets = const [],
  });

  final EpisodeMetadata metadata;
  final List<EpisodeStep> steps;
  final List<EpisodeAsset> assets;

  Map<String, dynamic> toJson() => {
        'metadata': metadata.toJson(),
        'steps': steps.map((s) => s.toJson()).toList(),
        if (assets.isNotEmpty) 'assets': assets.map((a) => a.toJson()).toList(),
      };
}
