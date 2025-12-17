class EpisodeMetadata {
  EpisodeMetadata({
    required this.xrMode,
    required this.controlRole,
    required this.environmentId,
    this.tags = const [],
    this.softwareVersions,
    this.safety,
    this.share,
    this.schemaVersion,
  });

  final String xrMode;
  final String controlRole;
  final String environmentId;
  final List<String> tags;
  final SoftwareVersions? softwareVersions;
  final SafetyMetadata? safety;
  final ShareMetadata? share;
  final String? schemaVersion;

  EpisodeMetadata copyWith({
    String? xrMode,
    String? controlRole,
    String? environmentId,
    List<String>? tags,
    SoftwareVersions? softwareVersions,
    SafetyMetadata? safety,
    ShareMetadata? share,
    String? schemaVersion,
  }) {
    return EpisodeMetadata(
      xrMode: xrMode ?? this.xrMode,
      controlRole: controlRole ?? this.controlRole,
      environmentId: environmentId ?? this.environmentId,
      tags: tags ?? this.tags,
      softwareVersions: softwareVersions ?? this.softwareVersions,
      safety: safety ?? this.safety,
      share: share ?? this.share,
      schemaVersion: schemaVersion ?? this.schemaVersion,
    );
  }

  Map<String, dynamic> toJson() => {
        'xr_mode': xrMode,
        'control_role': controlRole,
        'environment_id': environmentId,
        'tags': tags,
        if (softwareVersions != null) 'software': softwareVersions!.toJson(),
        if (safety != null) 'safety': safety!.toJson(),
        if (share != null) 'share': share!.toJson(),
        if (schemaVersion != null) 'schema_version': schemaVersion,
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

class SafetyMetadata {
  const SafetyMetadata({
    required this.contentRating,
    required this.intendedAudience,
    required this.piiAttested,
    required this.piiCleared,
    required this.piiRedacted,
    required this.pendingReview,
    this.piiAttestation,
  });

  final ContentRating contentRating;
  final String intendedAudience;
  final bool piiAttested;
  final bool piiCleared;
  final bool piiRedacted;
  final bool pendingReview;
  final PiiAttestation? piiAttestation;

  SafetyMetadata copyWith({
    ContentRating? contentRating,
    String? intendedAudience,
    bool? piiAttested,
    bool? piiCleared,
    bool? piiRedacted,
    bool? pendingReview,
    PiiAttestation? piiAttestation,
  }) {
    return SafetyMetadata(
      contentRating: contentRating ?? this.contentRating,
      intendedAudience: intendedAudience ?? this.intendedAudience,
      piiAttested: piiAttested ?? this.piiAttested,
      piiCleared: piiCleared ?? this.piiCleared,
      piiRedacted: piiRedacted ?? this.piiRedacted,
      pendingReview: pendingReview ?? this.pendingReview,
      piiAttestation: piiAttestation ?? this.piiAttestation,
    );
  }

  Map<String, dynamic> toJson() => {
        'content_rating': contentRating.toJson(),
        'intended_audience': intendedAudience,
        'pii_attested': piiAttested,
        'pii_cleared': piiCleared,
        'pii_redacted': piiRedacted,
        'pending_review': pendingReview,
        if (piiAttestation != null) 'pii_attestation': piiAttestation!.toJson(),
      };
}

class ContentRating {
  const ContentRating({
    required this.audience,
    required this.violence,
    required this.language,
  });

  final String audience;
  final String violence;
  final String language;

  Map<String, dynamic> toJson() => {
        'audience': audience,
        'violence': violence,
        'language': language,
      };
}

class PiiAttestation {
  const PiiAttestation({
    required this.piiPresent,
    required this.facesPresent,
    required this.namesPresent,
    required this.consent,
  });

  final bool piiPresent;
  final bool facesPresent;
  final bool namesPresent;
  final bool consent;

  PiiAttestation copyWith({
    bool? piiPresent,
    bool? facesPresent,
    bool? namesPresent,
    bool? consent,
  }) {
    return PiiAttestation(
      piiPresent: piiPresent ?? this.piiPresent,
      facesPresent: facesPresent ?? this.facesPresent,
      namesPresent: namesPresent ?? this.namesPresent,
      consent: consent ?? this.consent,
    );
  }

  Map<String, dynamic> toJson() => {
        'pii_present': piiPresent,
        'faces_present': facesPresent,
        'name_present': namesPresent,
        'consent': consent,
      };
}

class ShareMetadata {
  const ShareMetadata({
    required this.isPublic,
    required this.slug,
    required this.title,
    required this.license,
    required this.tags,
  });

  final bool isPublic;
  final String slug;
  final String title;
  final String license;
  final List<String> tags;

  ShareMetadata copyWith({
    bool? isPublic,
    String? slug,
    String? title,
    String? license,
    List<String>? tags,
  }) {
    return ShareMetadata(
      isPublic: isPublic ?? this.isPublic,
      slug: slug ?? this.slug,
      title: title ?? this.title,
      license: license ?? this.license,
      tags: tags ?? this.tags,
    );
  }

  Map<String, dynamic> toJson() => {
        'public': isPublic,
        'slug': slug,
        'title': title,
        'license': license,
        'tags': tags,
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
