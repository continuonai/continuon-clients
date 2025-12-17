class ShareMetadata {
  const ShareMetadata({
    required this.isPublic,
    required this.slug,
    required this.title,
    required this.license,
    required this.tags,
    required this.contentRating,
    required this.intendedAudience,
    required this.piiAttested,
    this.piiCleared = false,
    this.piiRedacted = false,
    this.pendingReview = true,
    this.description,
  });

  final bool isPublic;
  final String slug;
  final String title;
  final String license;
  final List<String> tags;
  final String contentRating;
  final String intendedAudience;
  final bool piiAttested;
  final bool piiCleared;
  final bool piiRedacted;
  final bool pendingReview;
  final String? description;

  ShareMetadata copyWith({
    bool? isPublic,
    bool? piiCleared,
    bool? piiRedacted,
    bool? pendingReview,
  }) {
    return ShareMetadata(
      isPublic: isPublic ?? this.isPublic,
      slug: slug,
      title: title,
      license: license,
      tags: tags,
      contentRating: contentRating,
      intendedAudience: intendedAudience,
      piiAttested: piiAttested,
      piiCleared: piiCleared ?? this.piiCleared,
      piiRedacted: piiRedacted ?? this.piiRedacted,
      pendingReview: pendingReview ?? this.pendingReview,
      description: description,
    );
  }

  Map<String, dynamic> toJson() => {
        'public': isPublic,
        'slug': slug,
        'title': title,
        'license': license,
        'tags': tags,
        'content_rating': contentRating,
        'intended_audience': intendedAudience,
        'pii_attested': piiAttested,
        'pii_cleared': piiCleared,
        'pii_redacted': piiRedacted,
        'pending_review': pendingReview,
        if (description != null) 'description': description,
      };

  factory ShareMetadata.fromJson(Map<String, dynamic> json) => ShareMetadata(
        isPublic: json['public'] as bool? ?? false,
        slug: json['slug'] as String,
        title: json['title'] as String,
        license: json['license'] as String,
        tags: List<String>.from(json['tags'] as List? ?? const []),
        contentRating: json['content_rating'] as String? ?? 'general',
        intendedAudience: json['intended_audience'] as String? ?? 'general',
        piiAttested: json['pii_attested'] as bool? ?? false,
        piiCleared: json['pii_cleared'] as bool? ?? false,
        piiRedacted: json['pii_redacted'] as bool? ?? false,
        pendingReview: json['pending_review'] as bool? ?? true,
        description: json['description'] as String?,
      );
}

class SignedAssetUrls {
  const SignedAssetUrls({
    this.previewSignedUrl,
    this.timelineSignedUrl,
    this.downloadSignedUrl,
    this.redactedDownloadUrl,
  });

  final String? previewSignedUrl;
  final String? timelineSignedUrl;
  final String? downloadSignedUrl;
  final String? redactedDownloadUrl;

  factory SignedAssetUrls.fromJson(Map<String, dynamic> json) => SignedAssetUrls(
        previewSignedUrl: json['preview'] as String?,
        timelineSignedUrl: json['timeline'] as String?,
        downloadSignedUrl: json['download'] as String?,
        redactedDownloadUrl: json['redacted_download'] as String?,
      );

  Map<String, dynamic> toJson() => {
        if (previewSignedUrl != null) 'preview': previewSignedUrl,
        if (timelineSignedUrl != null) 'timeline': timelineSignedUrl,
        if (downloadSignedUrl != null) 'download': downloadSignedUrl,
        if (redactedDownloadUrl != null) 'redacted_download': redactedDownloadUrl,
      };
}

class PublicEpisode {
  final String slug;
  final String title;
  final List<String> tags;
  final String license;
  final String ownerHandle;
  final String robotModel;
  final String xrMode;
  final String controlRole;
  final int durationMs;
  final DateTime createdAt;
  final String? previewThumbUrl;
  final String? previewVideoUrl;
  final ShareMetadata share;
  final SignedAssetUrls assets;

  const PublicEpisode({
    required this.slug,
    required this.title,
    required this.tags,
    required this.license,
    required this.ownerHandle,
    required this.robotModel,
    required this.xrMode,
    required this.controlRole,
    required this.durationMs,
    required this.createdAt,
    required this.share,
    required this.assets,
    this.previewThumbUrl,
    this.previewVideoUrl,
  });

  factory PublicEpisode.fromJson(Map<String, dynamic> json) {
    final shareJson = Map<String, dynamic>.from(json['share'] as Map? ?? {});
    final share = ShareMetadata.fromJson({
      ...shareJson,
      if (!shareJson.containsKey('slug') && json['slug'] != null)
        'slug': json['slug'],
      if (!shareJson.containsKey('title') && json['title'] != null)
        'title': json['title'],
      if (!shareJson.containsKey('license') && json['license'] != null)
        'license': json['license'],
      if (!shareJson.containsKey('tags') && json['tags'] != null)
        'tags': json['tags'],
    });
    final assets = SignedAssetUrls.fromJson(
        Map<String, dynamic>.from(json['assets'] as Map? ?? {}));
    return PublicEpisode(
      slug: share.slug,
      title: share.title,
      tags: share.tags,
      license: share.license,
      ownerHandle: json['owner_handle'] as String? ?? '',
      robotModel: json['robot_model'] as String? ?? 'unknown',
      xrMode: json['xr_mode'] as String? ?? 'unknown',
      controlRole: json['control_role'] as String? ?? 'unknown',
      durationMs: json['duration_ms'] is int
          ? json['duration_ms'] as int
          : int.tryParse('${json['duration_ms']}') ?? 0,
      createdAt: DateTime.tryParse(json['created_at'] as String? ?? '') ??
          DateTime.now(),
      previewThumbUrl: json['preview_thumb_url'] as String?,
      previewVideoUrl:
          json['preview_video_url'] as String? ?? assets.previewSignedUrl,
      share: share,
      assets: assets,
    );
  }
}

