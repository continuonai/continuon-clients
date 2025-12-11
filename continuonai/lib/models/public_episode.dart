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
    this.previewThumbUrl,
    this.previewVideoUrl,
  });
}

