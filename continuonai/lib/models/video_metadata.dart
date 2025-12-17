class VideoMetadata {
  final String title;
  final String authorName;
  final String providerUrl;

  VideoMetadata({
    required this.title,
    required this.authorName,
    required this.providerUrl,
  });

  factory VideoMetadata.fromJson(Map<String, dynamic> json) {
    return VideoMetadata(
      title: json['title'] as String? ?? '',
      authorName: json['author_name'] as String? ?? '',
      providerUrl: json['provider_url'] as String? ?? '',
    );
  }
}
