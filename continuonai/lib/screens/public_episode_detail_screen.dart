import 'package:flutter/material.dart';

import '../models/public_episode.dart';
import '../services/public_episodes_service.dart';
import '../theme/continuon_theme.dart';

class PublicEpisodeDetailScreen extends StatefulWidget {
  const PublicEpisodeDetailScreen({super.key, required this.slug});

  static const routeName = '/episodes/detail';

  final String slug;

  @override
  State<PublicEpisodeDetailScreen> createState() =>
      _PublicEpisodeDetailScreenState();
}

class _PublicEpisodeDetailScreenState extends State<PublicEpisodeDetailScreen> {
  final _service = PublicEpisodesService();
  late Future<PublicEpisode?> _episodeFuture;

  @override
  void initState() {
    super.initState();
    _episodeFuture = _service.fetchBySlug(widget.slug);
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final brand = theme.extension<ContinuonBrandExtension>();
    return Scaffold(
      appBar: AppBar(
        title: const Text('Episode'),
      ),
      body: Container(
        decoration: brand != null
            ? BoxDecoration(gradient: brand.waveGradient)
            : null,
        child: FutureBuilder<PublicEpisode?>(
          future: _episodeFuture,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return Center(
                child: Text(
                  'Failed to load: ${snapshot.error}',
                  style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white),
                ),
              );
            }
            final episode = snapshot.data;
            if (episode == null) {
              return Center(
                child: Text(
                  'Episode not found.',
                  style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white),
                ),
              );
            }
            final durationMinutes = (episode.durationMs / 60000).toStringAsFixed(1);
            final playbackUrl =
                episode.previewVideoUrl ?? episode.assets.previewSignedUrl;
            return SingleChildScrollView(
              padding: const EdgeInsets.all(20),
              child: Column(
                crossAxisAlignment: CrossAxisAlignment.start,
                children: [
                  Row(
                    crossAxisAlignment: CrossAxisAlignment.center,
                    children: [
                      Container(
                        padding: const EdgeInsets.all(12),
                        decoration: BoxDecoration(
                          color: Colors.white.withValues(alpha: 0.12),
                          borderRadius: BorderRadius.circular(12),
                        ),
                        child: const Icon(Icons.public, color: Colors.white, size: 28),
                      ),
                      const SizedBox(width: 12),
                      Expanded(
                        child: Text(
                          episode.title,
                          style: theme.textTheme.headlineSmall
                              ?.copyWith(color: Colors.white, fontWeight: FontWeight.w700),
                        ),
                      ),
                    ],
                  ),
                  const SizedBox(height: 8),
                  Text(
                    '${episode.xrMode} • ${episode.controlRole} • ${episode.robotModel}',
                    style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white70),
                  ),
                  const SizedBox(height: 4),
                  Text(
                    '${episode.share.license} • ${episode.ownerHandle} • ${durationMinutes} min',
                    style: theme.textTheme.bodySmall?.copyWith(color: Colors.white70),
                  ),
                  const SizedBox(height: 12),
                  Wrap(
                    spacing: 8,
                    runSpacing: 6,
                    children: episode.share.tags
                        .map((t) => Chip(
                              label: Text(t),
                              padding: EdgeInsets.zero,
                              visualDensity: VisualDensity.compact,
                            ))
                        .toList(),
                  ),
                  const SizedBox(height: 16),
                  _section(
                    context,
                    title: 'Playback',
                    child: _PlaybackPlaceholder(previewUrl: playbackUrl),
                  ),
                  _section(
                    context,
                    title: 'Download / Timeline',
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        if (episode.assets.timelineSignedUrl != null)
                          Text('Timeline: ${episode.assets.timelineSignedUrl}'),
                        if (episode.assets.downloadSignedUrl != null)
                          Text('Download: ${episode.assets.downloadSignedUrl}'),
                        if (episode.assets.redactedDownloadUrl != null)
                          Text('Redacted: ${episode.assets.redactedDownloadUrl}'),
                        if (episode.assets.timelineSignedUrl == null &&
                            episode.assets.downloadSignedUrl == null)
                          const Text('Signed URLs will populate after ingestion.'),
                      ],
                    ),
                  ),
                  _section(
                    context,
                    title: 'About',
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text('Slug: ${episode.slug}'),
                        Text('Created: ${episode.createdAt.toIso8601String()}'),
                        Text('Owner: ${episode.ownerHandle}'),
                        Text('License: ${episode.share.license}'),
                        Text('Content rating: ${episode.share.contentRating}'),
                        Text('Audience: ${episode.share.intendedAudience}'),
                        Text('Public: ${episode.share.isPublic}'),
                        Text('PII cleared: ${episode.share.piiCleared}'),
                        Text('Redacted assets: ${episode.share.piiRedacted}'),
                      ],
                    ),
                  ),
                ],
              ),
            );
          },
        ),
      ),
    );
  }

  Widget _section(BuildContext context, {required String title, required Widget child}) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.only(top: 16),
      child: Container(
        width: double.infinity,
        padding: const EdgeInsets.all(16),
        decoration: BoxDecoration(
          color: Colors.white.withValues(alpha: 0.06),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.white.withValues(alpha: 0.1)),
        ),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            Text(
              title,
              style: theme.textTheme.titleMedium?.copyWith(
                color: Colors.white,
                fontWeight: FontWeight.w600,
              ),
            ),
            const SizedBox(height: 8),
            DefaultTextStyle.merge(
              style: theme.textTheme.bodyMedium?.copyWith(color: Colors.white70),
              child: child,
            ),
          ],
        ),
      ),
    );
  }
}

class _PlaybackPlaceholder extends StatelessWidget {
  const _PlaybackPlaceholder({this.previewUrl});

  final String? previewUrl;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Container(
      height: 180,
      decoration: BoxDecoration(
        color: Colors.black.withValues(alpha: 0.2),
        borderRadius: BorderRadius.circular(12),
        border: Border.all(color: Colors.white.withValues(alpha: 0.08)),
      ),
      child: Center(
        child: Column(
          mainAxisAlignment: MainAxisAlignment.center,
          children: [
            const Icon(Icons.ondemand_video, color: Colors.white70, size: 36),
            const SizedBox(height: 8),
            Text(
              previewUrl == null
                  ? 'Preview video will appear here (signed URL).'
                  : 'Preview available (mock).',
              style: theme.textTheme.bodySmall?.copyWith(color: Colors.white70),
            ),
          ],
        ),
      ),
    );
  }
}

