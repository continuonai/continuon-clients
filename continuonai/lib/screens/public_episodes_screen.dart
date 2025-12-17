import 'package:flutter/material.dart';

import '../models/public_episode.dart';
import '../services/public_episodes_service.dart';
import '../theme/continuon_theme.dart';
import 'public_episode_detail_screen.dart';
import '../widgets/layout/continuon_card.dart';
import '../widgets/layout/continuon_layout.dart';

class PublicEpisodesScreen extends StatefulWidget {
  const PublicEpisodesScreen({super.key});

  static const routeName = '/episodes';

  @override
  State<PublicEpisodesScreen> createState() => _PublicEpisodesScreenState();
}

class _PublicEpisodesScreenState extends State<PublicEpisodesScreen> {
  final _service = PublicEpisodesService();
  late Future<List<PublicEpisode>> _episodesFuture;
  String? _selectedLicense;
  String? _selectedTag;

  @override
  void initState() {
    super.initState();
    _episodesFuture = _service.fetchPublicEpisodes();
  }

  void _applyFilters(String? license, String? tag) {
    setState(() {
      _selectedLicense = license;
      _selectedTag = tag;
    });
  }

  List<PublicEpisode> _filter(List<PublicEpisode> episodes) {
    return episodes.where((e) {
      final licOk =
          _selectedLicense == null || e.share.license == _selectedLicense;
      final tagOk = _selectedTag == null || e.share.tags.contains(_selectedTag);
      return licOk && tagOk;
    }).toList();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final brand = theme.extension<ContinuonBrandExtension>();
    return ContinuonLayout(
      appBarActions: [
        IconButton(
          tooltip: 'Refresh',
          icon: const Icon(Icons.refresh),
          onPressed: () {
            setState(() {
              _episodesFuture = _service.fetchPublicEpisodes();
            });
          },
        ),
      ],
      body: Container(
        decoration:
            brand != null ? BoxDecoration(gradient: brand.waveGradient) : null,
        child: FutureBuilder<List<PublicEpisode>>(
          future: _episodesFuture,
          builder: (context, snapshot) {
            if (snapshot.connectionState == ConnectionState.waiting) {
              return const Center(child: CircularProgressIndicator());
            }
            if (snapshot.hasError) {
              return Center(
                child: Text(
                  'Failed to load episodes: ${snapshot.error}',
                  style:
                      theme.textTheme.bodyMedium?.copyWith(color: Colors.white),
                  textAlign: TextAlign.center,
                ),
              );
            }
            final episodes = _filter(snapshot.data ?? []);
            if (episodes.isEmpty) {
              return Center(
                child: Text(
                  'No public episodes yet.\nPublish from Continuon Brain Studio or Continuon AI uploader with sharing enabled.',
                  style: theme.textTheme.bodyMedium
                      ?.copyWith(color: Colors.white70),
                  textAlign: TextAlign.center,
                ),
              );
            }
            final allTags = <String>{
              for (final e in snapshot.data ?? []) ...e.share.tags,
            }.toList()
              ..sort();
            final licenses = <String>{
              for (final e in snapshot.data ?? []) e.share.license,
            }.toList()
              ..sort();
            return Column(
              children: [
                _Filters(
                  tags: allTags,
                  licenses: licenses,
                  selectedLicense: _selectedLicense,
                  selectedTag: _selectedTag,
                  onChanged: _applyFilters,
                ),
                Expanded(
                  child: ListView.builder(
                    padding: const EdgeInsets.all(16),
                    itemCount: episodes.length,
                    itemBuilder: (context, index) {
                      return _EpisodeCard(episode: episodes[index]);
                    },
                  ),
                ),
              ],
            );
          },
        ),
      ),
    );
  }
}

class _Filters extends StatelessWidget {
  const _Filters({
    required this.tags,
    required this.licenses,
    required this.selectedLicense,
    required this.selectedTag,
    required this.onChanged,
  });

  final List<String> tags;
  final List<String> licenses;
  final String? selectedLicense;
  final String? selectedTag;
  final void Function(String? license, String? tag) onChanged;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    return Padding(
      padding: const EdgeInsets.fromLTRB(16, 12, 16, 4),
      child: Wrap(
        spacing: 12,
        runSpacing: 8,
        crossAxisAlignment: WrapCrossAlignment.center,
        children: [
          _filterChip(
            context,
            label: selectedLicense == null ? 'All licenses' : selectedLicense!,
            items: licenses,
            onClear: () => onChanged(null, selectedTag),
            onSelect: (value) => onChanged(value, selectedTag),
          ),
          _filterChip(
            context,
            label: selectedTag == null ? 'All tags' : selectedTag!,
            items: tags,
            onClear: () => onChanged(selectedLicense, null),
            onSelect: (value) => onChanged(selectedLicense, value),
          ),
          Text(
            'Fetched from Continuon Cloud (public share only; PII cleared)',
            style: theme.textTheme.bodySmall?.copyWith(color: Colors.white70),
          ),
        ],
      ),
    );
  }

  Widget _filterChip(
    BuildContext context, {
    required String label,
    required List<String> items,
    required VoidCallback onClear,
    required ValueChanged<String> onSelect,
  }) {
    return PopupMenuButton<String>(
      tooltip: 'Filter',
      onSelected: onSelect,
      itemBuilder: (context) => [
        ...items.map(
          (i) => PopupMenuItem<String>(
            value: i,
            child: Text(i),
          ),
        ),
        const PopupMenuDivider(),
        PopupMenuItem<String>(
          value: '_clear',
          child: const Text('Clear'),
          onTap: onClear,
        ),
      ],
      child: Chip(
        label: Text(label),
        deleteIcon: const Icon(Icons.close),
        onDeleted: onClear,
      ),
    );
  }
}

class _EpisodeCard extends StatelessWidget {
  const _EpisodeCard({required this.episode});

  final PublicEpisode episode;

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final durationMinutes = (episode.durationMs / 60000).toStringAsFixed(1);
    final share = episode.share;
    return ContinuonCard(
      margin: const EdgeInsets.only(bottom: 12),
      padding: EdgeInsets.zero,
      backgroundColor: theme.cardColor
          .withOpacity(0.9), // Keep opacity for glass effect if needed
      child: ListTile(
        contentPadding: const EdgeInsets.all(16),
        leading: Container(
          width: 56,
          height: 56,
          decoration: BoxDecoration(
            color: ContinuonColors.primaryBlue.withValues(alpha: 0.1),
            borderRadius: BorderRadius.circular(12),
          ),
          child: const Icon(Icons.play_circle_fill,
              color: ContinuonColors.primaryBlue),
        ),
        title: Text(
          episode.title,
          style: theme.textTheme.titleMedium
              ?.copyWith(fontWeight: FontWeight.w600),
        ),
        subtitle: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            const SizedBox(height: 4),
            Text(
              '${episode.xrMode} • ${episode.controlRole} • ${episode.robotModel} • ${share.contentRating}',
              style: theme.textTheme.bodySmall,
            ),
            const SizedBox(height: 4),
            Text(
              '${share.license} • ${episode.ownerHandle} • ${durationMinutes} min',
              style:
                  theme.textTheme.bodySmall?.copyWith(color: Colors.grey[600]),
            ),
            const SizedBox(height: 6),
            Wrap(
              spacing: 6,
              runSpacing: 4,
              children: share.tags
                  .map((t) => Chip(
                        label: Text(t),
                        padding: EdgeInsets.zero,
                        visualDensity: VisualDensity.compact,
                      ))
                  .toList(),
            ),
          ],
        ),
        onTap: () {
          Navigator.push(
            context,
            MaterialPageRoute(
              builder: (_) => PublicEpisodeDetailScreen(slug: episode.slug),
            ),
          );
        },
      ),
    );
  }
}
