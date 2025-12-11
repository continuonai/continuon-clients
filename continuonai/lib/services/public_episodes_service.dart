import 'dart:async';

import '../models/public_episode.dart';

/// Temporary mock service; replace with real HTTPS client once the public
/// episodes API is live.
class PublicEpisodesService {
  Future<PublicEpisode?> fetchBySlug(String slug) async {
    final list = await fetchPublicEpisodes();
    return list.cast<PublicEpisode?>().firstWhere(
          (e) => e?.slug == slug,
          orElse: () => null,
        );
  }

  Future<List<PublicEpisode>> fetchPublicEpisodes() async {
    await Future<void>.delayed(const Duration(milliseconds: 300));
    final now = DateTime.now();
    return [
      PublicEpisode(
        slug: 'assembly-demo-1',
        title: 'Drawer Assembly with Bimanual Assist',
        tags: const ['assembly', 'bimanual', 'workspace'],
        license: 'continuon-eval-only',
        ownerHandle: '@continuon.ai/demo',
        robotModel: 'Franka Emika Panda',
        xrMode: 'workstation',
        controlRole: 'human_teleop',
        durationMs: 420000,
        createdAt: now.subtract(const Duration(days: 2)),
        previewThumbUrl: null,
        previewVideoUrl: null,
      ),
      PublicEpisode(
        slug: 'kitchen-pick-vision',
        title: 'Kitchen Pick and Place (Vision-only)',
        tags: const ['pick-place', 'kitchen', 'vision'],
        license: 'cc-by-4.0',
        ownerHandle: '@continuon.ai/labs',
        robotModel: 'UR5e',
        xrMode: 'trainer',
        controlRole: 'human_supervisor',
        durationMs: 210000,
        createdAt: now.subtract(const Duration(days: 6)),
        previewThumbUrl: null,
        previewVideoUrl: null,
      ),
    ];
  }
}

