import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../models/public_episode.dart';
import '../models/rlds_models.dart' as rlds;
import 'public_episodes_service.dart';
import 'training_queue_service.dart';

class YoutubeImportResult {
  const YoutubeImportResult({
    required this.record,
    this.publishedEpisode,
  });

  final rlds.EpisodeRecord record;
  final PublicEpisode? publishedEpisode;
}

class YoutubeImportService {
  YoutubeImportService({
    http.Client? httpClient,
    PublicEpisodesService? publicEpisodesService,
    TrainingQueueService? trainingQueueService,
    Uri? importEndpoint,
  })  : _client = httpClient ?? http.Client(),
        _publicEpisodesService = publicEpisodesService ??
            PublicEpisodesService(httpClient: httpClient),
        _trainingQueueService = trainingQueueService ??
            TrainingQueueService(httpClient: httpClient),
        _importEndpoint = importEndpoint ??
            Uri.parse('https://cloud.continuonai.com/api/import/youtube');

  final http.Client _client;
  final PublicEpisodesService _publicEpisodesService;
  final TrainingQueueService _trainingQueueService;
  final Uri _importEndpoint;

  Future<YoutubeImportResult> importAndPublish({
    required String youtubeUrl,
    required ShareMetadata share,
    bool requestPublicListing = false,
    String? robotHost,
    int robotHttpPort = 8080,
    String? robotAuthToken,
    bool robotUseTls = false,
  }) async {
    final record = await _fetchTranscodeAndBuildRecord(youtubeUrl);
    final normalizedShare = share.copyWith(
      isPublic: requestPublicListing || share.isPublic,
      piiCleared: share.piiCleared,
      piiRedacted: share.piiRedacted,
      pendingReview: share.piiCleared ? share.pendingReview : true,
    );

    final uploadSession =
        await _publicEpisodesService.prepareUpload(normalizedShare);
    final publishedEpisode = await _publicEpisodesService.publishEpisode(
      record: record,
      share: normalizedShare,
      uploadSession: uploadSession,
    );

    if (robotHost != null && robotHost.isNotEmpty) {
      final rldsShare = rlds.ShareMetadata(
        isPublic: normalizedShare.isPublic,
        slug: normalizedShare.slug,
        title: normalizedShare.title,
        license: normalizedShare.license,
        tags: normalizedShare.tags,
      );

      await _trainingQueueService.enqueueEpisode(
        robotHost: robotHost,
        httpPort: robotHttpPort,
        useTls: robotUseTls,
        authToken: robotAuthToken,
        record: record,
        share: rldsShare,
        requestPublicListing: requestPublicListing,
      );
    }

    return YoutubeImportResult(
        record: record, publishedEpisode: publishedEpisode);
  }

  Future<rlds.EpisodeRecord> _fetchTranscodeAndBuildRecord(
      String youtubeUrl) async {
    final response = await _client.post(
      _importEndpoint,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'youtube_url': youtubeUrl}),
    );
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
          'Import failed: HTTP ${response.statusCode} ${response.body}');
    }
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    final classification = rlds.TaskClassification(
      tasks: List<String>.from(
          payload['task_labels'] as List? ?? const ['unlabeled']),
      motionPrimitives: List<String>.from(
          payload['motion_primitives'] as List? ?? const ['unknown_motion']),
      confidence: (payload['task_confidence'] as Map?)?.map(
        (key, value) => MapEntry('$key', double.tryParse('$value') ?? 0.0),
      ),
    );
    final metadata = rlds.EpisodeMetadata(
      xrMode: payload['xr_mode'] as String? ?? 'youtube_tv',
      controlRole: payload['control_role'] as String? ?? 'observer',
      environmentId: payload['environment_id'] as String? ?? 'youtube_import',
      tags: List<String>.from(
          payload['tags'] as List? ?? const ['youtube', 'import']),
      source: 'youtube',
      provenance: {
        'video_url': youtubeUrl,
        if (payload['job_id'] != null) 'job_id': payload['job_id'],
        if (payload['transcode_bucket'] != null)
          'transcode_bucket': payload['transcode_bucket'],
      },
    );
    final steps = <rlds.EpisodeStep>[];
    final rawSteps = payload['steps'] as List?;
    if (rawSteps != null) {
      for (final step in rawSteps.whereType<Map>()) {
        steps.add(
          rlds.EpisodeStep(
            observation:
                Map<String, dynamic>.from(step['observation'] as Map? ?? {}),
            action: Map<String, dynamic>.from(step['action'] as Map? ?? {}),
            isTerminal: step['is_terminal'] as bool? ?? false,
            stepMetadata: Map<String, String>.from(
                step['step_metadata'] as Map? ??
                    const {'source': 'youtube_import'}),
          ),
        );
      }
    }
    return rlds.EpisodeRecord(
      metadata: metadata,
      steps: steps,
      taskClassification: classification,
      sourceContext: {
        'youtube_url': youtubeUrl,
        'transcoded_video': payload['transcoded_url'],
      },
    );
  }
}
