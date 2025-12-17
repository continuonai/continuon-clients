import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../models/public_episode.dart';
import '../models/rlds_models.dart';

class TrainingQueueService {
  TrainingQueueService({http.Client? httpClient})
      : _client = httpClient ?? http.Client();

  final http.Client _client;

  Future<void> enqueueEpisode({
    required String robotHost,
    int httpPort = 8080,
    bool useTls = false,
    String? authToken,
    required EpisodeRecord record,
    ShareMetadata? share,
    bool requestPublicListing = false,
  }) async {
    final uri = Uri(
      scheme: useTls ? 'https' : 'http',
      host: robotHost,
      port: httpPort,
      path: '/api/training/queue/episodes',
    );
    final response = await _client.post(
      uri,
      headers: {
        'Content-Type': 'application/json',
        if (authToken != null) 'authorization': 'Bearer $authToken',
      },
      body: jsonEncode({
        'episode': record.toJson(),
        'request_public_listing': requestPublicListing,
        if (share != null) 'share': share.toJson(),
      }),
    );
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
          'Failed to queue episode: HTTP ${response.statusCode} ${response.body}');
    }
  }
}
