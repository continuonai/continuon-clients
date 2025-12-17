import 'dart:async';
import 'dart:convert';
import 'dart:io';

import 'package:http/http.dart' as http;

import '../models/public_episode.dart';
import '../models/rlds_models.dart' as rlds;
import 'cloud_uploader.dart';

class PublicEpisodesService {
  PublicEpisodesService({
    http.Client? httpClient,
    CloudUploader? uploader,
    Uri? apiBase,
    Uri? uploadSigner,
  })  : _client = httpClient ?? http.Client(),
        _uploader = uploader ?? CloudUploader(),
        _apiBase = apiBase ??
            Uri.parse('https://cloud.continuonai.com/api/public-episodes'),
        _uploadSigner = uploadSigner ??
            Uri.parse(
                'https://cloud.continuonai.com/api/public-episodes/sign-uploads');

  final http.Client _client;
  final CloudUploader _uploader;
  final Uri _apiBase;
  final Uri _uploadSigner;

  Future<PublicEpisode?> fetchBySlug(String slug) async {
    final uri = _apiBase.replace(path: '${_apiBase.path}/$slug');
    final response = await _client.get(uri);
    if (response.statusCode == 404) return null;
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
          'Failed to fetch episode: HTTP ${response.statusCode}');
    }
    final decoded = jsonDecode(response.body);
    final json = decoded is Map<String, dynamic>
        ? decoded
        : Map<String, dynamic>.from(decoded as Map);
    final episode = PublicEpisode.fromJson(json);
    if (!_isPublicReady(episode)) return null;
    return episode;
  }

  Future<List<PublicEpisode>> fetchPublicEpisodes() async {
    final response = await _client.get(_apiBase);
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
          'Failed to fetch episodes: HTTP ${response.statusCode}');
    }
    final decoded = jsonDecode(response.body);
    final rawList = decoded is List
        ? decoded
        : (decoded is Map<String, dynamic>
            ? decoded['episodes'] as List? ?? []
            : []);
    final episodes = rawList
        .whereType<Map>()
        .map((e) => PublicEpisode.fromJson(Map<String, dynamic>.from(e)))
        .where(_isPublicReady)
        .toList();
    return episodes;
  }

  Future<PublicEpisodeUploadSession> prepareUpload(ShareMetadata share) async {
    final response = await _client.post(
      _uploadSigner,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'share': share.toJson()}),
    );
    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
          'Failed to sign uploads: HTTP ${response.statusCode}');
    }
    final payload = jsonDecode(response.body) as Map<String, dynamic>;
    return PublicEpisodeUploadSession.fromJson(payload);
  }

  Future<PublicEpisode> publishEpisode({
    required rlds.EpisodeRecord record,
    required ShareMetadata share,
    required PublicEpisodeUploadSession uploadSession,
    List<int>? previewBytes,
    List<int>? timelineBytes,
  }) async {
    final uploadedEpisodeUrl =
        await _uploader.uploadEpisode(record, uploadSession.rldsUploadUrl);
    if (uploadSession.previewUploadUrl != null && previewBytes != null) {
      await _putBytes(uploadSession.previewUploadUrl!, previewBytes,
          contentType: 'video/mp4');
    }
    if (uploadSession.timelineUploadUrl != null && timelineBytes != null) {
      await _putBytes(uploadSession.timelineUploadUrl!, timelineBytes,
          contentType: 'application/json');
    }

    final assets = uploadSession.assets.toJson();
    assets['download'] =
        uploadSession.assets.downloadSignedUrl ?? uploadedEpisodeUrl.toString();

    final response = await _client.post(
      _apiBase,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({
        'share': share.toJson(),
        'rlds_url': uploadedEpisodeUrl.toString(),
        'assets': assets,
      }),
    );

    if (response.statusCode < 200 || response.statusCode >= 300) {
      throw HttpException(
          'Failed to publish episode: HTTP ${response.statusCode}');
    }
    return PublicEpisode.fromJson(
        jsonDecode(response.body) as Map<String, dynamic>);
  }

  bool _isPublicReady(PublicEpisode episode) {
    final share = episode.share;
    return share.isPublic && share.piiCleared && !share.pendingReview;
  }

  Future<void> _putBytes(Uri url, List<int> bytes,
      {String? contentType}) async {
    final request = http.Request('PUT', url)
      ..bodyBytes = bytes
      ..headers.addAll({
        if (contentType != null) 'Content-Type': contentType,
      });
    final streamed = await _client.send(request);
    if (streamed.statusCode < 200 || streamed.statusCode >= 300) {
      throw HttpException('Upload failed: HTTP ${streamed.statusCode}');
    }
  }
}

class PublicEpisodeUploadSession {
  const PublicEpisodeUploadSession({
    required this.rldsUploadUrl,
    required this.assets,
    this.previewUploadUrl,
    this.timelineUploadUrl,
  });

  final Uri rldsUploadUrl;
  final Uri? previewUploadUrl;
  final Uri? timelineUploadUrl;
  final SignedAssetUrls assets;

  factory PublicEpisodeUploadSession.fromJson(Map<String, dynamic> json) {
    final uploads = json['upload_urls'] as Map? ?? {};
    final assets = json['assets'] as Map? ?? {};
    final rldsString = (uploads['rlds'] ?? json['rlds_upload_url']) as String?;
    if (rldsString == null) {
      throw StateError('Missing RLDS upload URL in signer response');
    }
    return PublicEpisodeUploadSession(
      rldsUploadUrl: Uri.parse(rldsString),
      previewUploadUrl:
          (uploads['preview'] ?? json['preview_upload_url']) != null
              ? Uri.parse(
                  (uploads['preview'] ?? json['preview_upload_url']) as String)
              : null,
      timelineUploadUrl: (uploads['timeline'] ?? json['timeline_upload_url']) !=
              null
          ? Uri.parse(
              (uploads['timeline'] ?? json['timeline_upload_url']) as String)
          : null,
      assets: SignedAssetUrls.fromJson(Map<String, dynamic>.from(assets)),
    );
  }
}
