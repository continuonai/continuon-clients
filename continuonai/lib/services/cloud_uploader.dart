import 'dart:convert';
import 'dart:io';

import 'package:googleapis_auth/auth_io.dart' as auth;
import 'package:http/http.dart' as http;
import 'package:path_provider/path_provider.dart';

import '../models/rlds_models.dart';

class CloudUploader {
  CloudUploader({http.Client? httpClient}) : _httpClient = httpClient ?? http.Client();

  final http.Client _httpClient;
  auth.AutoRefreshingAuthClient? _authenticatedClient;

  Future<void> signInWithServiceAccount(String jsonCredentials, List<String> scopes) async {
    final credentials = auth.ServiceAccountCredentials.fromJson(jsonCredentials);
    _authenticatedClient = await auth.clientViaServiceAccount(credentials, scopes, baseClient: _httpClient);
  }

  Future<Uri> uploadEpisode(EpisodeRecord record, Uri signedUrl) async {
    final directory = await getTemporaryDirectory();
    final episodeFile = File('${directory.path}/episode.json');
    await episodeFile.writeAsString(jsonEncode(record.toJson()));
    final bytes = await episodeFile.readAsBytes();
    final request = http.Request('PUT', signedUrl)
      ..headers['Content-Type'] = 'application/json'
      ..bodyBytes = bytes;
    final response = await _httpClient.send(request);
    if (response.statusCode >= 200 && response.statusCode < 300) {
      return signedUrl;
    }
    throw HttpException('Upload failed: ${response.statusCode}');
  }

  Future<Uri> fetchSignedUploadUrl(Uri brokerEndpoint, EpisodeMetadata metadata) async {
    final client = _authenticatedClient ?? _httpClient;
    final response = await client.post(
      brokerEndpoint,
      headers: {'Content-Type': 'application/json'},
      body: jsonEncode({'metadata': metadata.toJson()}),
    );
    if (response.statusCode >= 200 && response.statusCode < 300) {
      final body = jsonDecode(response.body) as Map<String, dynamic>;
      return Uri.parse(body['upload_url'] as String);
    }
    throw HttpException('Failed to fetch upload URL: ${response.statusCode}');
  }
}
