/// Ralph Service for ContinuonAI Flutter App
/// ==========================================
///
/// Integrates the Ralph Layer system with the Flutter companion app.
/// Provides secure API key storage and interfaces with the Meta Ralph Agent.

import 'dart:async';
import 'dart:convert';

import 'package:flutter/foundation.dart';
import 'package:flutter_secure_storage/flutter_secure_storage.dart';
import 'package:http/http.dart' as http;
import 'package:shared_preferences/shared_preferences.dart';

/// Available CLI providers for the Ralph layer
enum CLIProvider {
  claude('claude', 'Claude Code (Opus 4.5)'),
  gemini('gemini', 'Gemini CLI'),
  ollama('ollama', 'Ollama (Local)'),
  openai('openai', 'OpenAI API');

  const CLIProvider(this.value, this.displayName);
  final String value;
  final String displayName;
}

/// Secure storage keys for API credentials
class _StorageKeys {
  static const anthropicApiKey = 'ralph_anthropic_api_key';
  static const geminiApiKey = 'ralph_gemini_api_key';
  static const openaiApiKey = 'ralph_openai_api_key';
  static const ollamaEndpoint = 'ralph_ollama_endpoint';
  static const defaultProvider = 'ralph_default_provider';
}

/// Response from a Ralph chat interaction
class RalphChatResponse {
  final String text;
  final List<String> actionsTaken;
  final List<Map<String, dynamic>> guardrailsTriggered;
  final Map<String, dynamic> loopStates;
  final String? provider;
  final double? latencyMs;
  final String? error;
  final String sessionId;
  final String timestamp;

  RalphChatResponse({
    required this.text,
    this.actionsTaken = const [],
    this.guardrailsTriggered = const [],
    this.loopStates = const {},
    this.provider,
    this.latencyMs,
    this.error,
    required this.sessionId,
    required this.timestamp,
  });

  factory RalphChatResponse.fromJson(Map<String, dynamic> json) {
    final response = json['response'] as Map<String, dynamic>? ?? {};
    return RalphChatResponse(
      text: response['text'] as String? ?? '',
      actionsTaken:
          (response['actions'] as List<dynamic>?)?.cast<String>() ?? [],
      guardrailsTriggered:
          (response['guardrails_triggered'] as List<dynamic>?)
              ?.cast<Map<String, dynamic>>() ??
          [],
      loopStates:
          response['loop_states'] as Map<String, dynamic>? ?? {},
      provider: response['provider'] as String?,
      latencyMs: (response['latency_ms'] as num?)?.toDouble(),
      error: response['error'] as String?,
      sessionId: json['session_id'] as String? ?? 'default',
      timestamp: json['timestamp'] as String? ?? DateTime.now().toIso8601String(),
    );
  }

  bool get hasError => error != null && error!.isNotEmpty;
  bool get hasGuardrails => guardrailsTriggered.isNotEmpty;
}

/// Ralph introspection/status data
class RalphStatus {
  final String status;
  final Map<String, dynamic> introspection;
  final int sessionCount;
  final List<String> activeSessions;

  RalphStatus({
    required this.status,
    required this.introspection,
    required this.sessionCount,
    required this.activeSessions,
  });

  factory RalphStatus.fromJson(Map<String, dynamic> json) {
    final sessions = json['sessions'] as Map<String, dynamic>? ?? {};
    return RalphStatus(
      status: json['status'] as String? ?? 'unknown',
      introspection:
          json['introspection'] as Map<String, dynamic>? ?? {},
      sessionCount: sessions['count'] as int? ?? 0,
      activeSessions:
          (sessions['active'] as List<dynamic>?)?.cast<String>() ?? [],
    );
  }

  bool get isHealthy => status == 'healthy';
  bool get isHalted => status == 'halted';
}

/// Chat message in Ralph history
class RalphMessage {
  final String role;
  final String content;
  final String timestamp;
  final Map<String, dynamic> metadata;

  RalphMessage({
    required this.role,
    required this.content,
    required this.timestamp,
    this.metadata = const {},
  });

  factory RalphMessage.fromJson(Map<String, dynamic> json) {
    return RalphMessage(
      role: json['role'] as String? ?? 'unknown',
      content: json['content'] as String? ?? '',
      timestamp: json['timestamp'] as String? ?? '',
      metadata: json['metadata'] as Map<String, dynamic>? ?? {},
    );
  }

  bool get isUser => role == 'user';
  bool get isAssistant => role == 'assistant';
}

/// Service for interacting with the Ralph Layer system
class RalphService {
  static const _secureStorage = FlutterSecureStorage();

  String? _host;
  int _httpPort = 8081;
  bool _useHttps = false;

  /// Current session ID
  String sessionId = 'flutter-default';

  /// Currently active CLI provider
  CLIProvider? _activeProvider;

  /// Callback for status changes
  final StreamController<RalphStatus> _statusController =
      StreamController<RalphStatus>.broadcast();
  Stream<RalphStatus> get statusStream => _statusController.stream;

  /// Initialize the service with connection parameters
  Future<void> connect({
    required String host,
    int httpPort = 8081,
    bool useHttps = false,
  }) async {
    _host = host;
    _httpPort = httpPort;
    _useHttps = useHttps;

    // Load saved provider preference
    await _loadDefaultProvider();
  }

  Uri _buildUri(String path, {Map<String, String>? queryParameters}) {
    if (_host == null) {
      throw StateError('RalphService not connected');
    }
    return Uri(
      scheme: _useHttps ? 'https' : 'http',
      host: _host,
      port: _httpPort,
      path: path,
      queryParameters: queryParameters,
    );
  }

  // ============================================================
  // API Key Management
  // ============================================================

  /// Save Anthropic API key securely
  Future<void> setAnthropicApiKey(String apiKey) async {
    await _secureStorage.write(key: _StorageKeys.anthropicApiKey, value: apiKey);
    debugPrint('Anthropic API key saved');
  }

  /// Get Anthropic API key
  Future<String?> getAnthropicApiKey() async {
    return await _secureStorage.read(key: _StorageKeys.anthropicApiKey);
  }

  /// Check if Anthropic API key is configured
  Future<bool> hasAnthropicApiKey() async {
    final key = await getAnthropicApiKey();
    return key != null && key.isNotEmpty;
  }

  /// Delete Anthropic API key
  Future<void> deleteAnthropicApiKey() async {
    await _secureStorage.delete(key: _StorageKeys.anthropicApiKey);
  }

  /// Save Gemini API key securely
  Future<void> setGeminiApiKey(String apiKey) async {
    await _secureStorage.write(key: _StorageKeys.geminiApiKey, value: apiKey);
    debugPrint('Gemini API key saved');
  }

  /// Get Gemini API key
  Future<String?> getGeminiApiKey() async {
    return await _secureStorage.read(key: _StorageKeys.geminiApiKey);
  }

  /// Check if Gemini API key is configured
  Future<bool> hasGeminiApiKey() async {
    final key = await getGeminiApiKey();
    return key != null && key.isNotEmpty;
  }

  /// Delete Gemini API key
  Future<void> deleteGeminiApiKey() async {
    await _secureStorage.delete(key: _StorageKeys.geminiApiKey);
  }

  /// Save OpenAI API key securely
  Future<void> setOpenAIApiKey(String apiKey) async {
    await _secureStorage.write(key: _StorageKeys.openaiApiKey, value: apiKey);
    debugPrint('OpenAI API key saved');
  }

  /// Get OpenAI API key
  Future<String?> getOpenAIApiKey() async {
    return await _secureStorage.read(key: _StorageKeys.openaiApiKey);
  }

  /// Check if OpenAI API key is configured
  Future<bool> hasOpenAIApiKey() async {
    final key = await getOpenAIApiKey();
    return key != null && key.isNotEmpty;
  }

  /// Delete OpenAI API key
  Future<void> deleteOpenAIApiKey() async {
    await _secureStorage.delete(key: _StorageKeys.openaiApiKey);
  }

  /// Save Ollama endpoint
  Future<void> setOllamaEndpoint(String endpoint) async {
    await _secureStorage.write(key: _StorageKeys.ollamaEndpoint, value: endpoint);
    debugPrint('Ollama endpoint saved: $endpoint');
  }

  /// Get Ollama endpoint
  Future<String> getOllamaEndpoint() async {
    final endpoint = await _secureStorage.read(key: _StorageKeys.ollamaEndpoint);
    return endpoint ?? 'http://localhost:11434';
  }

  /// Get configured API key status for all providers
  Future<Map<CLIProvider, bool>> getApiKeyStatus() async {
    return {
      CLIProvider.claude: await hasAnthropicApiKey(),
      CLIProvider.gemini: await hasGeminiApiKey(),
      CLIProvider.openai: await hasOpenAIApiKey(),
      CLIProvider.ollama: true, // Ollama doesn't require API key
    };
  }

  /// Save default provider preference
  Future<void> setDefaultProvider(CLIProvider provider) async {
    final prefs = await SharedPreferences.getInstance();
    await prefs.setString(_StorageKeys.defaultProvider, provider.value);
    _activeProvider = provider;
    debugPrint('Default CLI provider set to: ${provider.displayName}');
  }

  /// Load default provider preference
  Future<void> _loadDefaultProvider() async {
    final prefs = await SharedPreferences.getInstance();
    final providerValue = prefs.getString(_StorageKeys.defaultProvider);
    if (providerValue != null) {
      _activeProvider = CLIProvider.values.firstWhere(
        (p) => p.value == providerValue,
        orElse: () => CLIProvider.claude,
      );
    } else {
      _activeProvider = CLIProvider.claude;
    }
  }

  /// Get current active provider
  CLIProvider get activeProvider => _activeProvider ?? CLIProvider.claude;

  // ============================================================
  // Ralph Chat API
  // ============================================================

  /// Send a chat message through the Ralph layer
  Future<RalphChatResponse> sendMessage(String message) async {
    try {
      final uri = _buildUri('/api/ralph/chat');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'message': message,
          'session_id': sessionId,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return RalphChatResponse.fromJson(data);
      }

      return RalphChatResponse(
        text: '',
        error: 'HTTP ${response.statusCode}: ${response.body}',
        sessionId: sessionId,
        timestamp: DateTime.now().toIso8601String(),
      );
    } catch (e) {
      return RalphChatResponse(
        text: '',
        error: e.toString(),
        sessionId: sessionId,
        timestamp: DateTime.now().toIso8601String(),
      );
    }
  }

  /// Stream a chat response
  Stream<String> streamMessage(String message) async* {
    try {
      final uri = _buildUri('/api/ralph/chat/stream');
      final client = http.Client();
      final request = http.Request('POST', uri);
      request.headers['Content-Type'] = 'application/json';
      request.headers['Accept'] = 'text/event-stream';
      request.body = jsonEncode({
        'message': message,
        'session_id': sessionId,
      });

      final response = await client.send(request);
      if (response.statusCode == 200) {
        await for (final line in response.stream
            .transform(utf8.decoder)
            .transform(const LineSplitter())) {
          if (line.isNotEmpty) {
            try {
              final data = jsonDecode(line) as Map<String, dynamic>;
              if (data['type'] == 'chunk') {
                yield data['content'] as String;
              }
            } catch (_) {
              // Skip malformed lines
            }
          }
        }
      }
      client.close();
    } catch (e) {
      yield '[Error: $e]';
    }
  }

  /// Teach the agent a correct response
  Future<bool> teach({
    required String userInput,
    required String correctResponse,
  }) async {
    try {
      final uri = _buildUri('/api/ralph/teach');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({
          'session_id': sessionId,
          'input': userInput,
          'correct_response': correctResponse,
        }),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return data['success'] == true;
      }
      return false;
    } catch (e) {
      debugPrint('Teaching failed: $e');
      return false;
    }
  }

  /// Get pending questions from the teaching system
  Future<List<String>> getPendingQuestions() async {
    try {
      final uri = _buildUri('/api/ralph/questions');
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return (data['questions'] as List<dynamic>?)?.cast<String>() ?? [];
      }
      return [];
    } catch (e) {
      debugPrint('Get questions failed: $e');
      return [];
    }
  }

  // ============================================================
  // Provider Management
  // ============================================================

  /// Switch the CLI provider on the server
  Future<bool> switchProvider(CLIProvider provider) async {
    try {
      final uri = _buildUri('/api/ralph/provider');
      final response = await http.post(
        uri,
        headers: {'Content-Type': 'application/json'},
        body: jsonEncode({'provider': provider.value}),
      );

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        if (data['success'] == true) {
          _activeProvider = provider;
          await setDefaultProvider(provider);
          return true;
        }
      }
      return false;
    } catch (e) {
      debugPrint('Switch provider failed: $e');
      return false;
    }
  }

  /// Get available providers from the server
  Future<List<CLIProvider>> getAvailableProviders() async {
    try {
      final uri = _buildUri('/api/ralph/providers');
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final available = data['available'] as List<dynamic>? ?? [];
        return available
            .map((v) => CLIProvider.values.firstWhere(
                  (p) => p.value == v,
                  orElse: () => CLIProvider.claude,
                ))
            .toList();
      }
      return [];
    } catch (e) {
      debugPrint('Get providers failed: $e');
      return [];
    }
  }

  // ============================================================
  // Status and Introspection
  // ============================================================

  /// Get Ralph system status
  Future<RalphStatus?> getStatus() async {
    try {
      final uri = _buildUri('/api/ralph/status');
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final status = RalphStatus.fromJson(data);
        _statusController.add(status);
        return status;
      }
      return null;
    } catch (e) {
      debugPrint('Get status failed: $e');
      return null;
    }
  }

  /// Get chat history
  Future<List<RalphMessage>> getChatHistory() async {
    try {
      final uri = _buildUri('/api/ralph/history', queryParameters: {
        'session_id': sessionId,
      });
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        final history = data['history'] as List<dynamic>? ?? [];
        return history
            .map((m) => RalphMessage.fromJson(m as Map<String, dynamic>))
            .toList();
      }
      return [];
    } catch (e) {
      debugPrint('Get history failed: $e');
      return [];
    }
  }

  /// Get guardrails
  Future<List<Map<String, dynamic>>> getGuardrails() async {
    try {
      final uri = _buildUri('/api/ralph/guardrails');
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        final data = jsonDecode(response.body) as Map<String, dynamic>;
        return (data['guardrails'] as List<dynamic>?)
                ?.cast<Map<String, dynamic>>() ??
            [];
      }
      return [];
    } catch (e) {
      debugPrint('Get guardrails failed: $e');
      return [];
    }
  }

  /// Get loop state
  Future<Map<String, dynamic>?> getLoopState(String loopName) async {
    try {
      final uri = _buildUri('/api/ralph/loops/$loopName');
      final response = await http.get(uri);

      if (response.statusCode == 200) {
        return jsonDecode(response.body) as Map<String, dynamic>;
      }
      return null;
    } catch (e) {
      debugPrint('Get loop state failed: $e');
      return null;
    }
  }

  // ============================================================
  // Session Management
  // ============================================================

  /// Clear the current session
  Future<bool> clearSession() async {
    try {
      final uri = _buildUri('/api/ralph/session/$sessionId/clear');
      final response = await http.post(uri);
      return response.statusCode == 200;
    } catch (e) {
      debugPrint('Clear session failed: $e');
      return false;
    }
  }

  /// Create a new session
  void newSession() {
    sessionId = 'flutter-${DateTime.now().millisecondsSinceEpoch}';
  }

  /// Dispose resources
  void dispose() {
    _statusController.close();
  }
}
