// Gemma 3n E4B Web Integration Example
// Uses LiteRT-LM runtime for optimized web inference
// https://github.com/google-ai-edge/LiteRT-LM

import 'package:flutter_gemma/flutter_gemma.dart';

/// Example: Download and use Gemma 3n E4B model for web
/// 
/// Model specs:
/// - 4B parameters
/// - 4096 token context window
/// - 4-bit per-channel quantization
/// - ~4.2GB download size
/// - Web-optimized .litert format
Future<void> setupGemma3nE4B() async {
  // 1. Initialize with HuggingFace token
  // Get token from: https://huggingface.co/settings/tokens
  // Accept license: https://huggingface.co/google/gemma-3n-E4B-it-litert-lm
  const token = String.fromEnvironment('HUGGINGFACE_TOKEN');
  
  if (token.isEmpty) {
    print('⚠️  HUGGINGFACE_TOKEN not set');
    print('Set with: flutter run --dart-define=HUGGINGFACE_TOKEN=hf_...');
    return;
  }
  
  await FlutterGemma.initialize(huggingFaceToken: token);
  print('✅ FlutterGemma initialized');
  
  // 2. Download Gemma 3n E4B (litert-lm format for web)
  print('📥 Downloading Gemma 3n E4B (4.2GB, 4-bit)...');
  
  await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
    .fromNetwork(
      'https://huggingface.co/google/gemma-3n-E4B-it-litert-lm/resolve/main/gemma-3n-E4B-it-int4.litert',
      token: token,
    )
    .withProgress((progress) {
      print('Download progress: $progress%');
    })
    .install();
  
  print('✅ Gemma 3n E4B installed');
  
  // 3. Create model instance with GPU backend
  final model = await FlutterGemma.getActiveModel(
    maxTokens: 2048,
    preferredBackend: PreferredBackend.gpu,  // WebGL/WebGPU
    temperature: 0.7,
    topK: 40,
    topP: 0.95,
  );
  
  print('✅ Model loaded with GPU backend');
  
  // 4. Create chat session
  final chat = await model.createChat();
  
  // 5. Send message and get response
  await chat.addQueryChunk(
    Message.text(
      text: 'What is the status of the robot?',
      isUser: true,
    ),
  );
  
  print('🤖 Generating response...');
  final response = await chat.generateChatResponse();
  print('Response: ${response.text}');
  
  // 6. Continue conversation with context
  await chat.addQueryChunk(
    Message.text(
      text: 'How do I control the arm?',
      isUser: true,
    ),
  );
  
  final response2 = await chat.generateChatResponse();
  print('Response: ${response2.text}');
}

/// Example: Multimodal chat with vision input
/// Gemma 3n E4B supports vision + text
Future<void> gemmaVisionExample(List<int> imageBytes) async {
  final model = await FlutterGemma.getActiveModel();
  final chat = await model.createChat();
  
  // Send image + text query
  await chat.addQueryChunk(
    Message.withImage(
      text: 'What do you see in this camera frame?',
      imageBytes: imageBytes,
      isUser: true,
    ),
  );
  
  final response = await chat.generateChatResponse();
  print('Vision response: ${response.text}');
}

/// Example: Streaming responses for real-time feel
Future<void> gemmaStreamingExample() async {
  final model = await FlutterGemma.getActiveModel();
  final chat = await model.createChat();
  
  await chat.addQueryChunk(
    Message.text(
      text: 'Explain how to record a training episode',
      isUser: true,
    ),
  );
  
  // Stream tokens as they're generated
  await for (final token in chat.generateChatResponseStream()) {
    print(token.text);  // Print each token as it arrives
  }
}

/// Example: Web-specific configuration
void configureForWeb() {
  // Enable cache for persistent storage across browser restarts
  FlutterGemma.configure(
    enableWebCache: true,  // Uses Cache API
    cacheMaxAge: Duration(days: 30),
  );
  
  print('✅ Web cache enabled (persists model across sessions)');
}

/// Example: Check model info
Future<void> checkModelInfo() async {
  final info = await FlutterGemma.getModelInfo();
  
  print('Model Info:');
  print('  Name: ${info.name}');
  print('  Size: ${info.sizeBytes / 1024 / 1024 / 1024:.2f} GB');
  print('  Type: ${info.modelType}');
  print('  Backend: ${info.backend}');
  print('  Context: ${info.maxTokens} tokens');
}

/// Complete example for robot control integration
class RobotChatWidget extends StatefulWidget {
  @override
  _RobotChatWidgetState createState() => _RobotChatWidgetState();
}

class _RobotChatWidgetState extends State<RobotChatWidget> {
  FlutterGemmaModel? _model;
  FlutterGemmaChat? _chat;
  bool _isLoading = false;
  List<ChatMessage> _messages = [];
  
  @override
  void initState() {
    super.initState();
    _initializeGemma();
  }
  
  Future<void> _initializeGemma() async {
    setState(() => _isLoading = true);
    
    try {
      // Initialize if not already done
      await FlutterGemma.initialize(
        huggingFaceToken: const String.fromEnvironment('HUGGINGFACE_TOKEN'),
      );
      
      // Get active model (assumes already downloaded)
      _model = await FlutterGemma.getActiveModel(
        maxTokens: 2048,
        preferredBackend: PreferredBackend.gpu,
      );
      
      // Create chat session
      _chat = await _model!.createChat();
      
      setState(() {
        _isLoading = false;
        _messages.add(ChatMessage(
          text: 'Chat with Gemma 3n E4B about robot control',
          isUser: false,
          isSystem: true,
        ));
      });
    } catch (e) {
      setState(() {
        _isLoading = false;
        _messages.add(ChatMessage(
          text: 'Error initializing Gemma: $e',
          isUser: false,
          isSystem: true,
        ));
      });
    }
  }
  
  Future<void> _sendMessage(String text) async {
    if (_chat == null || text.trim().isEmpty) return;
    
    setState(() {
      _messages.add(ChatMessage(text: text, isUser: true));
      _isLoading = true;
    });
    
    try {
      // Add user message to chat
      await _chat!.addQueryChunk(
        Message.text(text: text, isUser: true),
      );
      
      // Get response
      final response = await _chat!.generateChatResponse();
      
      setState(() {
        _messages.add(ChatMessage(
          text: response.text ?? 'No response',
          isUser: false,
        ));
        _isLoading = false;
      });
    } catch (e) {
      setState(() {
        _messages.add(ChatMessage(
          text: 'Error: $e',
          isUser: false,
          isSystem: true,
        ));
        _isLoading = false;
      });
    }
  }
  
  @override
  Widget build(BuildContext context) {
    return Column(
      children: [
        Expanded(
          child: ListView.builder(
            itemCount: _messages.length,
            itemBuilder: (context, index) {
              final msg = _messages[index];
              return ChatBubble(message: msg);
            },
          ),
        ),
        if (_isLoading)
          LinearProgressIndicator(),
        ChatInput(onSend: _sendMessage),
      ],
    );
  }
}

class ChatMessage {
  final String text;
  final bool isUser;
  final bool isSystem;
  
  ChatMessage({
    required this.text,
    required this.isUser,
    this.isSystem = false,
  });
}

// Note: This example requires Flutter widgets (StatefulWidget)
// For web-only implementation, use standard HTML/JavaScript
// with flutter_gemma compiled to JS via flutter build web

/*
 * Web Deployment Checklist:
 * 
 * 1. Build for web: flutter build web --release
 * 2. Set environment: --dart-define=HUGGINGFACE_TOKEN=hf_...
 * 3. Enable CORS on HuggingFace model URLs
 * 4. Configure web server for WebAssembly/WebGL
 * 5. Test GPU backend availability (WebGL 2.0 required)
 * 6. Enable cache for persistence across reloads
 * 7. Set appropriate memory limits (4.2GB model + inference)
 * 
 * Performance Notes:
 * - First load: ~30s download + compile (cached after)
 * - Inference: 200-800ms per response (GPU-dependent)
 * - Context: 4096 tokens (vs 2048 for E2B)
 * - Quality: Better than E2B, comparable to desktop models
 * 
 * Browser Requirements:
 * - Chrome/Edge 90+, Firefox 90+, Safari 15+
 * - WebGL 2.0 support
 * - 8GB+ RAM recommended
 * - Stable internet for initial download
 */
