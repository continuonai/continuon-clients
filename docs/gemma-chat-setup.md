# Gemma 3 Nano Chat Integration

This guide explains how to set up and use the Gemma 3 Nano AI chat assistant integrated into the robot control interface.

## Overview

The Gemma 3 Nano chat provides an on-device AI assistant that can help with:
- Robot status and control information
- Tips for manual control and safe operation
- Answering questions about robot state and hardware
- Guidance on movement and recording

## Quick Start (Mock Mode)

The chat works out-of-the-box with mock responses - no setup required!

1. Start the robot server:
   ```bash
   cd /home/craigm26/ContinuonXR
   python3 continuonbrain/robot_api_server.py
   ```

2. Open http://192.168.68.86:8080/control in your browser

3. Find the chat panel in the bottom-right (🤖 Gemma 3n Assistant)

4. Ask questions like:
   - "What's the robot status?"
   - "How do I control the arm?"
   - "How fast is the car going?"

## Real Gemma Model Setup (Optional)

For production-quality AI responses using the actual Gemma 3 Nano model:

### 1. Install Dependencies

```bash
# Install transformers and PyTorch
pip3 install transformers torch

# For GPU acceleration (CUDA)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# For CPU-only (smaller download)
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2. Get HuggingFace Access Token

Gemma 3 Nano is a gated model requiring authentication:

1. Create account at https://huggingface.co
2. Accept Gemma model license at https://huggingface.co/google/gemma-3n-E2B-it
3. Generate access token at https://huggingface.co/settings/tokens
4. Copy your token (starts with `hf_...`)

### 3. Set Environment Variable

```bash
# Add to ~/.bashrc or ~/.bash_profile
export HUGGINGFACE_TOKEN="hf_YourTokenHere"

# Or set for current session
export HUGGINGFACE_TOKEN="hf_YourTokenHere"
```

### 4. Start Server with Gemma

```bash
cd /home/craigm26/ContinuonXR
python3 continuonbrain/robot_api_server.py
```

The server will automatically:
- Detect transformers library
- Load Gemma 3n-E2B-it model (2B parameters, ~3-6GB download)
- Use GPU if available, fall back to CPU
- Provide real AI responses with context awareness

## Flutter App Integration

The Flutter companion app can also use Gemma 3 Nano directly:

### 1. Update Dependencies

Already added to `continuonai/pubspec.yaml`:
```yaml
dependencies:
  flutter_gemma: ^0.11.13
```

Install:
```bash
cd continuonai
flutter pub get
```

### 2. Download Model in Flutter

Create a config.json for security:
```json
{
  "huggingface_token": "hf_YourTokenHere"
}
```

Use in Flutter:
```dart
import 'package:flutter_gemma/flutter_gemma.dart';

// Initialize with token
await FlutterGemma.initialize(huggingFaceToken: token);

// Download Gemma 3 Nano E4B (web-optimized litert-lm format)
await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
  .fromNetwork(
    'https://huggingface.co/google/gemma-3n-E4B-it-litert-lm/resolve/main/gemma-3n-E4B-it-int4.litert',
    token: token
  )
  .withProgress((progress) => print('Download: $progress%'))
  .install();

// Create model instance
final model = await FlutterGemma.getActiveModel(
  maxTokens: 2048,
  preferredBackend: PreferredBackend.gpu
);

// Create chat and send message
final chat = await model.createChat();
await chat.addQueryChunk(Message.text(text: 'Hello!', isUser: true));
final response = await chat.generateChatResponse();
```

### 3. Platform-Specific Setup

**Android:**
- Minimum SDK: 21
- Recommended: 8GB+ RAM
- GPU support automatic

**iOS:**
- iOS 16.0+
- Add memory entitlements in Xcode
- 8GB+ RAM recommended

**Web:**
- Uses LiteRT-LM runtime (https://github.com/google-ai-edge/LiteRT-LM)
- GPU backend required (WebGL/WebGPU)
- .litert format optimized for web
- Cache API for persistence
- CORS configuration needed
- Recommended: Gemma 3n E4B 4-bit (4.2GB)

## Model Options

### Gemma 3 Nano E2B
- **Size**: 2B parameters, ~3-6GB download
- **Features**: Multimodal (vision + text), function calling
- **Speed**: Fast on modern hardware
- **URL**: https://huggingface.co/google/gemma-3n-E2B-it-litert-lm
- **Format**: .task (mobile), .litert (web)

### Gemma 3 Nano E4B (Recommended for Web)
- **Size**: 4B parameters, 4096 tokens context, ~4.2GB download (4-bit quantized)
- **Features**: Better quality, multimodal, function calling
- **Speed**: Optimized for web with LiteRT-LM
- **URL**: https://huggingface.co/google/gemma-3n-E4B-it-litert-lm
- **Implementation**: https://github.com/google-ai-edge/LiteRT-LM
- **Format**: .litert (web-optimized)

## Chat API Usage

### Python Backend

```python
from continuonbrain.gemma_chat import create_gemma_chat

# Create chat instance (auto-detects transformers availability)
chat = create_gemma_chat(use_mock=False)

# Generate response
response = chat.chat(
    message="What's the robot status?",
    system_context="Current mode: manual_control | Hardware: REAL"
)

print(response)  # AI-generated response

# Get model info
info = chat.get_model_info()
print(f"Model: {info['model_name']}, Loaded: {info['loaded']}")
```

### HTTP API

```bash
# Send chat message
curl -X POST http://192.168.68.86:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{
    "message": "How do I control the arm?",
    "history": []
  }'

# Response
{
  "response": "The arm has 6 joints controlled via sliders or arrow buttons..."
}
```

### JavaScript Frontend

```javascript
function sendChatMessage() {
  const message = document.getElementById('chat-input').value;
  
  fetch('/api/chat', {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({
      message: message,
      history: chatHistory.slice(-10)
    })
  })
  .then(res => res.json())
  .then(data => {
    addChatMessage(data.response, 'assistant');
  });
}
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                  Web Browser Client                     │
│  ┌─────────────────────────────────────────────────┐   │
│  │  /control Page                                   │   │
│  │  ┌──────────────┐  ┌─────────────────────────┐ │   │
│  │  │ Camera Feed  │  │  Chat Panel (bottom-right)│ │   │
│  │  │ Arm Controls │  │  - Message input          │ │   │
│  │  │ Car Controls │  │  - Message history        │ │   │
│  │  └──────────────┘  │  - Minimize toggle        │ │   │
│  │                     └─────────────────────────┘ │   │
│  └─────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │ HTTP POST /api/chat
                         ▼
┌─────────────────────────────────────────────────────────┐
│           Python Robot API Server                       │
│  ┌─────────────────────────────────────────────────┐   │
│  │  RobotService.ChatWithGemma()                    │   │
│  │  - Get robot status for context                  │   │
│  │  - Call gemma_chat.chat(message, context)        │   │
│  │  - Return AI response                            │   │
│  └──────────────────────┬──────────────────────────┘   │
└────────────────────────┬────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────┐
│              GemmaChat Module                           │
│  ┌─────────────────────────────────────────────────┐   │
│  │  Real: transformers + Gemma 3n model             │   │
│  │  - Load model from HuggingFace                   │   │
│  │  - Tokenize prompt with history + context        │   │
│  │  - Generate response via model.generate()        │   │
│  │  - Extract and return assistant text             │   │
│  ├─────────────────────────────────────────────────┤   │
│  │  Mock: Pattern-matching fallback                 │   │
│  │  - Keyword detection for topics                  │   │
│  │  - Pre-written helpful responses                 │   │
│  │  - Always available (no deps)                    │   │
│  └─────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────┘
```

## Troubleshooting

### "transformers not available, using mock chat"
- **Cause**: transformers library not installed
- **Fix**: `pip3 install transformers torch`

### "HUGGINGFACE_TOKEN not set - gated models will not be accessible"
- **Cause**: No HuggingFace token in environment
- **Fix**: `export HUGGINGFACE_TOKEN="hf_YourToken"`

### Model download slow or fails
- **Check**: Internet connection (3-6GB download)
- **Try**: Use CPU-only torch for smaller download
- **Alternative**: Download model manually and load from local path

### Out of memory errors
- **Cause**: Model too large for available RAM
- **Fix**: Use quantized E2B variant (smaller)
- **Alternative**: Increase swap space or use mock mode

### Chat not responding
- **Check**: Browser console for errors
- **Verify**: Server running on port 8080
- **Test**: `curl http://192.168.68.86:8080/api/status`

## Features

### Context Awareness
The chat includes current robot state:
- Current mode (manual_control, autonomous, etc.)
- Motion enabled status
- Recording status  
- Hardware mode (REAL vs MOCK)
- Joint positions
- Detected hardware devices

### Chat History
- Last 10 messages preserved for context
- Sent with each request for continuity
- Trimmed automatically to prevent bloat

### Minimizable UI
- Toggle with header click
- Bottom-right overlay doesn't block controls
- Persists across page refreshes

### Error Handling
- Falls back to mock responses if model unavailable
- Graceful handling of network errors
- Input validation and sanitization

## Future Enhancements

- [ ] Multimodal vision: Send camera frame with query
- [ ] Function calling: Let Gemma control robot directly
- [ ] Voice input/output via Web Speech API
- [ ] Streaming responses for real-time feel
- [ ] Fine-tune on robot-specific Q&A dataset
- [ ] Add thinking mode for complex reasoning
- [ ] Integrate with episode recording annotations

## References

- **flutter_gemma**: https://pub.dev/packages/flutter_gemma
- **Gemma 3 Nano E2B**: https://huggingface.co/google/gemma-3n-E2B-it-litert-lm
- **Gemma 3 Nano E4B**: https://huggingface.co/google/gemma-3n-E4B-it-litert-lm (4.2GB, 4-bit)
- **LiteRT-LM**: https://github.com/google-ai-edge/LiteRT-LM
- **HuggingFace Transformers**: https://huggingface.co/docs/transformers
- **Model Types**: .task (mobile), .litert (web), .bin (custom)
- **Authentication**: https://huggingface.co/settings/tokens
