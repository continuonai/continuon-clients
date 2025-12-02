# Gemma 3 Nano Chat - Quick Reference

## 🚀 Immediate Use (No Setup Required)

### Access the Chat
1. Start robot server: `python3 continuonbrain/robot_api_server.py`
2. Open browser: http://192.168.68.86:8080/control
3. Look for **🤖 Gemma 3n Assistant** in bottom-right corner

### Chat Commands (Mock Mode)

| Question Type | Example | Response |
|--------------|---------|----------|
| Status | "What's the robot status?" | Current mode, motion state, hardware info |
| Controls | "How do I move the arm?" | Joint sliders, arrows, keyboard controls |
| Joints | "Tell me about the joints" | J0-J5 descriptions, value ranges |
| Car | "How fast is the car?" | Speed presets, safety defaults |
| Recording | "How do I record episodes?" | Manual training mode instructions |
| Safety | "How do I stop?" | Emergency stop, safety features |
| Camera | "Tell me about the camera" | OAK-D Lite specs, resolution |
| Help | "help" | General assistance overview |

### UI Controls
- **Send Message**: Press Enter or click ➤ button
- **Minimize**: Click header "🤖 Gemma 3n Assistant"
- **Maximize**: Click header again
- **Scroll**: Auto-scrolls to latest message

## ⚡ Upgrade to Real AI (Optional)

### Quick Setup
```bash
# 1. Install dependencies
pip3 install transformers torch

# 2. Get HuggingFace token
# Visit: https://huggingface.co/settings/tokens
# Accept Gemma license: https://huggingface.co/google/gemma-3n-E2B-it

# 3. Set token
export HUGGINGFACE_TOKEN="hf_YourTokenHere"

# 4. Restart server (model auto-downloads ~3-6GB)
python3 continuonbrain/robot_api_server.py
```

### What Changes
- ✨ True natural language understanding
- 🧠 Context-aware reasoning
- 💬 Personalized conversational responses
- 🎯 Better understanding of complex questions

## 📱 Flutter App Integration

### Setup
```bash
cd continuonai
flutter pub get  # Already done!
```

### Usage Example
```dart
import 'package:flutter_gemma/flutter_gemma.dart';

// Initialize
await FlutterGemma.initialize(huggingFaceToken: token);

// Download Gemma 3n E4B (web-optimized, 4.2GB, 4-bit)
await FlutterGemma.installModel(modelType: ModelType.gemmaIt)
  .fromNetwork(
    'https://huggingface.co/google/gemma-3n-E4B-it-litert-lm/resolve/main/gemma-3n-E4B-it-int4.litert',
    token: token
  )
  .install();

// Chat
final model = await FlutterGemma.getActiveModel();
final chat = await model.createChat();
await chat.addQueryChunk(Message.text(text: 'Hello!', isUser: true));
final response = await chat.generateChatResponse();
```

## 🔌 API Reference

### Endpoint: /api/chat

**Request**
```json
POST /api/chat
Content-Type: application/json

{
  "message": "What's the robot status?",
  "history": [
    {"role": "User", "content": "Previous question"},
    {"role": "Assistant", "content": "Previous answer"}
  ]
}
```

**Response (Success)**
```json
{
  "response": "Robot status: Current mode: manual_control | Motion allowed: True | Hardware: REAL. The robot is ready for your commands."
}
```

**Response (Error)**
```json
{
  "error": "Error message"
}
```

### JavaScript Example
```javascript
fetch('/api/chat', {
  method: 'POST',
  headers: {'Content-Type': 'application/json'},
  body: JSON.stringify({
    message: 'How do I control the arm?',
    history: chatHistory.slice(-10)
  })
})
.then(res => res.json())
.then(data => console.log(data.response));
```

### Python Example
```python
from continuonbrain.gemma_chat import create_gemma_chat

chat = create_gemma_chat()
response = chat.chat(
    message="What's the robot status?",
    system_context="Mode: manual_control | Motion: enabled"
)
print(response)
```

## 🎨 UI Customization

### Change Chat Position
Edit `robot_api_server.py` line ~1028:
```css
.chat-overlay {
    position: fixed;
    bottom: 20px;    /* Distance from bottom */
    right: 20px;     /* Distance from right */
    /* Change to: left: 20px; for left side */
}
```

### Change Chat Width
```css
.chat-overlay {
    width: 350px;    /* Increase for wider chat */
}
```

### Change Message Colors
```css
.chat-message.user {
    background: #007aff;    /* Blue for user */
}
.chat-message.assistant {
    background: #e8e8e8;    /* Gray for assistant */
}
```

## 🐛 Troubleshooting

### Chat panel not visible
- Check browser console for JavaScript errors
- Verify server running: `curl http://192.168.68.86:8080/api/status`
- Refresh page (Ctrl+F5 for hard refresh)

### Send button not working
- Check browser console for network errors
- Verify /api/chat endpoint: `curl -X POST http://192.168.68.86:8080/api/chat -H "Content-Type: application/json" -d '{"message":"test"}'`
- Check CORS headers enabled

### No response from assistant
- Check server logs for errors
- Verify gemma_chat module loaded: `python3 -c "from continuonbrain.gemma_chat import create_gemma_chat; print('OK')"`
- Try restarting server

### "transformers not available" warning
- This is normal for mock mode
- Install transformers to enable real AI: `pip3 install transformers torch`

## 📚 Learn More

- **Full Documentation**: `docs/gemma-chat-setup.md`
- **Integration Summary**: `GEMMA_INTEGRATION.md`
- **Example Script**: `continuonbrain/examples/gemma_chat_example.py`
- **Source Code**: `continuonbrain/gemma_chat.py`

## 💡 Pro Tips

1. **Start simple**: Try "status", "help", "controls" first
2. **Be specific**: "How do I move J2?" vs "How do I move?"
3. **Use context**: Chat remembers last 10 messages
4. **Minimize when not needed**: Click header to save screen space
5. **Combine with controls**: Chat while controlling robot
6. **Try voice**: Use browser voice input (microphone button in some keyboards)

## 🎯 Common Use Cases

### Learning Robot Controls
```
You: "How do I control the arm?"
Gemma: "Control the arm with joint sliders or arrow buttons..."
```

### Checking Status
```
You: "What's the current speed?"
Gemma: "Speed is preset to SLOW (0.3) for safety..."
```

### Getting Help
```
You: "I'm stuck, help!"
Gemma: "I can help with robot status, camera info, control instructions..."
```

### Recording Episodes
```
You: "How do I record a training episode?"
Gemma: "Make sure you're in manual_training mode and motion is enabled..."
```

---

**Quick Start**: Just open http://192.168.68.86:8080/control and start chatting! 🚀
