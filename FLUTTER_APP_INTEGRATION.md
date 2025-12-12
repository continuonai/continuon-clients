# ContinuonAI Flutter App Integration Guide

This guide explains how to access the ContinuonBrain robot API from your Flutter app on web, Linux, iOS, and Android platforms.

## Network Configuration

### Robot Server Details
- **IP Address**: `192.168.68.90` (on your local network)
- **Port**: `8080`
- **Base URL**: `http://192.168.68.90:8080`
- **Web UI**: `http://192.168.68.90:8080/ui`

### Server Binding
The server binds to `0.0.0.0:8080`, which means it accepts connections from any network interface, including:
- Local network (LAN) devices
- WiFi-connected devices
- Any device on the same subnet

## API Endpoints for Flutter App

### Core Endpoints

#### 1. Robot Status
```
GET http://192.168.68.90:8080/api/status
```
Returns the current robot status and mode.

#### 1b. Mobile Summary (optimized for quick status cards)
```
GET http://192.168.68.90:8080/api/mobile/summary
```
Returns a small JSON payload that combines the robot status, most recent safety gate snapshot, and the first few eligible tasks. The Continuon AI app can poll this endpoint to populate a lightweight mobile dashboard without loading the web UI.

#### 2. Chat with AI
```
POST http://192.168.68.90:8080/api/chat
Content-Type: application/json

{
  "message": "Your message here"
}
```

#### 3. Camera Stream
```
GET http://192.168.68.90:8080/api/camera/stream
```
Returns MJPEG stream for live video.

#### 4. Single Camera Frame
```
GET http://192.168.68.90:8080/api/camera/frame
```
Returns a single JPEG frame.

#### 5. Drive Control
```
POST http://192.168.68.90:8080/api/robot/drive
Content-Type: application/json

{
  "steering": 0.0,  // -1.0 to 1.0
  "throttle": 0.0   // -1.0 to 1.0
}
```

#### 6. Settings
```
GET http://192.168.68.90:8080/api/settings
POST http://192.168.68.90:8080/api/settings
```

#### 7. HOPE Brain Structure
```
GET http://192.168.68.90:8080/api/hope/structure
```
Returns the brain topology and state for visualization.

#### 8. Task Library
```
GET http://192.168.68.90:8080/api/tasks/library
```
Returns available tasks with eligibility checks.

#### 9. Network Management
```
GET http://192.168.68.90:8080/api/network/wifi/status
GET http://192.168.68.90:8080/api/network/wifi/scan
POST http://192.168.68.90:8080/api/network/wifi/connect
GET http://192.168.68.90:8080/api/network/bluetooth/paired
POST http://192.168.68.90:8080/api/network/bluetooth/connect
```

## Flutter Integration Example

### 1. Add HTTP Package
```yaml
# pubspec.yaml
dependencies:
  http: ^1.1.0
```

### 2. Create API Service
```dart
import 'package:http/http.dart' as http;
import 'dart:convert';

class ContinuonAPI {
  static const String baseUrl = 'http://192.168.68.90:8080';
  
  // Get robot status
  Future<Map<String, dynamic>> getStatus() async {
    final response = await http.get(Uri.parse('$baseUrl/api/status'));
    if (response.statusCode == 200) {
      return json.decode(response.body);
    }
    throw Exception('Failed to load status');
  }
  
  // Send chat message
  Future<Map<String, dynamic>> sendChat(String message) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/chat'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({'message': message}),
    );
    if (response.statusCode == 200) {
      return json.decode(response.body);
    }
    throw Exception('Failed to send message');
  }
  
  // Control robot drive
  Future<Map<String, dynamic>> drive(double steering, double throttle) async {
    final response = await http.post(
      Uri.parse('$baseUrl/api/robot/drive'),
      headers: {'Content-Type': 'application/json'},
      body: json.encode({
        'steering': steering,
        'throttle': throttle,
      }),
    );
    if (response.statusCode == 200) {
      return json.decode(response.body);
    }
    throw Exception('Failed to control drive');
  }
  
  // Get camera frame URL
  String getCameraFrameUrl() {
    return '$baseUrl/api/camera/frame';
  }
  
  // Get camera stream URL
  String getCameraStreamUrl() {
    return '$baseUrl/api/camera/stream';
  }
}
```

### 3. Display Camera Stream
```dart
import 'package:flutter/material.dart';

class CameraView extends StatelessWidget {
  final String streamUrl = 'http://192.168.68.90:8080/api/camera/stream';
  
  @override
  Widget build(BuildContext context) {
    return Image.network(
      streamUrl,
      loadingBuilder: (context, child, progress) {
        if (progress == null) return child;
        return CircularProgressIndicator();
      },
      errorBuilder: (context, error, stackTrace) {
        return Text('Camera unavailable');
      },
    );
  }
}
```

## Platform-Specific Considerations

### Web
- Works out of the box
- May need CORS headers (already configured with `Access-Control-Allow-Origin: *`)

### Android
Add internet permission to `android/app/src/main/AndroidManifest.xml`:
```xml
<uses-permission android:name="android.permission.INTERNET" />
```

For HTTP (non-HTTPS) connections, add to `AndroidManifest.xml`:
```xml
<application
    android:usesCleartextTraffic="true"
    ...>
```

### iOS
Add to `ios/Runner/Info.plist`:
```xml
<key>NSAppTransportSecurity</key>
<dict>
    <key>NSAllowsArbitraryLoads</key>
    <true/>
</dict>
```

### Linux
- Works out of the box
- Ensure network connectivity to the robot's IP

## Network Discovery

The robot broadcasts its presence on UDP port 5555. You can implement discovery in your Flutter app:

```dart
import 'dart:io';
import 'dart:convert';

Future<List<String>> discoverRobots() async {
  final socket = await RawDatagramSocket.bind(InternetAddress.anyIPv4, 5555);
  List<String> robots = [];
  
  socket.listen((event) {
    if (event == RawSocketEvent.read) {
      final datagram = socket.receive();
      if (datagram != null) {
        final message = utf8.decode(datagram.data);
        // Parse robot info from broadcast message
        robots.add(message);
      }
    }
  });
  
  await Future.delayed(Duration(seconds: 3));
  socket.close();
  return robots;
}
```

## Testing Network Access

### From Command Line
```bash
# Test basic connectivity
curl http://192.168.68.90:8080/api/status

# Test chat endpoint
curl -X POST http://192.168.68.90:8080/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Hello robot"}'

# Get camera frame
curl http://192.168.68.90:8080/api/camera/frame -o frame.jpg
```

### From Browser
Open: `http://192.168.68.90:8080/ui`

## Troubleshooting

### Cannot Connect
1. Verify robot is on the same network
2. Check IP address: `ip addr show` on robot
3. Verify firewall allows port 8080
4. Test with: `ping 192.168.68.90`

### Slow Response
- The Gemma AI model takes time to load (first request may be slow)
- Camera streams are bandwidth-intensive

### CORS Issues (Web)
- Server already configured with `Access-Control-Allow-Origin: *`
- If issues persist, check browser console for specific errors

## Security Notes

⚠️ **Important**: This server is currently configured for local network use only. For production:
1. Add authentication
2. Use HTTPS
3. Implement rate limiting
4. Validate all inputs
5. Restrict CORS to specific origins

## Next Steps

1. Implement service discovery in your Flutter app
2. Add error handling and retry logic
3. Implement offline mode
4. Add authentication when ready for production
5. Consider WebSocket for real-time updates
