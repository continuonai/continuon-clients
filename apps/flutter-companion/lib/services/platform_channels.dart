import 'dart:async';

import 'package:flutter/services.dart';

class PlatformBrainBridge {
  const PlatformBrainBridge();

  static const MethodChannel _channel = MethodChannel('continuonbrain_bridge');
  static const EventChannel _stateEvents = EventChannel('continuonbrain_bridge/state');

  bool get isConnected => _connected;
  static bool _connected = false;

  Future<void> initConnection(String host, int port, {bool useTls = false}) async {
    final response = await _channel.invokeMethod<bool>('connect', {
      'host': host,
      'port': port,
      'useTls': useTls,
    });
    _connected = response ?? false;
  }

  Future<void> sendCommand(String payload) async {
    await _channel.invokeMethod<void>('sendCommand', {'payload': payload});
  }

  Stream<String> subscribeState(String clientId) {
    return _stateEvents
        .receiveBroadcastStream({'client_id': clientId}).map((event) => event as String);
  }
}
