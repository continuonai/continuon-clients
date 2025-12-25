import 'package:flutter/foundation.dart' show kIsWeb;
import 'dart:io' show Platform;

class PlatformInfo {
  static bool get isWeb => kIsWeb;
  
  static bool get isAndroid => !kIsWeb && Platform.isAndroid;
  static bool get isIOS => !kIsWeb && Platform.isIOS;
  static bool get isDesktop => !kIsWeb && (Platform.isLinux || Platform.isMacOS || Platform.isWindows);
  
  static String get name {
    if (isWeb) return 'Web';
    if (isAndroid) return 'Android';
    if (isIOS) return 'iOS';
    if (isDesktop) return 'Desktop';
    return 'Unknown';
  }
}
