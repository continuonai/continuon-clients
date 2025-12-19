// Stub file for web platform where webview_flutter is not available
// This file is used when dart.library.html is available (web platform)

import 'package:flutter/material.dart';

dynamic createWebViewController(
  String url,
  void Function(bool) onLoadingChanged,
  void Function(String) onError,
) {
  // This should never be called on web
  throw UnsupportedError('WebView is not supported on web platform');
}

Widget buildWebViewWidget(dynamic controller) {
  // This should never be called on web
  return const Center(
    child: Text('WebView not available on web platform'),
  );
}

