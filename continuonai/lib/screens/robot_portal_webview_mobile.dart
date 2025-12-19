// Mobile implementation using webview_flutter
// This file is used when dart.library.io is available (mobile platforms)

import 'package:flutter/material.dart';
import 'package:webview_flutter/webview_flutter.dart';

dynamic createWebViewController(
  String url,
  void Function(bool) onLoadingChanged,
  void Function(String) onError,
) {
  return WebViewController()
    ..setJavaScriptMode(JavaScriptMode.unrestricted)
    ..setNavigationDelegate(
      NavigationDelegate(
        onPageStarted: (String url) {
          onLoadingChanged(true);
        },
        onPageFinished: (String url) {
          onLoadingChanged(false);
        },
        onWebResourceError: (WebResourceError error) {
          onLoadingChanged(false);
          onError(error.description);
        },
      ),
    )
    ..loadRequest(Uri.parse(url));
}

Widget buildWebViewWidget(dynamic controller) {
  if (controller == null) {
    return const Center(
      child: Text('Initializing robot portal...'),
    );
  }
  return SizedBox.expand(
    child: WebViewWidget(controller: controller as WebViewController),
  );
}

