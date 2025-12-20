import 'package:flutter/foundation.dart';
import 'package:flutter/material.dart';
// ignore: avoid_web_libraries_in_flutter
import 'dart:html' as html;
// ignore: avoid_web_libraries_in_flutter
import 'dart:ui_web' as ui_web;

import '../widgets/layout/continuon_layout.dart';

// Conditional import: webview_flutter only available on mobile
// On web, we'll use a stub that prevents compilation errors
import 'robot_portal_webview_stub.dart'
    if (dart.library.io) 'robot_portal_webview_mobile.dart'
    as webview;

class RobotPortalScreen extends StatefulWidget {
  final String host;
  final int httpPort;
  final String robotName;

  const RobotPortalScreen({
    super.key,
    required this.host,
    required this.httpPort,
    required this.robotName,
  });

  static const routeName = '/robot-portal';

  @override
  State<RobotPortalScreen> createState() => _RobotPortalScreenState();
}

class _RobotPortalScreenState extends State<RobotPortalScreen> {
  static int _iframeCounter = 0;
  late final String _iframeId;
  bool _isLoading = true;
  bool _isMixedContentBlocked = false;
  dynamic _webViewController;
  late final String _url;

  @override
  void initState() {
    super.initState();
    _url = _buildPortalUrl();
    if (kIsWeb) {
      _iframeId = 'robot-portal-iframe-${_iframeCounter++}';
      _registerIframe();
    } else {
      // Initialize WebView for mobile platforms
      _webViewController = webview.createWebViewController(_url, (loading) {
        if (mounted) {
          setState(() => _isLoading = loading);
        }
      }, (error) {
        if (mounted) {
          setState(() => _isLoading = false);
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text('Failed to load robot portal: $error'),
              backgroundColor: Colors.red,
            ),
          );
        }
      });
    }
  }

  String _buildPortalUrl() {
    final bool isSecureContext = kIsWeb && html.window.isSecureContext;
    final String protocol =
        kIsWeb ? html.window.location.protocol.replaceAll(':', '') : '';

    final String scheme;
    if (isSecureContext || protocol == 'https') {
      scheme = 'https';
    } else {
      scheme = 'http';
    }

    return Uri(
      scheme: scheme,
      host: widget.host,
      port: widget.httpPort,
      path: '/',
    ).toString();
  }

  void _registerIframe() {
    final bool isSecureContext = html.window.isSecureContext;
    final bool willBeBlocked = isSecureContext && _url.startsWith('http://');

    if (willBeBlocked) {
      if (mounted) {
        setState(() {
          _isLoading = false;
          _isMixedContentBlocked = true;
        });
      }
      return;
    }

    final iframe = html.IFrameElement()
      ..src = _url
      ..style.border = 'none'
      ..style.width = '100%'
      ..style.height = '100%'
      ..onLoad.listen((_) {
        if (mounted) {
          setState(() => _isLoading = false);
        }
      })
      ..onError.listen((_) {
        if (mounted) {
          setState(() {
            _isLoading = false;
            _isMixedContentBlocked =
                isSecureContext && _url.startsWith('http://');
          });
          ScaffoldMessenger.of(context).showSnackBar(
            SnackBar(
              content: Text(
                _isMixedContentBlocked
                    ? 'Portal blocked: HTTPS page cannot load insecure (HTTP) content.'
                    : 'Failed to load robot portal. Check connection.',
              ),
              backgroundColor: Colors.red,
            ),
          );
        }
      });

    // Register the iframe with Flutter's platform view registry
    ui_web.platformViewRegistry.registerViewFactory(
      _iframeId,
      (int viewId) => iframe,
    );
  }

  @override
  Widget build(BuildContext context) {
    return ContinuonLayout(
      body: Stack(
        children: [
          // WebView or iframe depending on platform
          if (kIsWeb)
            SizedBox.expand(
              child: HtmlElementView(viewType: _iframeId),
            )
          else
            webview.buildWebViewWidget(_webViewController),
          // Loading indicator
          if (_isMixedContentBlocked)
            Container(
              color: Colors.white,
              child: const Center(
                child: Padding(
                  padding: EdgeInsets.symmetric(horizontal: 24.0),
                  child: Text(
                    'The robot portal is blocked because this page is HTTPS but the portal is served over HTTP. '
                    'Open the portal over HTTPS or access this page from a non-secure origin to continue.',
                    textAlign: TextAlign.center,
                  ),
                ),
              ),
            )
          else if (_isLoading)
            Container(
              color: Colors.white,
              child: const Center(
                child: Column(
                  mainAxisAlignment: MainAxisAlignment.center,
                  children: [
                    CircularProgressIndicator(),
                    SizedBox(height: 16),
                    Text('Loading robot portal...'),
                  ],
                ),
              ),
            ),
        ],
      ),
    );
  }
}
