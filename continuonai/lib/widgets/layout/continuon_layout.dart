import 'package:flutter/material.dart';
import 'continuon_app_bar.dart';

class ContinuonLayout extends StatelessWidget {
  final Widget body;
  final Widget? floatingActionButton;
  final Widget? bottomNavigationBar;
  final Color? backgroundColor;

  /// If true, the body will be placed directly behind the app bar (if transparent).
  final bool extendBodyBehindAppBar;
  final List<Widget>? appBarActions;

  const ContinuonLayout({
    super.key,
    required this.body,
    this.floatingActionButton,
    this.bottomNavigationBar,
    this.backgroundColor,
    this.extendBodyBehindAppBar = false,
    this.appBarActions,
  });

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      backgroundColor:
          backgroundColor ?? Theme.of(context).scaffoldBackgroundColor,
      extendBodyBehindAppBar: extendBodyBehindAppBar,
      appBar: ContinuonAppBar(actions: appBarActions),
      body: body,
      floatingActionButton: floatingActionButton,
      bottomNavigationBar: bottomNavigationBar,
    );
  }
}
