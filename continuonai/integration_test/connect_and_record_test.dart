import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';
import 'package:integration_test/integration_test.dart';

import 'package:flutter_companion/main.dart' as app;

void main() {
  IntegrationTestWidgetsFlutterBinding.ensureInitialized();

  testWidgets('connect form validates and record screen renders',
      (WidgetTester tester) async {
    app.main();
    await tester.pumpAndSettle();

    expect(find.text('ContinuonBrain connection'), findsOneWidget);
    await tester.enterText(find.byType(TextFormField).first, 'brain.local');
    await tester.enterText(find.byType(TextFormField).at(1), '50051');

    await tester.tap(find.text('Skip to record'));
    await tester.pumpAndSettle();

    expect(find.text('Record task / RLDS episodes'), findsOneWidget);
    await tester.enterText(find.byType(TextField), 'note');
    await tester.tap(find.text('Record step'));
    await tester.pump();

    expect(find.textContaining('Recorded step'), findsOneWidget);
  });
}
