import 'package:flutter/material.dart';
import 'package:flutter_bloc/flutter_bloc.dart';
import '../services/brain_client.dart';
import '../widgets/layout/continuon_layout.dart';

class ModelManagerScreen extends StatefulWidget {
  final BrainClient brainClient;
  const ModelManagerScreen({super.key, required this.brainClient});

  static const routeName = '/model-manager';

  @override
  State<ModelManagerScreen> createState() => _ModelManagerScreenState();
}

class _ModelManagerScreenState extends State<ModelManagerScreen> {
  List<dynamic> _models = [];
  bool _isLoading = false;

  @override
  void initState() {
    super.initState();
    _fetchModels();
  }

  Future<void> _fetchModels() async {
    setState(() => _isLoading = true);
    final res = await widget.brainClient.listModels();
    if (mounted) {
      setState(() {
        _models = res['models'] ?? [];
        _isLoading = false;
      });
    }
  }

  Future<void> _activateModel(String id) async {
    final res = await widget.brainClient.activateModel(id);
    if (mounted) {
      if (res['success'] == true) {
        ScaffoldMessenger.of(context).showSnackBar(
          SnackBar(content: Text('Model $id activated')),
        );
        _fetchModels();
      } else {
        ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(
              content: Text('Failed to activate model'),
              backgroundColor: Colors.red),
        );
      }
    }
  }

  @override
  Widget build(BuildContext context) {
    return ContinuonLayout(
      body: _isLoading
          ? const Center(child: CircularProgressIndicator())
          : RefreshIndicator(
              onRefresh: _fetchModels,
              child: ListView.builder(
                padding: const EdgeInsets.all(16),
                itemCount: _models.length,
                itemBuilder: (context, index) {
                  final model = _models[index];
                  final id = model['id'] ?? 'unknown';
                  final name = model['name'] ?? id;
                  return Card(
                    child: ListTile(
                      leading: const Icon(Icons.psychology),
                      title: Text(name),
                      subtitle: Text(model['backend'] ?? 'Native'),
                      trailing: ElevatedButton(
                        onPressed: () => _activateModel(id),
                        child: const Text('Activate'),
                      ),
                    ),
                  );
                },
              ),
            ),
    );
  }
}
