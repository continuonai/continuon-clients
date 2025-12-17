import 'package:flutter/material.dart';

import '../models/public_episode.dart';
import '../services/youtube_import_service.dart';
import '../theme/continuon_theme.dart';
import '../widgets/layout/continuon_layout.dart';
import '../widgets/layout/continuon_card.dart';

class YoutubeImportScreen extends StatefulWidget {
  const YoutubeImportScreen({super.key});

  static const routeName = '/import/youtube';

  @override
  State<YoutubeImportScreen> createState() => _YoutubeImportScreenState();
}

class _YoutubeImportScreenState extends State<YoutubeImportScreen> {
  final _formKey = GlobalKey<FormState>();
  final _youtubeUrlController = TextEditingController();
  final _slugController = TextEditingController();
  final _titleController = TextEditingController();
  final _licenseController = TextEditingController(text: 'cc-by-4.0');
  final _tagsController = TextEditingController(text: 'youtube,import');
  final _descriptionController = TextEditingController();
  final _robotHostController = TextEditingController();
  final _robotPortController = TextEditingController(text: '8080');
  final _robotAuthController = TextEditingController();

  final _service = YoutubeImportService();

  bool _requestPublicListing = false;
  String _contentRating = 'general';
  String _audience = 'general';
  bool _isSubmitting = false;

  String? _status;
  PublicEpisode? _publishedEpisode;

  int _step = 0; // 0=URL, 1=Review
  bool _isLoadingMetadata = false;

  @override
  void dispose() {
    _youtubeUrlController.dispose();
    _slugController.dispose();
    _titleController.dispose();
    _licenseController.dispose();
    _tagsController.dispose();
    _descriptionController.dispose();
    _robotHostController.dispose();
    _robotPortController.dispose();
    _robotAuthController.dispose();
    super.dispose();
  }

  @override
  Widget build(BuildContext context) {
    final theme = Theme.of(context);
    final brand = theme.extension<ContinuonBrandExtension>();
    return ContinuonLayout(
      // No specific actions
      body: Container(
        decoration:
            brand != null ? BoxDecoration(gradient: brand.waveGradient) : null,
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 900),
            child: ContinuonCard(
              margin: const EdgeInsets.all(16),
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: SingleChildScrollView(
                  child: AnimatedSwitcher(
                    duration: const Duration(milliseconds: 300),
                    child: _step == 0
                        ? _buildStepUrl(theme)
                        : _buildStepForm(theme),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildStepUrl(ThemeData theme) {
    return Column(
      key: const ValueKey('step0'),
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Text(
          'Step 1: Import from YouTube',
          style: theme.textTheme.titleMedium
              ?.copyWith(fontWeight: FontWeight.bold),
        ),
        const SizedBox(height: 12),
        Text(
          'Paste a video link to begin. We’ll fetch details to prefill the training form.',
          style: theme.textTheme.bodyMedium,
        ),
        const SizedBox(height: 24),
        _buildTextField(
          controller: _youtubeUrlController,
          label: 'YouTube URL',
          validator: (v) =>
              v == null || !v.contains('http') ? 'Provide a valid link' : null,
        ),
        const SizedBox(height: 24),
        Row(
          mainAxisAlignment: MainAxisAlignment.end,
          children: [
            if (_isLoadingMetadata)
              const Padding(
                  padding: EdgeInsets.only(right: 16),
                  child: CircularProgressIndicator()),
            ElevatedButton(
              onPressed: _isLoadingMetadata ? null : _onNextStep,
              child: const Text('Next'),
            ),
          ],
        ),
      ],
    );
  }

  Widget _buildStepForm(ThemeData theme) {
    return Form(
      key: _formKey,
      child: Column(
        key: const ValueKey('step1'),
        crossAxisAlignment: CrossAxisAlignment.start,
        children: [
          Row(
            children: [
              IconButton(
                  onPressed: () => setState(() => _step = 0),
                  icon: const Icon(Icons.arrow_back)),
              const SizedBox(width: 8),
              Text(
                'Step 2: Review & Import',
                style: theme.textTheme.titleMedium,
              ),
            ],
          ),
          const SizedBox(height: 12),
          Row(
            children: [
              Expanded(
                child: _buildTextField(
                  controller: _slugController,
                  label: 'Slug',
                  validator: _required,
                ),
              ),
              const SizedBox(width: 12),
              Expanded(
                child: _buildTextField(
                  controller: _titleController,
                  label: 'Title',
                  validator: _required,
                ),
              ),
            ],
          ),
          _buildTextField(
            controller: _licenseController,
            label: 'License',
            validator: _required,
          ),
          _buildTextField(
            controller: _tagsController,
            label: 'Tags (comma separated)',
          ),
          _buildTextField(
            controller: _descriptionController,
            label: 'Description (optional)',
            maxLines: 3,
          ),
          const SizedBox(height: 8),
          Wrap(
            spacing: 12,
            runSpacing: 8,
            children: [
              DropdownButton<String>(
                value: _contentRating,
                items: const [
                  DropdownMenuItem(
                      value: 'general', child: Text('Content rating: general')),
                  DropdownMenuItem(
                      value: '13+', child: Text('Content rating: 13+')),
                  DropdownMenuItem(
                      value: '18+', child: Text('Content rating: 18+')),
                ],
                onChanged: (value) {
                  if (value != null) {
                    setState(() => _contentRating = value);
                  }
                },
              ),
              DropdownButton<String>(
                value: _audience,
                items: const [
                  DropdownMenuItem(
                      value: 'general', child: Text('Audience: general')),
                  DropdownMenuItem(
                      value: 'research', child: Text('Audience: research')),
                  DropdownMenuItem(
                      value: 'enterprise', child: Text('Audience: enterprise')),
                ],
                onChanged: (value) {
                  if (value != null) {
                    setState(() => _audience = value);
                  }
                },
              ),
            ],
          ),
          const SizedBox(height: 12),
          SwitchListTile(
            value: _requestPublicListing,
            onChanged: (v) => setState(() => _requestPublicListing = v),
            title: const Text('Contribute to public dataset'),
            subtitle: const Text(
                'Episodes will be queued for automatic PII scrubbing and safety checks.'),
            activeColor: ContinuonColors.primaryBlue,
            contentPadding: EdgeInsets.zero,
          ),
          if (_requestPublicListing)
            Padding(
              padding: const EdgeInsets.only(top: 8, bottom: 8),
              child: Row(
                children: [
                  Icon(Icons.shield_outlined,
                      size: 16, color: Colors.green.shade700),
                  const SizedBox(width: 8),
                  Expanded(
                    child: Text(
                      'Safe for public use: PII will be redacted automatically.',
                      style: theme.textTheme.bodySmall?.copyWith(
                          color: Colors.green.shade800,
                          fontStyle: FontStyle.italic),
                    ),
                  ),
                ],
              ),
            ),
          const Divider(height: 32),
          Text('Optional: push to robot training queue',
              style: theme.textTheme.titleSmall),
          Row(
            children: [
              Expanded(
                child: _buildTextField(
                  controller: _robotHostController,
                  label: 'Robot host (LAN)',
                ),
              ),
              const SizedBox(width: 12),
              SizedBox(
                width: 120,
                child: _buildTextField(
                  controller: _robotPortController,
                  label: 'HTTP port',
                  keyboardType: TextInputType.number,
                ),
              ),
            ],
          ),
          _buildTextField(
            controller: _robotAuthController,
            label: 'Auth token (optional)',
            obscureText: true,
          ),
          const SizedBox(height: 16),
          ElevatedButton.icon(
            onPressed: _isSubmitting ? null : _onSubmit,
            icon: const Icon(Icons.cloud_upload),
            label: Text(_isSubmitting ? 'Importing…' : 'Import and queue'),
          ),
          if (_status != null) ...[
            const SizedBox(height: 12),
            Text(_status!,
                style: theme.textTheme.bodyMedium
                    ?.copyWith(color: Colors.green[700])),
          ],
          if (_publishedEpisode != null) ...[
            const SizedBox(height: 12),
            Text(
                'Published: ${_publishedEpisode!.slug} (${_publishedEpisode!.share.license})'),
          ],
          const SizedBox(height: 12),
          Text(
            'Only episodes with share.public=true, pii_cleared=true, and pending_review=false will list. '
            'Uploads include signed URLs only; raw bucket paths stay private.',
            style: theme.textTheme.bodySmall,
          ),
        ],
      ),
    );
  }

  Future<void> _onNextStep() async {
    final url = _youtubeUrlController.text.trim();
    if (url.isEmpty || !url.contains('http')) {
      ScaffoldMessenger.of(context).showSnackBar(
          const SnackBar(content: Text('Please enter a valid YouTube URL.')));
      return;
    }
    setState(() => _isLoadingMetadata = true);
    final meta = await _service.fetchMetadata(url);
    if (meta != null) {
      _titleController.text = meta.title;
      _slugController.text = _slugify(meta.title);
      if (meta.authorName.isNotEmpty) {
        if (_tagsController.text.isEmpty) {
          _tagsController.text = 'youtube,import,${meta.authorName}';
        } else if (!_tagsController.text.contains(meta.authorName)) {
          _tagsController.text += ',${meta.authorName}';
        }
      }
    } else {
      // Fallback or just empty
    }
    setState(() {
      _isLoadingMetadata = false;
      _step = 1;
    });
  }

  String _slugify(String input) {
    return input
        .toLowerCase()
        .replaceAll(RegExp(r'[^a-z0-9]+'), '-')
        .replaceAll(RegExp(r'^-+|-+$'), '');
  }

  Future<void> _onSubmit() async {
    if (!_formKey.currentState!.validate()) return;
    setState(() {
      _isSubmitting = true;
      _status = 'Starting YouTube download and transcode…';
      _publishedEpisode = null;
    });
    try {
      final share = ShareMetadata(
        isPublic: _requestPublicListing,
        slug: _slugController.text.trim(),
        title: _titleController.text.trim(),
        license: _licenseController.text.trim(),
        tags: _parseTags(_tagsController.text),
        contentRating: _contentRating,
        intendedAudience: _audience,
        // Automatic PII logic:
        // If public, we attest it's safe-ish but request system checks (cleared=false, review=true).
        // If private (local), we don't force checks.
        piiAttested: _requestPublicListing,
        piiCleared: false, // Let the system/admin clear it
        piiRedacted: false, // System produces redacted assets
        pendingReview: _requestPublicListing, // If public, needs review
        description: _descriptionController.text.isEmpty
            ? null
            : _descriptionController.text.trim(),
      );
      final port = int.tryParse(_robotPortController.text) ?? 8080;
      final result = await _service.importAndPublish(
        youtubeUrl: _youtubeUrlController.text.trim(),
        share: share,
        requestPublicListing: _requestPublicListing,
        robotHost: _robotHostController.text.trim().isEmpty
            ? null
            : _robotHostController.text.trim(),
        robotHttpPort: port,
        robotAuthToken: _robotAuthController.text.isEmpty
            ? null
            : _robotAuthController.text,
      );
      setState(() {
        _status = 'Transcoded, wrapped as RLDS, and uploaded.';
        _publishedEpisode = result.publishedEpisode;
      });
    } catch (e) {
      setState(() {
        _status = 'Failed: $e';
      });
    } finally {
      setState(() => _isSubmitting = false);
    }
  }

  String? _required(String? value) {
    if (value == null || value.trim().isEmpty) return 'Required';
    return null;
  }

  List<String> _parseTags(String raw) {
    return raw
        .split(',')
        .map((t) => t.trim())
        .where((t) => t.isNotEmpty)
        .toList();
  }

  Widget _buildTextField({
    required TextEditingController controller,
    required String label,
    String? Function(String?)? validator,
    int maxLines = 1,
    TextInputType? keyboardType,
    bool obscureText = false,
  }) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 6),
      child: TextFormField(
        controller: controller,
        validator: validator,
        maxLines: maxLines,
        keyboardType: keyboardType,
        obscureText: obscureText,
        decoration: InputDecoration(labelText: label),
      ),
    );
  }
}
