import 'package:flutter/material.dart';

import '../models/public_episode.dart';
import '../services/youtube_import_service.dart';
import '../theme/continuon_theme.dart';

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

  bool _piiAttested = true;
  bool _piiCleared = false;
  bool _piiRedacted = false;
  bool _requestPublicListing = false;
  String _contentRating = 'general';
  String _audience = 'general';
  bool _isSubmitting = false;
  String? _status;
  PublicEpisode? _publishedEpisode;

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
    return Scaffold(
      appBar: AppBar(
        title: const Text('Import from YouTube'),
      ),
      body: Container(
        decoration: brand != null ? BoxDecoration(gradient: brand.waveGradient) : null,
        child: Center(
          child: ConstrainedBox(
            constraints: const BoxConstraints(maxWidth: 900),
            child: Card(
              margin: const EdgeInsets.all(16),
              shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(16)),
              child: Padding(
                padding: const EdgeInsets.all(20),
                child: SingleChildScrollView(
                  child: Form(
                    key: _formKey,
                    child: Column(
                      crossAxisAlignment: CrossAxisAlignment.start,
                      children: [
                        Text(
                          'Paste a YouTube link to download/transcode, convert to RLDS, and queue training.',
                          style: theme.textTheme.titleMedium,
                        ),
                        const SizedBox(height: 12),
                        _buildTextField(
                          controller: _youtubeUrlController,
                          label: 'YouTube URL',
                          validator: (v) => v == null || !v.contains('http')
                              ? 'Provide a valid link'
                              : null,
                        ),
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
                                DropdownMenuItem(value: 'general', child: Text('Content rating: general')),
                                DropdownMenuItem(value: '13+', child: Text('Content rating: 13+')),
                                DropdownMenuItem(value: '18+', child: Text('Content rating: 18+')),
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
                                DropdownMenuItem(value: 'general', child: Text('Audience: general')),
                                DropdownMenuItem(value: 'research', child: Text('Audience: research')),
                                DropdownMenuItem(value: 'enterprise', child: Text('Audience: enterprise')),
                              ],
                              onChanged: (value) {
                                if (value != null) {
                                  setState(() => _audience = value);
                                }
                              },
                            ),
                          ],
                        ),
                        CheckboxListTile(
                          value: _piiAttested,
                          onChanged: (v) => setState(() => _piiAttested = v ?? false),
                          title: const Text('PII attested (faces/plates blur + OCR/ASR scan triggered)'),
                          controlAffinity: ListTileControlAffinity.leading,
                        ),
                        CheckboxListTile(
                          value: _piiCleared,
                          onChanged: (v) => setState(() => _piiCleared = v ?? false),
                          title: const Text('PII cleared and pending_review=false (safe to list publicly)'),
                          controlAffinity: ListTileControlAffinity.leading,
                        ),
                        CheckboxListTile(
                          value: _piiRedacted,
                          onChanged: (v) => setState(() => _piiRedacted = v ?? false),
                          title: const Text('Provide redacted assets (serve redacted_signed_url when present)'),
                          controlAffinity: ListTileControlAffinity.leading,
                        ),
                        SwitchListTile(
                          value: _requestPublicListing,
                          onChanged: (v) => setState(() => _requestPublicListing = v),
                          title: const Text('Flag for public listing after PII/safety checks'),
                        ),
                        const Divider(),
                        Text('Optional: push to robot training queue', style: theme.textTheme.titleSmall),
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
                          Text(_status!, style: theme.textTheme.bodyMedium?.copyWith(color: Colors.green[700])),
                        ],
                        if (_publishedEpisode != null) ...[
                          const SizedBox(height: 12),
                          Text('Published: ${_publishedEpisode!.slug} (${_publishedEpisode!.share.license})'),
                        ],
                        const SizedBox(height: 12),
                        Text(
                          'Only episodes with share.public=true, pii_cleared=true, and pending_review=false will list. '
                          'Uploads include signed URLs only; raw bucket paths stay private.',
                          style: theme.textTheme.bodySmall,
                        ),
                      ],
                    ),
                  ),
                ),
              ),
            ),
          ),
        ),
      ),
    );
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
        piiAttested: _piiAttested,
        piiCleared: _piiCleared,
        piiRedacted: _piiRedacted,
        pendingReview: !_piiCleared,
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
        robotAuthToken:
            _robotAuthController.text.isEmpty ? null : _robotAuthController.text,
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
