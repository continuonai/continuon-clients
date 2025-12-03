import 'package:flutter/material.dart';
import '../services/brain_client.dart';
import '../theme/continuon_theme.dart';

class ChatOverlay extends StatefulWidget {
  const ChatOverlay({super.key, required this.brainClient});

  final BrainClient brainClient;

  @override
  State<ChatOverlay> createState() => _ChatOverlayState();
}

class _ChatOverlayState extends State<ChatOverlay> {
  bool _minimized = false;
  final TextEditingController _controller = TextEditingController();
  final ScrollController _scrollController = ScrollController();
  final List<Map<String, String>> _messages = [
    {'role': 'system', 'content': 'Chat with Gemma about robot control and status.'}
  ];
  bool _loading = false;

  void _toggleMinimize() {
    setState(() => _minimized = !_minimized);
  }

  Future<void> _sendMessage() async {
    final text = _controller.text.trim();
    if (text.isEmpty || _loading) return;

    setState(() {
      _messages.add({'role': 'user', 'content': text});
      _loading = true;
      _controller.clear();
    });
    _scrollToBottom();

    // Prepare history for API (exclude system message if API doesn't support it, but here we keep it simple)
    final history = _messages
        .where((m) => m['role'] != 'system')
        .map((m) => {'role': m['role']!, 'content': m['content']!})
        .toList();

    final result = await widget.brainClient.chatWithGemma(text, history);

    if (mounted) {
      setState(() {
        _loading = false;
        if (result.containsKey('response')) {
          _messages.add({'role': 'assistant', 'content': result['response']});
        } else {
          _messages.add({'role': 'system', 'content': 'Error: ${result['error']}'});
        }
      });
      _scrollToBottom();
    }
  }

  void _scrollToBottom() {
    WidgetsBinding.instance.addPostFrameCallback((_) {
      if (_scrollController.hasClients) {
        _scrollController.animateTo(
          _scrollController.position.maxScrollExtent,
          duration: const Duration(milliseconds: 300),
          curve: Curves.easeOut,
        );
      }
    });
  }

  @override
  Widget build(BuildContext context) {
    return Positioned(
      bottom: 20,
      right: 20,
      child: Container(
        width: 400,
        height: _minimized ? 50 : 600,
        decoration: BoxDecoration(
          color: ContinuonColors.gray900.withOpacity(0.95),
          borderRadius: BorderRadius.circular(12),
          border: Border.all(color: Colors.grey[800]!),
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.5),
              blurRadius: 32,
              offset: const Offset(0, 8),
            ),
          ],
        ),
        child: Column(
          children: [
            GestureDetector(
              onTap: _toggleMinimize,
              child: Container(
                padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 12),
                decoration: BoxDecoration(
                  color: ContinuonColors.gray800,
                  borderRadius: const BorderRadius.vertical(top: Radius.circular(12)),
                  border: Border(bottom: BorderSide(color: Colors.grey[800]!)),
                ),
                child: Row(
                  mainAxisAlignment: MainAxisAlignment.spaceBetween,
                  children: [
                    const Text(
                      '💬 Chat with Gemma',
                      style: TextStyle(
                        color: Colors.white,
                        fontWeight: FontWeight.bold,
                        fontSize: 14,
                      ),
                    ),
                    Icon(
                      _minimized ? Icons.expand_less : Icons.expand_more,
                      color: ContinuonColors.gray500,
                    ),
                  ],
                ),
              ),
            ),
            if (!_minimized)
              Expanded(
                child: Column(
                  children: [
                    Expanded(
                      child: ListView.builder(
                        controller: _scrollController,
                        padding: const EdgeInsets.all(16),
                        itemCount: _messages.length,
                        itemBuilder: (context, index) {
                          final msg = _messages[index];
                          final isUser = msg['role'] == 'user';
                          final isSystem = msg['role'] == 'system';
                          
                          if (isSystem) {
                            return Padding(
                              padding: const EdgeInsets.symmetric(vertical: 8),
                              child: Text(
                                msg['content']!,
                                textAlign: TextAlign.center,
                                style: const TextStyle(color: ContinuonColors.gray500, fontSize: 12),
                              ),
                            );
                          }

                          return Align(
                            alignment: isUser ? Alignment.centerRight : Alignment.centerLeft,
                            child: Container(
                              margin: const EdgeInsets.only(bottom: 12),
                              padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 10),
                              constraints: const BoxConstraints(maxWidth: 280),
                              decoration: BoxDecoration(
                                color: isUser ? ContinuonColors.primaryBlue : ContinuonColors.gray800,
                                borderRadius: BorderRadius.circular(8),
                              ),
                              child: Text(
                                msg['content']!,
                                style: const TextStyle(color: Colors.white, fontSize: 13),
                              ),
                            ),
                          );
                        },
                      ),
                    ),
                    Container(
                      padding: const EdgeInsets.all(12),
                      decoration: BoxDecoration(
                        border: Border(top: BorderSide(color: Colors.grey[800]!)),
                      ),
                      child: Row(
                        children: [
                          Expanded(
                            child: TextField(
                              controller: _controller,
                              style: const TextStyle(color: Colors.white),
                              decoration: InputDecoration(
                                hintText: 'Type a message...',
                                hintStyle: TextStyle(color: Colors.grey[600]),
                                border: InputBorder.none,
                                isDense: true,
                              ),
                              onSubmitted: (_) => _sendMessage(),
                            ),
                          ),
                          IconButton(
                            onPressed: _loading ? null : _sendMessage,
                            icon: _loading
                                ? const SizedBox(
                                    width: 16,
                                    height: 16,
                                    child: CircularProgressIndicator(strokeWidth: 2, color: Colors.white),
                                  )
                                : const Icon(Icons.send, color: ContinuonColors.primaryBlue),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
          ],
        ),
      ),
    );
  }
}
