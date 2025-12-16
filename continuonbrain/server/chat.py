"""
Chat service selection.

NOTE: The canonical builder now lives in `continuonbrain.gemma_chat.build_chat_service`
so both the web server and ChatAdapter/eval runners share the same policy.
"""

from typing import Any, Optional

from continuonbrain.gemma_chat import build_chat_service as _build_chat_service


def build_chat_service() -> Optional[Any]:
    return _build_chat_service()

