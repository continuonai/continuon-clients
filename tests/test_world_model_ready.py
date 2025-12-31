import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from continuonbrain.services.agent_hope import HOPEAgent
from continuonbrain.services.brain_service import BrainService
from continuonbrain.services.runtime_context import (
    HardwareCapabilities,
    RuntimeContext,
    WorldModelCapabilities,
)


def test_world_model_adapter_seeded_when_jax_available(tmp_path, monkeypatch):
    """Ensure HOPEAgent receives a world_model when JAX is available."""
    runtime_mgr = MagicMock()
    wm_caps = WorldModelCapabilities(
        jax_available=True, world_model_type="jax_core", can_predict=True, can_plan=True
    )
    hardware_caps = HardwareCapabilities(world_model=wm_caps)
    runtime_context = RuntimeContext()
    runtime_context.hardware = hardware_caps
    runtime_mgr.get_context.return_value = runtime_context
    runtime_mgr.mark_world_model_ready = MagicMock()

    monkeypatch.setattr(
        "continuonbrain.services.runtime_context.get_runtime_context_manager",
        lambda config_dir=None: runtime_mgr,
    )

    class DummyChat:
        model_name = "gemma-2b"

    monkeypatch.setattr(
        "continuonbrain.services.brain_service.build_chat_service",
        lambda: DummyChat(),
    )

    adapter = object()

    with patch.object(BrainService, "_init_jax_search", autospec=True) as mock_init_jax:
        def _fake_init(self):
            self.jax_adapter = adapter

        mock_init_jax.side_effect = _fake_init
        service = BrainService(config_dir=str(tmp_path), auto_detect=False)

    assert service.jax_adapter is adapter
    runtime_mgr.mark_world_model_ready.assert_called()

    hope_agent = HOPEAgent(MagicMock(), world_model=service.jax_adapter)
    assert hope_agent.world_model is adapter
