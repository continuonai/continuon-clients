import pytest
import asyncio
import queue
from continuonbrain.services.brain_service import BrainService

@pytest.mark.asyncio
async def test_end_to_end_lesson(tmp_path):
    # Setup
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    (config_dir / "ownership.json").write_text("{}")
    
    # Mock Hardware/Chat
    service = BrainService(
        config_dir=str(config_dir),
        prefer_real_hardware=False,
        auto_detect=False,
        allow_mock_fallback=True
    )
    # Ensure init
    await service.initialize()
    
    # Run Math Lesson
    res = await service.RunCurriculumLesson("math-basics")
    assert res["success"]
    assert res["all_passed"]
    
    # Check if events were pushed
    events = []
    try:
        while True:
            events.append(service.chat_event_queue.get_nowait())
    except queue.Empty:
        pass
        
    # Verify thoughts
    thoughts = [e for e in events if e.get("type") == "thought"]
    assert len(thoughts) > 0
    assert any("Starting Lesson" in t["text"] for t in thoughts)
    
    # Verify tool usage (wrapped in cognitive)
    # The payload structure is { "cognitive": { "type": "tool_use", ... } }
    tool_uses = [e for e in events if e.get("cognitive", {}).get("type") == "tool_use"]
    assert len(tool_uses) > 0
    assert any(tu["cognitive"]["name"] == "calculator" for tu in tool_uses)

    # Clean up
    if hasattr(service, "shutdown"):
        service.shutdown()
