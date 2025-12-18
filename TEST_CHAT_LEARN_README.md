# Agent Chat Manager Multi-Turn Training Tests

This directory contains test scripts for validating the agent chat manager multi-turn agentic learning process with JAX and other models.

## Overview

The agent chat manager implements a multi-turn learning process where:
- **Agent Manager (HOPE)**: Primary orchestrator that asks curious questions about the system
- **Subagents**: Specialized models (e.g., Gemma) that provide technical answers
- **RLDS Logging**: Conversations are logged to RLDS format for training data
- **JAX Integration**: Supports JAX models for training pipeline integration

## Test Scripts

### 1. `quick_test_chat_learn.py` - Quick Verification

Simple test script for quick verification of the chat learning endpoint.

**Usage:**
```bash
# Test with default settings (localhost:8082)
python3 quick_test_chat_learn.py

# Test with custom server URL
CONTINUON_API_URL=http://192.168.1.100:8082 python3 quick_test_chat_learn.py
```

**What it tests:**
- Basic HOPE agent manager with Gemma subagent
- 4-turn conversation about JAX training pipeline
- Response validation and conversation quality

### 2. `test_agent_chat_manager_training.py` - Comprehensive Test Suite

Full test suite covering multiple scenarios and model combinations.

**Usage:**
```bash
# Run all tests
python3 test_agent_chat_manager_training.py

# Run specific test
python3 test_agent_chat_manager_training.py --test basic
python3 test_agent_chat_manager_training.py --test hope-gemma
python3 test_agent_chat_manager_training.py --test jax
python3 test_agent_chat_manager_training.py --test extended

# Custom test with parameters
python3 test_agent_chat_manager_training.py \
    --test basic \
    --turns 6 \
    --model hope-v1 \
    --delegate google/gemma-370m \
    --topic "CMS compaction and memory management"

# Check RLDS logging status only
python3 test_agent_chat_manager_training.py --check-rlds

# JSON output for programmatic use
JSON_OUTPUT=1 python3 test_agent_chat_manager_training.py > results.json
```

**Test Scenarios:**
1. **basic_hope**: Basic HOPE agent manager (no subagent)
2. **hope_with_gemma**: HOPE with Gemma subagent
3. **gemma_direct**: Direct Gemma model (no HOPE)
4. **jax_integration**: JAX model integration test
5. **multi_topic**: Multiple topics to test learning diversity
6. **extended**: Extended 10-turn conversation

## Prerequisites

1. **Server Running**: The ContinuonBrain server must be running
   ```bash
   # Start server (adjust path as needed)
   python3 -m continuonbrain.server.routes
   # Or use your startup script
   ```

2. **RLDS Logging (Optional)**: To test RLDS logging, enable it:
   ```bash
   export CONTINUON_LOG_CHAT_RLDS=1
   # Or set in settings.json: chat.log_rlds = true
   ```

3. **Python Dependencies**:
   ```bash
   pip install requests
   ```

## Expected Behavior

### Successful Test Output

```
✓ Server is accessible at http://localhost:8082
✓ Completed in 12.34s
  History length: 8
  Outputs: 4
  ✓ Dynamic content detected
```

### Conversation Structure

Each turn should show:
- **Agent Manager** asking curious questions about system internals
- **Subagent** providing technical answers
- **Synthesis** of insights and improvements

Example:
```
[1] user: We are training the HOPE Agent Manager...
[2] assistant: I'm curious: How does CMS compaction actually consolidate memories?
[3] user: You are the internal Subagent...
[4] assistant: CMS compaction uses energy transfer from episodic memory...
```

## RLDS Training Data

When `CONTINUON_LOG_CHAT_RLDS=1` is set, conversations are logged to:
- Default: `~/.continuon_config/rlds/episodes/`
- Or: `/opt/continuonos/brain/rlds/episodes/`

Each session creates JSON files with:
- Conversation history
- Model metadata
- Session information
- Training context

## Model Options

### Available Models

- **hope-v1**: HOPE Agent Manager (default)
- **google/gemma-3n-2b**: Gemma 3n 2B model
- **google/gemma-370m**: Gemma 370M model (lightweight)
- **google/gemma-3-270m-it**: Gemma 3 270M instruction-tuned

### Model Combinations

1. **HOPE only**: `model_hint=hope-v1`, no delegate
2. **HOPE + Subagent**: `model_hint=hope-v1`, `delegate_model_hint=google/gemma-370m`
3. **Direct model**: `model_hint=google/gemma-3n-2b`, no delegate
4. **Consult pattern**: `delegate_model_hint=consult:google/gemma-370m`

## Troubleshooting

### Server Not Accessible
```
✗ Cannot connect to server at http://localhost:8082
```
**Solution**: Ensure server is running and check the URL/port.

### Mock Responses Detected
```
⚠ Warning: Detected mock/fallback responses
```
**Solution**: Model backend may not be available. Check:
- Model files are downloaded
- Sufficient memory/GPU
- Model configuration in settings

### Timeout Errors
```
✗ Request timed out
```
**Solution**: 
- Increase timeout: `--timeout 600`
- Reduce turns: `--turns 3`
- Check server performance

### RLDS Not Logging
```
⚠ RLDS directory not found
```
**Solution**:
- Set `CONTINUON_LOG_CHAT_RLDS=1`
- Check `chat.log_rlds` in settings.json
- Verify write permissions to RLDS directory

## Integration with JAX Training

The chat learning process generates RLDS data that can be used for JAX training:

1. **Generate Data**: Run chat learning sessions
2. **Convert to TFRecord**: Use `continuonbrain.jax_models.data.tfrecord_converter`
3. **Train**: Use `continuonbrain.jax_models.train.local_sanity_check` or cloud TPU training

Example workflow:
```bash
# 1. Generate training data
python3 test_agent_chat_manager_training.py --test extended

# 2. Convert to TFRecord (if needed)
python3 -m continuonbrain.jax_models.data.tfrecord_converter \
    --input-dir ~/.continuon_config/rlds/episodes \
    --output-dir ~/.continuon_config/rlds/tfrecord

# 3. Run local sanity check
python3 -m continuonbrain.jax_models.train.local_sanity_check \
    --data-dir ~/.continuon_config/rlds/tfrecord
```

## Next Steps

After successful testing:
1. Review generated RLDS episodes
2. Validate training data quality
3. Integrate with JAX training pipeline
4. Monitor training metrics
5. Deploy trained models

## See Also

- `docs/jax-training-pipeline.md`: JAX training pipeline documentation
- `continuonbrain/robot_api_server.py`: `RunChatLearn` implementation
- `continuonbrain/services/brain_service.py`: BrainService chat learning
- `AGENTS.md`: Agent instructions and guidelines
