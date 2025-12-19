# How to Train HOPE on New Tools

This guide explains how to teach the HOPE Agent Manager to use new tools and improve its chat capabilities.

## Architecture

- **Tool Execution**: Defined in `continuonbrain/services/brain_service.py` (inside `RunChatLearn` or `_process_chat_response`).
- **Tool Knowledge**: Tools are exposed to the model via the **System Prompt**.
- **Training Data**: RLDS episodes containing tool usage examples.
- **Training Loop**: The JAX trainer consumes these episodes to update the model.

## Step-by-Step Guide

### 1. Implement the Tool Logic

You need to define what the tool *does*. Locate `continuonbrain/services/brain_service.py` and find the tool execution block (search for `ASK_GEMINI` or `Tool Router`).

Add your new tool logic:

```python
# continuonbrain/services/brain_service.py

# Inside the tool processing loop:
elif action == "SEARCH_WIKI":
    query = parts[1] # Parse arguments (e.g. from regex or JSON)
    # Execute the actual logic
    import wikipedia
    try:
        summary = wikipedia.summary(query, sentences=3)
        result = f"Wiki Summary: {summary}"
        status_updates.append(f"Tool: Searched Wiki for '{query}'")
    except Exception as e:
        result = f"Wiki Error: {e}"
```

### 2. Update the System Prompt

The model needs to know the tool is available. Update the system prompt string in `continuonbrain/services/brain_service.py` (usually in `RunChatLearn` or `_process_chat_response`).

```python
system_prompt = """
You are HOPE. You have access to the following tools:
- ASK_GEMINI "question": Consult a smarter model for advice.
- SEARCH_WIKI "topic": Search Wikipedia for factual information.
...
"""
```

### 3. Generate Training Data

You generally have two ways to create the Data that trains the model to uses these tools:

#### Method A: Interactive "Learning Session" (Teacher Mode)
This is the "Show, Don't Tell" method where you act as the teacher.

1.  Run the learning session script:
    ```bash
    python3 scripts/run_learning_session.py
    ```
2.  When HOPE asks a question that *should* be answered with the tool, **intervene**.
3.  As the Teacher (or via Gemini delegation), explicitly instruct HOPE:
    > "You should use the SEARCH_WIKI tool to find that out."
4.  In the next turn, if HOPE follows the instruction (or if you force the action), that successful interaction is logged as an RLDS episode.

#### Method B: Import External Dataset
If you have an existing dataset of function calls (e.g., from Glaive, xLAM, or a custom JSONL), import it directly into RLDS.

```bash
python3 continuonbrain/tools/import_tool_calling_dataset_to_rlds.py \
  --input my_tool_dataset.jsonl \
  --format jsonl \
  --source wiki_tool_v1
```

### 4. Run Training

Once episodes are in `rlds/episodes/`, run the trainer. It automatically picks up new files.

```bash
# On Pi5 or Desktop
python -m continuonbrain.run_trainer --trainer jax --mode local --config-preset pi5
```

## Verification output

After training, start a new chat session and ask a question that requires the tool.

- **User**: "Who is the CEO of DeepMind?"
- **HOPE** (Autonomous): `[TOOL: SEARCH_WIKI "DeepMind CEO"]` -> *Executes* -> "Demis Hassabis is the CEO..."

## Troubleshooting

- **Tool not triggering?** Check if the system prompt clearly defines the trigger syntax.
- **Syntax errors?** Ensure your regex/parser in `BrainService.py` matches what the model outputs.
- **Low performance?** Generate more training examples (10-50 examples are often enough for consistent tool usage).
