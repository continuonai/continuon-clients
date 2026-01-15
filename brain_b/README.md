# Brain B - Simplified ContinuonBrain

A minimal, teachable robot brain. Option B while the full ContinuonBrain matures.

## Philosophy

- **Talk** to your robot naturally
- **Teach** it behaviors by demonstration
- **Invoke** learned behaviors by name
- **Recover** from crashes via checkpoints

## Quick Start

```bash
cd brain_b
python main.py
```

Then:
```
You: drive forward
Bot: Moving forward.

You: stop
Bot: Stopped.

You: teach patrol
Bot: Ready to learn 'patrol'. Show me what to do.

You: forward
You: left
You: forward
You: left
You: done
Bot: Learned 'patrol': forward -> left -> forward -> left

You: patrol
Bot: Running 'patrol'...
```

## Architecture

```
brain_b/
├── actor_runtime/
│   ├── agent.py        # Agent with state, mailbox, budget
│   ├── event_log.py    # Append-only log, checkpoints, replay
│   ├── runtime.py      # spawn, send, checkpoint, suspend/resume
│   └── teaching.py     # teach(), invoke(), behavior storage
├── conversation/
│   ├── intents.py      # Intent classification
│   └── handler.py      # Natural language interface
├── hardware/
│   ├── motors.py       # Motor control abstraction
│   └── safety.py       # Basic safety checks
└── main.py             # Entry point
```

## Comparison to Full ContinuonBrain

| Feature | Brain B | Full ContinuonBrain |
|---------|---------|---------------------|
| Lines of code | ~500 | ~50,000 |
| Time to understand | 30 min | Days |
| Time to get running | Minutes | Hours |
| Neural network | No | 12.8M params |
| Cloud training | No | Yes |
| Multi-loop HOPE | No | Yes |
| Ralph layers | No | 4 layers |
| Good enough for RC car | Yes | Overkill |

## When to Graduate to Full Brain

- When you need learned behaviors to generalize
- When you want cloud-trained skills
- When you need multi-robot coordination
- When simple rules aren't enough

Until then, Brain B gets you driving.
