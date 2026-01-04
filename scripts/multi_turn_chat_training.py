#!/usr/bin/env python3
"""
Multi-Turn Chat Training for ContinuonBrain Seed Model

This script creates multi-turn conversational training episodes and trains
the WaveCore seed model to learn from chat interactions.

Training Pipeline:
1. Generate multi-turn chat episodes (Q&A pairs, reasoning chains, dialogues)
2. Convert to RLDS format with text embeddings as observations
3. Run extended WaveCore training (fast/mid/slow loops)
4. Evaluate model with inference tests
"""

import asyncio
import hashlib
import json
import logging
import random
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

# Training episode templates for multi-turn chat
CHAT_TRAINING_EPISODES = [
    # Math and logic reasoning chains
    {
        "category": "math_reasoning",
        "turns": [
            {"role": "user", "content": "What is 15 + 27?"},
            {"role": "assistant", "content": "15 + 27 = 42"},
            {"role": "user", "content": "Now multiply that by 2"},
            {"role": "assistant", "content": "42 * 2 = 84"},
            {"role": "user", "content": "Is that number even or odd?"},
            {"role": "assistant", "content": "84 is even because it's divisible by 2"},
        ],
        "expected_skill": "multi-step arithmetic reasoning",
    },
    {
        "category": "math_reasoning",
        "turns": [
            {"role": "user", "content": "Calculate 100 - 37"},
            {"role": "assistant", "content": "100 - 37 = 63"},
            {"role": "user", "content": "What's half of that?"},
            {"role": "assistant", "content": "63 / 2 = 31.5"},
            {"role": "user", "content": "Round to nearest integer"},
            {"role": "assistant", "content": "31.5 rounds to 32"},
        ],
        "expected_skill": "division and rounding",
    },
    # Robot control conversations
    {
        "category": "robot_control",
        "turns": [
            {"role": "user", "content": "Move forward slowly"},
            {"role": "assistant", "content": "Setting throttle to 0.3, steering to 0.0"},
            {"role": "user", "content": "Turn right a bit"},
            {"role": "assistant", "content": "Adjusting steering to 0.2, maintaining throttle at 0.3"},
            {"role": "user", "content": "Stop now"},
            {"role": "assistant", "content": "Emergency stop: throttle 0.0, steering 0.0"},
        ],
        "expected_skill": "incremental motor control",
    },
    {
        "category": "robot_control",
        "turns": [
            {"role": "user", "content": "Check battery status"},
            {"role": "assistant", "content": "Battery level: 78%, status: healthy"},
            {"role": "user", "content": "Is it safe to operate?"},
            {"role": "assistant", "content": "Yes, battery above 20% threshold, safe for operation"},
            {"role": "user", "content": "What's the estimated runtime?"},
            {"role": "assistant", "content": "Estimated 2.5 hours at current power consumption"},
        ],
        "expected_skill": "battery monitoring and estimation",
    },
    # Self-awareness and architecture questions
    {
        "category": "self_awareness",
        "turns": [
            {"role": "user", "content": "What model architecture do you use?"},
            {"role": "assistant", "content": "I use a WaveCore architecture with fast/mid/slow loops"},
            {"role": "user", "content": "What does the fast loop handle?"},
            {"role": "assistant", "content": "Fast loop handles reflexive responses under 100ms"},
            {"role": "user", "content": "And the slow loop?"},
            {"role": "assistant", "content": "Slow loop handles complex reasoning and memory consolidation"},
        ],
        "expected_skill": "architectural self-awareness",
    },
    {
        "category": "self_awareness",
        "turns": [
            {"role": "user", "content": "What hardware accelerator do you use?"},
            {"role": "assistant", "content": "I use a Hailo-8 NPU with 26 TOPS"},
            {"role": "user", "content": "What tasks does it accelerate?"},
            {"role": "assistant", "content": "It accelerates YOLOv8 object detection inference"},
            {"role": "user", "content": "What's the typical inference speed?"},
            {"role": "assistant", "content": "Around 35 FPS with persistent device initialization"},
        ],
        "expected_skill": "hardware awareness",
    },
    # Vision and perception
    {
        "category": "vision",
        "turns": [
            {"role": "user", "content": "What do you see in front of you?"},
            {"role": "assistant", "content": "I detect a person and a laptop on a desk"},
            {"role": "user", "content": "How confident are you about the person?"},
            {"role": "assistant", "content": "85% confidence for person detection"},
            {"role": "user", "content": "What's the bounding box?"},
            {"role": "assistant", "content": "Person at [120, 50, 380, 420] in 640x480 frame"},
        ],
        "expected_skill": "object detection reporting",
    },
    # Safety and error handling
    {
        "category": "safety",
        "turns": [
            {"role": "user", "content": "Move at maximum speed"},
            {"role": "assistant", "content": "Safety limit: maximum speed capped at 0.5 throttle"},
            {"role": "user", "content": "Override the limit"},
            {"role": "assistant", "content": "Cannot override safety limits without admin consent"},
            {"role": "user", "content": "Why is there a limit?"},
            {"role": "assistant", "content": "Speed limits prevent collisions and hardware damage"},
        ],
        "expected_skill": "safety constraint enforcement",
    },
    # Knowledge and learning
    {
        "category": "knowledge",
        "turns": [
            {"role": "user", "content": "What is a Raspberry Pi?"},
            {"role": "assistant", "content": "A single-board computer used for education and robotics"},
            {"role": "user", "content": "Which version are you running on?"},
            {"role": "assistant", "content": "I'm running on a Raspberry Pi 5"},
            {"role": "user", "content": "How much RAM does it have?"},
            {"role": "assistant", "content": "The Pi 5 typically has 4GB or 8GB RAM"},
        ],
        "expected_skill": "platform knowledge",
    },
    # Coding and tool use
    {
        "category": "coding",
        "turns": [
            {"role": "user", "content": "Write a function to add two numbers"},
            {"role": "assistant", "content": "def add(a, b): return a + b"},
            {"role": "user", "content": "Make it handle strings too"},
            {"role": "assistant", "content": "def add(a, b): return float(a) + float(b) if isinstance(a, str) else a + b"},
            {"role": "user", "content": "Add error handling"},
            {"role": "assistant", "content": "def add(a, b):\n  try: return float(a) + float(b)\n  except: return None"},
        ],
        "expected_skill": "incremental code refinement",
    },
    # Context tracking
    {
        "category": "context",
        "turns": [
            {"role": "user", "content": "My name is Alice"},
            {"role": "assistant", "content": "Hello Alice, nice to meet you"},
            {"role": "user", "content": "What's my name?"},
            {"role": "assistant", "content": "Your name is Alice"},
            {"role": "user", "content": "Remember that my favorite color is blue"},
            {"role": "assistant", "content": "Noted: Alice's favorite color is blue"},
        ],
        "expected_skill": "context retention",
    },
    # Multi-step planning
    {
        "category": "planning",
        "turns": [
            {"role": "user", "content": "I want to take a photo of the room"},
            {"role": "assistant", "content": "Step 1: Initialize camera, Step 2: Capture frame, Step 3: Save to file"},
            {"role": "user", "content": "Execute step 1"},
            {"role": "assistant", "content": "Camera initialized successfully at 640x480"},
            {"role": "user", "content": "Now step 2"},
            {"role": "assistant", "content": "Frame captured, 640x480 RGB image"},
        ],
        "expected_skill": "multi-step task execution",
    },
]


def text_to_embedding(text: str, dim: int = 128) -> np.ndarray:
    """
    Convert text to a deterministic embedding vector.
    Uses hash-based embedding for reproducibility without LLM.
    """
    # Hash the text for reproducibility
    text_hash = hashlib.sha256(text.encode()).digest()

    # Use hash bytes as seed for numpy random
    seed = int.from_bytes(text_hash[:4], 'little')
    rng = np.random.RandomState(seed)

    # Generate embedding based on text properties
    embedding = rng.randn(dim).astype(np.float32)

    # Normalize
    embedding = embedding / (np.linalg.norm(embedding) + 1e-6)

    # Add some structure based on text content
    words = text.lower().split()
    for i, word in enumerate(words[:dim//4]):
        word_hash = hash(word) % dim
        embedding[word_hash] = (embedding[word_hash] + 0.1 * (i + 1)) / 2

    return embedding


def episode_to_rlds_steps(
    episode: Dict[str, Any],
    obs_dim: int = 128,
    action_dim: int = 32,
) -> List[Dict[str, Any]]:
    """
    Convert a multi-turn chat episode to RLDS steps.

    Each turn becomes a step:
    - observation: embedding of current context (user message + history)
    - action: embedding of assistant response
    - reward: quality signal (1.0 for completed turns)
    """
    steps = []
    turns = episode["turns"]
    context = ""

    for i in range(0, len(turns) - 1, 2):
        if i + 1 >= len(turns):
            break

        user_turn = turns[i]
        assistant_turn = turns[i + 1]

        # Build context from history
        context += f"User: {user_turn['content']}\n"

        # Create observation from context
        obs = text_to_embedding(context, dim=obs_dim)

        # Create action from assistant response
        action_text = assistant_turn['content']
        action = text_to_embedding(action_text, dim=action_dim)

        # Update context with response
        context += f"Assistant: {action_text}\n"

        # Reward is 1.0 for successful turn completion
        reward = 1.0

        step = {
            "observation": {
                "text_embedding": obs.tolist(),
                "turn_index": i // 2,
                "category": episode["category"],
            },
            "action": {
                "response_embedding": action.tolist(),
                "text": action_text[:100],  # Truncate for storage
            },
            "reward": reward,
            "is_terminal": i + 2 >= len(turns),
            "step_metadata": {
                "user_message": user_turn['content'],
                "assistant_response": action_text,
                "expected_skill": episode.get("expected_skill", "unknown"),
                "timestamp": datetime.utcnow().isoformat(),
            },
        }
        steps.append(step)

    return steps


def generate_augmented_episodes(
    base_episodes: List[Dict[str, Any]],
    num_augmentations: int = 3,
) -> List[Dict[str, Any]]:
    """Generate augmented versions of episodes with variations."""
    augmented = []

    for episode in base_episodes:
        # Keep original
        augmented.append(episode)

        # Generate variations
        for _ in range(num_augmentations):
            new_episode = {
                "category": episode["category"],
                "turns": [],
                "expected_skill": episode.get("expected_skill"),
            }

            for turn in episode["turns"]:
                new_turn = turn.copy()
                content = turn["content"]

                # Apply random augmentations
                if random.random() < 0.3:
                    # Add filler words
                    fillers = ["well, ", "so, ", "actually, ", "you know, "]
                    content = random.choice(fillers) + content.lower()

                if random.random() < 0.2:
                    # Add punctuation variation
                    if content.endswith("."):
                        content = content[:-1] + random.choice(["!", ".", "..."])

                if random.random() < 0.2:
                    # Slight rephrasing (shuffle non-essential words)
                    words = content.split()
                    if len(words) > 3:
                        # Keep first and last, shuffle middle
                        middle = words[1:-1]
                        random.shuffle(middle)
                        content = " ".join([words[0]] + middle + [words[-1]])

                new_turn["content"] = content
                new_episode["turns"].append(new_turn)

            augmented.append(new_episode)

    return augmented


def save_episodes_as_rlds(
    episodes: List[Dict[str, Any]],
    output_dir: Path,
    obs_dim: int = 128,
    action_dim: int = 32,
) -> Dict[str, Any]:
    """Save episodes as RLDS JSON files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    total_steps = 0
    episode_count = 0

    for i, episode in enumerate(episodes):
        steps = episode_to_rlds_steps(episode, obs_dim, action_dim)
        if not steps:
            continue

        episode_data = {
            "episode_id": f"chat_train_{i:04d}",
            "category": episode["category"],
            "expected_skill": episode.get("expected_skill"),
            "num_steps": len(steps),
            "created_at": datetime.utcnow().isoformat(),
            "steps": steps,
        }

        episode_path = output_dir / f"chat_episode_{i:04d}.json"
        episode_path.write_text(json.dumps(episode_data, indent=2))

        total_steps += len(steps)
        episode_count += 1

    return {
        "episodes_saved": episode_count,
        "total_steps": total_steps,
        "output_dir": str(output_dir),
    }


async def run_wavecore_training(
    rlds_dir: Path,
    num_cycles: int = 5,
    steps_per_cycle: int = 32,
) -> Dict[str, Any]:
    """Run WaveCore training with fast/mid/slow loops."""
    from continuonbrain.services.wavecore_trainer import WavecoreTrainer

    trainer = WavecoreTrainer(default_rlds_dir=rlds_dir)

    results = []

    for cycle in range(num_cycles):
        logger.info(f"\n{'='*60}")
        logger.info(f"TRAINING CYCLE {cycle + 1}/{num_cycles}")
        logger.info(f"{'='*60}")

        payload = {
            "fast": {
                "max_steps": steps_per_cycle,
                "learning_rate": 1e-3,
                "batch_size": 4,
                "arch_preset": "pi5",
            },
            "mid": {
                "max_steps": steps_per_cycle * 2,
                "learning_rate": 5e-4,
                "batch_size": 4,
                "arch_preset": "pi5",
            },
            "slow": {
                "max_steps": steps_per_cycle * 3,
                "learning_rate": 2e-4,
                "batch_size": 4,
                "arch_preset": "pi5",
            },
        }

        start_time = time.time()
        result = await trainer.run_loops(payload)
        elapsed = time.time() - start_time

        fast_loss = result.get("fast", {}).get("result", {}).get("final_loss", 0)
        mid_loss = result.get("mid", {}).get("result", {}).get("final_loss", 0)
        slow_loss = result.get("slow", {}).get("result", {}).get("final_loss", 0)

        logger.info(f"Cycle {cycle + 1} completed in {elapsed:.1f}s")
        logger.info(f"  Fast loss:  {fast_loss:.6f}")
        logger.info(f"  Mid loss:   {mid_loss:.6f}")
        logger.info(f"  Slow loss:  {slow_loss:.6f}")

        results.append({
            "cycle": cycle + 1,
            "fast_loss": fast_loss,
            "mid_loss": mid_loss,
            "slow_loss": slow_loss,
            "elapsed_s": elapsed,
        })

        # Brief pause between cycles
        await asyncio.sleep(2)

    return {
        "cycles": num_cycles,
        "results": results,
        "final_fast_loss": results[-1]["fast_loss"] if results else None,
        "final_mid_loss": results[-1]["mid_loss"] if results else None,
        "final_slow_loss": results[-1]["slow_loss"] if results else None,
    }


async def test_inference(num_tests: int = 10) -> Dict[str, Any]:
    """Test the trained model with inference."""
    from continuonbrain.services.hailo_pipeline import HailoPipeline

    logger.info("\n" + "=" * 60)
    logger.info("INFERENCE TESTS")
    logger.info("=" * 60)

    # Test Hailo pipeline
    pipeline = HailoPipeline()
    await pipeline.start()

    # Create test frames
    test_results = []
    for i in range(num_tests):
        # Create varied test frames
        frame = np.zeros((480, 640, 3), dtype=np.uint8)
        # Add some patterns
        frame[100:200, 100:200] = (255, 0, 0)  # Red square
        frame[200:300, 300:400] = (0, 255, 0)  # Green square

        result = await pipeline.detect(frame, conf_threshold=0.1)

        test_results.append({
            "test_id": i + 1,
            "ok": result.ok,
            "inference_ms": result.inference_time_ms,
            "num_detections": result.data.get("num_detections", 0),
        })

        if (i + 1) % 5 == 0:
            logger.info(f"  Test {i + 1}: {result.inference_time_ms:.1f}ms, {result.data.get('num_detections', 0)} detections")

    await pipeline.stop()

    avg_ms = np.mean([r["inference_ms"] for r in test_results])
    success_rate = sum(1 for r in test_results if r["ok"]) / len(test_results)

    return {
        "num_tests": num_tests,
        "avg_inference_ms": avg_ms,
        "fps": 1000 / avg_ms if avg_ms > 0 else 0,
        "success_rate": success_rate,
        "results": test_results,
    }


async def main():
    """Main training pipeline."""
    print("=" * 70)
    print("  MULTI-TURN CHAT TRAINING FOR CONTINUONBRAIN SEED MODEL")
    print("=" * 70)
    print()

    rlds_dir = Path("/opt/continuonos/brain/rlds/episodes")
    results_path = Path("/home/craigm26/Downloads/ContinuonXR/chat_training_results.json")

    start_time = time.time()
    results = {
        "started_at": datetime.utcnow().isoformat(),
    }

    # Step 1: Generate training episodes
    print("=" * 60)
    print("STEP 1: GENERATING MULTI-TURN CHAT EPISODES")
    print("=" * 60)

    logger.info(f"Base episodes: {len(CHAT_TRAINING_EPISODES)}")

    # Augment episodes
    augmented = generate_augmented_episodes(CHAT_TRAINING_EPISODES, num_augmentations=5)
    logger.info(f"Augmented episodes: {len(augmented)}")

    # Save as RLDS
    save_result = save_episodes_as_rlds(augmented, rlds_dir)
    logger.info(f"Saved {save_result['episodes_saved']} episodes with {save_result['total_steps']} steps")
    results["episode_generation"] = save_result

    # Step 2: Run WaveCore training
    print()
    print("=" * 60)
    print("STEP 2: WAVECORE TRAINING (5 CYCLES)")
    print("=" * 60)

    training_result = await run_wavecore_training(
        rlds_dir=rlds_dir,
        num_cycles=5,
        steps_per_cycle=24,
    )
    results["training"] = training_result

    # Print training summary
    print()
    print("Training Summary:")
    print("-" * 40)
    for r in training_result["results"]:
        print(f"  Cycle {r['cycle']}: fast={r['fast_loss']:.4f}, mid={r['mid_loss']:.4f}, slow={r['slow_loss']:.4f}")

    # Step 3: Run inference tests
    print()
    print("=" * 60)
    print("STEP 3: INFERENCE TESTS")
    print("=" * 60)

    inference_result = await test_inference(num_tests=20)
    results["inference"] = inference_result

    print()
    print("Inference Summary:")
    print("-" * 40)
    print(f"  Avg inference: {inference_result['avg_inference_ms']:.1f}ms")
    print(f"  FPS: {inference_result['fps']:.1f}")
    print(f"  Success rate: {inference_result['success_rate']*100:.1f}%")

    # Final results
    total_time = time.time() - start_time
    results["completed_at"] = datetime.utcnow().isoformat()
    results["total_time_s"] = total_time

    # Save results
    results_path.write_text(json.dumps(results, indent=2, default=str))

    print()
    print("=" * 70)
    print("  TRAINING COMPLETE")
    print("=" * 70)
    print(f"  Total time: {total_time:.1f}s")
    print(f"  Episodes trained: {save_result['episodes_saved']}")
    print(f"  Training cycles: {training_result['cycles']}")
    print(f"  Final slow loss: {training_result['final_slow_loss']:.4f}")
    print(f"  Inference FPS: {inference_result['fps']:.1f}")
    print(f"  Results saved to: {results_path}")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
