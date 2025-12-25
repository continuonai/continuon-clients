
import os
import json
import shutil
import time
from pathlib import Path
from datetime import datetime, timedelta
import numpy as np

# Mock encoder to avoid heavy dependencies in simple verify script
class MockEncoder:
    def encode(self, text, **kwargs):
        if "move" in text.lower() or "drive" in text.lower():
            return np.array([1.0, 0.0, 0.0])
        if "camera" in text.lower() or "see" in text.lower():
            return np.array([0.0, 1.0, 0.0])
        return np.array([0.0, 0.0, 1.0])

# Inject mock encoder
import continuonbrain.services.experience_logger as exp_logger
exp_logger.get_encoder = lambda: MockEncoder()

from continuonbrain.services.experience_logger import ExperienceLogger

def setup_test_env():
    test_dir = Path("tmp/verify_memory")
    if test_dir.exists():
        shutil.rmtree(test_dir)
    test_dir.mkdir(parents=True)
    return test_dir

def test_redundancy_consolidation(test_dir):
    print("--- Testing Redundancy Consolidation ---")
    logger = ExperienceLogger(test_dir)
    
    # Generate 100 similar memories
    print("Generating 100 similar 'move' memories...")
    for i in range(100):
        logger.log_conversation(
            question=f"How do I move the robot? (variation {i})",
            answer=f"Use the drive API to move. ({i})",
            agent="llm",
            confidence=0.8
        )
    
    stats = logger.get_statistics()
    print(f"Total before consolidation: {stats['total_conversations']}")
    
    # Manual trigger consolidation
    res = logger.consolidate_memories(similarity_threshold=0.9)
    print(f"Consolidation result: {res}")
    
    stats_after = logger.get_statistics()
    print(f"Total after consolidation: {stats_after['total_conversations']}")
    
    if stats_after['total_conversations'] < 20:
        print("✅ SUCCESS: Redundancy significantly reduced.")
    else:
        print("❌ FAILURE: Redundancy not reduced enough.")

def test_confidence_decay(test_dir):
    print("\n--- Testing Recency Decay ---")
    logger = ExperienceLogger(test_dir)
    
    # Log a memory and manually backdate it
    conv_id = logger.log_conversation("Old memory", "Old answer", "llm", 0.9)
    
    # Backdate last_accessed to 40 days ago
    old_date = (datetime.now() - timedelta(days=40)).isoformat()
    
    temp_file = logger.conversations_file.with_suffix('.tmp')
    with open(logger.conversations_file, 'r') as f_in, open(temp_file, 'w') as f_out:
        for line in f_in:
            conv = json.loads(line)
            if conv['conversation_id'] == conv_id:
                conv['last_accessed'] = old_date
                conv['timestamp'] = old_date
            f_out.write(json.dumps(conv) + '\n')
    temp_file.replace(logger.conversations_file)
    
    # Apply decay (factor 0.95 per day over 30) -> 10 days of decay
    # 0.9 * (0.95^10) approx 0.9 * 0.59 = 0.53
    logger.apply_confidence_decay(decay_factor=0.95, max_age_days=30)
    
    # Verify
    with open(logger.conversations_file, 'r') as f:
        for line in f:
            conv = json.loads(line)
            if conv['conversation_id'] == conv_id:
                print(f"New confidence: {conv['confidence']}")
                if conv['confidence'] < 0.8:
                    print("✅ SUCCESS: Confidence decayed correctly.")
                else:
                    print(f"❌ FAILURE: Confidence did not decay enough ({conv['confidence']})")

def test_model_evolution_penalty(test_dir):
    print("\n--- Testing Model Evolution Penalty ---")
    logger = ExperienceLogger(test_dir)
    
    logger.log_conversation("New memory", "New answer", "llm", 1.0)
    
    # Apply 10% penalty
    logger.apply_model_evolution_penalty(penalty=0.10)
    
    # Verify
    with open(logger.conversations_file, 'r') as f:
        for line in f:
            conv = json.loads(line)
            if conv['question'] == "New memory":
                print(f"New confidence: {conv['confidence']}")
                if conv['confidence'] == 0.9:
                    print("✅ SUCCESS: Model evolution penalty applied.")
                else:
                    print(f"❌ FAILURE: Penalty not applied correctly ({conv['confidence']})")

def test_validated_immunity(test_dir):
    print("\n--- Testing Validated Immunity ---")
    logger = ExperienceLogger(test_dir)
    
    conv_id = logger.log_conversation("Validated memory", "Validated answer", "llm", 1.0)
    logger.validate_conversation(conv_id, True)
    
    # Try to decay and apply penalty
    logger.apply_model_evolution_penalty(penalty=0.50)
    
    # Verify
    with open(logger.conversations_file, 'r') as f:
        for line in f:
            conv = json.loads(line)
            if conv['conversation_id'] == conv_id:
                print(f"Confidence: {conv['confidence']}")
                if conv['confidence'] == 1.0:
                    print("✅ SUCCESS: Validated memory is immune to decay.")
                else:
                    print(f"❌ FAILURE: Validated memory was affected ({conv['confidence']})")

if __name__ == "__main__":
    t_dir = setup_test_env()
    try:
        test_redundancy_consolidation(t_dir)
        test_confidence_decay(t_dir)
        test_model_evolution_penalty(t_dir)
        test_validated_immunity(t_dir)
    finally:
        # shutil.rmtree(t_dir)
        pass
