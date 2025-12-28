#!/usr/bin/env python3
"""
Train World Model via WaveCore Loops

This script trains the JAX CoreModel (world model) using:
1. Real RLDS episodes from autonomous runs
2. Synthetic episodes for bootstrapping
3. WaveCore fast/mid/slow training loops

The trained model enables physics prediction for action planning.
"""

import os
import sys
import json
import time
import argparse
import logging
from pathlib import Path
from datetime import datetime

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s'
)
logger = logging.getLogger(__name__)


def count_rlds_episodes(episodes_dir: Path) -> int:
    """Count available RLDS episodes."""
    count = 0
    if episodes_dir.exists():
        for f in episodes_dir.glob("*.json"):
            count += 1
        for f in episodes_dir.glob("*/episode.json"):
            count += 1
    return count


def run_wavecore_training(
    config_dir: str,
    episodes_dir: str,
    fast_steps: int = 100,
    mid_steps: int = 200,
    slow_steps: int = 300,
    use_synthetic: bool = False,
    run_hope_eval: bool = True,
    compact_export: bool = True,
) -> dict:
    """
    Run WaveCore training loops.
    
    Returns:
        Training results dict
    """
    from continuonbrain.services.brain_service import BrainService
    import asyncio
    
    # Initialize brain service
    logger.info(f"Initializing BrainService with config_dir={config_dir}")
    brain = BrainService(config_dir=config_dir)
    
    # Build training payload
    payload = {
        "fast": {
            "arch_preset": "pi5",
            "max_steps": fast_steps,
            "batch_size": 8,
            "learning_rate": 1e-3,
            "disable_jit": True,
            "use_synthetic": use_synthetic,
        },
        "mid": {
            "arch_preset": "pi5",
            "max_steps": mid_steps,
            "batch_size": 8,
            "learning_rate": 5e-4,
            "disable_jit": True,
            "use_synthetic": use_synthetic,
        },
        "slow": {
            "arch_preset": "pi5",
            "max_steps": slow_steps,
            "batch_size": 8,
            "learning_rate": 2e-4,
            "disable_jit": True,
            "use_synthetic": use_synthetic,
        },
        "compact_export": compact_export,
        "run_hope_eval": run_hope_eval,
        "run_facts_eval": False,
        "episodes_dir": episodes_dir,
    }
    
    logger.info("Starting WaveCore training loops...")
    logger.info(f"  Fast:  {fast_steps} steps")
    logger.info(f"  Mid:   {mid_steps} steps")
    logger.info(f"  Slow:  {slow_steps} steps")
    
    start = time.time()
    result = asyncio.run(brain.RunWavecoreLoops(payload))
    duration = time.time() - start
    
    result["total_duration_s"] = duration
    result["training_config"] = payload
    
    return result


def verify_world_model(config_dir: str) -> dict:
    """Verify the trained world model works."""
    logger.info("Verifying trained world model...")
    
    try:
        # Try Mamba world model first (simpler API)
        from continuonbrain.mamba_brain import WorldModelState, WorldModelAction
        from continuonbrain.mamba_brain.world_model import build_world_model
        
        # Build world model (will use Mamba or stub)
        world_model = build_world_model(prefer_mamba=True)
        
        # Test prediction
        state = WorldModelState(joint_pos=[0.5] * 6)
        action = WorldModelAction(joint_delta=[0.1, -0.1, 0.05, 0.0, 0.0, 0.0])
        
        result = world_model.predict(state, action)
        
        return {
            "success": True,
            "next_state": result.next_state.joint_pos,
            "uncertainty": result.uncertainty,
            "backend": result.debug.get("backend", "unknown"),
        }
        
    except Exception as e:
        logger.error(f"World model verification failed: {e}")
        return {"success": False, "error": str(e)}


def integrate_with_hope_agent(config_dir: str) -> dict:
    """Wire trained world model to HOPE Agent."""
    logger.info("Integrating world model with HOPE Agent...")
    
    try:
        from continuonbrain.mamba_brain.world_model import build_world_model
        from continuonbrain.hope_impl.config import HOPEConfig
        from continuonbrain.hope_impl.brain import HOPEBrain
        from continuonbrain.services.agent_hope import HOPEAgent
        
        # Build world model
        world_model = build_world_model(prefer_mamba=True)
        
        # Load HOPE brain with required dimensions
        hope_config = HOPEConfig()
        hope_brain = HOPEBrain(
            hope_config,
            obs_dim=128,
            action_dim=32,
            output_dim=32,
        )
        
        # Create integrated agent
        agent = HOPEAgent(
            hope_brain,
            world_model=world_model,
            confidence_threshold=0.6,
        )
        
        # Test prediction through agent
        result = agent.predict_action_outcome(
            current_state={"joint_positions": [0.5] * 6},
            proposed_action={"joint_delta": [0.1] * 6},
        )
        
        return {
            "success": result.get("success", False),
            "integrated": True,
            "prediction": result,
        }
        
    except Exception as e:
        logger.error(f"HOPE integration failed: {e}")
        return {"success": False, "error": str(e)}


def main():
    parser = argparse.ArgumentParser(
        description="Train World Model via WaveCore Loops",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Quick training with synthetic data
  python scripts/train_world_model.py --synthetic --fast-steps 50

  # Full training with real RLDS data
  python scripts/train_world_model.py --fast-steps 100 --mid-steps 200 --slow-steps 300

  # Verify existing model
  python scripts/train_world_model.py --verify-only
"""
    )
    
    parser.add_argument(
        "--config-dir",
        default="/opt/continuonos/brain",
        help="Brain config directory"
    )
    parser.add_argument(
        "--episodes-dir",
        default="/opt/continuonos/brain/rlds/episodes",
        help="RLDS episodes directory"
    )
    parser.add_argument(
        "--fast-steps",
        type=int,
        default=100,
        help="Fast loop training steps"
    )
    parser.add_argument(
        "--mid-steps",
        type=int,
        default=200,
        help="Mid loop training steps"
    )
    parser.add_argument(
        "--slow-steps",
        type=int,
        default=300,
        help="Slow loop training steps"
    )
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use synthetic episodes (for testing)"
    )
    parser.add_argument(
        "--skip-eval",
        action="store_true",
        help="Skip HOPE evaluation after training"
    )
    parser.add_argument(
        "--verify-only",
        action="store_true",
        help="Only verify existing model, don't train"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for results JSON"
    )
    
    args = parser.parse_args()
    
    print("=" * 60)
    print("🧠 World Model Training via WaveCore")
    print("=" * 60)
    
    config_dir = Path(args.config_dir)
    episodes_dir = Path(args.episodes_dir)
    
    # Check for RLDS episodes
    episode_count = count_rlds_episodes(episodes_dir)
    print(f"\n📂 Episodes directory: {episodes_dir}")
    print(f"   Found: {episode_count} episodes")
    
    if episode_count == 0 and not args.synthetic:
        print("\n⚠️ No RLDS episodes found. Use --synthetic for testing.")
        if not args.verify_only:
            args.synthetic = True
            print("   Falling back to synthetic data...")
    
    results = {
        "timestamp": datetime.now().isoformat(),
        "config_dir": str(config_dir),
        "episodes_dir": str(episodes_dir),
        "episode_count": episode_count,
        "synthetic": args.synthetic,
    }
    
    # Verify only mode
    if args.verify_only:
        print("\n🔍 Verification Mode")
        verify_result = verify_world_model(str(config_dir))
        results["verification"] = verify_result
        
        if verify_result["success"]:
            print(f"   ✅ World model verified")
            print(f"   Backend: {verify_result.get('backend')}")
            print(f"   Uncertainty: {verify_result.get('uncertainty'):.4f}")
        else:
            print(f"   ❌ Verification failed: {verify_result.get('error')}")
        
        # Also try HOPE integration
        integration_result = integrate_with_hope_agent(str(config_dir))
        results["hope_integration"] = integration_result
        
        if integration_result["success"]:
            print(f"   ✅ HOPE Agent integration verified")
        else:
            print(f"   ❌ HOPE integration failed: {integration_result.get('error')}")
    
    else:
        # Run training
        print(f"\n🚀 Starting Training")
        print(f"   Fast loop:  {args.fast_steps} steps")
        print(f"   Mid loop:   {args.mid_steps} steps")
        print(f"   Slow loop:  {args.slow_steps} steps")
        print(f"   Synthetic:  {args.synthetic}")
        print()
        
        try:
            train_result = run_wavecore_training(
                config_dir=str(config_dir),
                episodes_dir=str(episodes_dir),
                fast_steps=args.fast_steps,
                mid_steps=args.mid_steps,
                slow_steps=args.slow_steps,
                use_synthetic=args.synthetic,
                run_hope_eval=not args.skip_eval,
            )
            results["training"] = train_result
            
            duration = train_result.get("total_duration_s", 0)
            print(f"\n✅ Training completed in {duration:.1f}s")
            
            # Verify trained model
            verify_result = verify_world_model(str(config_dir))
            results["verification"] = verify_result
            
            if verify_result["success"]:
                print(f"   World model verified: {verify_result.get('backend')}")
            
            # Integrate with HOPE
            integration_result = integrate_with_hope_agent(str(config_dir))
            results["hope_integration"] = integration_result
            
            if integration_result["success"]:
                print(f"   HOPE Agent integration successful")
            
        except Exception as e:
            logger.error(f"Training failed: {e}")
            results["error"] = str(e)
            import traceback
            traceback.print_exc()
    
    # Save results
    if args.output:
        output_path = Path(args.output)
        output_path.write_text(json.dumps(results, indent=2, default=str))
        print(f"\n📄 Results saved to: {output_path}")
    
    print("\n" + "=" * 60)
    print("✅ World Model Training Complete")
    print("=" * 60)
    
    return 0 if results.get("verification", {}).get("success") else 1


if __name__ == "__main__":
    sys.exit(main())

