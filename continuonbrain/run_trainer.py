"""
Launch the Nested Learning Sidecar Trainer.
"""
import sys
import time
from pathlib import Path
import logging

# Ensure repo root is on path
REPO_ROOT = Path(__file__).parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

# Import Sidecar Deps
from continuonbrain.trainer.sidecar_runner import SidecarTrainer
from continuonbrain.trainer.local_lora_trainer import ModelHooks, SafetyGateConfig, GatingSensors

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("TrainerSidecar")

def main():
    config_dir = Path("/tmp/continuonbrain_demo")
    config_dir.mkdir(parents=True, exist_ok=True)
    
    # Paths for trainer config
    trainer_config_path = config_dir / "trainer_config.json"
    
    # Create default config if missing
    if not trainer_config_path.exists():
        # ... logic to create default config if we had the class schema handy ...
        # For now, we assume the user/setup ensures it, or we rely on defaults
        pass

    # Stub hooks for now (Gemma hooks require real model path)
    hooks = ModelHooks() 
    safety = SafetyGateConfig()
    gating = GatingSensors()

    logger.info("Starting Nested Learning Sidecar...")
    
    try:
        if not trainer_config_path.exists():
            logger.warning(f"Trainer config not found at {trainer_config_path}, waiting...")
            # Ideally we write a default one here
            
        trainer = SidecarTrainer(
            cfg_path=trainer_config_path,
            hooks=hooks,
            safety_cfg=safety,
            gating=gating
        )

        while True:
            # Run the mid-loop training check
            result = trainer.train_if_new_data()
            if result.status == "ok":
                logger.info(f"Training success: {result.steps} steps, loss={result.avg_loss}")
            elif result.reason != "No new RLDS episodes since last promotion":
                # Only log interesting skips
                logger.info(f"Training skipped: {result.reason}")
            
            # Sleep loop (e.g. 1 minute)
            time.sleep(60)
            
    except KeyboardInterrupt:
        logger.info("Trainer stopping...")

if __name__ == "__main__":
    main()
