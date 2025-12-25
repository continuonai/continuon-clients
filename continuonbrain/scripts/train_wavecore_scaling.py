from __future__ import annotations
import time
import torch
import logging
from pathlib import Path
from continuonbrain.wavecore.config import WaveCoreConfig
from continuonbrain.wavecore.utils.trainer import WaveCoreTrainer
from continuonbrain.system_health import SystemHealthChecker

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ScalingRunner")

def run_scaling_sweep():
    checker = SystemHealthChecker()
    # Sweep d_model from 64 to 512
    d_models = [64, 128, 256, 512]
    
    for d in d_models:
        logger.info(f"--- Testing Scale: d_model={d} ---")
        config = WaveCoreConfig(d_model=d, n_layers=4, batch_size=8)
        
        # Check health before starting
        status, _ = checker.run_all_checks(quick_mode=True)
        if status.value == "critical":
            logger.error("System health critical. Aborting sweep.")
            break
            
        try:
            trainer = WaveCoreTrainer(config)
            # Run a 50-step stress test
            res = trainer.run_sanity_check(steps=50)
            logger.info(f"Scale {d} success: avg_loss={res['avg_loss']:.4f}")
            
            # Log metrics
            health_report = checker.run_all_checks(quick_mode=True)[1]
            logger.info(f"Health stats: {health_report}")
            
        except RuntimeError as e:
            if "out of memory" in str(e):
                logger.warning(f"Out of Memory reached at d_model={d}")
                break
            else:
                logger.error(f"Error at scale {d}: {e}")
                break
        except Exception as e:
            logger.error(f"Unexpected error at scale {d}: {e}")
            break

if __name__ == "__main__":
    run_scaling_sweep()
