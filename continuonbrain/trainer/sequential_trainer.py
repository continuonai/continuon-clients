import time
import logging
import gc
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from typing import Dict, Any, Optional, Callable
from continuonbrain.resource_monitor import ResourceMonitor, ResourceLevel
from continuonbrain.trainer.datasets import FastWindowDataset, MidTrajectoryDataset

logger = logging.getLogger(__name__)

class SequentialTrainer:
    def __init__(self, 
                 checkpoint_dir: str, 
                 data_dir: str, 
                 resource_monitor: Optional[ResourceMonitor] = None):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.data_dir = Path(data_dir)
        self.resource_monitor = resource_monitor or ResourceMonitor()
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    def train_component(self, 
                        component_name: str, 
                        model_factory: Callable[[], nn.Module], 
                        dataset_class: Any, 
                        epochs: int = 1,
                        batch_size: int = 32,
                        learning_rate: float = 1e-4,
                        collate_fn: Optional[Callable] = None):
        """
        Train a single component sequentially.
        """
        logger.info(f"Starting sequential training for: {component_name}")
        
        # 1. Check Resources
        status = self.resource_monitor.check_resources()
        if not status.can_allocate:
            logger.error(f"Cannot start training {component_name}: {status.message}")
            return False
            
        # 2. Load Dataset
        try:
            logger.info(f"Loading dataset for {component_name}...")
            dataset = dataset_class(str(self.data_dir))
            dataloader = torch.utils.data.DataLoader(dataset, 
                                                   batch_size=batch_size, 
                                                   shuffle=True,
                                                   collate_fn=collate_fn)
            logger.info(f"Dataset loaded. {len(dataset)} samples.")
        except Exception as e:
            logger.error(f"Failed to load dataset: {e}")
            return False
            
        # 3. Load Model
        try:
            logger.info(f"Initializing model for {component_name}...")
            model = model_factory()
            optimizer = optim.AdamW(model.parameters(), lr=learning_rate)
            # Load checkpoint if exists
            ckpt_path = self.checkpoint_dir / f"{component_name}_latest.pt"
            if ckpt_path.exists():
                logger.info(f"Resuming from checkpoint: {ckpt_path}")
                checkpoint = torch.load(ckpt_path)
                model.load_state_dict(checkpoint['model_state_dict'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception as e:
            logger.error(f"Failed to initialize model: {e}")
            return False
            
        # 4. Training Loop
        model.train()
        # criterion = nn.MSELoss() # defined by model or passed in?
        
        for epoch in range(epochs):
            total_loss = 0
            steps = 0
            for batch in dataloader:
                # Monitor resources during training
                if steps % 10 == 0:
                     status = self.resource_monitor.check_resources()
                     if status.level in [ResourceLevel.CRITICAL, ResourceLevel.EMERGENCY]:
                         logger.warning(f"Resource pressure during training: {status.message}. Saving and stopping.")
                         self._save_checkpoint(component_name, model, optimizer, epoch, steps)
                         return False

                optimizer.zero_grad()
                
                # Abstracting the forward/loss pass
                if hasattr(model, 'train_step'):
                    loss = model.train_step(batch)
                else:
                    # Fallback or error
                    logger.warning(f"Model {component_name} does not implement train_step")
                    loss = torch.tensor(0.0, requires_grad=True)

                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                steps += 1
            
            avg_loss = total_loss / steps if steps > 0 else 0
            logger.info(f"Epoch {epoch+1}/{epochs} completed. Avg Loss: {avg_loss:.4f}")
            
        # 5. Save Checkpoint
        self._save_checkpoint(component_name, model, optimizer, epochs, 0)
        
        # 6. Unload
        logger.info(f"Unloading {component_name}...")
        del model
        del optimizer
        del dataloader
        del dataset
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        return True

    def _save_checkpoint(self, name, model, optimizer, epoch, step):
        path = self.checkpoint_dir / f"{name}_latest.pt"
        torch.save({
            'epoch': epoch,
            'step': step,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, path)
        logger.info(f"Saved checkpoint to {path}")
