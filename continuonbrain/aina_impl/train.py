import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from .policy import AINAPolicy

# Configure Logger for this module
logger = logging.getLogger(__name__)

def save_checkpoint(model, optimizer, epoch, loss, path):
    """Save training state."""
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }, path)
    logger.info(f"Checkpoint saved to {path}")

def run_training_session(config=None, status_callback=None):
    """
    Run a full training session.
    Args:
        config (dict): Configuration overrides.
        status_callback (callable): Optional callback for status updates (msg, progress).
    """
    if config is None:
        config = {}
        
    lr = config.get("lr", 1e-4)
    epochs = config.get("epochs", 5) # Short default for testing
    batch_size = config.get("batch_size", 4)
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints/aina_vision")
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    # Setup Logging
    log_file = os.path.join(checkpoint_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)
    
    msg = f"Starting AINA Policy Training. Epochs: {epochs}, LR: {lr}"
    logger.info(msg)
    if status_callback: status_callback(msg)

    # Model
    model = AINAPolicy(n_fingers=5, n_obj_points=100)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()
    
    model.train()
    
    for epoch in range(epochs):
        # Generate Dummy Batch (Replace with Data Loader in Prod)
        fingertip_hist = torch.randn(batch_size, 10, 5, 3) 
        object_pcd = torch.randn(batch_size, 10, 100, 3)
        target_fingertips = torch.randn(batch_size, 5, 5, 3)
        
        optimizer.zero_grad()
        pred_fingertips = model(fingertip_hist, object_pcd)
        loss = criterion(pred_fingertips, target_fingertips)
        loss.backward()
        optimizer.step()
        
        # Log Progress
        progress_msg = f"Epoch {epoch+1}/{epochs}, Loss: {loss.item():.6f}"
        logger.info(progress_msg)
        if status_callback: status_callback(progress_msg)
        
        # Save Checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"aina_policy_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, loss, ckpt_path)

    complete_msg = "Training Complete."
    logger.info(complete_msg)
    if status_callback: status_callback(complete_msg)
    
    logger.removeHandler(file_handler) # Clean up
    return model

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO) # Console out for manual run
    run_training_session()
