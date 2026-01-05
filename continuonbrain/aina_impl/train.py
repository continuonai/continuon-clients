import os
import time
import logging
import torch
import torch.nn as nn
import torch.optim as optim
from .policy import AINAPolicy
from .dataset import AINADataset, create_dataloader

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
            - rlds_dir: Path to RLDS episodes directory
            - lr: Learning rate (default: 1e-4)
            - epochs: Number of epochs (default: 5)
            - batch_size: Batch size (default: 4)
            - obs_horizon: Observation window (default: 10)
            - pred_horizon: Prediction window (default: 5)
            - checkpoint_dir: Checkpoint directory
            - use_real_data: Use real RLDS data (default: True if rlds_dir exists)
        status_callback (callable): Optional callback for status updates (msg, progress).
    """
    if config is None:
        config = {}

    lr = config.get("lr", 1e-4)
    epochs = config.get("epochs", 5)
    batch_size = config.get("batch_size", 4)
    obs_horizon = config.get("obs_horizon", 10)
    pred_horizon = config.get("pred_horizon", 5)
    n_obj_points = config.get("n_obj_points", 100)
    checkpoint_dir = config.get("checkpoint_dir", "./checkpoints/aina_vision")
    rlds_dir = config.get("rlds_dir", "/opt/continuonos/brain/rlds/episodes")

    os.makedirs(checkpoint_dir, exist_ok=True)

    # Setup Logging
    log_file = os.path.join(checkpoint_dir, "training.log")
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    # Check if real data is available
    use_real_data = config.get("use_real_data", os.path.isdir(rlds_dir))

    msg = f"Starting AINA Policy Training. Epochs: {epochs}, LR: {lr}, Real Data: {use_real_data}"
    logger.info(msg)
    if status_callback: status_callback(msg)

    # Model - adapted for SO-ARM101 (1 end-effector instead of 5 fingers)
    # Output: 7 values (6 joints + gripper) instead of 5*3 fingertips
    model = AINAPolicy(
        n_fingers=1,  # Single end-effector
        n_obj_points=n_obj_points,
        obs_horizon=obs_horizon,
        pred_horizon=pred_horizon,
    )
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    model.train()

    # Create dataloader
    dataloader = None
    if use_real_data:
        try:
            dataset = AINADataset(
                episodes_dir=rlds_dir,
                obs_horizon=obs_horizon,
                pred_horizon=pred_horizon,
                n_obj_points=n_obj_points,
            )
            if len(dataset) > 0:
                dataloader = torch.utils.data.DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=True,
                    drop_last=True,
                )
                logger.info(f"Using real data: {len(dataset)} samples from {rlds_dir}")
            else:
                logger.warning("No samples found in RLDS directory, falling back to synthetic data")
                use_real_data = False
        except Exception as e:
            logger.warning(f"Failed to load real data: {e}, falling back to synthetic data")
            use_real_data = False

    for epoch in range(epochs):
        epoch_loss = 0.0
        num_batches = 0

        if use_real_data and dataloader:
            # Use real RLDS data
            for batch in dataloader:
                ee_traj = batch["ee_trajectory"]  # [B, T_obs, 3]
                obj_pcd = batch["object_pcd"]     # [B, T_obs, N_points, 3]
                target = batch["target_joints"]    # [B, T_pred, 7]

                # Reshape for model: expand EE to match expected finger dims
                # [B, T_obs, 1, 3]
                fingertip_hist = ee_traj.unsqueeze(2)
                # [B, T_pred, 1, 3] - use first 3 coords of target as position
                target_fingertips = target[:, :, :3].unsqueeze(2)

                optimizer.zero_grad()
                pred_fingertips = model(fingertip_hist, obj_pcd)
                loss = criterion(pred_fingertips, target_fingertips)
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()
                num_batches += 1
        else:
            # Generate synthetic batch for testing
            fingertip_hist = torch.randn(batch_size, obs_horizon, 1, 3)
            object_pcd = torch.randn(batch_size, obs_horizon, n_obj_points, 3)
            target_fingertips = torch.randn(batch_size, pred_horizon, 1, 3)

            optimizer.zero_grad()
            pred_fingertips = model(fingertip_hist, object_pcd)
            loss = criterion(pred_fingertips, target_fingertips)
            loss.backward()
            optimizer.step()

            epoch_loss = loss.item()
            num_batches = 1

        avg_loss = epoch_loss / max(num_batches, 1)

        # Log Progress
        progress_msg = f"Epoch {epoch+1}/{epochs}, Loss: {avg_loss:.6f}, Batches: {num_batches}"
        logger.info(progress_msg)
        if status_callback: status_callback(progress_msg)

        # Save Checkpoint
        if (epoch + 1) % 5 == 0 or (epoch + 1) == epochs:
            ckpt_path = os.path.join(checkpoint_dir, f"aina_policy_epoch_{epoch+1}.pt")
            save_checkpoint(model, optimizer, epoch, avg_loss, ckpt_path)

    complete_msg = "Training Complete."
    logger.info(complete_msg)
    if status_callback: status_callback(complete_msg)

    logger.removeHandler(file_handler)
    return model

if __name__ == "__main__":
    # Test run
    logging.basicConfig(level=logging.INFO) # Console out for manual run
    run_training_session()
