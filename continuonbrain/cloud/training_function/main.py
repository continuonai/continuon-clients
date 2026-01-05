"""
Cloud Function for ContinuonBrain Training.

This Cloud Function is triggered by Firestore document creation in the
'training_triggers' collection. It downloads episodes from GCS, runs
training using the WaveCore JAX model, and uploads the trained model
back to GCS.

Deployment:
    gcloud functions deploy train_model \
        --runtime python311 \
        --trigger-event providers/cloud.firestore/eventTypes/document.create \
        --trigger-resource "projects/YOUR_PROJECT/databases/(default)/documents/training_triggers/{job_id}" \
        --memory 16GB \
        --timeout 3600s \
        --region us-central1

For GPU/TPU training, use Cloud Run or Vertex AI instead.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import tempfile
import time
import zipfile
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

# Google Cloud imports
from google.cloud import firestore
from google.cloud import storage

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Firestore client (reused across invocations)
db = firestore.Client()
storage_client = storage.Client()


@dataclass
class TrainingConfig:
    """Training configuration."""
    model_type: str = "wavecore"
    epochs: int = 100
    batch_size: int = 32
    learning_rate: float = 1e-4
    obs_dim: int = 128
    action_dim: int = 32
    output_dim: int = 32
    arch_preset: str = "cloud"
    use_tpu: bool = False
    use_gpu: bool = True
    mixed_precision: bool = True
    sparsity_lambda: float = 0.0
    gradient_checkpointing: bool = False
    shuffle_buffer_size: int = 10000
    prefetch_buffer_size: int = 4

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrainingConfig":
        """Create config from dictionary."""
        return cls(**{k: v for k, v in data.items() if k in cls.__dataclass_fields__})


@dataclass
class TrainingResult:
    """Training result."""
    success: bool
    model_uri: Optional[str] = None
    final_loss: Optional[float] = None
    training_time_s: float = 0.0
    epochs_completed: int = 0
    best_checkpoint_step: Optional[int] = None
    metrics: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None


def train_model(event: Dict[str, Any], context: Any) -> None:
    """
    Cloud Function entry point triggered by Firestore document creation.

    Args:
        event: Firestore trigger event containing document data
        context: Cloud Function context with event metadata
    """
    try:
        # Extract document data
        # Event structure for Firestore triggers:
        # event['value']['fields'] contains the document fields
        doc_fields = event.get("value", {}).get("fields", {})

        job_id = _extract_field(doc_fields, "job_id")
        bucket_name = _extract_field(doc_fields, "bucket")
        blob_name = _extract_field(doc_fields, "name")
        config_data = _extract_field(doc_fields, "config", default={})

        logger.info(f"Training triggered for job: {job_id}")
        logger.info(f"Episodes: gs://{bucket_name}/{blob_name}")

        # Parse config
        config = TrainingConfig.from_dict(config_data)

        # Update job status to running
        update_job_status(job_id, "running")

        # Run training
        result = run_training(job_id, bucket_name, blob_name, config)

        # Update job with result
        if result.success:
            update_job_status(
                job_id,
                "completed",
                result={
                    "model_uri": result.model_uri,
                    "final_loss": result.final_loss,
                    "training_time_s": result.training_time_s,
                    "epochs_completed": result.epochs_completed,
                    "best_checkpoint_step": result.best_checkpoint_step,
                    "metrics": result.metrics,
                }
            )
            logger.info(f"Training completed for job: {job_id}")
        else:
            update_job_status(
                job_id,
                "failed",
                result={"error": result.error}
            )
            logger.error(f"Training failed for job: {job_id}: {result.error}")

    except Exception as e:
        logger.error(f"Training function error: {e}", exc_info=True)
        # Try to update job status
        try:
            job_id = event.get("value", {}).get("fields", {}).get("job_id", {}).get("stringValue")
            if job_id:
                update_job_status(job_id, "failed", result={"error": str(e)})
        except Exception:
            pass
        raise


def run_training(
    job_id: str,
    bucket_name: str,
    blob_name: str,
    config: TrainingConfig,
) -> TrainingResult:
    """
    Run the actual training process.

    Args:
        job_id: Training job ID
        bucket_name: GCS bucket containing episodes
        blob_name: GCS blob path for episodes
        config: Training configuration

    Returns:
        TrainingResult with model URI and metrics
    """
    start_time = time.time()

    with tempfile.TemporaryDirectory() as tmpdir:
        work_dir = Path(tmpdir)
        episodes_dir = work_dir / "episodes"
        checkpoint_dir = work_dir / "checkpoints"
        output_dir = work_dir / "output"

        episodes_dir.mkdir()
        checkpoint_dir.mkdir()
        output_dir.mkdir()

        try:
            # 1. Download episodes from GCS
            logger.info(f"Downloading episodes from gs://{bucket_name}/{blob_name}")
            download_episodes(bucket_name, blob_name, episodes_dir)

            # 2. Load and validate data
            episode_files = list(episodes_dir.glob("**/*.json"))
            logger.info(f"Found {len(episode_files)} episode files")

            if not episode_files:
                return TrainingResult(
                    success=False,
                    error="No episode files found in download",
                    training_time_s=time.time() - start_time,
                )

            # 3. Initialize JAX trainer
            trainer = WaveCoreTrainer(
                episodes_dir=episodes_dir,
                checkpoint_dir=checkpoint_dir,
                config=config,
            )

            # 4. Run training
            logger.info(f"Starting training: epochs={config.epochs}, batch_size={config.batch_size}")
            train_result = trainer.train()

            if not train_result.get("success"):
                return TrainingResult(
                    success=False,
                    error=train_result.get("error", "Training failed"),
                    training_time_s=time.time() - start_time,
                )

            # 5. Export model
            export_path = trainer.export(output_dir)

            # 6. Upload trained model to GCS
            model_gcs_path = f"models/{job_id}"
            upload_model(bucket_name, model_gcs_path, output_dir)

            model_uri = f"gs://{bucket_name}/{model_gcs_path}"

            training_time = time.time() - start_time

            return TrainingResult(
                success=True,
                model_uri=model_uri,
                final_loss=train_result.get("final_loss"),
                training_time_s=training_time,
                epochs_completed=train_result.get("epochs_completed", config.epochs),
                best_checkpoint_step=train_result.get("best_step"),
                metrics=train_result.get("metrics", {}),
            )

        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            return TrainingResult(
                success=False,
                error=str(e),
                training_time_s=time.time() - start_time,
            )


def download_episodes(bucket_name: str, blob_name: str, dest_dir: Path) -> None:
    """Download episodes from GCS."""
    bucket = storage_client.bucket(bucket_name)

    if blob_name.endswith(".zip"):
        # Download and extract zip
        blob = bucket.blob(blob_name)
        zip_path = dest_dir / "episodes.zip"
        blob.download_to_filename(str(zip_path))

        with zipfile.ZipFile(zip_path, 'r') as zf:
            zf.extractall(dest_dir)

        zip_path.unlink()
    else:
        # Download directory of files
        blobs = bucket.list_blobs(prefix=blob_name)
        for blob in blobs:
            if blob.name.endswith("/"):
                continue
            rel_path = blob.name.replace(blob_name, "").lstrip("/")
            local_path = dest_dir / rel_path
            local_path.parent.mkdir(parents=True, exist_ok=True)
            blob.download_to_filename(str(local_path))


def upload_model(bucket_name: str, gcs_path: str, source_dir: Path) -> None:
    """Upload trained model to GCS."""
    bucket = storage_client.bucket(bucket_name)

    for file_path in source_dir.rglob("*"):
        if file_path.is_file():
            rel_path = file_path.relative_to(source_dir)
            blob_path = f"{gcs_path}/{rel_path}"
            blob = bucket.blob(blob_path)
            blob.upload_from_filename(str(file_path))

    logger.info(f"Model uploaded to gs://{bucket_name}/{gcs_path}")


def update_job_status(
    job_id: str,
    status: str,
    result: Optional[Dict[str, Any]] = None,
) -> None:
    """Update job status in Firestore."""
    doc_ref = db.collection("training_jobs").document(job_id)

    update_data = {
        "status": status,
        "updated_at": firestore.SERVER_TIMESTAMP,
    }

    if status == "completed":
        update_data["completed_at"] = firestore.SERVER_TIMESTAMP

    if result:
        update_data["result"] = result

    doc_ref.update(update_data)


def _extract_field(fields: Dict, key: str, default: Any = None) -> Any:
    """Extract a field value from Firestore document fields structure."""
    if key not in fields:
        return default

    field_data = fields[key]

    # Firestore fields have type-specific keys
    if "stringValue" in field_data:
        return field_data["stringValue"]
    elif "integerValue" in field_data:
        return int(field_data["integerValue"])
    elif "doubleValue" in field_data:
        return float(field_data["doubleValue"])
    elif "booleanValue" in field_data:
        return field_data["booleanValue"]
    elif "mapValue" in field_data:
        # Recursively extract map fields
        map_fields = field_data["mapValue"].get("fields", {})
        return {k: _extract_field(map_fields, k) for k in map_fields}
    elif "arrayValue" in field_data:
        values = field_data["arrayValue"].get("values", [])
        return [_extract_value(v) for v in values]

    return default


def _extract_value(value_data: Dict) -> Any:
    """Extract a single value from Firestore value structure."""
    if "stringValue" in value_data:
        return value_data["stringValue"]
    elif "integerValue" in value_data:
        return int(value_data["integerValue"])
    elif "doubleValue" in value_data:
        return float(value_data["doubleValue"])
    elif "booleanValue" in value_data:
        return value_data["booleanValue"]
    return None


class WaveCoreTrainer:
    """
    JAX-based WaveCore model trainer for cloud execution.

    This is a simplified trainer that runs on Cloud Functions or
    Vertex AI for batch training of the ContinuonBrain model.
    """

    def __init__(
        self,
        episodes_dir: Path,
        checkpoint_dir: Path,
        config: TrainingConfig,
    ):
        self.episodes_dir = episodes_dir
        self.checkpoint_dir = checkpoint_dir
        self.config = config
        self.model = None
        self.optimizer = None
        self.state = None

    def train(self) -> Dict[str, Any]:
        """Run training loop."""
        try:
            # Import JAX (may not be available in all environments)
            import jax
            import jax.numpy as jnp
            from jax import random

            logger.info(f"JAX devices: {jax.devices()}")

            # Check for TPU/GPU
            if self.config.use_tpu:
                logger.info("Training on TPU")
            elif self.config.use_gpu and jax.devices()[0].platform == "gpu":
                logger.info("Training on GPU")
            else:
                logger.info("Training on CPU")

            # Initialize model
            self._init_model()

            # Load data
            dataset = self._load_dataset()

            if len(dataset) == 0:
                return {"success": False, "error": "No training data loaded"}

            # Training loop
            losses = []
            best_loss = float("inf")
            best_step = 0

            num_batches = max(1, len(dataset) // self.config.batch_size)
            total_steps = num_batches * self.config.epochs

            logger.info(f"Training for {self.config.epochs} epochs, {num_batches} batches per epoch")

            for epoch in range(self.config.epochs):
                epoch_losses = []

                for batch_idx in range(num_batches):
                    # Get batch
                    batch = self._get_batch(dataset, batch_idx)

                    # Training step
                    loss, self.state = self._train_step(self.state, batch)
                    epoch_losses.append(float(loss))

                    step = epoch * num_batches + batch_idx

                    # Log progress
                    if step % 100 == 0:
                        logger.info(f"Step {step}/{total_steps}, Loss: {loss:.6f}")

                    # Save best checkpoint
                    if loss < best_loss:
                        best_loss = loss
                        best_step = step
                        self._save_checkpoint(step, loss)

                avg_epoch_loss = sum(epoch_losses) / len(epoch_losses)
                losses.append(avg_epoch_loss)
                logger.info(f"Epoch {epoch + 1}/{self.config.epochs}, Avg Loss: {avg_epoch_loss:.6f}")

            # Save final checkpoint
            self._save_checkpoint(total_steps, losses[-1] if losses else 0.0, final=True)

            return {
                "success": True,
                "final_loss": losses[-1] if losses else None,
                "epochs_completed": self.config.epochs,
                "best_step": best_step,
                "best_loss": best_loss,
                "metrics": {
                    "loss_history": losses[-10:],  # Last 10 epochs
                    "total_steps": total_steps,
                },
            }

        except ImportError as e:
            logger.error(f"JAX import error: {e}")
            return {"success": False, "error": f"JAX not available: {e}"}
        except Exception as e:
            logger.error(f"Training error: {e}", exc_info=True)
            return {"success": False, "error": str(e)}

    def _init_model(self) -> None:
        """Initialize JAX model and optimizer."""
        import jax
        import jax.numpy as jnp
        from jax import random
        import optax

        # Initialize random key
        key = random.PRNGKey(42)

        # Create simple MLP model for demonstration
        # In production, this would use the actual WaveCore architecture
        def init_params(key):
            k1, k2, k3 = random.split(key, 3)
            return {
                "encoder": {
                    "w1": random.normal(k1, (self.config.obs_dim, 256)) * 0.01,
                    "b1": jnp.zeros(256),
                    "w2": random.normal(k2, (256, 128)) * 0.01,
                    "b2": jnp.zeros(128),
                },
                "decoder": {
                    "w1": random.normal(k3, (128 + self.config.action_dim, 256)) * 0.01,
                    "b1": jnp.zeros(256),
                    "w2": random.normal(key, (256, self.config.output_dim)) * 0.01,
                    "b2": jnp.zeros(self.config.output_dim),
                },
            }

        # Initialize parameters
        params = init_params(key)

        # Create optimizer
        if self.config.mixed_precision:
            # Use gradient scaling for mixed precision
            optimizer = optax.chain(
                optax.clip_by_global_norm(1.0),
                optax.adam(self.config.learning_rate),
            )
        else:
            optimizer = optax.adam(self.config.learning_rate)

        opt_state = optimizer.init(params)

        self.state = {
            "params": params,
            "opt_state": opt_state,
            "step": 0,
        }
        self.optimizer = optimizer

    def _load_dataset(self) -> List[Dict[str, Any]]:
        """Load episode data from JSON files."""
        episodes = []

        for ep_file in self.episodes_dir.glob("**/*.json"):
            try:
                with open(ep_file, "r") as f:
                    data = json.load(f)

                # Handle different episode formats
                if isinstance(data, list):
                    episodes.extend(data)
                elif isinstance(data, dict):
                    if "steps" in data:
                        episodes.extend(data["steps"])
                    else:
                        episodes.append(data)

            except Exception as e:
                logger.warning(f"Failed to load {ep_file}: {e}")

        logger.info(f"Loaded {len(episodes)} training examples")
        return episodes

    def _get_batch(self, dataset: List[Dict], batch_idx: int) -> Dict[str, Any]:
        """Get a training batch."""
        import jax.numpy as jnp
        import numpy as np

        start_idx = batch_idx * self.config.batch_size
        end_idx = min(start_idx + self.config.batch_size, len(dataset))

        batch_data = dataset[start_idx:end_idx]

        # Extract observations and actions
        # Adapt to actual data format
        observations = []
        actions = []
        targets = []

        for item in batch_data:
            # Handle various data formats
            obs = item.get("observation") or item.get("obs") or item.get("state", [])
            act = item.get("action") or item.get("actions", [])
            tgt = item.get("target") or item.get("next_obs") or obs

            # Pad/truncate to expected dimensions
            obs = self._pad_or_truncate(obs, self.config.obs_dim)
            act = self._pad_or_truncate(act, self.config.action_dim)
            tgt = self._pad_or_truncate(tgt, self.config.output_dim)

            observations.append(obs)
            actions.append(act)
            targets.append(tgt)

        return {
            "observations": jnp.array(observations, dtype=jnp.float32),
            "actions": jnp.array(actions, dtype=jnp.float32),
            "targets": jnp.array(targets, dtype=jnp.float32),
        }

    def _pad_or_truncate(self, data: Any, target_dim: int) -> List[float]:
        """Pad or truncate data to target dimension."""
        if isinstance(data, dict):
            # Flatten dict values
            flat = []
            for v in data.values():
                if isinstance(v, (list, tuple)):
                    flat.extend(v)
                elif isinstance(v, (int, float)):
                    flat.append(v)
            data = flat

        if not isinstance(data, (list, tuple)):
            data = [float(data)] if data else []

        data = [float(x) if x is not None else 0.0 for x in data]

        if len(data) < target_dim:
            data = data + [0.0] * (target_dim - len(data))
        elif len(data) > target_dim:
            data = data[:target_dim]

        return data

    def _train_step(self, state: Dict, batch: Dict) -> tuple:
        """Execute a single training step."""
        import jax
        import jax.numpy as jnp
        from jax import grad

        params = state["params"]
        opt_state = state["opt_state"]

        def loss_fn(params):
            # Forward pass
            obs = batch["observations"]
            actions = batch["actions"]
            targets = batch["targets"]

            # Encoder
            h = jnp.dot(obs, params["encoder"]["w1"]) + params["encoder"]["b1"]
            h = jax.nn.relu(h)
            h = jnp.dot(h, params["encoder"]["w2"]) + params["encoder"]["b2"]
            h = jax.nn.relu(h)

            # Decoder (concatenate with actions)
            h_with_action = jnp.concatenate([h, actions], axis=-1)
            out = jnp.dot(h_with_action, params["decoder"]["w1"]) + params["decoder"]["b1"]
            out = jax.nn.relu(out)
            out = jnp.dot(out, params["decoder"]["w2"]) + params["decoder"]["b2"]

            # MSE loss
            loss = jnp.mean((out - targets) ** 2)

            # Sparsity regularization
            if self.config.sparsity_lambda > 0:
                l1_norm = sum(jnp.sum(jnp.abs(p)) for p in jax.tree_util.tree_leaves(params))
                loss = loss + self.config.sparsity_lambda * l1_norm

            return loss

        # Compute gradients
        loss, grads = jax.value_and_grad(loss_fn)(params)

        # Update parameters
        updates, new_opt_state = self.optimizer.update(grads, opt_state, params)
        import optax
        new_params = optax.apply_updates(params, updates)

        new_state = {
            "params": new_params,
            "opt_state": new_opt_state,
            "step": state["step"] + 1,
        }

        return loss, new_state

    def _save_checkpoint(self, step: int, loss: float, final: bool = False) -> None:
        """Save model checkpoint."""
        import pickle

        ckpt_name = "final" if final else f"step_{step}"
        ckpt_path = self.checkpoint_dir / f"checkpoint_{ckpt_name}.pkl"

        checkpoint = {
            "step": step,
            "loss": loss,
            "params": self.state["params"],
            "config": {
                "obs_dim": self.config.obs_dim,
                "action_dim": self.config.action_dim,
                "output_dim": self.config.output_dim,
                "arch_preset": self.config.arch_preset,
            },
        }

        with open(ckpt_path, "wb") as f:
            pickle.dump(checkpoint, f)

        logger.info(f"Saved checkpoint: {ckpt_path}")

    def export(self, output_dir: Path) -> Path:
        """Export trained model for inference."""
        import json
        import pickle

        # Copy best checkpoint
        best_ckpt = None
        for ckpt_file in sorted(self.checkpoint_dir.glob("checkpoint_*.pkl")):
            best_ckpt = ckpt_file

        if best_ckpt:
            shutil.copy(best_ckpt, output_dir / "model.pkl")

        # Write model manifest
        manifest = {
            "model_type": "jax_wavecore",
            "checkpoint_path": "model.pkl",
            "config": {
                "obs_dim": self.config.obs_dim,
                "action_dim": self.config.action_dim,
                "output_dim": self.config.output_dim,
                "arch_preset": self.config.arch_preset,
            },
            "format": "pickle",
            "exported_at": datetime.utcnow().isoformat(),
        }

        manifest_path = output_dir / "model_manifest.json"
        manifest_path.write_text(json.dumps(manifest, indent=2))

        logger.info(f"Model exported to: {output_dir}")
        return output_dir


# For local testing
if __name__ == "__main__":
    # Simulate a training trigger event
    test_event = {
        "value": {
            "fields": {
                "job_id": {"stringValue": "test_job_001"},
                "bucket": {"stringValue": "continuon-training-data"},
                "name": {"stringValue": "episodes/test.zip"},
                "config": {
                    "mapValue": {
                        "fields": {
                            "epochs": {"integerValue": "10"},
                            "batch_size": {"integerValue": "8"},
                        }
                    }
                },
            }
        }
    }

    print("Testing Cloud Function locally...")
    try:
        train_model(test_event, None)
    except Exception as e:
        print(f"Test error (expected without GCS access): {e}")
