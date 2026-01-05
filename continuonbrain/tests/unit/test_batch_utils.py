"""Tests for JAX batch handling utilities."""
import pytest
import numpy as np

# Use numpy for testing since JAX may not be available
try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    jnp = np
    HAS_JAX = False

from continuonbrain.jax_models.batch_utils import (
    BatchSpec,
    ensure_batch_dim,
    ensure_batch_dim_reward,
    remove_batch_dim,
    remove_batch_dim_from_list,
    remove_batch_dim_from_dict,
    normalize_inputs,
    denormalize_outputs,
    validate_batch_consistency,
)


class TestBatchSpec:
    def test_auto_batched_ndim(self):
        spec = BatchSpec("test", unbatched_ndim=1)
        assert spec.batched_ndim == 2

    def test_explicit_batched_ndim(self):
        spec = BatchSpec("test", unbatched_ndim=2, batched_ndim=3)
        assert spec.batched_ndim == 3


class TestEnsureBatchDim:
    def test_adds_batch_to_1d(self):
        spec = BatchSpec("test", unbatched_ndim=1)
        tensor = jnp.zeros((10,))
        result, was_unbatched = ensure_batch_dim(tensor, spec)

        assert result.shape == (1, 10)
        assert was_unbatched is True

    def test_preserves_batched_2d(self):
        spec = BatchSpec("test", unbatched_ndim=1)
        tensor = jnp.zeros((4, 10))
        result, was_unbatched = ensure_batch_dim(tensor, spec)

        assert result.shape == (4, 10)
        assert was_unbatched is False

    def test_adds_batch_to_2d(self):
        spec = BatchSpec("test", unbatched_ndim=2, batched_ndim=3)
        tensor = jnp.zeros((5, 10))
        result, was_unbatched = ensure_batch_dim(tensor, spec)

        assert result.shape == (1, 5, 10)
        assert was_unbatched is True


class TestEnsureBatchDimReward:
    def test_scalar_reward(self):
        r = jnp.array(0.5)
        result, was_unbatched = ensure_batch_dim_reward(r)

        assert result.shape == (1, 1)
        assert was_unbatched is True

    def test_1d_reward(self):
        r = jnp.array([0.5])
        result, was_unbatched = ensure_batch_dim_reward(r)

        assert result.shape == (1, 1)
        assert was_unbatched is True

    def test_batched_1d_reward(self):
        r = jnp.array([0.5, 0.6, 0.7])
        result, was_unbatched = ensure_batch_dim_reward(r)

        assert result.shape == (3, 1)
        assert was_unbatched is False

    def test_already_2d_reward(self):
        r = jnp.array([[0.5], [0.6]])
        result, was_unbatched = ensure_batch_dim_reward(r)

        assert result.shape == (2, 1)
        assert was_unbatched is False


class TestRemoveBatchDim:
    def test_removes_when_unbatched(self):
        tensor = jnp.zeros((1, 10))
        result = remove_batch_dim(tensor, was_unbatched=True)

        assert result.shape == (10,)

    def test_preserves_when_batched(self):
        tensor = jnp.zeros((4, 10))
        result = remove_batch_dim(tensor, was_unbatched=False)

        assert result.shape == (4, 10)


class TestRemoveBatchDimFromList:
    def test_removes_from_all_tensors(self):
        tensors = [jnp.zeros((1, 5, 10)), jnp.zeros((1, 3, 8))]
        result = remove_batch_dim_from_list(tensors, was_unbatched=True)

        assert len(result) == 2
        assert result[0].shape == (5, 10)
        assert result[1].shape == (3, 8)

    def test_preserves_when_not_unbatched(self):
        tensors = [jnp.zeros((4, 5, 10))]
        result = remove_batch_dim_from_list(tensors, was_unbatched=False)

        assert result[0].shape == (4, 5, 10)


class TestRemoveBatchDimFromDict:
    def test_removes_from_tensor_values(self):
        info = {
            "query": jnp.zeros((1, 10)),
            "context": jnp.zeros((1, 20)),
            "scalar": 42,
        }
        result = remove_batch_dim_from_dict(info, was_unbatched=True)

        assert result["query"].shape == (10,)
        assert result["context"].shape == (20,)
        assert result["scalar"] == 42

    def test_handles_list_keys(self):
        info = {
            "memories": [jnp.zeros((1, 5, 10)), jnp.zeros((1, 3, 8))],
        }
        result = remove_batch_dim_from_dict(
            info, was_unbatched=True, list_keys=["memories"]
        )

        assert result["memories"][0].shape == (5, 10)
        assert result["memories"][1].shape == (3, 8)


class TestValidateBatchConsistency:
    def test_consistent_batches(self):
        tensors = {
            "a": jnp.zeros((4, 10)),
            "b": jnp.zeros((4, 20)),
        }
        batch_size = validate_batch_consistency(tensors)
        assert batch_size == 4

    def test_inconsistent_batches_raises(self):
        tensors = {
            "a": jnp.zeros((4, 10)),
            "b": jnp.zeros((3, 20)),
        }
        with pytest.raises(ValueError, match="Inconsistent"):
            validate_batch_consistency(tensors)

    def test_expected_batch_size(self):
        tensors = {"a": jnp.zeros((4, 10))}
        batch_size = validate_batch_consistency(tensors, expected_batch_size=4)
        assert batch_size == 4

    def test_wrong_expected_batch_size_raises(self):
        tensors = {"a": jnp.zeros((4, 10))}
        with pytest.raises(ValueError, match="Expected batch size"):
            validate_batch_consistency(tensors, expected_batch_size=8)
