# Plan 1: Fix Checkpoint Loading

## Overview
Complete the `SeedModel.load()` method to properly map loaded weights to JAX parameter structure.

## Current State
- `SeedModel.save()` works - saves weights as NPZ + manifest.json
- `SeedModel.load()` has TODO - loads NPZ but doesn't map to params tree

## Implementation

### Step 1: Update save() to preserve tree structure
```python
# seed/model.py - save()
def save(self, path: Path) -> None:
    # Flatten params with path keys
    flat_params = jax.tree_util.tree_map_with_path(
        lambda path, x: (self._path_to_key(path), np.array(x)),
        self._params
    )
    # Save with structure metadata
```

### Step 2: Implement load() weight mapping
```python
def load(self, path: Path) -> None:
    # Load manifest for config
    manifest = json.load(open(path / "manifest.json"))

    # Initialize model with same config
    self._initialize()

    # Load weights
    loaded = np.load(path / "seed_model.npz")

    # Rebuild tree structure
    self._params = self._rebuild_params_tree(loaded, self._params)
```

### Step 3: Add helper methods
- `_path_to_key(path)` - Convert JAX path to string key
- `_key_to_path(key)` - Convert string key back to path
- `_rebuild_params_tree(loaded, template)` - Map flat dict to tree

### Step 4: Add verification
- Verify param count matches
- Verify shapes match
- Log any mismatches

## Files to Modify
| File | Changes |
|------|---------|
| `seed/model.py` | Update save(), complete load(), add helpers |

## Testing
```python
seed = SeedModel(target='pi5')
seed.save('/tmp/test_ckpt')
seed2 = SeedModel(checkpoint_path='/tmp/test_ckpt')
assert seed.param_count == seed2.param_count
```

## Success Criteria
- Checkpoints round-trip correctly
- Param count preserved
- Model produces same output after load
