
"""
Optimized Seed Model Inference

Usage:
    from continuonbrain.seed.optimized_inference import get_inference_fn
    
    infer = get_inference_fn()
    output, state = infer(observation)
"""
import jax
import jax.numpy as jnp
import numpy as np
import pickle
from pathlib import Path

from continuonbrain.jax_models.config import CoreModelConfig
from continuonbrain.jax_models.core_model import CoreModel

_model = None
_params = None
_inference_fn = None
_state = None

def _load_model():
    global _model, _params, _inference_fn, _state
    
    if _model is not None:
        return
    
    STABLE_DIR = Path("/opt/continuonos/brain/model/seed_stable")
    
    with open(STABLE_DIR / "seed_model.pkl", 'rb') as f:
        data = pickle.load(f)
    
    _params = data['params']['params']
    obs_dim = data['metadata']['obs_dim']
    action_dim = data['metadata']['action_dim']
    output_dim = data['metadata']['output_dim']
    
    config = CoreModelConfig.pi5_optimized()
    _model = CoreModel(config=config, obs_dim=obs_dim, action_dim=action_dim, output_dim=output_dim)
    
    # Initialize state
    B = 1
    _state = {
        's': jnp.zeros((B, config.d_s)),
        'w': jnp.zeros((B, config.d_w)),
        'p': jnp.zeros((B, config.d_p)),
        'cms_memories': [jnp.zeros((B, sz, dim)) for sz, dim in zip(config.cms_sizes, config.cms_dims)],
        'cms_keys': [jnp.zeros((B, sz, config.d_k)) for sz in config.cms_sizes],
        'action_dim': action_dim,
    }
    
    @jax.jit
    def _fast_inference(params, obs, action, reward, s, w, p, cms_memories, cms_keys):
        output, info = _model.apply({'params': params}, x_obs=obs, a_prev=action, r_t=reward,
            s_prev=s, w_prev=w, p_prev=p, cms_memories=cms_memories, cms_keys=cms_keys)
        return output, info['fast_state'], info['wave_state'], info['particle_state'], info['cms_memories'], info['cms_keys']
    
    _inference_fn = _fast_inference

def get_inference_fn():
    """Get JIT-compiled inference function."""
    _load_model()
    
    def infer(observation: np.ndarray, reward: float = 0.0):
        global _state
        
        obs = jnp.array(observation.reshape(1, -1).astype(np.float32))
        action = jnp.zeros((1, _state['action_dim']))
        r = jnp.array([[reward]])
        
        output, s, w, p, cms_memories, cms_keys = _inference_fn(
            _params, obs, action, r, _state['s'], _state['w'], _state['p'],
            _state['cms_memories'], _state['cms_keys']
        )
        
        _state['s'] = s
        _state['w'] = w
        _state['p'] = p
        _state['cms_memories'] = cms_memories
        _state['cms_keys'] = cms_keys
        
        return np.array(output[0]), _state
    
    return infer

def reset_state():
    """Reset internal state."""
    global _state
    _load_model()
    config = CoreModelConfig.pi5_optimized()
    B = 1
    _state['s'] = jnp.zeros((B, config.d_s))
    _state['w'] = jnp.zeros((B, config.d_w))
    _state['p'] = jnp.zeros((B, config.d_p))
    _state['cms_memories'] = [jnp.zeros((B, sz, dim)) for sz, dim in zip(config.cms_sizes, config.cms_dims)]
    _state['cms_keys'] = [jnp.zeros((B, sz, config.d_k)) for sz in config.cms_sizes]
