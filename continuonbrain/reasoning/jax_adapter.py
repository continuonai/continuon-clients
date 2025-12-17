
from typing import Dict, Any, Optional
import jax.numpy as jnp
from continuonbrain.jax_models.core_model import CoreModel, CoreModelConfig
from continuonbrain.mamba_brain import BaseWorldModel, WorldModelState, WorldModelAction, WorldModelPredictResult

class JaxWorldModelAdapter(BaseWorldModel):
    """Adapts a JAX/Flax CoreModel to the BaseWorldModel interface for symbolic search."""

    def __init__(self, core_model: CoreModel, params: Dict[str, Any], config: CoreModelConfig):
        self.model = core_model
        self.params = params
        self.config = config
        # Cache dummy/initial states for single-step prediction
        self.dummy_metrics = {"uncertainty": 0.0}

    def predict(self, state: WorldModelState, action: WorldModelAction) -> WorldModelPredictResult:
        # 1. Convert WorldModelState/Action to JAX arrays
        # Assuming joint_pos maps to s_prev/obs in some way. 
        # For v0 seed model, we might just map joint_pos to Observation.
        
        # NOTE: This is a simplification. Real JAX model has complex state (s, w, p).
        # We will initialize them to zeros or carry them if we were doing rollouts (TODO: support stateful rollout).
        # For "single step" API, we assume a fresh start or use stored context if we extended this class.
        
        # Obs: [d_obs] from joint_pos
        # We need to map 7 joints to 128 dim (pad)
        obs = jnp.zeros((self.config.obs_dim,))
        joints = jnp.array(state.joint_pos)
        obs = obs.at[:len(joints)].set(joints)
        
        # Action: [d_action] from joint_delta
        act = jnp.zeros((self.config.action_dim,))
        deltas = jnp.array(action.joint_delta)
        act = act.at[:len(deltas)].set(deltas)
        
        # Reward: 0
        reward = jnp.array([0.0])
        
        # States: Zeros (Cold start for single step prediction - limitation of current Adapter)
        s_prev = jnp.zeros((self.config.d_s,))
        w_prev = jnp.zeros((self.config.d_w,))
        p_prev = jnp.zeros((self.config.d_p,))
        
        # CMS: Zeros
        cms_memories = [
            jnp.zeros((size, dim)) for size, dim in zip(self.config.cms_sizes, self.config.cms_dims)
        ]
        cms_keys = [
            jnp.zeros((size, self.config.d_k)) for size in self.config.cms_sizes
        ]

        # 2. Run JAX inference
        # We perform a single call to __call__
        # model.apply(params, ...)
        
        # Add batch dim [1, ...]
        y_t, info = self.model.apply(
            self.params,
            obs[None, :],
            act[None, :],
            reward[None, :],
            s_prev[None, :],
            w_prev[None, :],
            p_prev[None, :],
            [m[None, ...] for m in cms_memories], # This might be wrong dimension handling for list, check CoreModel
            # CoreModel expects list of arrays. The arrays inside might need batch dim if designed for it.
            # Looking at CoreModel code: "if s_prev.ndim == 1". It handles unbatched.
            # Let's pass unbatched for simplicity if supported.
            # CoreModel.__call__ signature supports unbatched? 
            # "if x_obs.ndim == 1: x_obs = x_obs[None, :]" -> It auto-batches inside!
            # So we pass 1D arrays.
            [k[None, ...] for k in cms_keys],  # Actually CoreModel expects lists of arrays.
        )
        
        # Wait, CoreModel auto-batches inputs inside `InputEncoder` but `CoreModel.__call__` arguments 
        # `cms_memories` are lists.
        # Let's retry passing Unbatched arrays and let CoreModel handle it/
        
        # Re-reading CoreModel:
        # def __call__(..., cms_memories: List[jnp.ndarray], ...)
        # Inside `CMSRead`: `if q_t.ndim == 1`...
        # So yes, it supports unbatched inputs.
        
        y_t, info = self.model.apply(
            self.params,
            obs,
            act,
            reward,
            s_prev,
            w_prev,
            p_prev,
            cms_memories,
            cms_keys
        )
        
        # 3. Decode output to Next State
        # y_t is [output_dim] (32).
        # We interpret prediction as "Next Joint Delta" or "Next Joint Pos".
        # For this prototype, let's assume y_t[:7] is delta to add to current state.
        
        pred_delta = y_t[:len(state.joint_pos)]
        next_joints = [j + d for j, d in zip(state.joint_pos, pred_delta)]
        
        # 4. Uncertainty
        # In v0 logic, we might use p_t magnitude or similar.
        # info['particle_state'] is available.
        p_state = info['particle_state']
        uncertainty = float(jnp.mean(jnp.abs(p_state))) # Heuristic

        return WorldModelPredictResult(
            next_state=WorldModelState(joint_pos=[float(x) for x in next_joints]),
            uncertainty=uncertainty,
            debug={"backend": "jax_core_v0"}
        )
