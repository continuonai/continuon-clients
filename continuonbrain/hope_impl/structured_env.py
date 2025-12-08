
import numpy as np
import math
from typing import Tuple, Optional, Dict

class StructuredTestEnv:
    """
    A predictable environment for verifying learning capability.
    
    Generates a sine wave pattern on the first dimension.
    Brain should learn to predict obs[t+1] from obs[t].
    """
    
    def __init__(
        self,
        obs_dim: int = 10,
        period: int = 20,
        noise_level: float = 0.05
    ):
        """
        Args:
            obs_dim: Observation dimension
            period: Steps per sine wave cycle
            noise_level: Amount of random noise to add
        """
        self.obs_dim = obs_dim
        self.period = period
        self.noise_level = noise_level
        self.step_count = 0
        self.current_obs = None
        
        # Track prediction error
        self.last_prediction_error = 0.0
        
    def reset(self) -> np.ndarray:
        self.step_count = 0
        self.current_obs = self._generate_obs(0)
        return self.current_obs
        
    def _generate_obs(self, t: int) -> np.ndarray:
        """Generate observation at time t."""
        # Dim 0: Sine wave
        val = math.sin(2 * math.pi * t / self.period)
        
        # Dim 1: Cosine wave (different phase)
        val2 = math.cos(2 * math.pi * t / self.period)
        
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[0] = val
        obs[1] = val2
        
        # Add noise
        noise = np.random.randn(self.obs_dim) * self.noise_level
        obs = obs + noise
        
        return obs.astype(np.float32)
        
    def step(
        self, 
        action: np.ndarray, 
        prediction: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool]:
        
        self.step_count += 1
        
        # Generate Ground Truth next state (without noise for error calc?)
        # Actually in RL we predict the noisy observation usually, or the underlying state.
        # We'll stick to predicting the next observation we see.
        next_obs = self._generate_obs(self.step_count)
        
        # Calculate Prediction Error (Surprise)
        reward = 0.0
        if prediction is not None:
            # Handle shape mismatch
            if prediction.shape != next_obs.shape:
                 prediction = prediction[:self.obs_dim] if len(prediction) > self.obs_dim else np.pad(prediction, (0, self.obs_dim-len(prediction)))
            
            error = np.linalg.norm(next_obs - prediction)
            self.last_prediction_error = error
            
            # Intrinsic reward: In curiosity we reward HIGH error.
            # But to MEASURE improvement, we want to see LOW error over time.
            # We return dummy reward here just to keep interface compatible.
            # The Brain's internal reward is typically exploration_bonus * error.
            # But the Brain's LEARNING minimizes prediction error.
            reward = error 
            
        self.current_obs = next_obs
        done = False
        
        return next_obs, reward, done

    def get_statistics(self) -> Dict:
        return {
            'step': self.step_count,
            'prediction_error': self.last_prediction_error
        }
