
import numpy as np
import math
from typing import Tuple, Optional, Dict

class LorenzAttractorEnv:
    """
    Chaotic but deterministic environment (Lorenz Attractor).
    Tests the brain's ability to model complex non-linear dynamics.
    
    System:
    dx/dt = sigma * (y - x)
    dy/dt = x * (rho - z) - y
    dz/dt = x * y - beta * z
    """
    
    def __init__(
        self,
        obs_dim: int = 10,
        sigma: float = 10.0,
        rho: float = 28.0,
        beta: float = 8.0/3.0,
        dt: float = 0.01,
        noise_level: float = 0.0
    ):
        self.obs_dim = obs_dim
        self.sigma = sigma
        self.rho = rho
        self.beta = beta
        self.dt = dt
        self.noise_level = noise_level
        
        self.state = np.array([1.0, 1.0, 1.0]) # Initial state
        self.step_count = 0
        self.last_prediction_error = 0.0
        
    def reset(self) -> np.ndarray:
        self.state = np.array([1.0, 1.0, 1.0])
        self.step_count = 0
        return self._get_obs()
        
    def _step_lorenz(self):
        x, y, z = self.state
        dx = self.sigma * (y - x)
        dy = x * (self.rho - z) - y
        dz = x * y - self.beta * z
        
        self.state += np.array([dx, dy, dz]) * self.dt
        
    def _get_obs(self) -> np.ndarray:
        # Map 3D Lorenz state to obs_dim
        # We'll just repeat it or pad it
        obs = np.zeros(self.obs_dim, dtype=np.float32)
        obs[:3] = self.state
        
        # Add some complex interactions for other dims if obs_dim > 3
        if self.obs_dim > 3:
            obs[3] = self.state[0] * self.state[1] * 0.1
            obs[4] = math.sin(self.step_count * 0.1)
        
        # Normalize to keep reasonable for NN inputs [-1, 1] range approx
        obs = obs / 20.0 
        
        if self.noise_level > 0:
            obs += np.random.randn(self.obs_dim).astype(np.float32) * self.noise_level
            
        return obs.astype(np.float32)
        
    def step(
        self, 
        action: np.ndarray, 
        prediction: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool]:
        
        self.step_count += 1
        self._step_lorenz()
        
        next_obs = self._get_obs()
        
        reward = 0.0
        if prediction is not None:
            if prediction.shape != next_obs.shape:
                 prediction = prediction[:self.obs_dim] if len(prediction) > self.obs_dim else np.pad(prediction, (0, self.obs_dim-len(prediction)))
            
            error = np.linalg.norm(next_obs - prediction)
            self.last_prediction_error = error
            reward = error 
            
        return next_obs, reward, False

    def get_statistics(self) -> Dict:
        return {
            'step': self.step_count,
            'prediction_error': self.last_prediction_error
        }
