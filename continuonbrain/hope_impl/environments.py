"""
Environment Adapters for HOPE Training

Provides standardized interfaces for different training environments.
"""

import numpy as np
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class Environment(ABC):
    """Base class for training environments."""
    
    @abstractmethod
    def reset(self) -> np.ndarray:
        """Reset environment and return initial observation."""
        pass
    
    @abstractmethod
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Execute action in environment.
        
        Returns:
            observation: Next observation
            reward: Scalar reward
            done: Whether episode is complete
        """
        pass
    
    @property
    @abstractmethod
    def obs_dim(self) -> int:
        """Observation dimension."""
        pass
    
    @property
    @abstractmethod
    def action_dim(self) -> int:
        """Action dimension."""
        pass


class RandomEnvironment(Environment):
    """
    Random environment for testing.
    Generates random observations and rewards.
    """
    
    def __init__(self, obs_dim: int = 10, action_dim: int = 4, max_steps: int = 100):
        self._obs_dim = obs_dim
        self._action_dim = action_dim
        self.max_steps = max_steps
        self.current_step = 0
    
    def reset(self) -> np.ndarray:
        """Reset to random initial state."""
        self.current_step = 0
        return np.random.randn(self._obs_dim).astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """Take random step."""
        self.current_step += 1
        obs = np.random.randn(self._obs_dim).astype(np.float32)
        reward = np.random.randn().item()
        done = self.current_step >= self.max_steps
        return obs, reward, done
    
    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    @property
    def action_dim(self) -> int:
        return self._action_dim


class CartPoleEnvironment(Environment):
    """
    CartPole-v1 environment from OpenAI Gym.
    Requires: pip install gymnasium
    """
    
    def __init__(self):
        try:
            import gymnasium as gym
            self.env = gym.make('CartPole-v1')
            self._obs_dim = 4
            self._action_dim = 2  # Discrete actions encoded as 2D
        except ImportError:
            raise ImportError("CartPole requires gymnasium: pip install gymnasium")
    
    def reset(self) -> np.ndarray:
        """Reset environment."""
        obs, _ = self.env.reset()
        return obs.astype(np.float32)
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        """
        Take step with continuous action.
        Convert to discrete: action[0] > 0 -> right, else left
        """
        discrete_action = 1 if action[0] > 0 else 0
        obs, reward, terminated, truncated, _ = self.env.step(discrete_action)
        done = terminated or truncated
        return obs.astype(np.float32), float(reward), done
    
    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    @property
    def action_dim(self) -> int:
        return self._action_dim
    
    def close(self):
        """Close environment."""
        self.env.close()


class CustomEnvironment(Environment):
    """
    Custom environment wrapper.
    Users can subclass this or pass a compatible object.
    """
    
    def __init__(self, env_object):
        """
        Wrap a custom environment object.
        
        The object must have:
        - reset() -> observation
        - step(action) -> (observation, reward, done)
        - obs_dim property
        - action_dim property
        """
        self.env = env_object
        self._obs_dim = env_object.obs_dim
        self._action_dim = env_object.action_dim
    
    def reset(self) -> np.ndarray:
        return self.env.reset()
    
    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool]:
        return self.env.step(action)
    
    @property
    def obs_dim(self) -> int:
        return self._obs_dim
    
    @property
    def action_dim(self) -> int:
        return self._action_dim


def create_environment(env_type: str, **kwargs) -> Environment:
    """
    Factory function to create environments.
    
    Args:
        env_type: One of 'random', 'cartpole', 'custom'
        **kwargs: Environment-specific arguments
    
    Returns:
        Environment instance
    """
    if env_type == 'random':
        obs_dim = kwargs.get('obs_dim', 10)
        action_dim = kwargs.get('action_dim', 4)
        max_steps = kwargs.get('max_steps', 100)
        return RandomEnvironment(obs_dim, action_dim, max_steps)
    
    elif env_type == 'cartpole':
        return CartPoleEnvironment()
    
    elif env_type == 'custom':
        env_object = kwargs.get('env_object')
        if env_object is None:
            raise ValueError("Custom environment requires 'env_object' kwarg")
        return CustomEnvironment(env_object)
    
    else:
        raise ValueError(f"Unknown environment type: {env_type}")
