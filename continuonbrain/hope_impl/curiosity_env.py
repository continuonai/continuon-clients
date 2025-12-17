"""
Curiosity-Driven Environment for Autonomous Learning

Generates novel experiences and rewards exploration of unfamiliar states.
"""

import numpy as np
import torch
from typing import Tuple, Optional, List
from collections import deque


class CuriosityEnvironment:
    """
    Self-directed exploration environment with intrinsic motivation.
    
    Rewards the brain for:
    - Exploring novel states
    - Making surprising predictions
    - Discovering new patterns
    """
    
    def __init__(
        self,
        obs_dim: int = 10,
        action_dim: int = 4,
        exploration_bonus: float = 0.1,
        novelty_threshold: float = 0.5,
        memory_size: int = 1000,
    ):
        """
        Initialize curiosity environment.
        
        Args:
            obs_dim: Observation dimension
            action_dim: Action dimension
            exploration_bonus: Weight for intrinsic reward
            novelty_threshold: Threshold for "novel" states
            memory_size: Number of past states to remember
        """
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.exploration_bonus = exploration_bonus
        self.novelty_threshold = novelty_threshold
        
        # Memory of seen states for novelty detection
        self.state_memory = deque(maxlen=memory_size)
        
        # Current state
        self.current_obs = None
        self.step_count = 0
        
        # Statistics
        self.total_novelty = 0.0
        self.total_surprise = 0.0
        self.novel_state_count = 0
    
    def reset(self) -> np.ndarray:
        """Reset to random initial state."""
        self.current_obs = np.random.randn(self.obs_dim).astype(np.float32)
        self.step_count = 0
        return self.current_obs
    
    def step(
        self,
        action: np.ndarray,
        prediction: Optional[np.ndarray] = None
    ) -> Tuple[np.ndarray, float, bool]:
        """
        Take step in curiosity environment.
        
        Args:
            action: Action taken by brain
            prediction: Brain's prediction of next state (for surprise)
        
        Returns:
            observation: Next observation
            reward: Intrinsic reward (novelty + surprise)
            done: Always False (continuous learning)
        """
        self.step_count += 1
        
        # Generate next observation influenced by action
        # Mix of random exploration and action-dependent dynamics
        noise = np.random.randn(self.obs_dim) * 0.5
        action_effect = action[:self.obs_dim] if len(action) >= self.obs_dim else np.pad(action, (0, self.obs_dim - len(action)))
        next_obs = (0.7 * noise + 0.3 * action_effect).astype(np.float32)
        
        # Compute novelty (how different from seen states)
        novelty = self._compute_novelty(next_obs)
        
        # Compute surprise (prediction error)
        surprise = 0.0
        if prediction is not None:
            surprise = self._compute_surprise(next_obs, prediction)
        
        # Intrinsic reward
        reward = self.exploration_bonus * (novelty + surprise)
        
        # Update statistics
        self.total_novelty += novelty
        self.total_surprise += surprise
        if novelty > self.novelty_threshold:
            self.novel_state_count += 1
        
        # Add to memory
        self.state_memory.append(next_obs.copy())
        
        # Update current state
        self.current_obs = next_obs
        
        # Never done (continuous learning)
        done = False
        
        return next_obs, reward, done
    
    def _compute_novelty(self, state: np.ndarray) -> float:
        """
        Compute novelty of state.
        
        Novelty = minimum distance to any seen state
        High novelty = far from all seen states
        """
        if len(self.state_memory) == 0:
            return 1.0  # First state is maximally novel
        
        # Compute distances to all remembered states
        distances = [np.linalg.norm(state - s) for s in self.state_memory]
        min_distance = min(distances)
        
        # Normalize to [0, 1]
        # Use sigmoid to bound
        novelty = 1.0 / (1.0 + np.exp(-min_distance + 2.0))
        
        return novelty
    
    def _compute_surprise(self, actual: np.ndarray, predicted: np.ndarray) -> float:
        """
        Compute surprise (prediction error).
        
        Surprise = ||actual - predicted||
        High surprise = bad prediction
        """
        # Handle dimension mismatch by padding or truncating
        if len(predicted) < len(actual):
            # Pad prediction with zeros
            predicted = np.pad(predicted, (0, len(actual) - len(predicted)))
        elif len(predicted) > len(actual):
            # Truncate prediction
            predicted = predicted[:len(actual)]
        
        error = np.linalg.norm(actual - predicted)
        
        # Normalize to [0, 1]
        surprise = 1.0 / (1.0 + np.exp(-error + 2.0))
        
        return surprise
    
    def get_statistics(self) -> dict:
        """Get curiosity statistics."""
        return {
            'total_steps': self.step_count,
            'novel_states': self.novel_state_count,
            'avg_novelty': self.total_novelty / max(self.step_count, 1),
            'avg_surprise': self.total_surprise / max(self.step_count, 1),
            'memory_size': len(self.state_memory),
            'novelty_rate': self.novel_state_count / max(self.step_count, 1),
        }
    
    def reset_statistics(self):
        """Reset statistics counters."""
        self.total_novelty = 0.0
        self.total_surprise = 0.0
        self.novel_state_count = 0
        self.step_count = 0
    
    @property
    def current_novelty(self) -> float:
        """Get novelty of current state."""
        if self.current_obs is None:
            return 0.0
        return self._compute_novelty(self.current_obs)


class AdaptiveCuriosityEnvironment(CuriosityEnvironment):
    """
    Curiosity environment that adapts difficulty based on performance.
    """
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.difficulty = 0.5  # 0 = easy, 1 = hard
        self.performance_history = deque(maxlen=100)
    
    def step(self, action: np.ndarray, prediction: Optional[np.ndarray] = None) -> Tuple[np.ndarray, float, bool]:
        """Step with adaptive difficulty."""
        obs, reward, done = super().step(action, prediction)
        
        # Track performance
        self.performance_history.append(reward)
        
        # Adapt difficulty
        if len(self.performance_history) >= 50:
            avg_reward = np.mean(self.performance_history)
            
            # If doing well, increase difficulty
            if avg_reward > 0.5:
                self.difficulty = min(1.0, self.difficulty + 0.01)
            # If struggling, decrease difficulty
            elif avg_reward < 0.2:
                self.difficulty = max(0.0, self.difficulty - 0.01)
        
        return obs, reward, done
    
    def reset(self) -> np.ndarray:
        """Reset with difficulty-adjusted initial state."""
        obs = super().reset()
        # Scale initial state by difficulty
        obs = obs * (0.5 + 0.5 * self.difficulty)
        self.current_obs = obs
        return obs
