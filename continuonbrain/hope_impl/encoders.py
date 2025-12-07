"""
HOPE Input Encoders

Neural network modules for encoding observations, actions, and rewards
into the unified feature representation e_t.

Math:
    e_t = E_φ(x_t^{obs}, a_{t-1}, r_t)
"""

import torch
import torch.nn as nn
from typing import Optional, Tuple


class InputEncoder(nn.Module):
    """
    Main input encoder combining observation, action, and reward.
    
    Encodes:
        x_obs: Observation (can be vector, image, etc.)
        a_prev: Previous action
        r_t: Scalar reward
    
    Output:
        e_t ∈ ℝ^{d_e}: Unified encoded feature
    
    Math:
        e_t = E_φ(x_obs, a_prev, r_t)
    """
    
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        d_e: int,
        obs_type: str = "vector",  # "vector" or "image"
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            obs_dim: Observation dimension (for vector) or channels (for image)
            action_dim: Action dimension
            d_e: Output encoding dimension
            obs_type: "vector" for state-based, "image" for visual observations
            hidden_dim: Hidden dimension for MLPs (default: 2 * d_e)
        """
        super().__init__()
        
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.d_e = d_e
        self.obs_type = obs_type
        self.hidden_dim = hidden_dim or (2 * d_e)
        
        # Observation encoder
        if obs_type == "vector":
            self.obs_encoder = nn.Sequential(
                nn.Linear(obs_dim, self.hidden_dim),
                nn.LayerNorm(self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, d_e),
            )
        elif obs_type == "image":
            # Simple CNN for image observations
            # Assumes input shape: [C, H, W]
            self.obs_encoder = nn.Sequential(
                nn.Conv2d(obs_dim, 32, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
                nn.Flatten(),
                nn.Linear(64, d_e),
            )
        else:
            raise ValueError(f"Unknown obs_type: {obs_type}")
        
        # Action encoder
        self.action_encoder = nn.Sequential(
            nn.Linear(action_dim, d_e // 2),
            nn.ReLU(),
        )
        
        # Reward encoder
        self.reward_encoder = nn.Sequential(
            nn.Linear(1, d_e // 4),
            nn.ReLU(),
        )
        
        # Fusion layer
        fusion_input_dim = d_e + (d_e // 2) + (d_e // 4)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input_dim, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, d_e),
            nn.LayerNorm(d_e),
        )
    
    def forward(
        self,
        x_obs: torch.Tensor,
        a_prev: torch.Tensor,
        r_t: torch.Tensor,
    ) -> torch.Tensor:
        """
        Encode inputs into unified feature.
        
        Args:
            x_obs: Observation [obs_dim] or [C, H, W]
            a_prev: Previous action [action_dim]
            r_t: Reward scalar [1] or []
        
        Returns:
            e_t: Encoded feature [d_e]
        """
        # Ensure reward is 1D
        if r_t.dim() == 0:
            r_t = r_t.unsqueeze(0)
        
        # Encode each component
        obs_feat = self.obs_encoder(x_obs)  # [d_e]
        action_feat = self.action_encoder(a_prev)  # [d_e // 2]
        if r_t.dim() == 1 and x_obs.dim() == 2:
            r_t = r_t.unsqueeze(-1)
        elif r_t.dim() == 0 and x_obs.dim() == 2:
             r_t = r_t.unsqueeze(0).unsqueeze(-1)
             
        reward_feat = r_t # Direct usage if simple, or linear projection?
        # Let's see what was there before
        reward_feat = self.reward_encoder(r_t)  # [d_e // 4]
        
        # Concatenate and fuse
        combined = torch.cat([obs_feat, action_feat, reward_feat], dim=-1)
        e_t = self.fusion(combined)
        
        return e_t


class QueryNetwork(nn.Module):
    """
    Query network for CMS read operation.
    
    Computes query vector from fast state and encoded input.
    
    Math:
        q_t = Q_ψ(s_{t-1}, e_t) ∈ ℝ^{d_k}
    """
    
    def __init__(self, d_s: int, d_e: int, d_k: int, hidden_dim: Optional[int] = None):
        """
        Args:
            d_s: Fast state dimension
            d_e: Encoded input dimension
            d_k: Query/key dimension
            hidden_dim: Hidden dimension (default: d_s + d_e)
        """
        super().__init__()
        
        self.d_s = d_s
        self.d_e = d_e
        self.d_k = d_k
        self.hidden_dim = hidden_dim or (d_s + d_e)
        
        self.net = nn.Sequential(
            nn.Linear(d_s + d_e, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, d_k),
        )
    
    def forward(self, s_prev: torch.Tensor, e_t: torch.Tensor) -> torch.Tensor:
        """
        Compute query vector.
        
        Args:
            s_prev: Previous fast state [d_s]
            e_t: Encoded input [d_e]
        
        Returns:
            q_t: Query vector [d_k]
        """
        combined = torch.cat([s_prev, e_t], dim=-1)
        q_t = self.net(combined)
        return q_t


class OutputDecoder(nn.Module):
    """
    Output decoder for producing predictions/actions.
    
    Decodes fast state and context into output.
    
    Math:
        y_t = H_ω(s_t, c_t)
    """
    
    def __init__(
        self,
        d_s: int,
        d_c: int,
        output_dim: int,
        output_type: str = "continuous",  # "continuous" or "discrete"
        hidden_dim: Optional[int] = None,
    ):
        """
        Args:
            d_s: Fast state dimension
            d_c: Context dimension
            output_dim: Output dimension
            output_type: "continuous" for regression, "discrete" for classification
            hidden_dim: Hidden dimension (default: d_s + d_c)
        """
        super().__init__()
        
        self.d_s = d_s
        self.d_c = d_c
        self.output_dim = output_dim
        self.output_type = output_type
        self.hidden_dim = hidden_dim or (d_s + d_c)
        
        layers = [
            nn.Linear(d_s + d_c, self.hidden_dim),
            nn.LayerNorm(self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self.hidden_dim // 2, output_dim),
        ]
        
        # Add activation based on output type
        if output_type == "discrete":
            # Logits for discrete outputs (use with cross-entropy loss)
            pass  # No activation, return logits
        elif output_type == "continuous":
            # Tanh for bounded continuous outputs
            layers.append(nn.Tanh())
        else:
            raise ValueError(f"Unknown output_type: {output_type}")
        
        self.net = nn.Sequential(*layers)
    
    def forward(self, s_t: torch.Tensor, c_t: torch.Tensor) -> torch.Tensor:
        """
        Decode state and context into output.
        
        Args:
            s_t: Fast state [d_s]
            c_t: Context [d_c]
        
        Returns:
            y_t: Output [output_dim]
        """
        combined = torch.cat([s_t, c_t], dim=-1)
        y_t = self.net(combined)
        return y_t
