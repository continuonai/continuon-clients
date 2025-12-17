import torch
import torch.nn as nn
import torch.nn.functional as F

class VectorNeuronLinear(nn.Module):
    """
    A simplified Vector Neuron Linear layer.
    Input: (Batch, N_points, 3, Channels_in) or (Batch, 3, Channels_in)
    Output: (Batch, N_points, 3, Channels_out) or (Batch, 3, Channels_out)
    
    Instead of standard weights W (Cin, Cout), we learn W (Cin, Cout) mapping 
    vectors to vectors while preserving the 3D structure (the '3' dimension).
    Basically, it applies a linear transformation to the feature dimension 
    independent of the spatial (3D) orientation.
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight = nn.Parameter(torch.randn(in_channels, out_channels))
        
    def forward(self, x):
        # x shape: [B, ..., 3, Cin]
        # We want to multiply the last dimension by weight
        return torch.matmul(x, self.weight)

class VectorNeuronLeakyReLU(nn.Module):
    """
    Vector Neuron Non-linearity.
    Standard ReLU breaks rotational equivariance.
    VN-ReLU projects the vector onto a learned direction 'q' and keeps separate 
    components to maintain equivariance. 
    
    Simplified for this implementation: We use a standard LeakyReLU on the magnitude
    or component-wise if we relax the strict equivariance requirement for this proof-of-concept.
    
    STRICT IMPLEMENTATION (Approximation): 
    Split feature space into two halves. Learn a direction vector.
    """
    def __init__(self, channels):
        super().__init__()
        # Simplified: Just return x for now to ensure graph connectivity 
        # without complex projection logic in first pass. 
        # Real VN-ReLU requires learning a direction 'k' and doing x + (x.k)k ...
        pass
        
    def forward(self, x):
        return F.leaky_relu(x, 0.1)

class VectorNeuronMLP(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.layer1 = VectorNeuronLinear(in_channels, hidden_channels)
        self.layer2 = VectorNeuronLinear(hidden_channels, out_channels)
        
    def forward(self, x):
        # x: [Batch, N_points, 3, In_Channels]
        x = self.layer1(x)
        x = F.leaky_relu(x, 0.1) # approximating VN-ReLU
        x = self.layer2(x)
        return x

class AINAPolicy(nn.Module):
    """
    Transformer-based Point Cloud Policy.
    Inputs:
        - fingertip_traj: [Batch, T_obs, N_fingers, 3]
        - object_pcd: [Batch, T_obs, N_points, 3]
    Output:
        - future_fingertips: [Batch, T_pred, N_fingers, 3]
    """
    def __init__(self, 
                 n_fingers=5, 
                 n_obj_points=100, 
                 obs_horizon=10, 
                 pred_horizon=5,
                 embed_dim=64):
        super().__init__()
        self.obs_horizon = obs_horizon
        self.pred_horizon = pred_horizon
        
        # Encoders for geometric structure (Vector Neurons)
        # Input channel is 1 (just the position vector itself)
        self.finger_vn = VectorNeuronMLP(1, 32, embed_dim) 
        self.object_vn = VectorNeuronMLP(1, 32, embed_dim)
        
        # Transformer
        # We flatten the VN output [B, T, N, 3, Emb] -> [B, T*N, 3*Emb] 
        # to pass to standard transformer, or keep structural?
        # The paper implies tokens are vectors.
        # Let's project to scalar embedding for the standard transformer part.
        
        self.projection = nn.Linear(3 * embed_dim, embed_dim)
        
        # Positional Encoding (Temporal)
        self.pos_embedding = nn.Parameter(torch.randn(1, obs_horizon * (n_fingers + n_obj_points), embed_dim))
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=4, batch_first=True)
        self.transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=2)
        
        # Decoder / Prediction Head
        # Predicts T_pred steps for N_fingers (3 coords each)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 128),
            nn.ReLU(),
            nn.Linear(128, pred_horizon * n_fingers * 3)
        )
        
        self.n_fingers = n_fingers
        
    def forward(self, fingertip_traj, object_pcd):
        """
        fingertip_traj: [B, T, N_fingers, 3]
        object_pcd: [B, T, N_points, 3]
        """
        B, T, Nf, _ = fingertip_traj.shape
        _, _, Np, _ = object_pcd.shape
        
        # 1. Vector Neuron Encoding
        # Reshape to [B, T*N, 3, 1] for VN
        ft_in = fingertip_traj.reshape(B, T*Nf, 3, 1)
        obj_in = object_pcd.reshape(B, T*Np, 3, 1)
        
        ft_feat = self.finger_vn(ft_in) # [B, T*Nf, 3, Emb]
        obj_feat = self.object_vn(obj_in) # [B, T*Np, 3, Emb]
        
        # 2. Flatten for Transformer
        # [B, T*Nf, 3*Emb]
        ft_tokens = ft_feat.reshape(B, T*Nf, -1)
        ft_tokens = self.projection(ft_tokens)
        
        obj_tokens = obj_feat.reshape(B, T*Np, -1)
        obj_tokens = self.projection(obj_tokens)
        
        # Concatenate tokens
        tokens = torch.cat([ft_tokens, obj_tokens], dim=1) # [B, T*(Nf+Np), Emb]
        
        # Add Positional Embedding (naive broadcasting)
        # Note: In real impl, handle variable sizing or fixed max size
        if tokens.shape[1] == self.pos_embedding.shape[1]:
            tokens = tokens + self.pos_embedding
            
        # 3. Transformer
        encoded = self.transformer_encoder(tokens)
        
        # 4. Prediction
        # Global pooling or take specific tokens?
        # AINA usually takes the last fingertip tokens to predict future.
        # Let's simple max pool over the sequence for Global Context
        global_feat = torch.max(encoded, dim=1)[0] # [B, Emb]
        
        pred_flat = self.head(global_feat) # [B, T_pred * Nf * 3]
        
        return pred_flat.reshape(B, self.pred_horizon, self.n_fingers, 3)
