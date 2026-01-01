import jax
import jax.numpy as jnp
from flax import linen as nn
import numpy as np
from typing import Any, Tuple

class VectorQuantizer(nn.Module):
    """Vector Quantization Layer."""
    embedding_dim: int
    num_embeddings: int
    commitment_cost: float = 0.25

    def setup(self):
        # Codebook: [num_embeddings, embedding_dim]
        init_fn = nn.initializers.variance_scaling(scale=1.0, mode='fan_in', distribution='uniform')
        self.embedding = self.param('embedding', init_fn, (self.num_embeddings, self.embedding_dim))

    def __call__(self, x):
        # x: [batch, height, width, embedding_dim]
        # Flatten input: [batch * height * width, embedding_dim]
        flat_x = x.reshape(-1, self.embedding_dim)

        # Calculate distances
        # (x - e)^2 = x^2 + e^2 - 2xe
        distances = (
            jnp.sum(flat_x**2, axis=1, keepdims=True)
            + jnp.sum(self.embedding**2, axis=1)
            - 2 * jnp.dot(flat_x, self.embedding.T)
        )

        # Encoding indices
        encoding_indices = jnp.argmin(distances, axis=1)
        encodings = jax.nn.one_hot(encoding_indices, self.num_embeddings, dtype=flat_x.dtype)

        # Quantize
        quantized = jnp.dot(encodings, self.embedding)
        quantized = quantized.reshape(x.shape)

        # Loss
        e_latent_loss = jnp.mean((jax.lax.stop_gradient(quantized) - x) ** 2)
        q_latent_loss = jnp.mean((quantized - jax.lax.stop_gradient(x)) ** 2)
        loss = q_latent_loss + self.commitment_cost * e_latent_loss

        # Straight-through estimator
        quantized = x + jax.lax.stop_gradient(quantized - x)

        return quantized, loss, encoding_indices

class Encoder(nn.Module):
    """Simple VQ-VAE Encoder."""
    latent_dim: int
    
    @nn.compact
    def __call__(self, x):
        # x: [batch, h, w, c]
        x = nn.Conv(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.Conv(features=self.latent_dim, kernel_size=(3, 3), strides=(1, 1))(x)
        return x

class Decoder(nn.Module):
    """Simple VQ-VAE Decoder."""
    output_channels: int
    
    @nn.compact
    def __call__(self, x):
        x = nn.ConvTranspose(features=64, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=32, kernel_size=(4, 4), strides=(2, 2))(x)
        x = nn.relu(x)
        x = nn.ConvTranspose(features=self.output_channels, kernel_size=(3, 3), strides=(1, 1))(x)
        return x

class VQVAE(nn.Module):
    latent_dim: int = 64
    num_embeddings: int = 512
    commitment_cost: float = 0.25
    output_channels: int = 3

    def setup(self):
        self.encoder = Encoder(latent_dim=self.latent_dim)
        self.vq = VectorQuantizer(
            embedding_dim=self.latent_dim,
            num_embeddings=self.num_embeddings,
            commitment_cost=self.commitment_cost
        )
        self.decoder = Decoder(output_channels=self.output_channels)

    def __call__(self, x):
        z_e = self.encoder(x)
        z_q, loss, indices = self.vq(z_e)
        x_recon = self.decoder(z_q)
        return x_recon, loss, indices
    
    def encode(self, x):
        z_e = self.encoder(x)
        _, _, indices = self.vq(z_e)
        return indices

if __name__ == "__main__":
    # Self-test
    print("Running VQ-VAE Self-Test...")
    key = jax.random.PRNGKey(0)
    model = VQVAE()
    dummy_input = jnp.ones((1, 64, 64, 3))
    params = model.init(key, dummy_input)
    recon, loss, indices = model.apply(params, dummy_input)
    print(f"Input shape: {dummy_input.shape}")
    print(f"Recon shape: {recon.shape}")
    print(f"Loss: {loss}")
    print(f"Indices shape: {indices.shape}")
    print("VQ-VAE Self-Test Passed!")
