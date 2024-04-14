import jax.numpy as jnp
import flax.linen as nn
from typing import Sequence

class Score_MLP(nn.Module):
    """
    Simple MLP neural model that predict score function
    
    Attr:
        features: hidden layer neuron numbers
    """
    features: Sequence[int]
    dim: int
    @nn.compact
    def __call__(self, x, t):
        # append the sample and time together
        if len(x.shape) == 1:
            x = jnp.append(x, t)
        else:
            x = jnp.column_stack([x, t])

        for i, feature in enumerate(self.features):
            x = nn.Dense(feature)(x)
            # Apply nonlinearity
            x = nn.tanh(x)

        return nn.Dense(self.dim)(x)