import jax
import jax.numpy as jnp

def score_matching_loss(score_fn, model_param, xs, ts):
    """
    The score matching loss can be compute as
    1/N \sum_i=1^N [1/2 \|s(x_i)\|_2^2 + Tr(\nabla_x s_theta(x_i))]
    
    Args:
    score_fn: Callable function, take model params, data and additional argument and output the score
    model_param: model parameters
    xs: data point
    ts: timesteps when the score is evaluated

    Return:
    Loss for score matching
    """
    score_joc_fn = jax.jacobian(score_fn, argnums=1)
    def score_matching_single_loss(x, t):
        return 1/2 * jnp.sum(score_fn(model_param, x, t)**2) + jnp.trace(score_joc_fn(model_param, x, t))    
    return jnp.mean(jax.vmap(score_matching_single_loss)(xs, ts))