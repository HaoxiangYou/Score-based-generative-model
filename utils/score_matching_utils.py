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
        loss for score matching
    """
    score_joc_fn = jax.jacobian(score_fn, argnums=1)
    def score_matching_single_loss(x, t):
        return 1/2 * jnp.sum(score_fn(model_param, x, t)**2) + jnp.trace(score_joc_fn(model_param, x, t))    
    return jnp.mean(jax.vmap(score_matching_single_loss)(xs, ts))

def sliced_score_matching_loss(score_fn, model_param, xs, ts, random_key):
    """
    The sliced_score_matching_loss introduced in https://arxiv.org/pdf/1905.07088.pdf
    
    Args:
        score_fn: Callable function, take model params, data and additional argument and output the score
        model_param: model parameters
        xs: data point
        ts: timesteps when the score is evaluated
        random_key: random_key to generate gaussian noisy
    
    Return:
        sliced_score_matching_loss
    """
    # generate random noize
    num_samples, sample_dim = xs.shape
    vs = jax.random.multivariate_normal(random_key, jnp.zeros(sample_dim), jnp.eye(sample_dim), shape=(num_samples,))

    def sliced_score_fn(x, t, v):
        return jnp.dot(v, score_fn(model_param, x, t))
    sliced_score_jac_fn = jax.grad(sliced_score_fn, argnums=0)
    def sliced_score_single_loss(x, t, v):
        return 1/2 * jnp.sum(score_fn(model_param, x, t)**2) + jnp.dot(sliced_score_jac_fn(x, t, v), v)
    return jnp.mean(jax.vmap(sliced_score_single_loss)(xs, ts, vs))