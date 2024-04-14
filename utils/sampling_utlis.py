import jax
import jax.numpy as jnp

def ula(score_fn, x0, random_key, dt, T):
    """
    Running unadjusted Langevin Algorithm in discrete time
    
    Args:
        score_fn: Callable function, take x and t, generate the score function at x and t
        x0: initial value
        random_key: the random key to generate noisy 
        dt: time-step
        T: the total time to run ula
    """
    x = x0
    ts = jnp.arange(T, 0-dt, -dt)
    noises = jax.random.multivariate_normal(random_key, jnp.zeros_like(x), jnp.diag(jnp.ones_like(x)), shape=(ts.shape[0],))
    for i in range(ts.shape[0]):
        x = x + dt * score_fn(x, ts[i]) + (2 * dt)**0.5 * noises[i]
    return x