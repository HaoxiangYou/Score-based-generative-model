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

def forward_OU_process(x0, t, random_key):
    """
    The forward Ornstein-Uhlenbeck(OU) process and the backward OU process in is 
    mainly based the theorem from lecture note 20-22 of CPSC 586 Spring 2024.

    The forward OU process is given by
        dX_t = -X_t dt + \sqrt{2} dW_t,
    The analytical solution to the SDE is given by
        X_t = (equal in distribution) e^{-t} X_0 + \sqrt{1 - e^{2t}} Z, 
    where Z is sample from standard Gaussian, i.e. \mathcal{N}(0, I)
    
    The X_t will converge to standard Gaussian as t big enough

    Args:
        x0: initial value
        random_key: the random key to generate noisy 
        t: time for SDE to envolve
    Returns:
        X_t: a realization of random variable drawn from rho_t, which is the distribution involve along OU process
    """
    z = jax.random.multivariate_normal(random_key, jnp.zeros_like(x0), jnp.diag(jnp.ones_like(x0)))
    return jnp.exp(-t) * x0 + (1-jnp.exp(-2*t)) * z

def backward_OU_process(score_fn, xT, random_key, dt, T, t0=0):
    """
    Run backward OU process to generate a sample.

    The backward OU process is given by
        dX_T-t = X_T-t + 2 score_fn(X_T-t, t) dt + \sqrt dW_T-t
    If we fixed score_fn(X_{T-t}, t), the one step discrete update for above SDE is given by
        X_t-1 = e^dt X_t + 2 (e^dt-1) score_fn(X_t, t) + \sqrt{e^{2dt} - 1} Z.
    When t is small, it can also be approximate by
        X_{t-1} = (1+dt) X_t + 2 dt score_fn(X_t, t) + \sqrt{2t} Z.
    
    Args:
        score_fn: Callable function, take x and t, generate the score function at x and t
        xT: initial value, which should be sample from a standard Gaussian.
        random_key: the random key to generate noisy 
        dt: time-step
        T: the time xT is draw from
        t0: the time to stop running backward OU, so we run backward OU for T-t time    
    Return:
        x0: a sample from target distribution
    """
    x = xT
    ts = jnp.arange(T, t0-dt, -dt)
    noises = jax.random.multivariate_normal(random_key, jnp.zeros_like(x), jnp.diag(jnp.ones_like(x)), shape=(ts.shape[0],))
    for i in range(ts.shape[0]):
        x = jnp.exp(dt) * x + 2 * (jnp.exp(dt)-1) * score_fn(x, ts[i]) + (jnp.exp(2 * dt) - 1)**0.5 * noises[i]
    return x