import jax
import jax.numpy as jnp

from svgp_preference import svgp_predict

#Select two candidate points via Thompson sampling. each candidate is argmax of an independent sample from GP posterior
def thompson_sampling(params, rng_key, n_candidates=2000):
    rng_key, k_grid, k_eps1, k_eps2 = jax.random.split(rng_key, 4)

    # Random candidates uniformly over the input domain
    candidates = jax.random.uniform(k_grid, shape=(n_candidates, 2),
                                    minval=-3.0, maxval=3.0)

    # Posterior mean and marginal variance at all candidates
    mu, var = svgp_predict(params, candidates, full_cov=False)  # both (n_candidates,)
    std = jnp.sqrt(jnp.clip(var, 1e-6))

    # Sample 1
    eps1   = jax.random.normal(k_eps1, shape=(n_candidates,))
    f1     = mu + std * eps1
    cand1  = candidates[jnp.argmax(f1)]                         # (2,)

    # Sample 2 — independent eps, same mu/std (marginal approximation)
    eps2   = jax.random.normal(k_eps2, shape=(n_candidates,))
    f2     = mu + std * eps2
    cand2  = candidates[jnp.argmax(f2)]                         # (2,)

    return cand1, cand2, rng_key
