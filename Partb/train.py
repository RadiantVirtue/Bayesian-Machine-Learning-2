import jax
import jax.numpy as jnp
import optax

from svgp_preference import svgp_elbo_preference


def train_preference_gp(params, X_i, X_j, y_ij, rng_key, num_epochs=200, lr=0.01, batch_size=64, n_samples=20):
    #Train the sparse GP surrogate on preference pair data using Adam.

    #Supports warm-starting
    N_total = y_ij.shape[0]
    opt = optax.adam(lr)
    opt_state = opt.init(params)

    for epoch in range(num_epochs):
        rng_key, perm_key = jax.random.split(rng_key)
        perm = jax.random.permutation(perm_key, N_total)

        for start in range(0, N_total, batch_size):
            idx       = perm[start : start + batch_size]
            X_i_b    = X_i[idx]
            X_j_b    = X_j[idx]
            y_b      = y_ij[idx]

            def neg_elbo(p):
                val, _ = svgp_elbo_preference(
                    p, X_i_b, X_j_b, y_b, N_total, rng_key, n_samples
                )
                return -val

            neg_val, grads = jax.value_and_grad(neg_elbo)(params)
            updates, opt_state = opt.update(grads, opt_state)
            params = optax.apply_updates(params, updates)

    return params, rng_key
