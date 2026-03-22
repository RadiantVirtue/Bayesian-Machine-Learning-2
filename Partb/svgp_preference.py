import jax
import jax.numpy as jnp
import jax.scipy as jsp


# Kernel + Cholesky utilities (copied from Part A, work for 2D inputs as-is)

def rbf_kernel(x1, x2, lengthscale, variance, diag=False):
    if x1.ndim == 1:
        x1 = x1.reshape(-1, 1)
        x2 = x2.reshape(-1, 1)
    if diag:
        dist_sq_diag = jnp.sum((x1 - x2) ** 2, axis=-1)
        return variance * jnp.exp(-0.5 * dist_sq_diag / (lengthscale ** 2))
    else:
        dist_sq = jnp.sum((x1[:, None, :] - x2[None, :, :]) ** 2, axis=-1)
        return variance * jnp.exp(-0.5 * dist_sq / (lengthscale ** 2))


def stable_cholesky(K, jitter=1e-5):
    return jnp.linalg.cholesky(K + jitter * jnp.eye(K.shape[0]))


def build_var_chol(var_chol_log_diag, var_chol_lower, M):
    L_v = jnp.zeros((M, M))
    lower_idx = jnp.tril_indices(M, k=-1)
    L_v = L_v.at[lower_idx].set(var_chol_lower)
    diag_idx = jnp.diag_indices(M)
    L_v = L_v.at[diag_idx].set(jnp.exp(var_chol_log_diag))
    return L_v


# Prediction (copied from Part A, works for 2D inputs as-is)

def svgp_predict(params, X_test, full_cov=False):
    ls  = jnp.exp(params["log_lengthscale"])
    var = jnp.exp(params["log_variance"])

    Z = params["inducing_inputs"]   # (M, 2)
    m = params["variational_mean"]  # (M,)
    M = Z.shape[0]

    L_v = build_var_chol(params["var_chol_log_diag"], params["var_chol_lower"], M)

    K_zz     = rbf_kernel(Z, Z, ls, var)                                      # (M, M)
    L_zz     = stable_cholesky(K_zz)                                          # (M, M)
    K_z_test = rbf_kernel(Z, X_test, ls, var)                                 # (M, N_test)
    alpha    = jsp.linalg.solve_triangular(L_zz, K_z_test, lower=True)        # (M, N_test)

    mu = alpha.T @ m                                                           # (N_test,)
    B  = L_v.T @ alpha                                                         # (M, N_test)

    if full_cov:
        K_tt = rbf_kernel(X_test, X_test, ls, var)                            # (N, N)
        cov  = K_tt - alpha.T @ alpha + B.T @ B
    else:
        k_diag = rbf_kernel(X_test, X_test, ls, var, diag=True)               # (N_test,)
        cov    = k_diag - jnp.sum(alpha ** 2, axis=0) + jnp.sum(B ** 2, axis=0)

    return mu, cov


# KL divergence (copied from Part A, unchanged)

def svgp_kl_divergence(params):
    m = params["variational_mean"]
    M = m.shape[0]
    L_v = build_var_chol(params["var_chol_log_diag"], params["var_chol_lower"], M)

    trace_S   = jnp.sum(L_v ** 2)
    quad      = jnp.dot(m, m)
    log_det_S = 2.0 * jnp.sum(params["var_chol_log_diag"])

    return 0.5 * (trace_S + quad - M - log_det_S)


# Bradley-Terry expected log likelihood for preference pairs

def svgp_ell_preference(params, X_i_batch, X_j_batch, y_ij_batch, N_total, rng_key, n_samples=20):
    ls  = jnp.exp(params["log_lengthscale"])
    var = jnp.exp(params["log_variance"])
    Z   = params["inducing_inputs"]   # (M, 2)
    m   = params["variational_mean"]  # (M,)
    M   = Z.shape[0]
    L_v  = build_var_chol(params["var_chol_log_diag"], params["var_chol_lower"], M)
    K_zz = rbf_kernel(Z, Z, ls, var)
    L_zz = stable_cholesky(K_zz)
    B_size = X_i_batch.shape[0]

    # Stack both halves to run the kernel and solve once
    X_pairs   = jnp.concatenate([X_i_batch, X_j_batch], axis=0)               # (2B, 2)
    K_z_pairs = rbf_kernel(Z, X_pairs, ls, var)                                # (M, 2B)
    alpha_all = jsp.linalg.solve_triangular(L_zz, K_z_pairs, lower=True)       # (M, 2B)

    alpha_i = alpha_all[:, :B_size]    # (M, B)
    alpha_j = alpha_all[:, B_size:]    # (M, B)
    B_i     = L_v.T @ alpha_i          # (M, B)
    B_j     = L_v.T @ alpha_j          # (M, B)

    # Posterior means for each side
    mu_i    = alpha_i.T @ m            # (B,)
    mu_j    = alpha_j.T @ m            # (B,)
    mu_diff = mu_i - mu_j              # (B,)

    # Posterior variances and cross-covariance — only diagonal elements needed
    k_ii = rbf_kernel(X_i_batch, X_i_batch, ls, var, diag=True)               # (B,)
    k_jj = rbf_kernel(X_j_batch, X_j_batch, ls, var, diag=True)               # (B,)
    k_ij = rbf_kernel(X_i_batch, X_j_batch, ls, var, diag=True)               # (B,)
    
    Sigma_ii = k_ii - jnp.sum(alpha_i ** 2, axis=0) + jnp.sum(B_i ** 2, axis=0)       # (B,)
    Sigma_jj = k_jj - jnp.sum(alpha_j ** 2, axis=0) + jnp.sum(B_j ** 2, axis=0)       # (B,)
    Sigma_ij = k_ij - jnp.sum(alpha_i * alpha_j, axis=0) + jnp.sum(B_i * B_j, axis=0) # (B,)

    var_diff = jnp.clip(Sigma_ii + Sigma_jj - 2.0 * Sigma_ij, 1e-6)           # (B,)
    std_diff = jnp.sqrt(var_diff)                                               # (B,)

    # Reparametrised Monte Carlo samples
    rng_key, subkey = jax.random.split(rng_key)
    eps    = jax.random.normal(subkey, shape=(n_samples, B_size))               # (S, B)
    f_diff = mu_diff[None, :] + std_diff[None, :] * eps                         # (S, B)

    # Log-Bernoulli via log-sigmoid (numerically stable)
    y       = y_ij_batch[None, :].astype(jnp.float32)                          # (1, B)
    log_lik = y * (-jax.nn.softplus(-f_diff)) + \
              (1.0 - y) * (-jax.nn.softplus(f_diff))                            # (S, B)

    ell = (N_total / B_size) * jnp.sum(jnp.mean(log_lik, axis=0))
    return ell, rng_key


# ELBO

def svgp_elbo_preference(params, X_i_batch, X_j_batch, y_ij_batch,
                          N_total, rng_key, n_samples=20):
    ell, rng_key = svgp_ell_preference(
        params, X_i_batch, X_j_batch, y_ij_batch, N_total, rng_key, n_samples
    )
    kl = svgp_kl_divergence(params)
    return ell - kl, rng_key


# Parameter initialisation - w/ inducinginputs in 2d, no log noise var, variational mean initailised at 0

def init_params_preference(num_inducing, rng_key):
    M = num_inducing
    rng_key, subkey = jax.random.split(rng_key)
    Z = jax.random.uniform(subkey, shape=(M, 2), minval=-3.0, maxval=3.0)

    return {
        "log_lengthscale":   jnp.array(0.0),
        "log_variance":      jnp.array(0.0),
        "inducing_inputs":   Z,
        "variational_mean":  jnp.zeros(M),
        "var_chol_log_diag": jnp.full((M,), jnp.log(0.1)),
        "var_chol_lower":    jnp.zeros(M * (M - 1) // 2),
    }
