import jax
import jax.numpy as jnp

def latent_function(x):
    x0 = x[..., 0]
    x1 = x[..., 1]
    
    background = (
        0.8 * jnp.sin(2.0 * x0 * x1) +
        0.6 * jnp.cos(3.0 * x0) +
        0.6 * jnp.sin(3.0 * x1) +
        0.5 * jnp.sin(1.5 * (x0**2 + x1**2))
    )
    
    dominant_peak = 4 * jnp.exp(
        -1.0 * ((x0 - 1.2)**2 + (x1 + 1.0)**2)
    )
    
    return background + dominant_peak

def generate_initial_preference_data(num_points=5, num_pairs=25, noise_std=0.5, seed=42):
    rng = jax.random.PRNGKey(seed)
    X = jax.random.uniform(rng, shape=(num_points, 2), minval=-3, maxval=3)
    
    latent_values = latent_function(X)
    # Sample pairs of points and generate preferences based on the latent function values with some noise
    rng, rng_subkey = jax.random.split(rng)
    indices_i = jax.random.randint(rng_subkey, shape=(num_pairs,), minval=0, maxval=num_points)
    rng, rng_subkey = jax.random.split(rng)
    indices_j = jax.random.randint(rng_subkey, shape=(num_pairs,), minval=0, maxval=num_points)
    X_i = X[indices_i]
    X_j = X[indices_j]
    latent_diff = latent_values[indices_i] - latent_values[indices_j]
    rng, rng_subkey = jax.random.split(rng)
    noise = noise_std * jax.random.normal(rng_subkey, shape=latent_diff.shape)
    y_ij = (latent_diff + noise > 0).astype(jnp.float32) # 1 if i preferred to j, 0 otherwise
    return X_i, X_j, y_ij, X, latent_values

# Vectorized Data Generation
def sample_preferences(rng_key, candidate_1, candidate_2, preference_samples_per_location=25):
    # Calculate the probability once
    latent_diff = latent_function(candidate_1) - latent_function(candidate_2)
    sigmoid_prob = jax.nn.sigmoid(latent_diff)
    
    # Draw all preference samples at once using the shape argument
    rng_key, rng_subkey = jax.random.split(rng_key)
    preferences = jax.random.bernoulli(
        rng_subkey, p=sigmoid_prob, shape=(preference_samples_per_location,)
    ).astype(jnp.float32)

    # Tile the candidates to match the number of samples
    new_X_i = jnp.tile(candidate_1, (preference_samples_per_location, 1)) # Shape: (preference_samples_per_location, 2)
    new_X_j = jnp.tile(candidate_2, (preference_samples_per_location, 1)) # Shape: (preference_samples_per_location, 2)
    new_y_ij = preferences # Shape: (preference_samples_per_location,)

    return new_X_i, new_X_j, new_y_ij