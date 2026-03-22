import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

os.environ["XLA_PYTHON_CLIENT_PREALLOCATE"] = "false"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np

from cw2_partB_data_generation import latent_function, generate_initial_preference_data, sample_preferences
from svgp_preference import svgp_predict, init_params_preference
from thompson_sampling import thompson_sampling
from train import train_preference_gp

# Select device — fall back to CPU if no CUDA device is found
try:
    _device = jax.devices("cuda")[0]
    print(f"Using device: {_device}")
except RuntimeError:
    _device = jax.devices("cpu")[0]
    print(f"WARNING: No CUDA device found — running on CPU ({_device}). Expect longer runtimes.")


def make_grid(n=80):
    x0 = jnp.linspace(-3, 3, n)
    x1 = jnp.linspace(-3, 3, n)
    X0, X1 = jnp.meshgrid(x0, x1)
    grid = jnp.stack([X0.ravel(), X1.ravel()], axis=-1)
    return grid, X0, X1


def plot_surrogate_and_truth(params, X_i_all, X_j_all, title_suffix, filename,
                              collection_order=None):
    grid, X0, X1 = make_grid(n=80)
    mu_grid, _  = svgp_predict(params, grid, full_cov=False)
    true_grid   = latent_function(grid)

    mu_img   = np.array(mu_grid).reshape(80, 80)
    true_img = np.array(true_grid).reshape(80, 80)
    all_locs = np.array(jnp.concatenate([X_i_all, X_j_all], axis=0))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, img, title in zip(axes,
                               [mu_img, true_img],
                               [f"Surrogate mean {title_suffix}",
                                f"True latent function {title_suffix}"]):
        im = ax.contourf(np.array(X0), np.array(X1), img, levels=30, cmap="viridis")
        fig.colorbar(im, ax=ax)
        if collection_order is not None:
            n_pts = len(all_locs)
            colors = cm.plasma(np.linspace(0, 1, n_pts))
            for pt, col in zip(all_locs, colors):
                ax.scatter(pt[0], pt[1], color=col, s=20, zorder=3, edgecolors="none")
        else:
            ax.scatter(all_locs[:, 0], all_locs[:, 1],
                       c="white", s=20, zorder=3, edgecolors="none", alpha=0.7)
        ax.set_xlim(-3, 3); ax.set_ylim(-3, 3)
        ax.set_xlabel("x₀"); ax.set_ylabel("x₁"); ax.set_title(title)

    plt.tight_layout()
    os.makedirs(os.path.join(os.path.dirname(__file__), "report"), exist_ok=True)
    save_path = os.path.join(os.path.dirname(__file__), "report", filename)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved: {save_path}")
    plt.show()


def main():
    rng_key = jax.random.PRNGKey(1)

    print("Generating initial preference data...")
    X_i, X_j, y_ij, _, _ = generate_initial_preference_data(seed=7)
    X_i  = jax.device_put(X_i,  _device)
    X_j  = jax.device_put(X_j,  _device)
    y_ij = jax.device_put(y_ij, _device)
    print(f"  Initial pairs: {y_ij.shape[0]}")

    print("Training initial surrogate model...")
    rng_key, init_key = jax.random.split(rng_key)
    params = init_params_preference(num_inducing=15, rng_key=init_key)
    params, rng_key = train_preference_gp(
        params, X_i, X_j, y_ij, rng_key, num_epochs=300, lr=0.01, batch_size=32
    )
    print("  Done.")

    print("Plotting initial surrogate...")
    plot_surrogate_and_truth(params, X_i, X_j, "(initial)", "plot_A_initial.png")

    print("\nStarting Bayesian Optimisation loop (50 rounds)...")
    for round_idx in range(50):
        rng_key, ts_key = jax.random.split(rng_key)
        cand1, cand2, _ = thompson_sampling(params, ts_key, n_candidates=2000)

        rng_key, sp_key = jax.random.split(rng_key)
        new_Xi, new_Xj, new_yij = sample_preferences(sp_key, cand1, cand2,
                                                       preference_samples_per_location=25)
        X_i  = jnp.concatenate([X_i,  new_Xi])
        X_j  = jnp.concatenate([X_j,  new_Xj])
        y_ij = jnp.concatenate([y_ij, new_yij])

        params, rng_key = train_preference_gp(
            params, X_i, X_j, y_ij, rng_key, num_epochs=20, lr=0.005, batch_size=64
        )

        print(f"  Round {round_idx + 1}/50  |  total pairs: {y_ij.shape[0]}  |  cand1=[{float(cand1[0]):.2f}, {float(cand1[1]):.2f}]  cand2=[{float(cand2[0]):.2f}, {float(cand2[1]):.2f}]")

    print("BO loop complete.")

    print("Plotting final surrogate...")
    all_locs_ordered = np.array(jnp.concatenate([X_i, X_j], axis=0))
    plot_surrogate_and_truth(params, X_i, X_j, "(after 50 rounds)", "plot_B_final.png",
                              collection_order=all_locs_ordered)

    grid, _, _ = make_grid(n=200)
    mu_grid, _ = svgp_predict(params, grid, full_cov=False)
    best_x = grid[jnp.argmax(mu_grid)]
    print(f"\nEstimated optimum: x = [{float(best_x[0]):.3f}, {float(best_x[1]):.3f}]")
    print(f"True optimum is near: x = [1.2, -1.0]")


if __name__ == "__main__":
    main()
