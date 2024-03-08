from typing import Optional
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from observers import GaussianEstimate


def plot_evolution(
    t: jax.Array,
    signal: jax.Array,
    estimate: Optional[GaussianEstimate] = None,
    name="s",
):
    _, channels = signal.shape
    plt.figure(figsize=(12, 3 * channels))

    for i in range(channels):
        plt.subplot(channels, 1, i + 1)
        if estimate is not None:
            mean = estimate.mean[:, i]
            std = jnp.sqrt(estimate.cov.diagonal(axis1=-2, axis2=-1))[:, i]
            plt.step(t, mean, label=f"$\hat{{{name}}}_{i}$")
            plt.fill_between(t, mean - std, mean + std, alpha=0.2)
        plt.step(t, signal[:, i], "k", linestyle="--", label=f"${name}_{i}$")
        plt.legend()
    plt.xlabel("Time [s]")
    plt.show()


def plot_sim_result(
    t: jax.Array,
    *,
    u_t: Optional[jax.Array] = None,
    y_t: Optional[jax.Array] = None,
    x_t: Optional[jax.Array] = None,
    z_t: Optional[jax.Array] = None,
    x_est: Optional[GaussianEstimate] = None,
    z_est: Optional[GaussianEstimate] = None,
):
    if x_t is not None:
        plot_evolution(t, x_t, x_est, name="x")
    if z_t is not None:
        plot_evolution(t, z_t, z_est, name="z")
    if u_t is not None:
        plot_evolution(t, u_t, name="u")
    if y_t is not None:
        plot_evolution(t, y_t, name="y")
