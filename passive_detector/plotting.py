from typing import Optional
import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
from observers import GaussianEstimate
from plotly import graph_objects as go, subplots


def plot_evolution(
    t: jax.Array,
    signal: Optional[jax.Array] = None,
    estimate: Optional[GaussianEstimate] = None,
    name="s",
):
    if signal is not None:
        _, channels = signal.shape
    if estimate is not None:
        _, channels = estimate.mean.shape

    fig = subplots.make_subplots(rows=channels, cols=1)

    for i in range(channels):
        if estimate is not None:
            mean = estimate.mean[:, i]
            std = jnp.sqrt(estimate.cov.diagonal(axis1=-2, axis2=-1))[:, i]

            fig.add_trace(
                go.Scatter(x=t, y=mean, mode="lines", name=f"estimate {name}_{i}"),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=mean - std,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                ),
                row=i + 1,
                col=1,
            )
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=mean + std,
                    mode="lines",
                    line=dict(width=0),
                    showlegend=False,
                    fill="tonexty",
                    fillcolor="rgba(0,0,255,0.2)",
                ),
                row=i + 1,
                col=1,
            )

        if signal is not None:
            fig.add_trace(
                go.Scatter(
                    x=t,
                    y=signal[:, i],
                    mode="lines",
                    line=dict(dash="dash"),
                    name=f"true {name}_{i}",
                ),
                row=i + 1,
                col=1,
            )

    fig.update_layout(height=300 * channels, width=800, title_text=name)
    fig.show()

    # plt.figure(figsize=(12, 3 * channels))
    # for i in range(channels):
    #     plt.subplot(channels, 1, i + 1)
    #     if estimate is not None:
    #         mean = estimate.mean[:, i]
    #         std = jnp.sqrt(estimate.cov.diagonal(axis1=-2, axis2=-1))[:, i]
    #         plt.step(t, mean, label=f"$\hat{{{name}}}_{i}$")
    #         plt.fill_between(t, mean - std, mean + std, alpha=0.2)
    #     if signal is not None:
    #         plt.step(t, signal[:, i], "k", linestyle="--", label=f"${name}_{i}$")
    #     plt.legend()
    # plt.xlabel("Time [s]")
    # plt.show()


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
    if u_t is not None:
        plot_evolution(t, u_t, name="u")
    if y_t is not None:
        plot_evolution(t, y_t, name="y")
    if z_t is not None or z_est is not None:
        plot_evolution(t, z_t, z_est, name="z")
    if x_t is not None or x_est is not None:
        plot_evolution(t, x_t, x_est, name="x")
