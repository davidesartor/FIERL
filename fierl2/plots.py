import matplotlib.pyplot as plt
import jax.random as jr


def plot_rollout(x, est, u, y, y_ref, u_ref, x_ref, tf=0.1):
    plt.figure(figsize=(15, 10))
    for i in range(x.shape[-1]):
        plt.subplot(x.shape[-1], 4, 4 * i + 1)
        mu, cov = est.mean[:, i], est.cov[:, i, i]
        plt.plot(mu, label=f"x{i}_est")
        plt.fill_between(range(len(mu)), mu - cov, mu + cov, alpha=0.5)
        plt.plot(x[:, i], label=f"x{i}")
        plt.plot(x_ref[:, i], ":k", label=f"x{i}_ref)")
        plt.vlines(
            tf * len(y_ref),
            plt.ylim()[0],
            plt.ylim()[1],
            colors="r",
            linestyles="dashed",
        )
        plt.legend()
        plt.grid()

    for i in range(u.shape[-1]):
        plt.subplot(u.shape[-1], 4, 4 * i + 3)
        plt.plot(u[:, i], label=f"u{i}")
        # plt.plot(a[:, i], label=f"a{i}")
        plt.plot(u_ref[:, i], ":k", label=f"u{i}_ref)")
        plt.vlines(
            tf * len(y_ref),
            plt.ylim()[0],
            plt.ylim()[1],
            colors="r",
            linestyles="dashed",
        )
        plt.legend()
        plt.grid()

    for i in range(y.shape[-1]):
        plt.subplot(y.shape[-1], 4, 4 * i + 4)
        plt.plot(y[:, i], label=f"y{i}")
        plt.plot(y_ref[:, i], ":k", label=f"y{i}_ref)")
        plt.vlines(
            tf * len(y_ref),
            plt.ylim()[0],
            plt.ylim()[1],
            colors="r",
            linestyles="dashed",
        )
        plt.legend()
        plt.grid()
    plt.show()
