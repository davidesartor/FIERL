from utils import *
from .systems import FDSSM, SysModule
from .controllers import MPC, ControlCostMatrices, References
from .observers import KalmanFilter
from flax import nnx


class Simulator(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        t_aux_on: int = 0,
        t_aux_off: int = -1,
        ref=References(y=1.0, u=0.0, x=0.0, z=0.0),
        Jcontrol=ControlCostMatrices(y=1.0, du=0.01, u=0.01, x=0.0, z=0.0),
        Jzest: Float[Array, "z z"] | Float[Array, "z"] | float = 1.0,
        Jxest: Float[Array, "x x"] | Float[Array, "x"] | float = 0.0,
        kalman_params: dict = dict(Qz=1e-8, Qx=1e-2, R=1e-2, mc_samples_init=0),
        mpc_params: dict = dict(horizon=16, discount=0.9, mc_samples_traj=0),
        *,
        rngs: nnx.Rngs,
    ):
        # sim params and state
        self.t_aux_on = t_aux_on
        self.t_aux_off = t_aux_off

        # costs and references
        self.ref = nnx.Param(ref)
        self.Jcontrol = nnx.Param(Jcontrol)
        self.Jzest = nnx.Param(Jzest)
        self.Jxest = nnx.Param(Jxest)

        # modules and state
        self.rngs = rngs
        self.t = nnx.Variable(0)
        self.sys = SysModule(sys, rngs=rngs)
        self.mpc = MPC(sys, Jcontrol, **mpc_params, rngs=rngs)
        self.kalman = KalmanFilter(sys, **kalman_params, rngs=rngs)

    @property
    def obs_dim(self):
        return self.obs().shape[-1]

    @property
    def a_dim(self):
        return self.sys.sys.u_dim

    def reset(self):
        self.t.value = 0
        self.sys.reset()
        self.mpc.reset()
        self.kalman.reset()

    def obs(self) -> Float[Array, "..."]:
        t = jnp.array([self.t.value])
        z_hat = self.kalman.z.value
        x_hat = self.kalman.x.value
        Psqrt = jnp.linalg.cholesky(self.kalman.P.value).flatten()
        return jnp.concat([t, z_hat, x_hat, Psqrt], axis=-1)

    def step(self, a: Float[Array, "u"] | None):
        # return initial state as info
        z, x = self.sys.z.value, self.sys.x.value  # hidden state
        z_hat, x_hat = (self.kalman.z.value, self.kalman.x.value)
        P = self.kalman.P.value

        # the actual simulation step
        u = self.mpc.step(z0=z_hat, x0=x_hat, ref=self.ref.value)
        a = a if a is not None else jnp.zeros_like(u)
        utot = u + a
        y = self.sys.step(utot)
        self.kalman.step(utot, y)
        self.t.value += 1
        return dict(z=z, x=x, z_hat=z_hat, x_hat=x_hat, P=P, u=u, a=a, y=y)

    def rewards_and_costs(self, outs: dict):
        z, x, z_hat, x_hat, P, u, a, y = (
            outs[k] for k in ["z", "x", "z_hat", "x_hat", "P", "u", "a", "y"]
        )
        reward = -(
            +quadratic_cost(z_hat - z, self.Jzest.value)
            + quadratic_cost(x_hat - x, self.Jxest.value)
        )
        cost = self.mpc.control_cost(zt=z, xt=x, ut=u + a, yt=y, ref=self.ref.value)

        return reward, cost

    @nnx.jit(static_argnames=("episode_length",))
    def rollout(self, policy, episode_length: int):
        @nnx.scan
        def aux_off_step(env, i):
            out = env.step(a=None)
            return env, out

        @nnx.scan
        def aux_on_step(carry, i):
            env, policy = carry
            obs = env.obs()
            a, log_p = policy.sample(obs)
            out = env.step(a)
            return (env, policy), (obs, a, log_p, out)

        # phase 1: aux off
        self.reset()
        Ton = (self.t_aux_on + episode_length) % episode_length
        self, outs1 = aux_off_step(self, jnp.arange(0, Ton))

        # phase 2: aux on
        Toff = (self.t_aux_off + episode_length) % episode_length
        (self, policy), (obs, a, log_p, outs2) = aux_on_step(
            (self, policy), jnp.arange(Ton, Toff)
        )
        next_obs = jnp.roll(obs, -1).at[-1].set(self.obs())
        rewards, costs = self.rewards_and_costs(outs2)

        # phase 3: aux off
        self, outs3 = aux_off_step(self, jnp.arange(Toff, episode_length))

        rollout = Rollout(obs, a, log_p, rewards, next_obs, costs)
        outs = jax.tree.map(lambda *x: jnp.concat(x), outs1, outs2, outs3)
        return rollout, outs

    def render(self, outs: dict, title=""):
        import matplotlib.pyplot as plt

        z, x, z_hat, x_hat, P, u, a, y = (
            outs[k] for k in ["z", "x", "z_hat", "x_hat", "P", "u", "a", "y"]
        )
        ref_y = jnp.broadcast_to(self.ref.y, y.shape)
        ref_u = jnp.broadcast_to(self.ref.u, u.shape)
        ref_x = jnp.broadcast_to(self.ref.x, x.shape)
        ref_z = jnp.broadcast_to(self.ref.z, z.shape)

        def plot(v, name: str, ref=None, est=None, color=None):
            t = list(range(len(v)))
            plt.plot(v, label=name, color=color)
            if est is not None:
                mean, cov = est
                plt.plot(mean, label="est")
                if cov is not None:
                    low, high = mean - cov**0.5, mean + cov**0.5
                    plt.fill_between(t, low, high, alpha=0.5, color="tab:grey")
            plt.hlines(0, 0, len(v), color="k", linestyle=":", alpha=0.5)
            if ref is not None:
                plt.plot(ref, ":", color="tab:red", label="ref")
            Ton, Toff = self.t_aux_on, self.t_aux_off
            plt.vlines([Ton, Toff], *plt.ylim(), colors=["g", "r"], linestyle="--")

            plt.legend()
            plt.grid(True)

        plt.figure(figsize=(20, 10))
        plt.suptitle(title)
        for i in range(z.shape[-1]):
            est = (z_hat[:, i], P[:, i, i])
            plt.subplot(z.shape[-1], 4, 4 * i + 1)
            plot(z[:, i], f"$z_{i}$", ref_z[:, i], est)
        for i in range(x.shape[-1]):
            est = (x_hat[:, -i - 1], P[:, -i - 1, -i - 1])
            plt.subplot(x.shape[-1], 4, 4 * (x.shape[-1] - i - 1) + 2)
            plot(x[:, -i - 1], f"$x_{x.shape[-1]-i}$", ref_x[:, -i - 1], est)
        for i in range(u.shape[-1]):
            plt.subplot(u.shape[-1], 4, 4 * i + 3)
            plot(v=u[:, i], name=f"$u_{i}$")
            plot(v=a[:, i], name=f"$a_{i}$")
            plot(v=u[:, i] + a[:, i], name=f"$tot$", ref=ref_u[:, i], color="tab:grey")
        for i in range(y.shape[-1]):
            plt.subplot(y.shape[-1], 4, 4 * i + 4)
            plot(v=y[:, i], name=f"$y_{i}$", ref=ref_y[:, i])
        plt.show()
