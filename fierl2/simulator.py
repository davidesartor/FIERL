from utils import *
from systems import FDSSM, SysModule
from controllers import MPC, ControlCostMatrices, References
from observers import EKF
from flax import nnx


class AuxCostMatrices(NamedTuple):
    z: Float[Array, "z z"] | Float[Array, "z"] | float
    x: Float[Array, "x x"] | Float[Array, "x"] | float


class Simulator(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        t_aux_on: int = 32,
        t_aux_off: int = 96,
        episode_length: int = 128,
        ref=References(y=1.0, u=0.0, a=0.0, x=0.0, z=0.0),
        Jcontrol=ControlCostMatrices(y=1.0, u=1.0, a=0.1, x=0.0, z=0.0),
        Jaux: AuxCostMatrices | None = AuxCostMatrices(z=1.0, x=0.1),
        obs_gramian_eps: Float[Array, "z"] | float | None = None,
        kalman_params: dict = dict(Qz=1e-8, Qx=1e-2, R=1e-2, mc_samples_init=16),
        mpc_params: dict = dict(horizon=16, discount=0.9, mc_samples_traj=0),
        rngs: nnx.Rngs = nnx.Rngs(0),
    ):
        assert (Jaux is None) or (obs_gramian_eps is None)
        assert (Jaux is not None) or (obs_gramian_eps is not None)

        # sim params and state
        self.t_aux_on = t_aux_on
        self.t_aux_off = t_aux_off
        self.episode_length = episode_length

        # costs rewards and references
        self.ref = nnx.Param(ref)
        self.Jcontrol = nnx.Param(Jcontrol)
        self.Jaux = None if Jaux is None else nnx.Param(Jaux)

        # modules and state
        self.rngs = rngs
        self.t = nnx.Variable(0)
        self.sys = SysModule(sys, rngs=rngs)
        self.mpc = MPC(sys, Jcontrol, **mpc_params, rngs=rngs)
        self.kalman = EKF(sys, **kalman_params, rngs=rngs)

        # auxiliary systems
        self.sys_plus_eps = [SysModule(sys, rngs=rngs) for _ in range(sys.z_dim)]
        self.sys_minus_eps = [SysModule(sys, rngs=rngs) for _ in range(sys.z_dim)]
        self.obs_gramian_eps = (
            obs_gramian_eps if obs_gramian_eps is None else nnx.Param(obs_gramian_eps)
        )

    @property
    def obs_dim(self):
        return self.obs().shape[-1]

    @property
    def a_dim(self):
        return self.sys.sys.u_dim

    def reset(self):
        self.t.value = 0
        self.sys.reset(deterministic=False)
        self.mpc.reset(deterministic=True)
        self.kalman.reset(deterministic=True)

        if self.obs_gramian_eps is not None:
            for i, sys in enumerate(self.sys_plus_eps):
                sys.reset(deterministic=True)
                sys.z.value = self.sys.z.value.at[i].add(self.obs_gramian_eps.value)
            for i, sys in enumerate(self.sys_minus_eps):
                sys.reset(deterministic=True)
                sys.z.value = self.sys.z.value.at[i].add(-self.obs_gramian_eps.value)

    def obs(self) -> Float[Array, "..."]:
        return self.kalman.flat_state()
        t = jnp.array([self.t.value])
        return jnp.concat([t, self.kalman.flat_state()], axis=-1)

    def step(self, a: Float[Array, "u"]):
        # return initial state as info
        z, x = self.sys.z.value, self.sys.x.value  # hidden state
        z_hat, x_hat = (self.kalman.z.value, self.kalman.x.value)
        P = self.kalman.P.value

        # the actual simulation step
        u = self.mpc.step(z0=z_hat, x0=x_hat, ref=self.ref.value)
        utot = u + a

        w = self.sys.sys.sample_w(rng=self.rngs())
        y = self.sys.step(utot, w=w)
        yp = [sys.step(utot, w=w) for sys in self.sys_plus_eps]
        ym = [sys.step(utot, w=w) for sys in self.sys_minus_eps]
        self.kalman.step(utot, y)
        self.t.value += 1
        return dict(
            z=z, x=x, z_hat=z_hat, x_hat=x_hat, P=P, u=u, a=a, y=y, yp=yp, ym=ym
        )

    def rewards(self, outs: dict):
        if self.Jaux is not None:
            zt, xt, zt_hat, xt_hat, P = (
                outs[k] for k in ["z", "x", "z_hat", "x_hat", "P"]
            )
            reward = -(
                +quadratic_cost(zt_hat - zt, self.Jaux.value.z)
                + quadratic_cost(xt_hat - xt, self.Jaux.value.x)
            )
            # add trace term to penalize the expected error
            # z_dim, x_dim = self.sys.sys.z_dim, self.sys.sys.x_dim
            # J = [
            #     [jnp.eye(z_dim) * self.Jaux.value.z, jnp.zeros((z_dim, x_dim))],
            #     [jnp.zeros((x_dim, z_dim)), jnp.eye(x_dim) * self.Jaux.value.x],
            # ]
            # reward = reward - jnp.trace(P @ jnp.block(J), axis1=-2, axis2=-1)

        if self.obs_gramian_eps is not None:
            dy = jnp.stack([yp - ym for yp, ym in zip(outs["yp"], outs["ym"])])
            noise_prec = jnp.linalg.inv(self.kalman.R.value)
            empirical_obs_gramian = jnp.cumsum(
                jnp.einsum("ita, jtb, ab->tij", dy, dy, noise_prec), axis=0
            )
            det_G = jnp.linalg.det(empirical_obs_gramian) ** (1 / len(dy))
            reward = det_G.at[1:].add(-det_G[:-1])
        return reward

    def costs(self, outs: dict):
        zt, xt, ut, at, yt = (outs[k] for k in ["z", "x", "u", "a", "y"])

        ref_u = self.ref.u
        if self.mpc.integral_action:
            ref_u = jnp.roll(ut, 1).at[0].set(self.ref.u)
        cost = (
            +quadratic_cost(yt - self.ref.y, self.Jcontrol.value.y)
            + quadratic_cost(ut - ref_u, self.Jcontrol.value.u)
            + quadratic_cost(at - 0.0 * self.ref.u, self.Jcontrol.value.u)
            + quadratic_cost(xt - self.ref.x, self.Jcontrol.value.x)
            + quadratic_cost(zt - self.ref.z, self.Jcontrol.value.z)
        )
        return cost

    @nnx.jit
    def rollout(self, policy) -> tuple[Rollout, dict]:
        Ton = (self.t_aux_on + self.episode_length) % self.episode_length
        Toff = (self.t_aux_off + self.episode_length) % self.episode_length
        self.reset()

        @nnx.scan
        def steps(carry, t):
            env, policy = carry
            obs = env.obs()
            a, log_p = policy.sample(obs)
            out = env.step(a * (Ton < t) * (t < Toff))
            return (env, policy), (obs, a, log_p, out)

        (self, policy), (obs, a, log_p, outs) = steps(
            (self, policy), jnp.arange(self.episode_length)
        )
        next_obs = jnp.roll(obs, -1).at[-1].set(self.obs())
        rewards = self.rewards(outs)
        costs = self.costs(outs)
        rollout = Rollout(obs, a, log_p, rewards, next_obs, costs)

        rollout = jax.tree.map(lambda x: x[Ton:], rollout)
        return rollout, outs

    def render(self, outs: dict):
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
            Ton, Toff = self.t_aux_on, self.t_aux_off
            plt.hlines(0, 0, len(v), color="k", linestyle=":", alpha=0.5)
            if ref is not None:
                plt.plot(ref, "--", color="k", label="ref")
            if est is not None:
                mean, cov = est
                plt.plot(mean, label="est")
                if cov is not None:
                    low, high = mean - cov**0.5, mean + cov**0.5
                    plt.fill_between(t, low, high, alpha=0.5, color="tab:grey")
            plt.plot(v, label=name, color=color)
            plt.vlines([Ton, Toff], *plt.ylim(), colors=["g", "r"], alpha=0.5)
            plt.legend()
            plt.grid(True)

        fig = plt.figure(figsize=(20, 10))
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
        return fig
