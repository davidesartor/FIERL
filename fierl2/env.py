from utils import *
from modules.systems import FDSSM, SysModule
from modules.controllers import MPC, ControlCostMatrices, References
from modules.observers import ExtendedKalmanFilter
from flax import nnx


class AuxCostMatrices(NamedTuple):
    a: Float[Array, "a a"] | Float[Array, "a"] | float
    z: Float[Array, "z z"] | Float[Array, "z"] | float
    x: Float[Array, "x x"] | Float[Array, "x"] | float


class Simulator(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        t_aux_on: int = 0,
        t_aux_off: int = -1,
        ref=References(y=1.0, u=0.0, x=0.0, z=0.0),
        Jcontrol=ControlCostMatrices(y=1.0, du=0.01, u=0.01, x=0.0, z=0.0),
        Jaux=AuxCostMatrices(a=0.1, z=1.0, x=0.0),
        kalman_params: dict = dict(Qz=1e-8, Qx=1e-2, R=1e-2, mc_samples_init=16),
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
        self.Jaux = nnx.Param(Jaux)

        # modules and state
        self.rngs = rngs
        self.t = nnx.Variable(0)
        self.sys = SysModule(sys, rngs=rngs)
        self.mpc = MPC(sys, Jcontrol, **mpc_params, rngs=rngs)
        self.kalman = ExtendedKalmanFilter(sys, **kalman_params, rngs=rngs)

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

    def obs(self) -> Float[Array, "..."]:
        t = jnp.array([self.t.value])
        return jnp.concat([t, self.kalman.flat_state()], axis=-1)

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
            +quadratic_cost(z_hat - z, self.Jaux.value.z)
            + quadratic_cost(x_hat - x, self.Jaux.value.x)
        )
        cost = self.mpc.control_cost(zt=z, xt=x, ut=u, yt=y, ref=self.ref.value)
        cost = cost + quadratic_cost(a, self.Jaux.value.a)
        return reward, cost

    @nnx.jit(static_argnames=("episode_length",))
    def rollout(self, policy, episode_length: int) -> tuple[Rollout, dict]:
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

        self.reset()
        if policy is None:
            self, outs = aux_off_step(self, jnp.arange(episode_length))
            return None, outs  # type: ignore

        Ton = (self.t_aux_on + episode_length) % episode_length
        Toff = (self.t_aux_off + episode_length) % episode_length

        # phase 1: aux off
        T1 = jnp.arange(0, Ton)
        self, outs1 = aux_off_step(self, T1)

        # phase 2: aux on
        T2 = jnp.arange(Ton, Toff)
        (self, policy), (obs, a, log_p, outs2) = aux_on_step((self, policy), T2)
        next_obs = jnp.roll(obs, -1).at[-1].set(self.obs())
        rewards, costs = self.rewards_and_costs(outs2)

        # phase 3: aux off
        T3 = jnp.arange(Toff, episode_length)
        self, outs3 = aux_off_step(self, T3)

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
            Ton, Toff = self.t_aux_on, self.t_aux_off
            plt.hlines(0, 0, len(v), color="k", linestyle=":", alpha=0.5)
            plt.plot(v, label=name, color=color)
            if ref is not None:
                plt.plot(ref, "--", color="tab:red", label="ref")
            if est is not None:
                mean, cov = est
                plt.plot(mean, label="est")
                if cov is not None:
                    low, high = mean - cov**0.5, mean + cov**0.5
                    plt.fill_between(t, low, high, alpha=0.5, color="tab:grey")
            plt.vlines([Ton, Toff], *plt.ylim(), colors=["g", "r"], alpha=0.5)

            plt.legend()
            plt.grid(True)

        fig = plt.figure(figsize=(20, 10))
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
        return fig


class Simulator2(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        t_aux_on: int = 0,
        t_aux_off: int = -1,
        ref=References(y=1.0, u=0.0, x=0.0, z=0.0),
        Jcontrol=ControlCostMatrices(y=1.0, du=0.01, u=0.001, x=0.0, z=0.0),
        Jaux=AuxCostMatrices(a=1.0, z=0.0, x=0.0),
        epsilon: float = 1e-2,
        kalman_params: dict = dict(Qz=1e-8, Qx=1e-2, R=1e-2, mc_samples_init=16),
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
        self.Jaux = nnx.Param(Jaux)

        # modules and state
        self.rngs = rngs
        self.t = nnx.Variable(0)
        self.sys = SysModule(sys, rngs=rngs)
        self.epsilon = epsilon
        self.sys_plus_eps = [SysModule(sys, rngs=rngs) for _ in range(sys.z_dim)]
        self.sys_minus_eps = [SysModule(sys, rngs=rngs) for _ in range(sys.z_dim)]
        self.mpc = MPC(sys, Jcontrol, **mpc_params, rngs=rngs)
        self.kalman = ExtendedKalmanFilter(sys, **kalman_params, rngs=rngs)

    @property
    def obs_dim(self):
        return self.obs().shape[-1]

    @property
    def a_dim(self):
        return self.sys.sys.u_dim

    def reset(self):
        self.t.value = 0
        self.sys.reset(deterministic=False)
        for sys in self.sys_plus_eps:
            sys.reset(deterministic=True)
            sys.z.value = self.sys.z.value + self.epsilon
        for sys in self.sys_minus_eps:
            sys.reset(deterministic=True)
            sys.z.value = self.sys.z.value - self.epsilon
        self.mpc.reset(deterministic=True)
        self.kalman.reset(deterministic=True)

    def obs(self) -> Float[Array, "..."]:
        t = jnp.array([self.t.value])
        return jnp.concat([t, self.kalman.flat_state()], axis=-1)

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
        yp = [sys.step(utot) for sys in self.sys_plus_eps]
        ym = [sys.step(utot) for sys in self.sys_minus_eps]
        self.kalman.step(utot, y)
        self.t.value += 1
        return dict(
            z=z, x=x, z_hat=z_hat, x_hat=x_hat, P=P, u=u, a=a, y=y, yp=yp, ym=ym
        )

    def rewards_and_costs(self, outs: dict):
        z, x, u, a, y, yp, ym = (outs[k] for k in ["z", "x", "u", "a", "y", "yp", "ym"])

        dy = jnp.stack([yp - ym for yp, ym in zip(yp, ym)])
        noise_prec = jnp.linalg.inv(self.kalman.R.value)
        empirical_obs_gramian = jnp.cumsum(
            jnp.einsum("...ita,...jtb, ab->...tij", dy, dy, noise_prec), axis=0
        )
        det_G = jnp.linalg.det(empirical_obs_gramian)
        reward = det_G.at[1:].add(-det_G[:-1])
        cost = self.mpc.control_cost(zt=z, xt=x, ut=u, yt=y, ref=self.ref.value)
        cost = cost + quadratic_cost(a, self.Jaux.value.a)
        return reward, cost

    @nnx.jit(static_argnames=("episode_length",))
    def rollout(self, policy, episode_length: int) -> tuple[Rollout, dict]:
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

        self.reset()
        if policy is None:
            self, outs = aux_off_step(self, jnp.arange(episode_length))
            return None, outs  # type: ignore

        Ton = (self.t_aux_on + episode_length) % episode_length
        Toff = (self.t_aux_off + episode_length) % episode_length

        # phase 1: aux off
        T1 = jnp.arange(0, Ton)
        self, outs1 = aux_off_step(self, T1)

        # phase 2: aux on
        T2 = jnp.arange(Ton, Toff)
        (self, policy), (obs, a, log_p, outs2) = aux_on_step((self, policy), T2)
        next_obs = jnp.roll(obs, -1).at[-1].set(self.obs())
        rewards, costs = self.rewards_and_costs(outs2)

        # phase 3: aux off
        T3 = jnp.arange(Toff, episode_length)
        self, outs3 = aux_off_step(self, T3)

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
            Ton, Toff = self.t_aux_on, self.t_aux_off
            plt.hlines(0, 0, len(v), color="k", linestyle=":", alpha=0.5)
            plt.plot(v, label=name, color=color)
            if ref is not None:
                plt.plot(ref, "--", color="tab:red", label="ref")
            if est is not None:
                mean, cov = est
                plt.plot(mean, label="est")
                if cov is not None:
                    low, high = mean - cov**0.5, mean + cov**0.5
                    plt.fill_between(t, low, high, alpha=0.5, color="tab:grey")
            plt.vlines([Ton, Toff], *plt.ylim(), colors=["g", "r"], alpha=0.5)

            plt.legend()
            plt.grid(True)

        fig = plt.figure(figsize=(20, 10))
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
        return fig
