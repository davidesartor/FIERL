from typing import NamedTuple, Any, Protocol
from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr

from modules.systems import FDSSM, SysModule
from modules.controllers import MPC, ControlCostMatrices, Signals, quadratic_cost
from modules.observers import ExtendedKalmanFilter
from flax import nnx
import utils


class AuxCostMatrices(NamedTuple):
    a: Float[Array, "a a"] | Float[Array, "a"] | float
    z: Float[Array, "z z"] | Float[Array, "z"] | float
    x: Float[Array, "x x"] | Float[Array, "x"] | float


class EOGReward(nnx.Module):
    def __init__(self, sys: FDSSM, eps: float, *, rngs: nnx.Rngs):
        self.sys = sys
        self.rngs = rngs
        self.eps = eps
        self.zs = nnx.Variable(jnp.zeros((2 * sys.z_dim, sys.z_dim)))
        self.xs = nnx.Variable(jnp.zeros((2 * sys.z_dim, sys.x_dim)))
        self.gramian = nnx.Variable(jnp.zeros((sys.z_dim, sys.z_dim)))
        self.determinant = nnx.Variable(jnp.linalg.det(self.gramian.value))

    def reset(self, z: Float[Array, "z"], x: Float[Array, "x"]):
        n = self.sys.z_dim
        Eps = jnp.eye(n) * self.eps
        self.gramian.value = jnp.zeros_like(self.gramian.value)
        self.xs.value = jnp.broadcast_to(x, self.xs.shape)
        self.zs.value = (
            jnp.broadcast_to(z, self.zs.shape).at[:n].add(Eps).at[n:].add(-Eps)
        )

    def step(self, u: Float[Array, "u"]):
        step = jax.vmap(lambda z, x: self.sys(z, x, u, w=None))
        zs, xs = self.zs.value, self.xs.value
        zs, xs, ys = step(zs, xs)
        self.zs.value, self.xs.value = zs, xs

        yp, ym = jnp.split(ys, 2)
        self.gramian.value += (yp - ym) @ (yp - ym).T
        new_determinant = jnp.linalg.det(self.gramian.value)
        reward = new_determinant - self.determinant.value
        self.determinant.value = new_determinant
        return reward


class Simulator(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        t_aux_on: int,
        t_aux_off: int,
        ref=Signals(y=1.0, u=0.0, x=0.0, z=0.0),
        Jcontrol=ControlCostMatrices(y=1.0, u=1.0, x=0.0, z=0.0),
        Jaux=AuxCostMatrices(a=0.1, z=1.0, x=0.0),
        reward_type: str = "quadratic",
        epsilon: float = 1e-2,
        kalman_params: dict = dict(Qz=1e-8, Qx=1e-2, R=1e-2),
        mpc_params: dict = dict(
            horizon=16, discount=0.9, newton_iters=1, integral_action=True
        ),
        *,
        rngs: nnx.Rngs,
    ):
        # sim params and state
        self.reward_type = reward_type
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
        self.mpc = MPC(sys, Jcontrol, **mpc_params)
        self.kalman = ExtendedKalmanFilter(sys, **kalman_params)
        self.reward_generator = EOGReward(sys, eps=epsilon, rngs=rngs)

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
        self.reward_generator.reset(self.sys.z.value, self.sys.x.value)

    def obs(self) -> Float[Array, "..."]:
        t = jnp.array([self.t.value])
        return jnp.concat([t, self.kalman.flat_state()], axis=-1)

    def step(self, a: Float[Array, "u"]):
        # return initial state as info
        ut_prev = self.mpc.ut.value
        z, x = self.sys.z.value, self.sys.x.value  # hidden state
        z_hat, x_hat = (self.kalman.z.value, self.kalman.x.value)
        P = self.kalman.P.value

        # the actual simulation step
        u = self.mpc.step(z0=z_hat, x0=x_hat, ref=self.ref.value)
        utot = u + a * (self.t.value >= self.t_aux_on) * (self.t.value < self.t_aux_off)
        y = self.sys.step(utot)
        self.kalman.step(utot, y)
        self.t.value += 1

        # compute reward and cost
        info = dict(z=z, x=x, z_hat=z_hat, x_hat=x_hat, P=P, u=u, a=a, y=y)
        if self.reward_type == "quadratic":
            reward = -(
                +quadratic_cost(z_hat - z, self.Jaux.value.z)
                + quadratic_cost(x_hat - x, self.Jaux.value.x)
            )
        else:
            reward = self.reward_generator.step(utot)

        ref_u = ut_prev[0] if self.mpc.integral_action else self.ref.u
        cy = quadratic_cost(y - self.ref.y, self.Jcontrol.value.y)
        cu = quadratic_cost(u - ref_u, self.Jcontrol.value.u)
        cx = quadratic_cost(x - self.ref.x, self.Jcontrol.value.x)
        cz = quadratic_cost(z - self.ref.z, self.Jcontrol.value.z)
        ca = quadratic_cost(a, self.Jaux.value.a)
        cost = cy + cu + cx + cz + ca
        return reward, cost, info

    def render(self, infos: dict, title=""):
        import matplotlib.pyplot as plt

        z, x, z_hat, x_hat, P, u, a, y = (
            infos[k] for k in ["z", "x", "z_hat", "x_hat", "P", "u", "a", "y"]
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
