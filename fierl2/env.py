from systems import FDSSM
from flax import nnx
from utils import *


class Simulator(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        ref_y: Float[Array, "#y"] | float = 1.0,
        ref_u: Float[Array, "#u"] | float = 0.0,
        ref_x: Float[Array, "#x"] | float = 0.0,
        ref_z: Float[Array, "#z"] | float = 0.0,
        Qx: Float[Array, "x x"] | Float[Array, "x"] | float = 1e-2,
        Qz: Float[Array, "x x"] | Float[Array, "x"] | float = 1e-2,
        R: Float[Array, "y y"] | Float[Array, "y"] | float = 1e-2,
        Jy: Float[Array, "x x"] | Float[Array, "x"] | float = 1.0,
        Ju: Float[Array, "u u"] | Float[Array, "u"] | float = 1.0,
        Jx: Float[Array, "x x"] | Float[Array, "x"] | float = 0.0,
        Jz: Float[Array, "z z"] | Float[Array, "z"] | float = 0.0,
        Ja: Float[Array, "u u"] | Float[Array, "u"] | float = 1.0,
        Jex: Float[Array, "x x"] | Float[Array, "x"] | float = 0.0,
        Jez: Float[Array, "z z"] | Float[Array, "z"] | float = 1.0,
        mpc_horizon: int = 16,
        mpc_discount: float = 0.9,
        mpc_integral_action: bool = True,
        t_aux_on: int = 0,
        t_aux_off: int = 0,
        *,
        rngs: nnx.Rngs,
    ):
        # sim params and state
        self.rngs = rngs
        self.t = nnx.Variable(0)
        self.t_aux_on = t_aux_on
        self.t_aux_off = t_aux_off

        # costs and references
        self.Jy = nnx.Param(jnp.asarray(Jy))
        self.Ju = nnx.Param(jnp.asarray(Ju))
        self.Jx = nnx.Param(jnp.asarray(Jx))
        self.Jz = nnx.Param(jnp.asarray(Jz))
        self.Ja = nnx.Param(jnp.asarray(Ja))
        self.Jez = nnx.Param(jnp.asarray(Jez))
        self.Jex = nnx.Param(jnp.asarray(Jex))
        self.ref_y = nnx.Param(ref_y)
        self.ref_u = nnx.Param(ref_u)
        self.ref_x = nnx.Param(ref_x)
        self.ref_z = nnx.Param(ref_z)

        # system params and state
        self.sys = sys
        self.z = nnx.Variable(jnp.zeros((sys.z_dim,)))
        self.x = nnx.Variable(jnp.zeros((sys.x_dim,)))

        # mpc params and state
        self.mpc_horizon = mpc_horizon
        self.mpc_discount = mpc_discount
        self.mpc_integral_action = mpc_integral_action
        self.ut = nnx.Variable(jnp.zeros((mpc_horizon, sys.u_dim)))

        # kalman params and state
        self.Qz = nnx.Param(jnp.eye(sys.z_dim) * Qz)
        self.Qx = nnx.Param(jnp.eye(sys.x_dim) * Qx)
        self.R = nnx.Param(jnp.eye(sys.y_dim) * R)
        self.x_hat = nnx.Variable(jnp.zeros((sys.z_dim + sys.x_dim,)))
        self.P = nnx.Variable(jnp.eye(sys.z_dim + sys.x_dim))

    @property
    def obs_dim(self):
        return self.obs().shape[-1]

    @property
    def a_dim(self):
        return self.sys.u_dim

    def reset(self):
        # reset system
        self.z.value, self.x.value = self.sys.reset(self.rngs())
        self.t.value = 0
        # reset kalman filter
        self.x_hat.value = jnp.concat(self.sys.reset(rng=None), axis=-1)
        self.P.value = jnp.eye(self.x_hat.shape[-1])
        # reset mpc
        self.ut.value = jnp.broadcast_to(self.ref_u.value, self.ut.shape)

    def obs(self) -> Float[Array, "..."]:
        t = jnp.array([self.t.value])
        x_hat = self.x_hat.value
        Psqrt = jnp.linalg.cholesky(self.P.value).flatten()
        return jnp.concat([t, x_hat, Psqrt], axis=-1)

    def step(self, a: Float[Array, "u"] | None):
        # return initial state as info
        z, x = self.z.value, self.x.value  # hidden state
        x_hat, P = self.x_hat.value, self.P.value  # observable state

        # the actual simulation step
        u = self.mpc_control()
        a = a if a is not None else jnp.zeros_like(u)
        utot = u + a
        y = self.system_step(utot)
        self.kalman_update(utot, y)
        return dict(z=z, x=x, x_hat=x_hat, P=P, u=u, a=a, y=y)

    def rewards(self, outs: dict):
        y, u, a, z, x, x_hat = (outs[k] for k in ["y", "u", "a", "z", "x", "x_hat"])
        est_z, est_x = jnp.split(x_hat, [self.sys.z_dim], axis=-1)
        c = self.control_cost(zt=z, xt=x, ut=u, yt=y)
        ca = self.quadratic_cost(a, self.Ja.value)
        cex = self.quadratic_cost(est_x - x, self.Jex.value)
        cez = self.quadratic_cost(est_z - z, self.Jez.value)
        return -(c + ca + cex + cez)

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
            a = policy.sample(obs)
            out = env.step(a)
            return (env, policy), (obs, a, out)

        self.reset()
        T1 = jnp.arange(0, self.t_aux_on)
        self, outs1 = aux_off_step(self, T1)

        T2 = jnp.arange(self.t_aux_on, self.t_aux_off)
        (self, policy), (obs, a, outs2) = aux_on_step((self, policy), T2)
        r = self.rewards(outs2)
        next_obs = jnp.roll(obs, -1).at[-1].set(self.obs())

        T3 = jnp.arange(self.t_aux_off, episode_length)
        self, outs3 = aux_off_step(self, T3)
        # extra_reward = self.rewards(outs3).sum()
        # r = r.at[-1].add(extra_reward)

        outs = jax.tree.map(lambda *x: jnp.concat(x), outs1, outs2, outs3)
        rollout = Rollout(obs=obs, a=a, r=r, next_obs=next_obs)
        return rollout, outs

    @staticmethod
    def quadratic_cost(
        s: Float[Array, "t d"],
        J: Float[Array, "d d"] | Float[Array, "d"] | Float[Array, ""],
        discount: float = 1.0,
    ) -> Float[Array, "t"]:
        gamma = (discount or 1.0) ** jnp.arange(len(s))
        if J.ndim == 2:
            return jnp.einsum("ti, ij, tj, t-> t", s, J, s, gamma)
        return (J * (s**2) * gamma[:, None]).sum(axis=-1)

    def control_cost(
        self,
        zt: Float[Array, "t z"],
        xt: Float[Array, "t x"],
        ut: Float[Array, "t u"],
        yt: Float[Array, "t y"],
        u0: Float[Array, "u"] = jnp.zeros(()),
        discount: float = 1.0,
    ) -> Float[Array, "t"]:
        c_z = self.quadratic_cost(zt - self.ref_z, self.Jz.value, discount)
        c_x = self.quadratic_cost(xt - self.ref_x, self.Jx.value, discount)
        if self.mpc_integral_action:
            ut = ut.at[1:].add(-ut[:-1]).at[0].add(-u0)
        c_u = self.quadratic_cost(ut - self.ref_u, self.Ju.value, discount)
        c_y = self.quadratic_cost(yt - self.ref_y, self.Jy.value, discount)
        return c_z + c_x + c_u + c_y

    def mpc_control(self) -> Float[Array, "u"]:
        # trajectory fn
        def trajectory(ut):
            def scan_fn(x_aug, u):
                z, x = jnp.split(x_aug, [self.sys.z_dim])
                z, x, y = self.sys(z, x, u, rng=None)
                x_aug = jnp.concatenate([z, x])
                return x_aug, (z, x, y)

            _, (zt, xt, yt) = jax.lax.scan(scan_fn, self.x_hat.value, ut)
            return zt, xt, yt

        # mpc cost over trajectory
        def cost(ut_flat):
            ut = ut_flat.reshape(self.ut.shape)
            zt, xt, yt = trajectory(ut)
            c = self.control_cost(zt, xt, ut, yt, self.ut[0], self.mpc_discount)
            return c.sum()

        # second order optimization step
        ut = jnp.roll(self.ut.value, -1).at[-1].set(self.ref_u)
        ut_flat = ut.flatten()
        H = jax.hessian(cost)(ut_flat)
        J = jax.grad(cost)(ut_flat)
        self.ut.value = jnp.linalg.solve(a=H, b=H @ ut_flat - J).reshape(ut.shape)
        u = self.ut.value[0]
        return u

    def kalman_update(self, u: Float[Array, "u"], y: Float[Array, "y"]):
        # linearize sys step
        def aug_step(x_hat):
            z, x = jnp.split(x_hat, [self.sys.z_dim])
            z, x, y = self.sys(z, x, u, rng=None)
            return jnp.concatenate([z, x]), y

        A, C = jax.jacobian(aug_step)(self.x_hat.value)
        dx, dy = aug_step(self.x_hat)
        dx = dx - A @ self.x_hat
        dy = dy - C @ self.x_hat

        # a posteriori update
        K = self.P @ C.T @ jnp.linalg.inv(C @ self.P @ C.T + self.R)
        self.x_hat.value = self.x_hat + K @ (y - C @ self.x_hat - dy)
        self.P.value = self.P - K @ C @ self.P

        # a priori update
        Zero = jnp.zeros((self.sys.z_dim, self.sys.z_dim))
        Q = jnp.block([[self.Qz.value, Zero], [Zero.T, self.Qx.value]])
        self.x_hat.value = A @ self.x_hat + dx
        self.P.value = A @ self.P @ A.T + Q

    def system_step(self, u_total: Float[Array, "u"]) -> Float[Array, "y"]:
        z, x = self.z.value, self.x.value
        z, x, y = self.sys(z, x, u_total, rng=self.rngs())
        self.z.value, self.x.value = z, x
        self.t.value += 1
        return y

    def render(self, outs: dict, title=""):
        import matplotlib.pyplot as plt

        z, x, x_hat, P, u, a, y = (
            outs[k] for k in ["z", "x", "x_hat", "P", "u", "a", "y"]
        )
        ref_y = jnp.broadcast_to(self.ref_y.value, y.shape)
        ref_u = jnp.broadcast_to(self.ref_u.value, u.shape)
        ref_x = jnp.broadcast_to(self.ref_x.value, x.shape)
        ref_z = jnp.broadcast_to(self.ref_z.value, z.shape)

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
            est = (x_hat[:, i], P[:, i, i])
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
