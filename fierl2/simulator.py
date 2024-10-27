from dataclasses import InitVar
from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax import Array
from systems import DSSM

KeyArray = Array
SqrMatrix = Array


class Gaussian(eqx.Module):
    mean: Array
    cov: Array

    def __call__(self, *, key: KeyArray, **kwargs) -> Array:
        return jr.multivariate_normal(key, self.mean, self.cov, **kwargs)

    def log_prob(self, x: Array):
        return jnp.array(
            jax.scipy.stats.multivariate_normal.logpdf(x, self.mean, self.cov)
        )


class KalmanFilter(eqx.Module):
    system: DSSM
    Q: SqrMatrix
    R: SqrMatrix
    Je: SqrMatrix

    def reset_est(
        self, mean=jnp.zeros(()), cov=jnp.ones(()), *, key: KeyArray
    ) -> Gaussian:
        return Gaussian(
            mean=mean * jnp.ones(self.system.x_dim),
            cov=cov * jnp.eye(self.system.x_dim),
        )

    def update_est(self, est: Gaussian, u: Array, y: Array) -> Gaussian:
        A, B, C, D, dx, dy = self.system.linearize(est.mean, u)
        # a priori update
        est = Gaussian(
            mean=A @ est.mean + B @ u + dx,
            cov=A @ est.cov @ A.T + self.Q,
        )
        # a posteriori update
        K = est.cov @ C.T @ jnp.linalg.inv(C @ est.cov @ C.T + self.R)
        y_pred = C @ est.mean + D @ u + dy
        return Gaussian(
            mean=est.mean + K @ (y - y_pred),
            cov=est.cov - K @ C @ est.cov,
        )


class MPC(eqx.Module):
    system: DSSM
    horizon: int
    Jy: SqrMatrix
    Ju: SqrMatrix
    Jx: SqrMatrix

    def windows(self, reference: Array) -> Array:
        ref = jnp.pad(reference, ((0, self.horizon), (0, 0)), mode="edge")
        windows = [ref[i : i + self.horizon] for i in range(len(ref) - self.horizon)]
        return jnp.stack(windows)

    def reset_ut(self, ut=jnp.zeros(()), *, key: KeyArray) -> Array:
        return ut * jnp.ones((self.horizon, self.system.u_dim))

    def update_ut(
        self, ut: Array, x0: Array, y_ref: Array, u_ref: Array, x_ref: Array
    ) -> Array:
        assert self.horizon == len(ut) == len(y_ref) == len(u_ref) == len(x_ref)
        Jy = jnp.kron(jnp.eye(self.horizon), self.Jy)
        Ju = jnp.kron(jnp.eye(self.horizon), self.Ju)
        Jx = jnp.kron(jnp.eye(self.horizon), self.Jx)

        ut = jnp.roll(ut, -1, axis=0).at[-1].set(ut[-1])
        A, B, C, D, dx, dy = self.system.linearize_on_trajectory(x0, ut)

        return jnp.linalg.solve(
            a=D.T @ Jy @ D + Ju + B.T @ Jx @ B,
            b=(
                D.T @ Jy @ (y_ref.flatten() - C @ x0 - dy)
                + Ju @ u_ref.flatten()
                + B.T @ Jx @ (x_ref.flatten() - A @ x0 - dx)
            ),
        ).reshape(ut.shape)


class SimStep(NamedTuple):
    x: Array
    est: Gaussian
    u: Array
    a: Array
    log_p: Array
    y: Array
    r: Array = jnp.zeros(())

    def set_reward(self, Jy, Ju, Jx, Je, y_ref, u_ref, x_ref):
        norm = lambda x, J: x.T @ J @ x
        cost_y = norm(self.y - y_ref, Jy)
        cost_u = norm(self.u + self.a - u_ref, Ju)
        cost_x = norm(self.x - x_ref, Jx)
        cost_e1 = norm(self.est.mean - self.x, Je)
        cost_e2 = jnp.trace(Je @ self.est.cov, axis1=-1, axis2=-2)
        return self._replace(r=-(cost_y + cost_u + cost_x + cost_e1 + cost_e2))


class Policy(eqx.Module):
    mlp: eqx.nn.MLP
    gamma: float = 0.99
    lambd: float = 0.9
    eps: float = 0.3

    def __init__(self, x_dim, u_dim, *args, key: KeyArray, **kwargs):
        self.mlp = eqx.nn.MLP(
            x_dim + x_dim**2, 2 * u_dim + 1, *args, **kwargs, key=key
        )

    def __call__(self, est: Gaussian):
        x = jnp.concat([est.mean.flatten(), est.cov.flatten()])
        x = self.mlp(x)
        (mean, std), V = jnp.split(x[:-1], 2, axis=-1), x[-1]
        pi = Gaussian(mean=mean, cov=jnp.diag(std**2))
        return pi, V

    def sample(self, est: Gaussian, *, key: KeyArray) -> tuple[Array, Array]:
        pi = self(est)[0]
        a = pi(key=key)
        return a, pi.log_prob(a)

    def loss(self, steps: SimStep) -> Array:
        policy, V = eqx.filter_vmap(self)(steps.est)

        def discount(x, g):
            x_disc = jnp.convolve(g ** jnp.arange(len(x)), x, mode="full")
            return x_disc[: len(x)]

        # compute value estimate loss
        V_monte_carlo = discount(steps.r, self.gamma)
        loss_V = (V_monte_carlo - V) ** 2

        # compute ppo loss
        A_est = steps.r[:-1] + self.gamma * V[1:] - V[:-1]
        A_est = discount(A_est, self.gamma * self.lambd)
        ratio_pi = jnp.exp(policy.log_prob(steps.a)[:-1] - steps.log_p[:-1])
        ratio_pi_clipped = jnp.clip(ratio_pi, 1 - self.eps, 1 + self.eps)
        loss_pi = -jnp.minimum(ratio_pi * A_est, ratio_pi_clipped * A_est)
        return loss_pi.mean() + loss_V.mean()


class Simulator(eqx.Module):
    system: DSSM
    mpc: MPC
    observer: KalmanFilter
    policy: Policy

    @property
    def cost_matrices(self):
        return self.mpc.Jy, self.mpc.Ju, self.mpc.Jx, self.observer.Je

    def __init__(
        self,
        system: DSSM,
        mpc_horizon: int,
        Jy=jnp.ones(()),
        Ju=jnp.zeros(()) + 1e-8,
        Jx=jnp.zeros(()),
        Je=jnp.ones(()),
        Q=jnp.ones(()),
        R=jnp.ones(()),
        mlp_hidden=256,
        mlp_depth=2,
        *,
        key: KeyArray,
    ):
        self.system = system
        Jy = Jy * jnp.eye(self.system.y_dim)
        Ju = Ju * jnp.eye(self.system.u_dim)
        Jx = Jx * jnp.eye(self.system.x_dim)
        self.mpc = MPC(self.system, mpc_horizon, Jy=Jy, Ju=Ju, Jx=Jx)
        Q = Q * jnp.eye(self.system.y_dim)
        R = R * jnp.eye(self.system.u_dim)
        Je = Je * jnp.eye(self.system.x_dim)
        self.observer = KalmanFilter(self.system, Q, R, Je=Je)
        self.policy = Policy(
            self.system.x_dim, self.system.u_dim, mlp_hidden, mlp_depth, key=key
        )

    def rollout(
        self, y_ref: Array, u_ref: Array, x_ref: Array, *, key: KeyArray
    ) -> SimStep:
        # prepare inputs
        assert len(y_ref) == len(u_ref) == len(x_ref)
        key_x0, key_est0, key_ut0, key_steps = jr.split(key, 4)
        keys_steps = jr.split(key_steps, len(y_ref))
        y_ref, u_ref, x_ref = map(self.mpc.windows, (y_ref, u_ref, x_ref))
        inputs = keys_steps, y_ref, u_ref, x_ref

        # initial state
        x = self.system.reset_state(key=key_x0)
        est = self.observer.reset_est(key=key_est0)
        ut = self.mpc.reset_ut(key=key_ut0)
        carry = x, est, ut

        # simulation loop
        def sim_step(carry, inputs):
            x, est, ut = carry
            key, y_ref, u_ref, x_ref = inputs

            key_step, key_act = jr.split(key)
            ut = self.mpc.update_ut(ut, est.mean, y_ref, u_ref, x_ref)
            u = ut[0]
            a, log_p = self.policy.sample(est, key=key_act)

            x, y = self.system.step(x, u + a, key=key_step)
            est = self.observer.update_est(est, u + a, y)

            out = SimStep(x, est, u, a, log_p, y).set_reward(
                *self.cost_matrices, y_ref[0], u_ref[0], x_ref[0]
            )
            return (x, est, ut), out

        _, rollout = jax.lax.scan(sim_step, carry, inputs)
        return rollout
