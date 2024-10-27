from typing import NamedTuple, Self
import jax
from jax import Array, numpy as jnp, random as jr
import equinox as eqx

from systems import DSSM


class Gaussian(NamedTuple):
    mean: Array
    cov: Array

    def sample(self, *, key, **kwargs):
        return jr.multivariate_normal(key, self.mean, self.cov, **kwargs)

    def expected_distance(self, point) -> Array:
        norm_delta = jnp.sum((point - self.mean) ** 2, axis=-1)
        trace_cov = jnp.trace(self.cov, axis1=-1, axis2=-2)
        return norm_delta + trace_cov


class KalmanFilter(eqx.Module):
    system: DSSM
    Q: Array
    R: Array

    def __init__(self, system: DSSM, Q=jnp.ones(()), R=jnp.ones(())):
        self.system = system
        self.Q = Q * jnp.eye(system.x_dim)
        self.R = R * jnp.eye(system.y_dim)

    def init_estimate(
        self, mean=jnp.zeros(()), cov=jnp.ones(()), *, key=None
    ) -> Gaussian:
        mean = mean * jnp.ones(self.system.x_dim)
        cov = cov * jnp.eye(self.system.x_dim)
        return Gaussian(mean, cov)

    def update(self, est: Gaussian, u: Array, y: Array) -> Gaussian:
        A, B, C, D, dx, dy = self.system.linearize(est.mean, u)
        # a priori update
        est = Gaussian(mean=A @ est.mean + B @ u + dx, cov=A @ est.cov @ A.T + self.Q)
        # a posteriori update
        K = est.cov @ C.T @ jnp.linalg.inv(C @ est.cov @ C.T + self.R)
        y_pred = C @ est.mean + D @ u + dy
        est = Gaussian(mean=est.mean + K @ (y - y_pred), cov=est.cov - K @ C @ est.cov)
        return est


class ReferenceSignals(NamedTuple):
    y: Array
    u: Array
    x: Array

    def __len__(self):
        assert len(self.y) == len(self.u) == len(self.x)
        return len(self.y)

    def stack_windows(self, size: int):
        def process_signal(x):
            x = jnp.pad(x, ((0, size), (0, 0)), mode="edge")
            x = [x[i : i + size] for i in range(len(x) - size)]
            return jnp.stack(x)

        return ReferenceSignals(*map(process_signal, self))


class MPC(eqx.Module):
    system: DSSM
    horizon: int
    Jy: Array
    Ju: Array
    Jx: Array

    def __init__(
        self,
        system: DSSM,
        horizon: int,
        Jy=jnp.ones(()),
        Ju=jnp.zeros(()),
        Jx=jnp.zeros(()),
        eps=1e-8,
    ):
        self.system = system
        self.horizon = horizon
        self.Jy = jnp.kron(jnp.eye(horizon), Jy * jnp.eye(system.y_dim))
        self.Ju = jnp.kron(jnp.eye(horizon), Ju * jnp.eye(system.u_dim))
        self.Jx = jnp.kron(jnp.eye(horizon), Jx * jnp.eye(system.x_dim))
        self.Ju = self.Ju + eps * jnp.eye(horizon * system.u_dim)

    def init_control(self, *, key=None) -> Array:
        ut = jnp.zeros((self.horizon, self.system.u_dim))
        return ut

    def optimal_control(
        self, ut: Array, x0: Array, ref: ReferenceSignals
    ) -> tuple[Array, Array]:
        Jy, Ju, Jx = self.Jy, self.Ju, self.Jx
        y_r, u_r, x_r = ref.y.flatten(), ref.u.flatten(), ref.x.flatten()
        A, B, C, D, dx, dy = self.system.linearize_on_trajectory(x0, ut)

        ut = jnp.linalg.solve(
            D.T @ Jy @ D + Ju + B.T @ Jx @ B,
            D.T @ Jy @ (y_r - C @ x0 - dy) + Ju @ u_r + B.T @ Jx @ (x_r - A @ x0 - dx),
        ).reshape(ut.shape)

        u = ut[0]
        ut = jnp.roll(ut, -1, axis=0).at[-1].set(ut[-1])
        return ut, u
