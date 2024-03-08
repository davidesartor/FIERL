from typing import NamedTuple, Optional
import jax
import jax.numpy as jnp
import systems


class GaussianEstimate(NamedTuple):
    mean: jax.Array
    cov: jax.Array


class KalmanFilter(systems.DSSM):
    @classmethod
    def from_dssm(cls, dssm: systems.DSSM):
        return cls(**dssm.__dict__)

    def init_estimate(
        self, mean_x0: jax.Array | float = 0.0, std_x0: jax.Array | float = 1.0
    ) -> GaussianEstimate:
        if isinstance(mean_x0, (int, float)):
            mean_x0 = mean_x0 * jnp.ones(self.state_dim)
        if isinstance(std_x0, (int, float)):
            std_x0 = std_x0 * jnp.ones(self.state_dim)
        return GaussianEstimate(mean_x0, jnp.diag(std_x0**2))

    def update_a_priori(
        self,
        estimate: GaussianEstimate,
        A: jax.Array,
        B: jax.Array,
        noise_std_x: jax.Array,
        u: jax.Array,
    ) -> GaussianEstimate:
        return GaussianEstimate(
            mean=A @ estimate.mean + B @ u,
            cov=A @ estimate.cov @ A.T + jnp.diag(noise_std_x**2),
        )

    def update_a_posteriori(
        self,
        estimate: GaussianEstimate,
        C: jax.Array,
        noise_std_y: jax.Array,
        y: jax.Array,
    ) -> GaussianEstimate:
        K = estimate.cov @ C.T @ jnp.linalg.inv(C @ estimate.cov @ C.T + jnp.diag(noise_std_y**2))
        return GaussianEstimate(
            mean=estimate.mean + K @ (y - C @ estimate.mean),
            cov=(jnp.eye(estimate.cov.shape[0]) - K @ C) @ estimate.cov,
        )

    def update(
        self, estimate: GaussianEstimate, u: jax.Array, y: jax.Array
    ) -> tuple[GaussianEstimate, GaussianEstimate]:
        estimate_after_measurement = self.update_a_posteriori(estimate, self.C, self.noise_std_y, y)
        estimate_next_a_priori = self.update_a_priori(
            estimate_after_measurement, self.A, self.B, self.noise_std_x, u
        )
        return estimate_next_a_priori, estimate_after_measurement

    def simulate(
        self,
        u_t: jax.Array,
        y_t: jax.Array,
        mean_x0: jax.Array | float = 0.0,
        std_x0: jax.Array | float = 1.0,
    ) -> GaussianEstimate:

        initial_estimate = self.init_estimate(mean_x0, std_x0)
        _, estimates_a_posteriori = jax.lax.scan(
            lambda est, u_y: self.update(est, *u_y), initial_estimate, (u_t, y_t)
        )
        return estimates_a_posteriori


class FaultObserver(KalmanFilter):
    fault_evol_std: jax.Array

    @classmethod
    def from_dssm(
        cls,
        dssm: systems.DSSM,
        fault_evol_std: jax.Array | float = 0.0,
    ):
        if isinstance(fault_evol_std, (int, float)):
            fault_evol_std = fault_evol_std * jnp.ones(dssm.input_dim)
        return cls(**dssm.__dict__, fault_evol_std=fault_evol_std)

    def init_estimate(
        self, mean_x0: jax.Array | float = 0.0, std_x0: jax.Array | float = 1.0
    ) -> GaussianEstimate:
        if isinstance(mean_x0, (int, float)):
            mean_x0 = mean_x0 * jnp.ones(self.state_dim)
        if isinstance(std_x0, (int, float)):
            std_x0 = std_x0 * jnp.ones(self.state_dim)
        mean_z0 = jnp.ones(self.input_dim)
        std_z0 = jnp.ones(self.input_dim)
        return super().init_estimate(
            jnp.concatenate([mean_x0, mean_z0]), jnp.concatenate([std_x0, std_z0])
        )

    def update(self, estimate: GaussianEstimate, u: jax.Array, y: jax.Array):
        A_bar = [
            [self.A, self.B @ jnp.diag(u)],
            [jnp.zeros_like(self.B.T), jnp.eye(self.input_dim)],
        ]
        B_bar = jnp.zeros((self.state_dim + self.input_dim, self.input_dim))
        C_bar = [self.C, jnp.zeros((self.output_dim, self.input_dim))]
        noise_std_x_bar = jnp.concatenate([self.noise_std_x, self.fault_evol_std])

        estimate_after_measurement = self.update_a_posteriori(
            estimate, jnp.block(C_bar), self.noise_std_y, y
        )
        estimate_next_a_priori = self.update_a_priori(
            estimate_after_measurement, jnp.block(A_bar), B_bar, noise_std_x_bar, u
        )
        return estimate_next_a_priori, estimate_after_measurement

    def split(self, estimate: GaussianEstimate) -> tuple[GaussianEstimate, GaussianEstimate]:
        state_estimate = GaussianEstimate(
            estimate.mean[: self.state_dim], estimate.cov[: self.state_dim, : self.state_dim]
        )
        fault_estimate = GaussianEstimate(
            estimate.mean[self.state_dim :], estimate.cov[self.state_dim :, self.state_dim :]
        )
        return state_estimate, fault_estimate
