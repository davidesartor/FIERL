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
        self, mean_x0: jax.Array | float = 0.0, cov_x0: jax.Array | float = 1.0
    ) -> GaussianEstimate:
        if isinstance(mean_x0, (int, float)):
            mean_x0 = mean_x0 * jnp.ones(self.state_dim)
        if isinstance(cov_x0, (int, float)):
            cov_x0 = cov_x0 * jnp.eye(self.state_dim)
        return GaussianEstimate(mean_x0, cov_x0)

    def update_a_priori(
        self,
        estimate: GaussianEstimate,
        A: jax.Array,
        B: jax.Array,
        noise_cov_x: jax.Array,
        u: jax.Array,
    ) -> GaussianEstimate:
        if isinstance(noise_cov_x, (int, float)):
            noise_cov_x = noise_cov_x * jnp.eye(len(estimate.mean))
        return GaussianEstimate(
            mean=A @ estimate.mean + B @ u,
            cov=A @ estimate.cov @ A.T + noise_cov_x,
        )

    def update_a_posteriori(
        self,
        estimate: GaussianEstimate,
        C: jax.Array,
        noise_cov_y: jax.Array,
        y: jax.Array,
    ) -> GaussianEstimate:
        if isinstance(noise_cov_y, (int, float)):
            noise_cov_y = noise_cov_y * jnp.eye(len(y))
        K = estimate.cov @ C.T @ jnp.linalg.inv(C @ estimate.cov @ C.T + noise_cov_y)
        return GaussianEstimate(
            mean=estimate.mean + K @ (y - C @ estimate.mean),
            cov=(jnp.eye(estimate.cov.shape[0]) - K @ C) @ estimate.cov,
        )

    def update(
        self, estimate: GaussianEstimate, u: jax.Array, y: jax.Array
    ) -> tuple[GaussianEstimate, GaussianEstimate]:
        estimate_after_measurement = self.update_a_posteriori(
            estimate, self.C, self.noise_cov_y, y
        )
        estimate_next_a_priori = self.update_a_priori(
            estimate_after_measurement, self.A, self.B, self.noise_cov_x, u
        )
        return estimate_next_a_priori, estimate_after_measurement

    def simulate(
        self,
        u_t: jax.Array,
        y_t: jax.Array,
        initial_estimate=None,
    ) -> GaussianEstimate:
        if initial_estimate is None:
            initial_estimate = self.init_estimate()
        _, estimates_a_posteriori = jax.lax.scan(
            lambda est, u_y: self.update(est, *u_y), initial_estimate, (u_t, y_t)
        )
        return estimates_a_posteriori


class FaultObserver(KalmanFilter):
    fault_evol_cov: jax.Array
    adaptive: bool = False

    @classmethod
    def from_dssm(
        cls,
        dssm: systems.DSSM,
        fault_evol_cov: jax.Array | float = 0.0,
        adaptive: bool = False,
    ):
        if isinstance(fault_evol_cov, (int, float)):
            fault_evol_cov = fault_evol_cov * jnp.eye(dssm.input_dim)
        return cls(**dssm.__dict__, fault_evol_cov=fault_evol_cov, adaptive=adaptive)

    def init_estimate(
        self,
        mean_x0: jax.Array | float = 0.0,
        cov_x0: jax.Array | float = 1.0,
        mean_z0: jax.Array | float = 1.0,
        cov_z0: jax.Array | float = 1.0,
    ) -> GaussianEstimate:
        if isinstance(mean_x0, (int, float)):
            mean_x0 = mean_x0 * jnp.ones(self.state_dim)
        if isinstance(cov_x0, (int, float)):
            cov_x0 = cov_x0 * jnp.eye(self.state_dim)
        if isinstance(mean_z0, (int, float)):
            mean_z0 = mean_z0 * jnp.ones(self.input_dim)
        if isinstance(cov_z0, (int, float)):
            cov_z0 = cov_z0 * jnp.eye(self.input_dim)

        zeros = jnp.zeros((cov_x0.shape[0], cov_z0.shape[1]))
        return super().init_estimate(
            jnp.concatenate([mean_x0, mean_z0]),
            jnp.block([[cov_x0, zeros], [zeros.T, cov_z0]]),
        )

    def update(self, estimate: GaussianEstimate, u: jax.Array, y: jax.Array):
        A_bar = [
            [self.A, self.B @ jnp.diag(u)],
            [jnp.zeros_like(self.B.T), jnp.eye(self.input_dim)],
        ]
        B_bar = jnp.zeros((self.state_dim + self.input_dim, self.input_dim))
        C_bar = [self.C, jnp.zeros((self.output_dim, self.input_dim))]

        fault_evol_cov = self.fault_evol_cov
        if self.adaptive:
            fault_evol_cov = fault_evol_cov * jnp.diagflat(
                jnp.diag(estimate.cov[self.state_dim :, self.state_dim :]) < 1.0
            )

        zeros = jnp.zeros((self.noise_cov_x.shape[0], self.fault_evol_cov.shape[1]))
        noise_cov_x_bar = jnp.block(
            [[self.noise_cov_x, zeros], [zeros.T, fault_evol_cov]]
        )

        estimate_after_measurement = self.update_a_posteriori(
            estimate, jnp.block(C_bar), self.noise_cov_y, y
        )
        estimate_next_a_priori = self.update_a_priori(
            estimate_after_measurement, jnp.block(A_bar), B_bar, noise_cov_x_bar, u
        )
        return estimate_next_a_priori, estimate_after_measurement

    def split(
        self, estimate: GaussianEstimate
    ) -> tuple[GaussianEstimate, GaussianEstimate]:
        state_estimate = GaussianEstimate(
            estimate.mean[: self.state_dim],
            estimate.cov[: self.state_dim, : self.state_dim],
        )
        fault_estimate = GaussianEstimate(
            estimate.mean[self.state_dim :],
            estimate.cov[self.state_dim :, self.state_dim :],
        )
        return state_estimate, fault_estimate
