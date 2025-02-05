from systems import FDSSM
from utils import *
from flax import nnx


class EKF(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        Qz: Float[Array, "z z"] | Float[Array, "z"] | float,
        Qx: Float[Array, "x x"] | Float[Array, "x"] | float,
        R: Float[Array, "y y"] | Float[Array, "y"] | float,
        mc_samples_init: int = 16,
        *,
        rngs: nnx.Rngs,
    ):
        self.sys = sys
        self.mc_samples_init = mc_samples_init
        Zeros = jnp.zeros((sys.z_dim, sys.x_dim))
        Q = [
            [jnp.eye(sys.z_dim) * Qz, Zeros],
            [Zeros.T, jnp.eye(sys.x_dim) * Qx],
        ]
        self.Q = nnx.Param(jnp.block(Q))
        self.R = nnx.Param(jnp.eye(sys.y_dim) * R)

        self.rngs = rngs
        self.z = nnx.Variable(jnp.zeros((sys.z_dim,)))
        self.x = nnx.Variable(jnp.zeros((sys.x_dim,)))
        self.P = nnx.Variable(jnp.eye(sys.z_dim + sys.x_dim))

    def reset(self, deterministic=False):
        if deterministic:
            self.z.value = self.sys.sample_z(rng=None)
            self.x.value = self.sys.sample_x(rng=None)
            self.P.value = jnp.eye(self.P.value.shape[-1])
        else:
            z = jax.vmap(self.sys.sample_z)(jr.split(self.rngs(), self.mc_samples_init))
            x = jax.vmap(self.sys.sample_x)(jr.split(self.rngs(), self.mc_samples_init))
            self.z.value = jnp.mean(z, axis=0)
            self.x.value = jnp.mean(x, axis=0)
            x_aug = jnp.concatenate([z, x], axis=-1)
            self.P.value = jnp.cov(x_aug.T)

    def flat_state(self):
        x_aug = jnp.concatenate([self.z.value, self.x.value], axis=-1)
        Psqrt = jnp.linalg.cholesky(self.P.value).reshape(*self.P.shape[:-2], -1)
        return jnp.concatenate([x_aug, Psqrt], axis=-1)

    def step(self, u: Float[Array, "u"], y: Float[Array, "y"]):
        def aug_step(x_aug):
            z, x = jnp.split(x_aug, [self.sys.z_dim])
            z, x, y = self.sys(z, x, u, w=None)
            return jnp.concatenate([z, x]), y

        # linearize sys step
        x_aug = jnp.concatenate([self.z.value, self.x.value], axis=-1)
        A, C = jax.jacobian(aug_step)(x_aug)
        dx, dy = aug_step(x_aug)
        dx = dx - A @ x_aug
        dy = dy - C @ x_aug

        # a posteriori update
        K = self.P @ C.T @ jnp.linalg.inv(C @ self.P @ C.T + self.R)
        x_aug = x_aug + K @ (y - C @ x_aug - dy)
        self.P.value = self.P - K @ C @ self.P

        # a priori update
        x_aug = A @ x_aug + dx
        self.P.value = A @ self.P @ A.T + self.Q
        self.z.value, self.x.value = jnp.split(x_aug, [self.sys.z_dim])
        return self.z.value, self.x.value


class UKF(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        Qz: Float[Array, "z z"] | Float[Array, "z"] | float,
        Qx: Float[Array, "x x"] | Float[Array, "x"] | float,
        R: Float[Array, "y y"] | Float[Array, "y"] | float,
        mc_samples_init: int = 0,
        alpha: float = 1e-3,
        beta: float = 2.0,
        kappa: float = 0.0,
        *,
        rngs: nnx.Rngs,
    ):
        self.sys = sys
        self.mc_samples_init = mc_samples_init
        self.alpha = alpha
        self.beta = beta
        self.kappa = kappa

        Zeros = jnp.zeros((sys.z_dim, sys.x_dim))
        Q = [
            [jnp.eye(sys.z_dim) * Qz, Zeros],
            [Zeros.T, jnp.eye(sys.x_dim) * Qx],
        ]
        self.Q = nnx.Param(jnp.block(Q))
        self.R = nnx.Param(jnp.eye(sys.y_dim) * R)

        self.rngs = rngs
        self.z = nnx.Variable(jnp.zeros((sys.z_dim,)))
        self.x = nnx.Variable(jnp.zeros((sys.x_dim,)))
        self.P = nnx.Variable(jnp.eye(sys.z_dim + sys.x_dim))

    def reset(self, deterministic=False):
        if deterministic:
            self.z.value = self.sys.sample_z(rng=None)
            self.x.value = self.sys.sample_x(rng=None)
            self.P.value = jnp.eye(self.P.value.shape[-1])
        else:
            z = jax.vmap(self.sys.sample_z)(jr.split(self.rngs(), self.mc_samples_init))
            x = jax.vmap(self.sys.sample_x)(jr.split(self.rngs(), self.mc_samples_init))
            self.z.value = jnp.mean(z, axis=0)
            self.x.value = jnp.mean(x, axis=0)
            x_aug = jnp.concatenate([z, x], axis=-1)
            self.P.value = jnp.cov(x_aug.T)

    def flat_state(self):
        x_aug = jnp.concatenate([self.z.value, self.x.value], axis=-1)
        Psqrt = jnp.linalg.cholesky(self.P.value).reshape(*self.P.shape[:-2], -1)
        return jnp.concatenate([x_aug, Psqrt], axis=-1)

    def step(self, u: Float[Array, "u"], y: Float[Array, "y"]):
        def aug_step(x_aug):
            z, x = jnp.split(x_aug, [self.sys.z_dim])
            z, x, y = self.sys(z, x, u, w=None)
            return jnp.concatenate([z, x]), y

        # linearize sys step
        x_aug = jnp.concatenate([self.z.value, self.x.value], axis=-1)
        A, C = jax.jacobian(aug_step)(x_aug)
        dx, dy = aug_step(x_aug)
        dx = dx - A @ x_aug
        dy = dy - C @ x_aug

        # a posteriori update
        K = self.P @ C.T @ jnp.linalg.inv(C @ self.P @ C.T + self.R)
        x_aug = x_aug + K @ (y - C @ x_aug - dy)
        self.P.value = self.P - K @ C @ self.P

        # a priori update
        x_aug = A @ x_aug + dx
        self.P.value = A @ self.P @ A.T + self.Q
        self.z.value, self.x.value = jnp.split(x_aug, [self.sys.z_dim])
        return self.z.value, self.x.value
