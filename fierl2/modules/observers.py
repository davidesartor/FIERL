from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from systems import DSSM


class KalmanFilter(nnx.Module):
    def __init__(
        self,
        sys: DSSM,
        Q: Float[Array, "x x"] | Float[Array, "x"] | float,
        R: Float[Array, "y y"] | Float[Array, "y"] | float,
    ):
        self.sys_step = sys.step
        self.Q = nnx.Param(jnp.eye(sys.x_dim) * Q)
        self.R = nnx.Param(jnp.eye(sys.y_dim) * R)

        self.x_hat = nnx.Variable(jnp.zeros((sys.x_dim,)))
        self.P = nnx.Variable(jnp.eye(sys.x_dim))

    def reset(self):
        # sys_reset = lambda m, _: (m.reset(), m.x.value)
        # _, x = nnx.scan(sys_reset)(self.sys, jnp.empty(10))
        # self.P.value = jnp.einsum("ki,kj->ij", x, x) + self.Q.value
        # self.x_hat.value = x.mean(axis=0)
        self.x_hat.value = jnp.zeros((self.x_hat.shape[-1],))
        self.P.value = jnp.eye(self.x_hat.shape[-1])

    def update(self, u: Float[Array, "u"], y: Float[Array, "y"]):
        # linearize sys step
        sys_step = lambda x: self.sys_step(x, u, rng=None)
        A, C = jax.jacobian(sys_step)(self.x_hat.value)
        dx, dy = sys_step(self.x_hat)
        dx = dx - A @ self.x_hat
        dy = dy - C @ self.x_hat
        # a posteriori update
        K = self.P @ C.T @ jnp.linalg.inv(C @ self.P @ C.T + self.R)
        self.x_hat.value = self.x_hat + K @ (y - C @ self.x_hat - dy)
        self.P.value = self.P - K @ C @ self.P
        # a priori update
        self.x_hat.value = A @ self.x_hat + dx
        self.P.value = A @ self.P @ A.T + self.Q

    def __call__(self) -> Float[Array, "x"]:
        return self.x_hat.value
