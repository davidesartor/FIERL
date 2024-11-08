from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import flax.linen as nn
from systems import DSSM


class KalmanFilter(nn.Module):
    sys: DSSM
    Q: Float[Array, "x x"]
    R: Float[Array, "y y"]

    def setup(self):
        self.variable("state", "x", self.sys.x0)
        self.variable("state", "P", lambda: jnp.eye(self.sys.x_dim))

    def __call__(self):
        return self.get_variable("state", "x")

    def step(self, u: Float[Array, "u"], y: Float[Array, "y"]):
        x = self.get_variable("state", "x")
        P = self.get_variable("state", "P")

        A, C, dx, dy = self.linearized_step(x, u)

        # a priori update
        x = A @ x + dx
        P = A @ P @ A.T + self.Q

        # a posteriori update
        K = P @ C.T @ jnp.linalg.inv(C @ P @ C.T + self.R)
        x = x + K @ (y - C @ x - dy)
        P = P - K @ C @ P

        self.put_variable("state", "x", x)
        self.put_variable("state", "P", P)

    @nn.nowrap
    def linearized_step(self, x: Float[Array, "x"], u: Float[Array, "u"]):
        A, C = jax.jacobian(lambda x: self.sys(x, u))(x)
        dx, dy = self.sys(x, u)
        dx = dx - A @ x
        dy = dy - C @ x
        return A, C, dx, dy
