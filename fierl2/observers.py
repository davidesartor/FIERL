from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import equinox as eqx
from utils import Module, RESET
from systems import DSSM


class KalmanFilter(Module):
    x: Float[Array, "x"] = eqx.field(init=False, default_factory=RESET)
    P: Float[Array, "x x"] = eqx.field(init=False, default_factory=RESET)

    sys: DSSM = eqx.field(static=True)
    Q: Float[Array, "x x"] = eqx.field(static=True)
    R: Float[Array, "y y"] = eqx.field(static=True)

    def __call__(self) -> Float[Array, "x"]:
        return self.x

    def reset(self, *, rng: Key | None):
        return self.replace(
            x=self.sys.reset(rng=None).x,
            P=jnp.eye(self.sys.x_dim),
        )

    def update(self, u: Float[Array, "u"], y: Float[Array, "y"]):
        A, C, dx, dy = self.linearized_step(self.x, u)
        # a posteriori update
        K = self.P @ C.T @ jnp.linalg.inv(C @ self.P @ C.T + self.R)
        self = self.replace(
            x=self.x + K @ (y - C @ self.x - dy),
            P=self.P - K @ C @ self.P,
        )
        # a priori update
        return self.replace(
            x=A @ self.x + dx,
            P=A @ self.P @ A.T + self.Q,
        )

    def linearized_step(self, x: Float[Array, "x"], u: Float[Array, "u"]):
        A, C = jax.jacobian(lambda x: self.sys(x, u, rng=None))(x)
        dx, dy = self.sys(x, u, rng=None)
        dx = dx - A @ x
        dy = dy - C @ x
        return A, C, dx, dy
