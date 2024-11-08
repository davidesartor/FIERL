from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import equinox as eqx
from systems import DSSM, UNINITIALIZED


class KalmanFilter(eqx.Module):
    x: Float[Array, "x"] = eqx.field(init=False, default_factory=UNINITIALIZED)
    P: Float[Array, "x x"] = eqx.field(init=False, default_factory=UNINITIALIZED)

    sys: DSSM = eqx.field(static=True)
    Q: Float[Array, "x x"] = eqx.field(static=True)
    R: Float[Array, "y y"] = eqx.field(static=True)

    def replace(self, *, x, P):
        return eqx.tree_at(lambda s: (s.x, s.P), self, (x, P))

    def reset(self, *, rng=None):
        x = self.sys.reset(rng=rng).x
        P = jnp.eye(x.shape[-1])
        return self.replace(x=x, P=P)

    def update(self, u: Float[Array, "u"], y: Float[Array, "y"]):
        A, C, dx, dy = self.linearized_step(self.x, u)

        # a priori update
        self = self.replace(
            x=A @ self.x + dx,
            P=A @ self.P @ A.T + self.Q,
        )

        # a posteriori update
        K = self.P @ C.T @ jnp.linalg.inv(C @ self.P @ C.T + self.R)
        return self.replace(
            x=self.x + K @ (y - C @ self.x - dy),
            P=self.P - K @ C @ self.P,
        )

    def linearized_step(self, x: Float[Array, "x"], u: Float[Array, "u"]):
        A, C = jax.jacobian(lambda x: self.sys.step(x, u))(x)
        dx, dy = self.sys.step(x, u)
        dx = dx - A @ x
        dy = dy - C @ x
        return A, C, dx, dy
