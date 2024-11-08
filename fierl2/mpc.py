from typing import Self, NamedTuple
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import flax.linen as nn
from systems import DSSM


class Costs(NamedTuple):
    y: Float[Array, "y y"]
    u: Float[Array, "u u"]
    x: Float[Array, "x x"]

    def calligraphic(self, horizon: int, discount: float = 1.0) -> Self:
        def cal(Ji: Float[Array, "d d"]) -> Float[Array, "n*d n*d"]:
            return jnp.kron(jnp.diag(discount ** jnp.arange(horizon)), Ji)

        return jax.tree_map(cal, self)


class References(NamedTuple):
    y: Float[Array, "t y"]
    u: Float[Array, "t u"]
    x: Float[Array, "t x"]

    def slice(self, t: int, horizon: int) -> Self:
        return jax.tree_map(
            lambda x: jax.lax.dynamic_slice(x, (t, 0), (horizon, x.shape[-1])), self
        )


class MPC(nn.Module):
    sys: DSSM
    horizon: int
    discount: float
    ref: References
    J: Costs

    @property
    def J_cal(self):
        return self.J.calligraphic(self.horizon, self.discount)

    def setup(self):
        self.variable("state", "t", lambda: 0)
        self.variable("state", "ut", lambda: self.ref.slice(0, self.horizon).u)

    def __call__(self):
        return self.get_variable("state", "ut")[0]

    def step(self, x: Float[Array, "x"]):
        t = self.get_variable("state", "t")
        ut = self.get_variable("state", "ut")

        ref = self.ref.slice(t, self.horizon)
        ut = jnp.roll(ut, -1, axis=-2).at[-1].set(ut[-1])

        # find optimal control
        B, D, dx, dy = self.linearized_trajectory(ut, x)
        ut = jnp.linalg.solve(
            a=D.T @ self.J_cal.y @ D + self.J_cal.u + B.T @ self.J_cal.x @ B,
            b=(
                D.T @ self.J_cal.y @ (ref.y.flatten() - dy)
                + self.J_cal.u @ ref.u.flatten()
                + B.T @ self.J_cal.x @ (ref.x.flatten() - dx)
            ),
        ).reshape(ut.shape)

        self.put_variable("state", "t", t + 1)
        self.put_variable("state", "ut", ut)

    @nn.nowrap
    def linearized_trajectory(self, ut: Float[Array, "t u"], x: Float[Array, "x"]):
        def scan_compatible_step(x, u):
            x, y = self.sys(x, u)
            return x, (x, y)

        trajectory = lambda ut: jax.lax.scan(scan_compatible_step, x, ut)[1]
        B, D = jax.jacobian(trajectory)(ut)
        B = B.reshape(-1, ut.flatten().size)
        D = D.reshape(-1, ut.flatten().size)
        dx, dy = trajectory(ut)
        dx = dx.flatten() - B @ ut.flatten()
        dy = dy.flatten() - D @ ut.flatten()
        return B, D, dx, dy
