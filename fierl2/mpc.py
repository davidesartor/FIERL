from typing import Self
from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import equinox as eqx
from systems import DSSM, UNINITIALIZED


class Costs(eqx.Module):
    y: Float[Array, "y y"]
    u: Float[Array, "u u"]
    x: Float[Array, "x x"]

    def calligraphic(self, horizon: int, discount: float = 1.0) -> Self:
        def cal(Ji: Float[Array, "d d"]) -> Float[Array, "n*d n*d"]:
            return jnp.kron(jnp.diag(discount ** jnp.arange(horizon)), Ji)

        return jax.tree_map(cal, self)


class References(eqx.Module):
    y: Float[Array, "t y"]
    u: Float[Array, "t u"]
    x: Float[Array, "t x"]

    def slice(self, t: int, horizon: int):
        return jax.tree_map(
            lambda x: jax.lax.dynamic_slice(x, (t, 0), (horizon, x.shape[-1])), self
        )


class MPC(eqx.Module):
    step: int = eqx.field(init=False, default_factory=UNINITIALIZED)
    u: Float[Array, "u"] = eqx.field(init=False, default_factory=UNINITIALIZED)
    ut: Float[Array, "n u"] = eqx.field(init=False, default_factory=UNINITIALIZED)

    sys: DSSM = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    discount: float = eqx.field(static=True)
    ref: References = eqx.field(static=True)
    J: Costs = eqx.field(static=True)
    J_cal: Costs = eqx.field(static=True, init=False)

    def __post_init__(self):
        self.J_cal = self.J.calligraphic(self.horizon, self.discount)

    def replace(self, *, step, u, ut):
        return eqx.tree_at(lambda s: (s.step, s.u, s.ut), self, (step, u, ut))

    def reset(self, *, rng=None):
        ut = jnp.roll(self.ref.u[: self.horizon, :], 1, axis=-2)
        return self.replace(step=0, u=ut[0], ut=ut)

    def update(self, x: Float[Array, "x"]):
        ref = self.ref.slice(self.step, self.horizon)
        ut = jnp.roll(self.ut, -1, axis=-2).at[-1, :].set(ref.u[-1, :])
        ut = self.optimal_control_sequence(ut, x, ref)
        return self.replace(step=self.step + 1, u=ut[0], ut=ut)

    def optimal_control_sequence(
        self, ut: Float[Array, "t u"], x: Float[Array, "x"], ref: References
    ):
        B, D, dx, dy = self.linearized_trajectory(ut, x)
        return jnp.linalg.solve(
            a=D.T @ self.J_cal.y @ D + self.J_cal.u + B.T @ self.J_cal.x @ B,
            b=(
                D.T @ self.J_cal.y @ (ref.y.flatten() - dy)
                + self.J_cal.u @ ref.u.flatten()
                + B.T @ self.J_cal.x @ (ref.x.flatten() - dx)
            ),
        ).reshape(ut.shape)

    def linearized_trajectory(self, ut: Float[Array, "t u"], x: Float[Array, "x"]):
        def scan_compatible_step(x, u):
            x, y = self.sys.step(x, u)
            return x, (x, y)

        trajectory = lambda ut: jax.lax.scan(scan_compatible_step, x, ut)[1]
        B, D = jax.jacobian(trajectory)(ut)
        B = B.reshape(-1, ut.flatten().size)
        D = D.reshape(-1, ut.flatten().size)
        dx, dy = trajectory(ut)
        dx = dx.flatten() - B @ ut.flatten()
        dy = dy.flatten() - D @ ut.flatten()
        return B, D, dx, dy
