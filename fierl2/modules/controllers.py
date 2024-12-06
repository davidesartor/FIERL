from .systems import FDSSM
from utils import *
from flax import nnx


class Signals(NamedTuple):
    y: Float[Array, "t y"] | Float[Array, "y"] | Float[Array, ""] | float
    u: Float[Array, "t u"] | Float[Array, "u"] | Float[Array, ""] | float
    x: Float[Array, "t x"] | Float[Array, "x"] | Float[Array, ""] | float
    z: Float[Array, "t z"] | Float[Array, "z"] | Float[Array, ""] | float


class ControlCostMatrices(NamedTuple):
    y: Float[Array, "y y"] | Float[Array, "y"] | float
    u: Float[Array, "u u"] | Float[Array, "u"] | float
    x: Float[Array, "x x"] | Float[Array, "x"] | float
    z: Float[Array, "z z"] | Float[Array, "z"] | float


class MPC(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        J: ControlCostMatrices,
        horizon: int = 16,
        discount: float = 0.9,
        newton_iters: int = 1,
        integral_action: bool = True,
        u_range: tuple[float, float] = (-5.0, 5.0),
    ):
        self.sys = sys
        self.horizon = horizon
        self.discount = discount
        self.u_range = u_range
        self.newton_iters = newton_iters
        self.integral_action = integral_action

        self.J = nnx.Param(J)
        self.ut = nnx.Variable(jnp.zeros((horizon, sys.u_dim)))

    def reset(self):
        self.ut.value = jnp.zeros_like(self.ut.value)

    def flat_state(self):
        return self.ut.value.reshape(*self.ut.shape[:-2], -1)

    def step(
        self, z0: Float[Array, "z"], x0: Float[Array, "x"], ref: Signals
    ) -> Float[Array, "u"]:
        # roll foward for warm start and preflatten
        ut_flat = jnp.roll(self.ut.value, -1).at[-1].set(self.ut.value[-1]).flatten()

        # optimization objective
        def cost_fn(ut_flat):
            ut = ut_flat.reshape(self.ut.shape)
            if self.integral_action:
                ut = ut.at[1:].add(-ut[:-1]).at[0].add(-self.ut[0])
            zt, xt, yt = self.sys.trajectory(z0, x0, ut, wt=None)
            cost_t = self.control_cost(zt, xt, ut, yt, ref)
            return jnp.sum(cost_t * self.discount ** jnp.arange(self.horizon))

        # second order optimization
        for _ in range(self.newton_iters):
            H = jax.hessian(cost_fn)(ut_flat) + 1e-8 * jnp.eye(len(ut_flat))
            J = jax.grad(cost_fn)(ut_flat)
            ut_flat = jnp.linalg.solve(a=H, b=H @ ut_flat - J)
            ut_flat = ut_flat.clip(*self.u_range)

        self.ut.value = ut_flat.reshape(*self.ut.shape)
        u = self.ut.value[0]
        return u

    def control_cost(
        self,
        zt: Float[Array, "t z"],
        xt: Float[Array, "t x"],
        ut: Float[Array, "t u"],
        yt: Float[Array, "t y"],
        ref: Signals,
    ) -> Float[Array, "t"]:
        cy = quadratic_cost(yt - ref.y, self.J.value.y)
        cu = quadratic_cost(ut - ref.u, self.J.value.u)
        cx = quadratic_cost(xt - ref.x, self.J.value.x)
        cz = quadratic_cost(zt - ref.z, self.J.value.z)
        return cy + cu + cx + cz
