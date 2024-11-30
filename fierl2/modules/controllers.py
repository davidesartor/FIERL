from .systems import FDSSM
from utils import *
from flax import nnx


class References(NamedTuple):
    y: Float[Array, "t y"] | Float[Array, "y"] | Float[Array, ""] | float
    u: Float[Array, "t u"] | Float[Array, "u"] | Float[Array, ""] | float
    x: Float[Array, "t x"] | Float[Array, "x"] | Float[Array, ""] | float
    z: Float[Array, "t z"] | Float[Array, "z"] | Float[Array, ""] | float


class ControlCostMatrices(NamedTuple):
    y: Float[Array, "y y"] | Float[Array, "y"] | float
    u: Float[Array, "u u"] | Float[Array, "u"] | float
    du: Float[Array, "u u"] | Float[Array, "u"] | float
    x: Float[Array, "x x"] | Float[Array, "x"] | float
    z: Float[Array, "z z"] | Float[Array, "z"] | float


class MPC(nnx.Module):
    def __init__(
        self,
        sys: FDSSM,
        J: ControlCostMatrices,
        horizon: int = 16,
        discount: float = 0.9,
        mc_samples_traj: int = 0,
        u_range: tuple[float, float] | None = (-10.0, 10.0),
        *,
        rngs: nnx.Rngs,
    ):
        self.sys = sys
        self.horizon = horizon
        self.discount = discount
        self.u_range = u_range
        self.mc_samples_traj = mc_samples_traj

        self.J = nnx.Param(J)

        self.rngs = rngs
        self.ut = nnx.Variable(jnp.zeros((horizon, sys.u_dim)))

    def reset(self):
        self.ut.value = jnp.zeros_like(self.ut.value)

    def step(
        self, z0: Float[Array, "z"], x0: Float[Array, "x"], ref: References
    ) -> Float[Array, "u"]:
        def cost_fn(ut_flat):
            def cost_single_sim(ut, rng):
                zt, xt, yt = self.sys.trajectory(z0, x0, ut, rng=rng)
                cost_t = self.control_cost(zt, xt, ut, yt, ref, u0=self.ut[0])
                return jnp.sum(cost_t * self.discount ** jnp.arange(self.horizon))

            ut = ut_flat.reshape(self.ut.shape)
            if not self.mc_samples_traj:
                return cost_single_sim(ut, rng=None)
            rngs = jr.split(self.rngs(), self.mc_samples_traj)
            return jnp.mean(jax.vmap(cost_single_sim, in_axes=(None, 0))(ut, rngs))

        # second order optimization step
        ut = jnp.roll(self.ut.value, -1).at[-1].set(self.ut.value[-1])
        ut_flat = ut.flatten()
        H = jax.hessian(cost_fn)(ut_flat)
        J = jax.grad(cost_fn)(ut_flat)
        ut = jnp.linalg.solve(a=H, b=H @ ut_flat - J).reshape(ut.shape)
        self.ut.value = ut.clip(*self.u_range)
        u = self.ut.value[0]
        return u

    def control_cost(
        self,
        zt: Float[Array, "t z"],
        xt: Float[Array, "t x"],
        ut: Float[Array, "t u"],
        yt: Float[Array, "t y"],
        ref: References,
        u0: Float[Array, "u"] = jnp.zeros(()),
    ) -> Float[Array, "t"]:
        dut = ut.at[1:].add(-ut[:-1]).at[0].add(-u0)
        cy = quadratic_cost(yt - ref.y, self.J.value.y)
        cu = quadratic_cost(ut - ref.u, self.J.value.u)
        cdu = quadratic_cost(dut, self.J.value.du)
        cx = quadratic_cost(xt - ref.x, self.J.value.x)
        cz = quadratic_cost(zt - ref.z, self.J.value.z)
        return cy + cu + cx + cz + cdu
