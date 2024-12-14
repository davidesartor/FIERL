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
        mc_samples_traj: int = 0,
        integral_action: bool = True,
        u_range: tuple[float, float] = (-5.0, 5.0),
        *,
        rngs: nnx.Rngs,
    ):
        self.sys = sys
        self.horizon = horizon
        self.discount = discount
        self.u_range = u_range
        self.newton_iters = newton_iters
        self.mc_samples_traj = mc_samples_traj
        self.integral_action = integral_action

        self.J = nnx.Param(J)

        self.rngs = rngs
        self.ut = nnx.Variable(jnp.zeros((horizon, sys.u_dim)))

    def reset(self, deterministic=False):
        if deterministic:
            self.ut.value = jnp.zeros_like(self.ut.value)
        else:
            umin, umax = self.u_range
            self.ut.value = 0.1 * jr.uniform(
                self.rngs(), self.ut.shape, minval=umin, maxval=umax
            )

    def flat_state(self):
        return self.ut.value.reshape(*self.ut.shape[:-2], -1)

    def step(
        self, z0: Float[Array, "z"], x0: Float[Array, "x"], ref: References
    ) -> Float[Array, "u"]:
        # roll foward for warm start and preflatten
        ut_flat = jnp.roll(self.ut.value, -1).at[-1].set(self.ut.value[-1]).flatten()

        # optimization objective (either deterministic or monte carlo estimation)
        def cost_fn(ut_flat):
            def cost_single_sim(ut, wt):
                zt, xt, yt = self.sys.trajectory(z0, x0, ut, wt)
                ref_u = self.ut.value if self.integral_action else ref.u
                cy = quadratic_cost(yt - ref.y, self.J.value.y)
                cu = quadratic_cost(ut - ref_u, self.J.value.u)
                cx = quadratic_cost(xt - ref.x, self.J.value.x)
                cz = quadratic_cost(zt - ref.z, self.J.value.z)
                cost = cy + cu + cx + cz
                return jnp.sum(cost * self.discount ** jnp.arange(self.horizon))

            ut = ut_flat.reshape(self.ut.shape)
            if not self.mc_samples_traj:
                return cost_single_sim(ut, wt=None)

            rngs = jr.split(self.rngs(), self.mc_samples_traj * len(self.ut))
            wt = jax.vmap(self.sys.sample_w)(rngs)
            wt = wt.reshape(self.mc_samples_traj, len(self.ut), -1)
            return jnp.mean(jax.vmap(cost_single_sim, in_axes=(None, 0))(ut, wt))

        # second order optimization
        for _ in range(self.newton_iters):
            H = jax.hessian(cost_fn)(ut_flat) + 1e-8 * jnp.eye(len(ut_flat))
            J = jax.grad(cost_fn)(ut_flat)
            ut_flat = jnp.linalg.solve(a=H, b=H @ ut_flat - J)
            ut_flat = ut_flat.clip(*self.u_range)

        self.ut.value = ut_flat.reshape(*self.ut.shape)
        u = self.ut.value[0]
        return u
