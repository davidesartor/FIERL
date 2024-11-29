from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from systems import DSSM


class MPC(nnx.Module):
    def __init__(
        self,
        sys: DSSM,
        Jy: Float[Array, "x x"] | Float[Array, "x"] | float,
        Ju: Float[Array, "u u"] | Float[Array, "u"] | float,
        Jx: Float[Array, "x x"] | Float[Array, "x"] | float,
        ref_y: Float[Array, "#y"] | float,
        ref_u: Float[Array, "#u"] | float,
        ref_x: Float[Array, "#x"] | float,
        horizon: int,
        discount: float,
        integral_action: bool,
    ):
        self.sys_step = sys.step
        self.horizon = horizon
        self.discount = discount
        self.integral_action = integral_action
        self.Jy = nnx.Param(jnp.eye(sys.y_dim) * Jy)
        self.Ju = nnx.Param(jnp.eye(sys.u_dim) * Ju)
        self.Jx = nnx.Param(jnp.eye(sys.x_dim) * Jx)
        self.ref_y = nnx.Param(ref_y)
        self.ref_u = nnx.Param(ref_u)
        self.ref_x = nnx.Param(ref_x)
        self.ut = nnx.Variable(jnp.zeros((horizon, sys.u_dim)))

    def reset(self):
        self.ut.value = jnp.broadcast_to(self.ref_u.value, self.ut.shape)

    def update(self, x: Float[Array, "x"]):
        def trajectory(ut):
            def scan_fn(x, u):
                x, y = self.sys_step(x, u, rng=None)
                return x, (x, y)

            _, (xt, yt) = jax.lax.scan(scan_fn, x, ut)
            return xt, yt

        def cost(ut_flat):
            ut = ut_flat.reshape(self.ut.shape)
            xt, yt = trajectory(ut)
            if self.integral_action:
                ut = ut.at[0].add(-self.ut.value[0]).at[1:].add(-ut[:-1])
            gamma = (self.discount or 1.0) ** jnp.arange(len(ut))
            norm = lambda x, J: jnp.einsum("ti, ij, tj, t->", x, J, x, gamma)
            cy = norm(yt - self.ref_y.value, self.Jy.value)
            cu = norm(ut - self.ref_u.value, self.Ju.value)
            cx = norm(xt - self.ref_x.value, self.Jx.value)
            return cy + cu + cx

        ut = jnp.roll(self.ut.value, -1).at[-1].set(self.ref_u)
        ut_flat = ut.flatten()
        H = jax.hessian(cost)(ut_flat)
        J = jax.grad(cost)(ut_flat)
        self.ut.value = jnp.linalg.solve(a=H, b=H @ ut_flat - J).reshape(ut.shape)

    def __call__(self) -> Float[Array, "u"]:
        return self.ut.value[0]
