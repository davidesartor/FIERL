from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import equinox as eqx
from utils import Module, RESET
from systems import DSSM


class MPC(Module):
    ut: Float[Array, "t u"] = eqx.field(init=False, default_factory=RESET)

    sys: DSSM = eqx.field(static=True)
    Jy: Float[Array, "y y"] = eqx.field(static=True)
    Ju: Float[Array, "u u"] = eqx.field(static=True)
    horizon: int = eqx.field(static=True)
    discount: float = eqx.field(static=True)

    def __call__(self) -> Float[Array, "u"]:
        return self.ut[0]

    def reset(self, *, rng: Key | None):
        return self.replace(ut=jnp.zeros((self.horizon, self.sys.u_dim)))

    def update(
        self,
        x: Float[Array, "x"],
        ref_y: Float[Array, "t y"],
        ref_u: Float[Array, "t u"],
    ):
        ut = jnp.roll(self.ut, -1, axis=-2).at[-1].set(ref_u[-1])

        # find optimal control
        D, dy = self.linearized_trajectory(ut, x)
        Ju = jnp.kron(jnp.diag(self.discount ** jnp.arange(self.horizon)), self.Ju)
        Jy = jnp.kron(jnp.diag(self.discount ** jnp.arange(self.horizon)), self.Jy)

        ut = jnp.linalg.solve(
            a=D.T @ Jy @ D + Ju,
            b=(D.T @ Jy @ (ref_y.flatten() - dy) + Ju @ ref_u.flatten()),
        ).reshape(ut.shape)
        return self.replace(ut=ut)

    def linearized_trajectory(self, ut: Float[Array, "t u"], x: Float[Array, "x"]):
        def trajectory(ut):
            step = lambda x, u: self.sys(x, u, rng=None)
            _, yt = jax.lax.scan(step, x, ut)
            return yt

        D = jax.jacobian(trajectory)(ut)
        D = D.reshape(-1, ut.flatten().size)
        dy = trajectory(ut).flatten() - D @ ut.flatten()
        return D, dy
