from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import equinox as eqx
from einops import einsum  # TODO use einsum instead of flatten + matmul
import systems


class Controller[ControllerState](eqx.Module):
    def reset(self) -> ControllerState:
        raise NotImplementedError

    def control(self, state: ControllerState, *args, **kwargs):
        raise NotImplementedError

    def update(self, state: ControllerState, *args, **kwargs) -> ControllerState:
        raise NotImplementedError


class MpcState(eqx.Module):
    u: Float[Array, "u"]
    previous_sol: Float[Array, "t u"]


class Mpc(Controller[MpcState]):
    sys: systems.Dssm
    horizon: int
    discount: float
    Jy: Float[Array, "y y"]
    Ju: Float[Array, "u u"]
    Jy_cal: Float[Array, "ny ny"] = eqx.field(init=False)
    Ju_cal: Float[Array, "nu nu"] = eqx.field(init=False)

    def __post_init__(self):
        disc = jnp.diag(self.discount ** jnp.arange(self.horizon))
        self.Jy_cal = jnp.kron(disc, self.Jy)
        self.Ju_cal = jnp.kron(disc, self.Ju)

    def reset(self):
        ut = jnp.zeros((self.horizon, self.sys.u_dim))
        return MpcState(previous_sol=ut, u=ut[0])

    def control(self, state: MpcState):
        return state.u

    def update(
        self,
        state: MpcState,
        x: Float[Array, "x"],
        ref_y: Float[Array, "n y"],
        ref_u: Float[Array, "n u"],
    ):
        ut = jnp.roll(state.previous_sol, -1, axis=-2).at[-1].set(ref_u[-1])

        # find optimal control
        D, dy = self.linearized_trajectory(ut, x)

        ut = jnp.linalg.solve(
            a=D.T @ self.Jy_cal @ D + self.Ju_cal,
            b=(
                D.T @ self.Jy_cal @ (ref_y.flatten() - dy)
                + self.Ju_cal @ ref_u.flatten()
            ),
        ).reshape(ut.shape)
        return MpcState(u=ut[0], previous_sol=ut)

    def linearized_trajectory(self, ut: Float[Array, "t u"], x: Float[Array, "x"]):
        trajectory = lambda ut: jax.lax.scan(self.sys.step, x, ut)[1]
        D = jax.jacobian(trajectory)(ut)
        D = D.reshape(-1, ut.flatten().size)
        dy = trajectory(ut).flatten() - D @ ut.flatten()
        return D, dy
