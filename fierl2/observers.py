from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import equinox as eqx
import systems


class Observer[ObserverState](eqx.Module):
    def reset(self) -> ObserverState:
        raise NotImplementedError

    def estimate(self, state: ObserverState, *args, **kwargs) -> systems.State:
        raise NotImplementedError

    def update(self, state: ObserverState, *args, **kwargs) -> ObserverState:
        raise NotImplementedError


class KalmanState(eqx.Module):
    x: Float[Array, "x"]
    P: Float[Array, "x x"]


class KalmanFilter(Observer[KalmanState]):
    sys: systems.Dssm
    Q: Float[Array, "x x"]
    R: Float[Array, "y y"]

    def reset(self):
        return KalmanState(
            x=self.sys.reset(rng=None),
            P=jnp.eye(self.sys.x_dim),
        )

    def estimate(self, state: KalmanState):
        return state.x

    def update(self, state: KalmanState, u: Float[Array, "u"], y: Float[Array, "y"]):
        A, C, dx, dy = self.linearized_step(state.x, u)
        # a posteriori update
        K = state.P @ C.T @ jnp.linalg.inv(C @ state.P @ C.T + self.R)
        state = KalmanState(
            x=state.x + K @ (y - C @ state.x - dy),
            P=state.P - K @ C @ state.P,
        )
        # a priori update
        state = KalmanState(
            x=A @ state.x + dx,
            P=A @ state.P @ A.T + self.Q,
        )
        return state

    def linearized_step(self, x: Float[Array, "x"], u: Float[Array, "u"]):
        A, C = jax.jacobian(lambda x: self.sys.step(x, u))(x)
        dx, dy = self.sys.step(x, u)
        dx = dx - A @ x
        dy = dy - C @ x
        return A, C, dx, dy
