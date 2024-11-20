from dataclasses import dataclass
from typing import Protocol
from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr

Fault = Float[Array, "z"]
State = Float[Array, "x"]
Input = Float[Array, "u"]
Output = Float[Array, "y"]


class DSSM(Protocol):
    x_dim: int
    u_dim: int
    y_dim: int

    def step(self, x: State, u: Input, *, rng: Key | None) -> tuple[State, Output]:
        raise NotImplementedError

    def reset(self, *, rng: Key | None) -> State:
        raise NotImplementedError


class FDSSM(Protocol):
    z_dim: int
    x_dim: int
    u_dim: int
    y_dim: int

    def step(
        self, z: Fault, x: State, u: Input, *, rng: Key | None
    ) -> tuple[Fault, State, Output]:
        raise NotImplementedError

    def reset(self, *, rng: Key | None) -> tuple[Fault, State]:
        raise NotImplementedError


@dataclass(frozen=True, slots=True)
class Cascade(DSSM):
    x_dim: int = 2
    u_dim: int = 1
    y_dim: int = 1

    flow_coeff: float = 0.5
    x_noise_std: float = 0.0
    y_noise_std: float = 0.0

    def step(self, x: State, u: Input, *, rng: Key | None) -> tuple[State, Output]:
        flows = self.flow_coeff * x
        x = x.at[0].add(u.sum() - flows[0])
        x = x.at[1:].add(flows[:-1] - flows[1:])
        y = flows[-1] * jnp.ones(self.y_dim)

        if rng is not None:
            rng_x, rng_y = jr.split(rng)
            x = x + self.x_noise_std * jr.normal(rng_x, x.shape)
            y = y + self.y_noise_std * jr.normal(rng_y, y.shape)
        return x, y

    def reset(self, *, rng: Key | None):
        x = jnp.zeros(self.x_dim)
        if rng is not None:
            x = x + jr.uniform(rng, x.shape)
        return x


def as_healty(wrapped: FDSSM) -> DSSM:
    class Wrapper(DSSM):
        x_dim = wrapped.x_dim + wrapped.z_dim
        u_dim = wrapped.u_dim
        y_dim = wrapped.y_dim

        def step(self, x: State, u: Input, *, rng: Key | None) -> tuple[State, Output]:
            z, x = jnp.split(x, (wrapped.z_dim,), axis=-1)
            z, x, y = wrapped.step(z, x, u, rng=rng)
            x = jnp.concat([z, x], axis=-1)
            return x, y

        def reset(self, *, rng: Key | None):
            z, x = wrapped.reset(rng=rng)
            return jnp.concat([z, x], axis=-1)

    return Wrapper()


def as_faulty(wrapped: DSSM) -> FDSSM:
    class Wrapper(FDSSM):
        z_dim = 0
        x_dim = wrapped.x_dim
        u_dim = wrapped.u_dim
        y_dim = wrapped.y_dim

        def step(
            self, z: Fault, x: State, u: Input, *, rng: Key | None
        ) -> tuple[Fault, State, Output]:
            x, y = wrapped.step(x, u, rng=rng)
            return z, x, y

        def reset(self, *, rng: Key | None):
            z = jnp.empty((0,))
            x = wrapped.reset(rng=rng)
            return z, x

    return Wrapper()


def faulty_actuators(wrapped: FDSSM) -> FDSSM:
    class Wrapper(FDSSM):
        z_dim = wrapped.u_dim
        x_dim = wrapped.x_dim
        u_dim = wrapped.u_dim
        y_dim = wrapped.y_dim

        def step(
            self, z: Fault, x: State, u: Input, *, rng: Key | None
        ) -> tuple[Fault, State, Output]:
            zu, z = jnp.split(z, (wrapped.u_dim,), axis=-1)
            u = u * zu
            z, x, y = wrapped.step(z, x, u, rng=rng)
            z = jnp.concat([zu, z], axis=-1)
            return z, x, y

        def reset(self, *, rng: Key | None):
            zu = jnp.ones(wrapped.u_dim)
            if rng is not None:
                rng, rng_idx, rng_val = jr.split(rng, 3)
                zu = zu.at[jr.choice(rng_idx, len(zu))].set(jr.uniform(rng_val))
            z, x = wrapped.reset(rng=rng)
            z = jnp.concat([zu, z], axis=-1)
            return z, x

    return Wrapper()


def faulty_sensors(wrapped: FDSSM) -> FDSSM:
    class Wrapper(FDSSM):
        z_dim = wrapped.y_dim
        x_dim = wrapped.x_dim
        u_dim = wrapped.u_dim
        y_dim = wrapped.y_dim

        def step(
            self, z: Fault, x: State, u: Input, *, rng: Key | None
        ) -> tuple[Fault, State, Output]:
            zy, z = jnp.split(z, (wrapped.y_dim,), axis=-1)
            z, x, y = wrapped.step(z, x, u, rng=rng)
            y = y + zy
            z = jnp.concat([zy, z], axis=-1)
            return z, x, y

        def reset(self, *, rng: Key | None):
            zy = jnp.zeros(wrapped.y_dim)
            if rng is not None:
                rng, rng_idx, rng_val = jr.split(rng, 3)
                zy = zy.at[jr.choice(rng_idx, len(zy))].set(jr.uniform(rng_val))
            z, x = wrapped.reset(rng=rng)
            z = jnp.concat([zy, z], axis=-1)
            return z, x

    return Wrapper()
