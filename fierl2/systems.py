from typing import Self
from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from utils import Module, RESET


class DSSM(Module):
    x: Float[Array, "x"] = eqx.field(init=False, default_factory=RESET)
    x_dim: int = eqx.field(static=True, init=False)
    u_dim: int = eqx.field(static=True, init=False)
    y_dim: int = eqx.field(static=True, init=False)

    def __call__(
        self, x: Float[Array, "x"], u: Float[Array, "u"], *, rng: Key | None
    ) -> tuple[Float[Array, "x"], Float[Array, "y"]]:
        raise NotImplementedError

    def step(
        self, u: Float[Array, "u"], *, rng: Key | None
    ) -> tuple[Self, Float[Array, "y"]]:
        x, y = self(self.x, u, rng=rng)
        return self.replace(x=x), y


class Cascade(DSSM):
    x_dim: int = eqx.field(static=True, default=3)
    u_dim: int = eqx.field(static=True, default=1)
    y_dim: int = eqx.field(static=True, default=1)

    flow_coeff: float = eqx.field(static=True, default=0.5)

    x_noise_std: float = eqx.field(static=True, default=1e-2)
    y_noise_std: float = eqx.field(static=True, default=1e-2)

    def __call__(
        self, x: Float[Array, "x"], u: Float[Array, "u"], *, rng: Key | None
    ) -> tuple[Float[Array, "x"], Float[Array, "y"]]:
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
            x = x + jr.normal(rng, x.shape)
        return self.replace(x=x)


class FDSSM(Module):
    z: Float[Array, "z"] = eqx.field(init=False, default_factory=RESET)
    x: Float[Array, "x"] = eqx.field(init=False, default_factory=RESET)
    z_dim: int = eqx.field(static=True, init=False)
    x_dim: int = eqx.field(static=True, init=False)
    u_dim: int = eqx.field(static=True, init=False)
    y_dim: int = eqx.field(static=True, init=False)

    def __call__(
        self,
        z: Float[Array, "z"],
        x: Float[Array, "x"],
        u: Float[Array, "u"],
        *,
        rng: Key | None,
    ) -> tuple[Float[Array, "z"], Float[Array, "x"], Float[Array, "y"]]:
        raise NotImplementedError

    def step(
        self, u: Float[Array, "u"], *, rng: Key | None
    ) -> tuple[Self, Float[Array, "y"]]:
        z, x, y = self(self.z, self.x, u, rng=rng)
        return self.replace(z=z, x=x), y


def as_healty(wrapped: FDSSM) -> DSSM:
    class Wrapper(DSSM):
        def __post_init__(self):
            self.x_dim = wrapped.x_dim + wrapped.z_dim
            self.u_dim = wrapped.u_dim
            self.y_dim = wrapped.y_dim
            return super().__post_init__()

        def __call__(
            self, x: Float[Array, "x"], u: Float[Array, "u"], *, rng: Key | None
        ) -> tuple[Float[Array, "x"], Float[Array, "y"]]:
            z, x = jnp.split(x, (wrapped.z_dim,), axis=-1)
            z, x, y = wrapped(z, x, u, rng=rng)
            x = jnp.concat([z, x], axis=-1)
            return x, y

        def reset(self, *, rng: Key | None):
            new = wrapped.reset(rng=rng)
            return self.replace(x=jnp.concat([new.z, new.x], axis=-1))

    return Wrapper()


def as_faulty(wrapped: DSSM) -> FDSSM:
    class Wrapper(FDSSM):
        def __post_init__(self):
            self.z_dim = 0
            self.x_dim = wrapped.x_dim
            self.u_dim = wrapped.u_dim
            self.y_dim = wrapped.y_dim
            return super().__post_init__()

        def __call__(
            self,
            z: Float[Array, "z"],
            x: Float[Array, "x"],
            u: Float[Array, "u"],
            *,
            rng: Key | None,
        ) -> tuple[Float[Array, "z"], Float[Array, "x"], Float[Array, "y"]]:
            x, y = wrapped(x, u, rng=rng)
            return z, x, y

        def reset(self, *, rng: Key | None):
            new = wrapped.reset(rng=rng)
            return self.replace(x=new.x, z=jnp.zeros((0,)))

    return Wrapper()


class ActuatorsFault(FDSSM):
    wrapped: FDSSM = eqx.field(static=True)

    def __post_init__(self):
        self.z_dim = self.wrapped.z_dim + self.wrapped.u_dim
        self.x_dim = self.wrapped.x_dim
        self.u_dim = self.wrapped.u_dim
        self.y_dim = self.wrapped.y_dim
        return super().__post_init__()

    def __call__(
        self,
        z: Float[Array, "z"],
        x: Float[Array, "x"],
        u: Float[Array, "u"],
        *,
        rng: Key | None,
    ) -> tuple[Float[Array, "z"], Float[Array, "x"], Float[Array, "y"]]:
        zu, z = jnp.split(z, (self.wrapped.u_dim,), axis=-1)
        u = u * zu
        z, x, y = self.wrapped(z, x, (u * zu), rng=rng)
        z = jnp.concat([zu, z], axis=-1)
        return z, x, y

    def reset(self, *, rng: Key | None):
        zu = jnp.ones(self.wrapped.u_dim)
        if rng is not None:
            rng, rng_idx, rng_val = jr.split(rng, 3)
            zu = zu.at[jr.choice(rng_idx, len(zu))].set(jr.uniform(rng_val))
        wrapped = self.wrapped.reset(rng=rng)
        return self.replace(z=jnp.concat([zu, wrapped.z], axis=-1), x=wrapped.x)


class SensorsFault(FDSSM):
    wrapped: FDSSM = eqx.field(static=True)

    def __post_init__(self):
        self.z_dim = self.wrapped.z_dim + self.wrapped.y_dim
        self.x_dim = self.wrapped.x_dim
        self.u_dim = self.wrapped.u_dim
        self.y_dim = self.wrapped.y_dim
        return super().__post_init__()

    def __call__(
        self,
        z: Float[Array, "z"],
        x: Float[Array, "x"],
        u: Float[Array, "u"],
        *,
        rng: Key | None,
    ) -> tuple[Float[Array, "z"], Float[Array, "x"], Float[Array, "y"]]:
        zy, z = jnp.split(z, (self.wrapped.y_dim,), axis=-1)
        z, x, y = self.wrapped(z, x, u, rng=rng)
        y = y + zy
        z = jnp.concat([zy, z], axis=-1)
        return z, x, y

    def reset(self, *, rng: Key | None):
        zy = jnp.ones(self.wrapped.y_dim)
        if rng is not None:
            rng, rng_idx, rng_val = jr.split(rng, 3)
            zy = zy.at[jr.choice(rng_idx, len(zy))].set(jr.uniform(rng_val))
        wrapped = self.wrapped.reset(rng=rng)
        return self.replace(z=jnp.concat([zy, wrapped.z], axis=-1), x=wrapped.x)
