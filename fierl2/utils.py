from typing import NamedTuple, Any, Protocol
from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr


class Logger(dict):
    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self:
                self[key] = []
            self[key].append(value)

    def to_array(self):
        return {key: jnp.array(value) for key, value in self.items()}


def quadratic_cost(
    s: Float[Array, "t d"], J: Float[Array, "d d"] | Float[Array, "d"] | float
) -> Float[Array, "t"]:
    J = jnp.asarray(J)
    if J.ndim == 2:
        return jnp.einsum("ti, ij, tj->t", s, J, s)
    return jnp.sum(J * s**2, axis=-1)


def tanh_clip(x, c):
    return jnp.tanh(x / c) * c


class Rollout(NamedTuple):
    obs: Float[Array, "..."]
    a: Float[Array, "u"]
    log_p: Float[Array, ""]
    r: Float[Array, ""]
    next_obs: Float[Array, "..."]


class DSSM(Protocol):
    x_dim: int
    u_dim: int
    y_dim: int

    def reset(self, rng: Key | None) -> Float[Array, "x"]:
        raise NotImplementedError

    def __call__(
        self, x: Float[Array, "x"], u: Float[Array, "u"], *, rng: Key | None
    ) -> tuple[Float[Array, "x"], Float[Array, "y"]]:
        raise NotImplementedError

    def trajectory(
        self, x0: Float[Array, "x"], ut: Float[Array, "t u"], *, rng: Key | None
    ):
        def scan_fn(x, inputs):
            u, k = inputs
            k = None if rng is None else k
            x, y = self(x, u, rng=k)
            return x, (x, y)

        keys = jr.split((jr.key(0) if rng is None else rng), len(ut))
        _, (xt, yt) = jax.lax.scan(scan_fn, x0, (ut, keys))
        return xt, yt


class FDSSM(Protocol):
    z_dim: int
    x_dim: int
    u_dim: int
    y_dim: int

    def reset(self, rng: Key | None) -> tuple[Float[Array, "z"], Float[Array, "x"]]:
        raise NotImplementedError

    def __call__(
        self,
        z: Float[Array, "z"],
        x: Float[Array, "x"],
        u: Float[Array, "u"],
        *,
        rng: Key | None
    ) -> tuple[Float[Array, "z"], Float[Array, "x"], Float[Array, "y"]]:
        raise NotImplementedError

    def trajectory(
        self,
        z0: Float[Array, "z"],
        x0: Float[Array, "x"],
        ut: Float[Array, "t u"],
        *,
        rng: Key | None
    ):
        def scan_fn(carry, inputs):
            z, x = carry
            u, k = inputs
            k = None if rng is None else k
            z, x, y = self(z, x, u, rng=k)
            return (z, x), (z, x, y)

        keys = jr.split((jr.key(0) if rng is None else rng), len(ut))
        _, (zt, xt, yt) = jax.lax.scan(scan_fn, (z0, x0), (ut, keys))
        return zt, xt, yt
