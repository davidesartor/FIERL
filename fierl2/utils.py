from typing import NamedTuple, Any, Protocol
from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx


class MLP(nnx.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim=64, *, rngs: nnx.Rngs):
        super().__init__(
            nnx.Linear(in_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.RMSNorm(hidden_dim, rngs=rngs),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.RMSNorm(hidden_dim, rngs=rngs),
            nnx.Linear(
                hidden_dim, out_dim, rngs=rngs, kernel_init=nnx.initializers.zeros
            ),
        )


class Logger(dict):
    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self:
                self[key] = []
            self[key].append(value)

    def to_array(self):
        return {key: jnp.array(value) for key, value in self.items()}


class Rollout(NamedTuple):
    obs: Float[Array, "..."]
    a: Float[Array, "u"]
    log_p: Float[Array, ""]
    r: Float[Array, ""]
    next_obs: Float[Array, "..."]
    c: Float[Array, ""] = jnp.zeros(())


def get_returns(
    r: Float[Array, "t"], discount: float
) -> tuple[Float[Array, ""], Float[Array, ""]]:
    def scansum_step(v, xi):
        v = xi + discount * v
        return v, v

    J0, Jt = jax.lax.scan(scansum_step, jnp.zeros(()), r, reverse=True)
    return J0, Jt


def quadratic_cost(
    s: Float[Array, "t d"], J: Float[Array, "d d"] | Float[Array, "d"] | float
) -> Float[Array, "t"]:
    J = jnp.asarray(J)
    if J.ndim == 2:
        return jnp.einsum("ti, ij, tj->t", s, J, s)
    return jnp.sum(J * s**2, axis=-1)


def tanh_clip(x: Float[Array, "..."], neg: float, pos: float | None = None):
    c = jnp.where(x < 0.0, neg, pos or neg)
    return jnp.tanh(x / c) * c
