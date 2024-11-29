from typing import NamedTuple, Any
from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr


def tanh_clip(x, c):
    return jnp.tanh(x / c) * c


class Rollout(NamedTuple):
    obs: Float[Array, "..."]
    a: Float[Array, "u"]
    r: Float[Array, ""]
    next_obs: Float[Array, "..."]


class Logger(dict):
    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self:
                self[key] = []
            self[key].append(value)

    def to_array(self):
        return {key: jnp.array(value) for key, value in self.items()}
