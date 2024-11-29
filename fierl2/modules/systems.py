from dataclasses import dataclass
from typing import Protocol
from utils import *
from flax import nnx


@dataclass
class Cascade(DSSM):
    x_dim: int = 2
    u_dim: int = 1
    y_dim: int = 1
    x_noise_std: float = 0.01
    y_noise_std: float = 0.01
    flow_coeff: float = 0.25
    action_range: tuple[float, float] | None = None

    def reset(self, rng: Key | None):
        x = jnp.ones((self.x_dim,)) / self.flow_coeff
        if rng is not None:
            x = x * jr.uniform(rng, x.shape, minval=0.5, maxval=2.0)
        return x

    def __call__(self, x, u, *, rng: Key | None):
        if self.action_range is not None:
            umin, umax = self.action_range
            assert umin < 0.0 < umax
            u = jax.lax.select(u > 0.0, tanh_clip(u, umax), tanh_clip(u, umin))

        flows = jnp.concat([u.sum(keepdims=True), self.flow_coeff * x], axis=-1)
        x = x + (flows[:-1] - flows[1:])
        y = flows[-1] * jnp.ones(self.y_dim)
        if rng is not None:
            rng_x, rng_y = jr.split(rng)
            x = x + self.x_noise_std * jr.normal(rng_x, x.shape)
            y = y + self.y_noise_std * jr.normal(rng_y, y.shape)
        return x, y


def actuator_faults(sys: DSSM, p: float = 0.0) -> FDSSM:
    class FaultySystem(FDSSM):
        z_dim: int = sys.u_dim
        x_dim: int = sys.x_dim
        u_dim: int = sys.u_dim
        y_dim: int = sys.y_dim

        def reset(self, rng: Key | None):
            z = jnp.ones((sys.u_dim,))
            if rng is not None and p == 0.0:
                rng, rng_idx, rng_val = jr.split(rng, 3)
                z = z.at[jr.choice(rng_idx, len(z))].set(jr.uniform(rng_val))
            x = sys.reset(rng=rng)
            return z, x

        def __call__(self, z, x, u, *, rng: Key | None):
            if rng is not None and p > 0.0:
                rng, rng_p, rng_val = jr.split(rng)
                z, _ = jax.lax.cond(
                    jr.uniform(rng_p) < p, self.reset, lambda rng: (z, _), rng_val
                )
            x, y = sys(x, u * z, rng=rng)
            return z, x, y

    return FaultySystem()


class SysModule(nnx.Module):
    def __init__(self, sys: FDSSM, *, rngs: nnx.Rngs):
        self.sys = sys
        self.rngs = rngs
        self.z = nnx.Variable(jnp.zeros((sys.z_dim,)))
        self.x = nnx.Variable(jnp.zeros((sys.x_dim,)))

    def reset(self):
        self.z.value, self.x.value = self.sys.reset(rng=self.rngs())

    def step(self, u: Float[Array, "u"]):
        z, x = self.z.value, self.x.value
        z, x, y = self.sys(z, x, u, rng=self.rngs())
        self.z.value, self.x.value = z, x
        return y
