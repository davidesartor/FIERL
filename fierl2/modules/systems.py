from dataclasses import dataclass
from utils import *
from flax import nnx


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
            assert self.action_range[0] < 0.0 < self.action_range[1]
            u = tanh_clip(u, *self.action_range)

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
            u = u * z
            x, y = sys(x, u, rng=rng)
            return z, x, y

    return FaultySystem()


def sensor_faults(sys: DSSM, p: float = 0.0) -> FDSSM:
    class FaultySystem(FDSSM):
        z_dim: int = sys.y_dim
        x_dim: int = sys.x_dim
        u_dim: int = sys.u_dim
        y_dim: int = sys.y_dim

        def reset(self, rng: Key | None):
            z = jnp.zeros((sys.y_dim,))
            if rng is not None and p == 0.0:
                rng, rng_idx, rng_val = jr.split(rng, 3)
                z = z.at[jr.choice(rng_idx, len(z))].set(jr.uniform(rng_val, minval=-1))
            x = sys.reset(rng=rng)
            return z, x

        def __call__(self, z, x, u, *, rng: Key | None):
            if rng is not None and p > 0.0:
                rng, rng_p, rng_val = jr.split(rng)
                z, _ = jax.lax.cond(
                    jr.uniform(rng_p) < p, self.reset, lambda rng: (z, _), rng_val
                )
            x, y = sys(x, u, rng=rng)
            y = y + z
            return z, x, y

    return FaultySystem()
