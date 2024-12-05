from dataclasses import dataclass
from utils import *
from flax import nnx


class FDSSM(Protocol):
    z_dim: int
    x_dim: int
    u_dim: int
    y_dim: int
    w_dim: int

    def sample_z(self, rng: Key | None) -> Float[Array, "z"]:
        raise NotImplementedError

    def sample_x(self, rng: Key | None) -> Float[Array, "x"]:
        raise NotImplementedError

    def sample_w(self, rng: Key | None) -> Float[Array, "w"]:
        raise NotImplementedError

    def __call__(
        self,
        z: Float[Array, "z"],
        x: Float[Array, "x"],
        u: Float[Array, "u"],
        w: Float[Array, "w"] | None,
    ) -> tuple[Float[Array, "z"], Float[Array, "x"], Float[Array, "y"]]:
        raise NotImplementedError

    def trajectory(
        self,
        z0: Float[Array, "z"],
        x0: Float[Array, "x"],
        ut: Float[Array, "t u"],
        wt: Float[Array, "t w"] | None,
    ):
        def scan_fn(carry, inputs):
            z, x = carry
            u, w = inputs
            w = None if wt is None else w
            z, x, y = self(z, x, u, w)
            return (z, x), (z, x, y)

        wt = jnp.zeros((ut.shape[0], self.w_dim)) if wt is None else wt
        _, (zt, xt, yt) = jax.lax.scan(scan_fn, (z0, x0), (ut, wt))
        return zt, xt, yt


class SysModule(nnx.Module):
    def __init__(self, sys: FDSSM, *, rngs: nnx.Rngs):
        self.sys = sys
        self.rngs = rngs
        self.z = nnx.Variable(jnp.zeros((sys.z_dim,)))
        self.x = nnx.Variable(jnp.zeros((sys.x_dim,)))

    def reset(self, deterministic=False):
        self.z.value = self.sys.sample_z(rng=None if deterministic else self.rngs())
        self.x.value = self.sys.sample_x(rng=None if deterministic else self.rngs())

    def step(self, u: Float[Array, "u"]):
        z, x = self.z.value, self.x.value
        w = self.sys.sample_w(rng=self.rngs())
        z, x, y = self.sys(z, x, u, w)
        self.z.value, self.x.value = z, x
        return y


@dataclass
class Cascade(FDSSM):
    x_dim: int = 2
    u_dim: int = 1
    y_dim: int = 1
    w_std: float | Float[Array, "w"] = 0.01
    flow_coeff: float = 0.25
    actuator_faults: bool = True
    sensor_faults: bool = False
    action_range: tuple[float, float] | None = None

    @property
    def z_dim(self):
        z_dim = self.u_dim if self.actuator_faults else 0
        z_dim += self.y_dim if self.sensor_faults else 0
        return z_dim

    @property
    def w_dim(self):
        return self.y_dim + self.x_dim

    def sample_z(self, rng: Key | None):
        z = jnp.ones(self.z_dim)
        if rng is not None:
            rng, rng_idx, rng_val = jr.split(rng, 3)
            z = z.at[jr.choice(rng_idx, len(z))].set(jr.uniform(rng_val))
        return z

    def sample_x(self, rng: Key | None):
        x = jnp.ones((self.x_dim,)) / self.flow_coeff
        if rng is not None:
            x = x * jr.uniform(rng, x.shape, minval=0.9, maxval=1.1)
        return x

    def sample_w(self, rng: Key | None):
        w = jnp.zeros(self.w_dim)
        if rng is not None:
            w = self.w_std * jr.normal(rng, (self.w_dim,))
        return w

    def __call__(self, z, x, u, w):
        # soft clip control before applying it
        if self.action_range is not None:
            assert self.action_range[0] < 0.0 < self.action_range[1]
            u = tanh_clip(u, *self.action_range)

        # apply actuator faults
        if self.actuator_faults:
            u = u * z[: self.u_dim]

        # apply system dynamics
        flows = jnp.concat([u.sum(keepdims=True), self.flow_coeff * x], axis=-1)
        x = x + (flows[:-1] - flows[1:])
        y = flows[-1] * jnp.ones(self.y_dim)

        # apply sensor faults
        if self.sensor_faults:
            y = y + z[-self.y_dim :]

        # apply noise
        if w is not None:
            y = y + w[: self.y_dim]
            x = x + w[self.y_dim :]
        return z, x, y
