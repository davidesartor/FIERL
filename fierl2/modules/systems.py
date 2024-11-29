from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx


class DSSM(nnx.Module):
    def __init__(self, x_dim: int, u_dim: int, y_dim: int, *, rng: nnx.Rngs):
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.rng = rng
        self.x = nnx.Variable(jnp.empty((self.x_dim,)))
        self.y = nnx.Variable(jnp.empty((self.y_dim,)))

    def reset(self):
        raise NotImplementedError

    def update(self, u: Float[Array, "u"]):
        x = self.x.value
        x, y = self.step(self.x.value, u, rng=self.rng.update())
        self.x.value, self.y.value = x, y

    def step(
        self, x: Float[Array, "x"], u: Float[Array, "u"], *, rng: Key | None
    ) -> tuple[Float[Array, "x"], Float[Array, "y"]]:
        raise NotImplementedError

    def __call__(self) -> Float[Array, "y"]:
        return self.y.value


class Cascade(DSSM):
    def __init__(
        self,
        x_dim: int = 2,
        u_dim: int = 1,
        y_dim: int = 1,
        x_noise_std: float = 0.0,
        y_noise_std: float = 0.0,
        flow_coeff: float = 0.5,
        *,
        rng: nnx.Rngs,
    ):
        super().__init__(x_dim, u_dim, y_dim, rng=rng)
        self.x_noise_std = x_noise_std
        self.y_noise_std = y_noise_std
        self.flow_coeff = flow_coeff

    def reset(self):
        self.x.value = jr.normal(self.rng.reset(), (self.x_dim,))

    def step(self, x, u, *, rng: Key | None):
        flows = jnp.concat([u.sum(keepdims=True), self.flow_coeff * x], axis=-1)
        x = x + (flows[:-1] - flows[1:])
        y = flows[-1] * jnp.ones(self.y_dim)
        if rng is not None:
            rng_x, rng_y = jr.split(rng)
            x = x + self.x_noise_std * jr.normal(rng_x, x.shape)
            y = y + self.y_noise_std * jr.normal(rng_y, y.shape)
        return x, y


class FDSSM(nnx.Module):
    def __init__(self, z_dim: int, x_dim: int, u_dim: int, y_dim: int, rng: nnx.Rngs):
        self.z_dim = z_dim
        self.x_dim = x_dim
        self.u_dim = u_dim
        self.y_dim = y_dim
        self.rng = rng
        self.z = nnx.Variable(jnp.empty((self.z_dim,)))
        self.x = nnx.Variable(jnp.empty((self.x_dim,)))
        self.y = nnx.Variable(jnp.empty((self.y_dim,)))

    def reset(self):
        raise NotImplementedError

    def update(self, u: Float[Array, "u"]):
        z, x = self.z.value, self.x.value
        z, x, y = self.step(z, x, u, rng=self.rng.update())
        self.z.value, self.x.value, self.y.value = z, x, y

    def step(
        self,
        z: Float[Array, "z"],
        x: Float[Array, "x"],
        u: Float[Array, "u"],
        *,
        rng: Key | None,
    ) -> tuple[Float[Array, "z"], Float[Array, "x"], Float[Array, "y"]]:
        raise NotImplementedError


class AsHealty(DSSM):
    def __init__(self, faulty: FDSSM):
        self.faulty = faulty
        super().__init__(
            x_dim=faulty.x_dim + faulty.z_dim,
            u_dim=faulty.u_dim,
            y_dim=faulty.y_dim,
            rng=faulty.rng,
        )

    def reset(self):
        self.faulty.reset()
        self.x = jnp.concatenate([self.faulty.z.value, self.faulty.x.value])

    def step(self, x, u, *, rng: Key | None):
        z, x = jnp.split(x, [self.faulty.z_dim])
        z, x, y = self.faulty.step(z, x, u, rng=rng)
        return jnp.concatenate([z, x]), y


class AsFaulty(FDSSM):
    def __init__(self, healty: DSSM):
        self.healty = healty
        super().__init__(
            z_dim=0,
            x_dim=healty.x_dim,
            u_dim=healty.u_dim,
            y_dim=healty.y_dim,
            rng=healty.rng,
        )

    def reset(self):
        self.healty.reset()
        self.x = self.healty.x.value
        self.z = jnp.empty((0,))

    def step(self, z, x, u, *, rng: Key | None):
        x, y = self.healty.step(x, u, rng=rng)
        return z, x, y


class ActuatorFaults(FDSSM):
    def __init__(self, sys: FDSSM):
        self.sys = sys
        super().__init__(
            z_dim=sys.z_dim + sys.u_dim,
            x_dim=sys.x_dim,
            u_dim=sys.u_dim,
            y_dim=sys.y_dim,
            rng=sys.rng,
        )

    def reset(self):
        zu = jnp.ones((self.sys.u_dim,))
        rng_idx, rng_val = jr.split(self.rng.reset(), 2)
        zu = zu.at[jr.choice(rng_idx, len(zu))].set(jr.uniform(rng_val))
        self.sys.reset()
        self.z = jnp.concatenate([zu, self.sys.z.value])
        self.x = self.sys.x.value

    def step(self, z, x, u, *, rng: Key | None):
        zu, z = jnp.split(z, [self.u_dim])
        z, x, y = self.sys.step(z, x, u * zu, rng=rng)
        z = jnp.concatenate([zu, z])
        return z, x, y
