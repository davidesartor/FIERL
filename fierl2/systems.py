from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

State = Float[Array, "x"]
Control = Float[Array, "u"]
Output = Float[Array, "y"]
Fault = Float[Array, "z"]


class Dssm(eqx.Module):
    x_dim: eqx.AbstractVar[int]
    u_dim: eqx.AbstractVar[int]
    y_dim: eqx.AbstractVar[int]

    def reset(self, *, rng=None) -> State:
        raise NotImplementedError

    def step(
        self, x: Float[Array, "x"], u: Control, *, rng=None
    ) -> tuple[State, Output]:
        raise NotImplementedError


class FaultyDssm(eqx.Module):
    z_dim: eqx.AbstractVar[int]
    x_dim: eqx.AbstractVar[int]
    u_dim: eqx.AbstractVar[int]
    y_dim: eqx.AbstractVar[int]

    def reset(self, *, rng=None) -> tuple[Fault, State]:
        raise NotImplementedError

    def step(
        self, z: Fault, x: State, u: Control, *, rng=None
    ) -> tuple[Fault, State, Output]:
        raise NotImplementedError

    def as_dssm(faulty) -> Dssm:
        class AugmentedSystem(Dssm):
            x_dim: int = faulty.z_dim + faulty.x_dim
            u_dim: int = faulty.u_dim
            y_dim: int = faulty.y_dim

            def step(self, x: State, u: Control, *, rng=None) -> tuple[State, Output]:
                z, x = jnp.split(x, (faulty.z_dim,), axis=-1)
                z, x, y = faulty.step(z, x, u, rng=rng)
                x = jnp.concatenate([z, x], axis=-1)
                return x, y

            def reset(self, *, rng=None):
                z, x = faulty.reset(rng=rng)
                return jnp.concatenate([z, x], axis=-1)

        return AugmentedSystem()


class ToyExample(FaultyDssm):
    x_dim: int = 5
    u_dim: int = 1
    y_dim: int = 1
    zu_dim: int = -1
    zy_dim: int = -1
    z_dim: int = 0

    input_coef: float = 1.0
    flow_coef: float = 0.5
    output_coef: float = 1.0

    x_noise_cov: float = 1e-3
    y_noise_cov: float = 1e-3

    def __post_init__(self):
        if self.zu_dim < 0:
            self.zu_dim = self.u_dim
        if self.zy_dim < 0:
            self.zy_dim = self.y_dim
        self.z_dim = self.zu_dim + self.zy_dim

    def step(self, z, x, u, *, rng=None):
        zu, zy = jnp.split(z, (self.zu_dim,), axis=-1)
        inflow = self.input_coef * (u.at[: self.zu_dim].mul(zu)).sum()
        mixflows = self.flow_coef * x[:-1]
        outflow = self.output_coef * x[-1]

        x = x.at[0].add(inflow)
        x = x.at[:-1].add(-mixflows)
        x = x.at[1:].add(mixflows)
        x = x.at[-1].add(-outflow)

        y = (outflow * jnp.ones(self.y_dim)).at[: self.zy_dim].add(zy)

        if rng is not None:
            kx, ky = jr.split(rng)
            x = x + jr.normal(kx, x.shape) * self.x_noise_cov**0.5
            y = y + jr.normal(ky, y.shape) * self.y_noise_cov**0.5
        return z, x, y

    def reset(self, *, rng=None):
        x = jnp.zeros(self.x_dim)
        zu = jnp.ones(self.zu_dim)
        zy = jnp.zeros(self.zy_dim)
        z = jnp.concatenate([zu, zy], axis=-1)

        if rng is not None:
            rng_x, rng_zi, rng_dz = jr.split(rng, 3)
            z = z.at[..., jr.choice(rng_zi, self.z_dim)].set(jr.uniform(rng_dz))
            x = x + jr.normal(rng_x, x.shape)
        return z, x
