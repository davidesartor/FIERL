from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

State = Float[Array, "x"]
Control = Float[Array, "u"]
Output = Float[Array, "y"]
Fault = Float[Array, "z"]

from dataclasses import replace


UNINITIALIZED = lambda: jnp.array(True)


class DSSM(eqx.Module):
    x: State = eqx.field(init=False, default_factory=UNINITIALIZED)

    x_dim: eqx.AbstractVar[int]
    u_dim: eqx.AbstractVar[int]
    y_dim: eqx.AbstractVar[int]

    def step(self, x: State, u: Control, *, rng=None) -> tuple[State, Output]:
        raise NotImplementedError

    def x0(self, *, rng=None):
        raise NotImplementedError

    def __post_init__(self):
        self.x = self.x0()

    def reset(self, *, rng=None):
        return self.replace(x=self.x0(rng=rng))

    def __call__(self, u: Control, *, rng=None):
        x, y = self.step(self.x, u, rng=rng)
        return self.replace(x=x), y

    def replace(self, *, x):
        return eqx.tree_at(lambda s: s.x, self, x)


class FaultyDSSM(eqx.Module):
    x: State = eqx.field(init=False, default=UNINITIALIZED)
    z: Fault = eqx.field(init=False, default=UNINITIALIZED)

    z_dim: eqx.AbstractVar[int]
    x_dim: eqx.AbstractVar[int]
    u_dim: eqx.AbstractVar[int]
    y_dim: eqx.AbstractVar[int]

    def step(
        self, z: Fault, x: State, u: Control, *, rng=None
    ) -> tuple[Fault, State, Output]:
        raise NotImplementedError

    def z0(self, *, rng=None):
        raise NotImplementedError

    def x0(self, *, rng=None):
        raise NotImplementedError

    def reset(self, *, rng=None):
        if rng is None:
            return self.replace(z=self.z0(), x=self.x0())
        z_rng, x_rng = jr.split(rng)
        return self.replace(z=self.z0(rng=z_rng), x=self.x0(rng=x_rng))

    def __call__(self, u: Control, *, rng=None):
        z, x, y = self.step(self.z, self.x, u, rng=rng)
        return self.replace(z=z, x=x), y

    def replace(self, *, x, z):
        return eqx.tree_at(lambda s: (s.z, s.x), self, (z, x))

    def as_dssm(faulty) -> DSSM:  # type: ignore
        class AugmentedSystem(DSSM):
            x_dim: int = faulty.z_dim + faulty.x_dim
            u_dim: int = faulty.u_dim
            y_dim: int = faulty.y_dim

            def step(self, x: State, u: Control, *, rng=None) -> tuple[State, Output]:
                z, x = jnp.split(x, (faulty.z_dim,), axis=-1)
                z, x, y = faulty.step(z, x, u, rng=rng)
                x = jnp.concatenate([z, x], axis=-1)
                return x, y

            def x0(self, rng=None):
                z_rng, x_rng = (None, None) if rng is None else jr.split(rng)
                z, x = faulty.z0(rng=z_rng), faulty.x0(rng=x_rng)
                return jnp.concatenate([x, z], axis=-1)

        return AugmentedSystem()


class ToyExample(FaultyDSSM):
    z_dim: int = eqx.field(init=False)
    x_dim: int = eqx.field(default=2)
    u_dim: int = eqx.field(default=1)
    y_dim: int = eqx.field(default=1)

    input_coef: float = 1.0
    flow_coef: float = 0.5
    output_coef: float = 1.0

    x_noise_std: float = 1e-1
    y_noise_std: float = 1e-1

    def __post_init__(self):
        self.z_dim = self.u_dim + self.y_dim

    def step(self, z, x, u, *, rng=None):
        zu, zy = jnp.split(z, (u.shape[-1],), axis=-1)
        inflow = self.input_coef * (u * zu).sum()
        mixflows = self.flow_coef * x[:-1]
        outflow = self.output_coef * x[-1]

        x = x.at[0].add(inflow)
        x = x.at[:-1].add(-mixflows)
        x = x.at[1:].add(mixflows)
        x = x.at[-1].add(-outflow)

        y = outflow + zy

        if rng is not None:
            kx, ky = jr.split(rng)
            x = x + jr.normal(kx, x.shape) * self.x_noise_std
            y = y + jr.normal(ky, y.shape) * self.y_noise_std
        return z, x, y

    def x0(self, *, rng=None):
        x = jnp.zeros(self.x_dim)
        if rng is not None:
            x = x + self.x_noise_std * jr.normal(rng, x.shape)
        return x

    def z0(self, *, rng=None):
        zu = jnp.ones(self.u_dim)
        zy = jnp.zeros(self.y_dim)
        z = jnp.concatenate([zu, zy], axis=-1)
        if rng is not None:
            rng_idx, rng_val = jr.split(rng)
            z = z.at[..., jr.choice(rng_idx, self.z_dim)].set(jr.uniform(rng_val))
        return z
