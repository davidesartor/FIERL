from jaxtyping import Float, Array
import jax
import jax.numpy as jnp
import jax.random as jr
import flax.linen as nn

State = Float[Array, "x"]
Control = Float[Array, "u"]
Output = Float[Array, "y"]
Fault = Float[Array, "z"]


class DSSM(nn.Module):
    x_dim: int
    u_dim: int
    y_dim: int

    def setup(self):
        self.variable("state", "x", self.x0)

    def step(self, u: Control, *, rng=None):
        x = self.get_variable("state", "x")
        x, y = self(x, u, rng=rng)
        self.put_variable("state", "x", x)
        return y

    @nn.nowrap
    def x0(self, rng=None):
        raise NotImplementedError

    @nn.nowrap
    def __call__(self, x: State, u: Control, *, rng=None) -> tuple[State, Output]:
        raise NotImplementedError


class FaultyDSSM(nn.Module):
    z_dim: int
    x_dim: int
    u_dim: int
    y_dim: int

    def setup(self):
        self.variable("state", "z", self.z0)
        self.variable("state", "x", self.x0)

    def step(self, u: Control, *, rng=None):
        z = self.get_variable("state", "z")
        x = self.get_variable("state", "x")
        z, x, y = self(z, x, u, rng=rng)
        self.put_variable("state", "z", z)
        self.put_variable("state", "x", x)
        return y

    @nn.nowrap
    def z0(self, *, rng=None):
        raise NotImplementedError

    @nn.nowrap
    def x0(self, *, rng=None):
        raise NotImplementedError

    @nn.nowrap
    def __call__(
        self, z: Fault, x: State, u: Control, *, rng=None
    ) -> tuple[Fault, State, Output]:
        raise NotImplementedError

    @nn.nowrap
    def as_dssm(faulty) -> DSSM:
        class AugmentedSystem(DSSM):
            @nn.nowrap
            def __call__(
                self, x: State, u: Control, *, rng=None
            ) -> tuple[State, Output]:
                z, x = jnp.split(x, (faulty.z_dim,), axis=-1)
                z, x, y = faulty(z, x, u, rng=rng)
                x = jnp.concatenate([z, x], axis=-1)
                return x, y

            @nn.nowrap
            def x0(self, rng=None):
                z_rng, x_rng = (None, None) if rng is None else jr.split(rng)
                z, x = faulty.z0(rng=z_rng), faulty.x0(rng=x_rng)
                return jnp.concatenate([z, x], axis=-1)

        return AugmentedSystem(
            x_dim=faulty.z_dim + faulty.x_dim, u_dim=faulty.u_dim, y_dim=faulty.y_dim
        )


class ToyExample(FaultyDSSM):
    input_coef: float = 1.0
    flow_coef: float = 0.5
    output_coef: float = 1.0

    x_noise_std: float = 1e-1
    y_noise_std: float = 1e-1

    @nn.nowrap
    def __call__(self, z, x, u, *, rng=None):
        zu, zy = jnp.split(z, (self.u_dim,), axis=-1)
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

    @nn.nowrap
    def x0(self, *, rng=None):
        x = jnp.zeros(self.x_dim)
        if rng is not None:
            x = x + self.x_noise_std * jr.normal(rng, x.shape)
        return x

    @nn.nowrap
    def z0(self, *, rng=None):
        zu = jnp.ones(self.u_dim)
        zy = jnp.zeros(self.y_dim)
        z = jnp.concatenate([zu, zy], axis=-1)
        if rng is not None:
            rng_idx, rng_val = jr.split(rng)
            z = z.at[..., jr.choice(rng_idx, self.z_dim)].set(jr.uniform(rng_val))
        return z
