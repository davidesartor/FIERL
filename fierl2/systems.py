from typing import ClassVar
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class DSSM(eqx.Module):
    x_dim: ClassVar[int]
    u_dim: ClassVar[int]
    y_dim: ClassVar[int]

    def init_state(self, *, key=None) -> jax.Array:
        raise NotImplementedError

    def step(self, x, u, *, key=None) -> jax.Array:
        raise NotImplementedError

    def measure(self, x, *, key=None) -> jax.Array:
        raise NotImplementedError

    def simulate(self, ut, x0=None, *, key=jr.key(0)):
        def scan_fn(x, input):
            u, key = input
            kx, ky = jr.split(key, 2)
            x = self.step(x, u, key=kx)
            y = self.measure(x, key=ky)
            return x, y

        key_x0, key_steps = jr.split(key)
        key_steps = jr.split(key_steps, len(ut))
        x0 = x0 or self.init_state(key=key_x0)
        xfinal, yt = jax.lax.scan(scan_fn, x0, (ut, key_steps))
        return xfinal, yt


class Car(DSSM):
    dt: float = 1.0
    mass: float = 1.0

    x_dim: ClassVar[int] = 5
    u_dim: ClassVar[int] = 2
    y_dim: ClassVar[int] = 2

    def init_state(self, *, key=None):
        theta = 0.0
        nx, ny = jnp.cos(theta), jnp.sin(theta)
        x, y = 0.0, 0.0
        v = 0.0
        return jnp.array([x, y, nx, ny, v])

    def step(self, x, u, *, key=None):
        def scan_fn(carry, dt):
            p1, p2, n1, n2, v = jnp.split(carry, 5, axis=-1)
            f, w = jnp.split(u, 2, axis=-1)

            theta = jnp.arctan2(n2, n1) + dt * w
            nx, ny = jnp.cos(theta), jnp.sin(theta)
            v = v + dt * f / self.mass
            p1 = p1 + dt * v * nx
            p2 = p2 + dt * v * ny
            carry = jnp.concatenate([p1, p2, nx, ny, v], axis=-1)
            return carry, None

        assert x.shape[-1] == 5 and u.shape[-1] == 2
        x, _ = jax.lax.scan(scan_fn, x, self.dt / 10 * jnp.ones(10))
        return x

    def measure(self, x, *, key=None):
        return x[..., :2]
