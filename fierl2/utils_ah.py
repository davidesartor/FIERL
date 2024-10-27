from typing import ClassVar
from systems import DSSM
import jax
import jax.numpy as jnp
import jax.random as jr


def parallel(*systems: DSSM):
    assert all(sys.x_dim == systems[0].x_dim for sys in systems)
    assert all(sys.u_dim == systems[0].u_dim for sys in systems)
    assert all(sys.y_dim == systems[0].y_dim for sys in systems)

    class MergedSystem(DSSM):
        x_dim: ClassVar[int] = systems[0].x_dim * len(systems)
        u_dim: ClassVar[int] = systems[0].u_dim
        y_dim: ClassVar[int] = systems[0].y_dim * len(systems)

        def init_state(self, *, key=None):
            x = jnp.concat([sys.init_state(key=key) for sys in systems], axis=-1)
            return x

        def step(self, x, u, *, key=None):
            xs = jnp.split(x, len(systems), axis=-1)
            x, y = zip(*(sys.step(x, u, key=key) for sys, x in zip(systems, xs)))
            return jnp.concatenate(x, axis=-1), jnp.concatenate(y, axis=-1)

    return MergedSystem()


def redundant_actuators(system_cls: type[DSSM], copies=2):
    if isinstance(copies, int):
        copies = (copies,) * system_cls.u_dim
    lims = jnp.cumsum(jnp.array((0, *copies)))

    class RedundantSystem(system_cls):
        u_dim: ClassVar[int] = sum(copies)

        def step(self, x, u, *, key=None):
            u = [u[..., start:stop] for start, stop in zip(lims[:-1], lims[1:])]
            u = jnp.split(u, copies, axis=-1)
            u = efficiencies.T * jnp.reshape(u, efficiencies.T.shape)
            return super().step(x, u.sum(axis=-1), key=key)

    return RedundantSystem
