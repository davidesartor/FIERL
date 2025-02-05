from typing import NamedTuple, Any, Protocol
from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal
from flax import nnx


class PickableLinear(nnx.Linear):
    # make the module pickable
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.kernel_init = None
        self.bias_init = None


class MLP(nnx.Module):
    def __init__(self, in_dim, hidden_dim, out_dim, *, rngs: nnx.Rngs):
        self.lin1 = PickableLinear(in_dim, hidden_dim, rngs=rngs)
        self.norm1 = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.lin2 = PickableLinear(hidden_dim, hidden_dim, rngs=rngs)
        self.norm2 = nnx.RMSNorm(hidden_dim, rngs=rngs)
        self.lin3 = PickableLinear(
            hidden_dim, out_dim, rngs=rngs, kernel_init=nnx.initializers.zeros
        )

    def __call__(self, x):
        x = jax.nn.silu(self.lin1(x))
        x = jax.nn.silu(self.lin2(self.norm1(x)))
        x = self.lin3(self.norm2(x))
        return x


class GaussianPolicy(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        a_dim: int,
        hidden_dim: int,
        stationary=False,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.mu = (
            MLP(obs_dim, hidden_dim, a_dim, rngs=rngs)
            if not stationary
            else nnx.Param(jnp.zeros((a_dim,)))
        )
        self.std = nnx.Param(jnp.eye(a_dim) if stationary else jnp.ones((a_dim,)))

    def __call__(self, obs: Float[Array, "o"]):
        mu = self.mu(obs) if isinstance(self.mu, MLP) else self.mu.value
        std = self.std.value
        if self.std.ndim == 1:
            std = jnp.diag(std)
        cov = std @ std.T + 1e-8 * jnp.eye(mu.shape[-1])
        return mu, cov

    def sample(self, obs: Float[Array, "o"]):
        mu, cov = self(obs)
        a = jr.multivariate_normal(self.rngs(), mu, cov)
        log_p = jnp.array(multivariate_normal.logpdf(a, mu, cov))
        return a, log_p

    def eval(self, obs: Float[Array, "o"], a: Float[Array, "a"]):
        mu, cov = self(obs)
        log_p = jnp.array(multivariate_normal.logpdf(a, mu, cov))
        ent = 0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * jnp.e * cov))
        return log_p, ent


class Logger(dict):
    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self:
                self[key] = []
            self[key].append(value)

    def to_array(self):
        return {key: jnp.array(value) for key, value in self.items()}


class Rollout(NamedTuple):
    obs: Float[Array, "..."]
    a: Float[Array, "u"]
    log_p: Float[Array, ""]
    r: Float[Array, ""]
    next_obs: Float[Array, "..."]
    c: Float[Array, ""] = jnp.zeros(())


@nnx.jit(static_argnames=("pool_size"))
def get_rollouts(env, policy, pool_size: int):
    def make_pool(m: nnx.Module):
        graph, rng, par, var = nnx.split(m, nnx.RngState, nnx.Param, nnx.Variable)
        var = jax.tree_map(lambda x: jnp.stack([x] * pool_size), var)
        return nnx.merge(graph, rng, par, var)

    map_axes = nnx.StateAxes({nnx.Param: None, (nnx.RngState, nnx.Variable): 0})

    @nnx.split_rngs(splits=pool_size)
    @nnx.vmap(in_axes=(map_axes, map_axes))
    def rollouts(env, policy):
        rollout, outs = env.rollout(policy)
        return rollout, outs

    steps, outs = rollouts(make_pool(env), make_pool(policy))
    return steps, outs


def get_returns(r: Float[Array, "t"], discount: float) -> Float[Array, "t"]:
    def accumulate(v, ri):
        v = ri + discount * v
        return v, v

    J0, Jt = jax.lax.scan(accumulate, jnp.zeros_like(r[0]), r, reverse=True)
    return Jt


def quadratic_cost(
    s: Float[Array, "... d"], J: Float[Array, "d d"] | Float[Array, "d"] | float
) -> Float[Array, "..."]:
    J = jnp.asarray(J)
    if J.ndim == 2:
        return jnp.einsum("...i, ij, ...j->...", s, J, s)
    return jnp.sum(J * s**2, axis=-1)


def tanh_clip(x: Float[Array, "..."], neg: float, pos: float | None = None):
    c = jnp.where(x < 0.0, neg, pos or neg)
    return jnp.tanh(x / c) * c
