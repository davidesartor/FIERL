from typing import NamedTuple, Any, Protocol
from jaxtyping import Float, Array, Key
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal
from flax import nnx


class MLP(nnx.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim=32, *, rngs: nnx.Rngs):
        super().__init__(
            nnx.Linear(in_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.RMSNorm(hidden_dim, rngs=rngs),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.RMSNorm(hidden_dim, rngs=rngs),
            nnx.Linear(
                hidden_dim, out_dim, rngs=rngs, kernel_init=nnx.initializers.zeros
            ),
        )


class GaussianPolicy(nnx.Module):
    def __init__(
        self,
        obs_dim: int,
        a_dim: int,
        variable_mu=True,
        correlations=False,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.mu = (
            MLP(obs_dim, a_dim, rngs=rngs)
            if variable_mu
            else nnx.Param(jnp.zeros((a_dim,)))
        )
        self.std = nnx.Param(jnp.ones((a_dim,)))
        self.corr = nnx.Param(jnp.zeros((a_dim, a_dim))) if correlations else None

    def __call__(self, obs: Float[Array, "o"]):
        mu = self.mu(obs) if isinstance(self.mu, MLP) else self.mu.value
        cov = jnp.diag(self.std**2 + 1e-8)
        if self.corr is not None:
            # cayley parameterization for orthogonal matrix
            A = self.corr.value - self.corr.value.T  # skew-symmetric matrix
            Q = (jnp.eye(len(mu)) - A / 2) @ jnp.linalg.inv(jnp.eye(len(mu)) + A / 2)
            cov = Q @ cov @ Q.T
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


def get_returns(r: Float[Array, "t"], discount: float) -> Float[Array, "t"]:
    def accumulate(v, ri):
        v = ri + discount * v
        return v, v

    J0, Jt = jax.lax.scan(accumulate, jnp.zeros_like(r[0]), r, reverse=True)
    return Jt


def quadratic_cost(
    s: Float[Array, "t d"], J: Float[Array, "d d"] | Float[Array, "d"] | float
) -> Float[Array, "t"]:
    J = jnp.asarray(J)
    if J.ndim == 2:
        return jnp.einsum("ti, ij, tj->t", s, J, s)
    return jnp.sum(J * s**2, axis=-1)


def tanh_clip(x: Float[Array, "..."], neg: float, pos: float | None = None):
    c = jnp.where(x < 0.0, neg, pos or neg)
    return jnp.tanh(x / c) * c
