from dataclasses import InitVar
from typing import Callable, NamedTuple
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
import jax.random as jr
from jax.scipy.stats import multivariate_normal
import equinox as eqx


class MLP(eqx.nn.MLP):
    layers: tuple[eqx.nn.Linear, ...]
    activation: Callable = eqx.field(static=True, default=jax.nn.gelu)
    final_activation: Callable = eqx.field(static=True, default=lambda x: x)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PPO(eqx.Module):
    mlp_mu: MLP = eqx.field(init=False)
    mlp_cov: MLP = eqx.field(init=False)
    mlp_V: MLP = eqx.field(init=False)
    x_dim: InitVar[int]
    u_dim: InitVar[int]
    mlp_kwargs: InitVar[dict] = {"width_size": 32, "depth": 2}
    seed: InitVar[int] = 0
    discount: float = eqx.field(static=True, default=0.99)
    gae_lambda: float = eqx.field(static=True, default=0.95)
    clip_range_pi: float = eqx.field(static=True, default=0.1)
    clip_range_vf: float = eqx.field(static=True, default=0.1)
    entropy_weight: float = eqx.field(static=True, default=0.0001)

    def __post_init__(self, x_dim, u_dim, mlp_kwargs, seed):
        rng_mu, rng_cov, rng_V = jr.split(jr.key(seed), 3)
        in_size = x_dim + x_dim**2
        self.mlp_V = MLP(in_size, 1, **mlp_kwargs, key=rng_V)
        self.mlp_mu = MLP(in_size, u_dim, **mlp_kwargs, key=rng_mu)
        self.mlp_cov = MLP(in_size, u_dim, **mlp_kwargs, key=rng_cov)

    def __call__(
        self, obs: Float[Array, "o"], *, rng=None, a=None
    ) -> tuple[Float[Array, "u"], Float[Array, ""], Float[Array, ""]]:
        # foward pass of the policy networks
        mu = self.mlp_mu(obs)
        cov = jnp.diag(jax.nn.sigmoid(self.mlp_cov(obs)) + 1e-8)

        # sample action, calculate log probability and entropy
        if a is None:
            a = mu if rng is None else jr.multivariate_normal(rng, mu, cov)

        log_p = jnp.array(multivariate_normal.logpdf(a, mu, cov))
        entropy = 0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * jnp.e * cov))
        return a, log_p, entropy

    def value(self, x: Float[Array, "x"]) -> Float[Array, ""]:
        return self.mlp_V(x).squeeze()

    def gen_advantage_estimation(
        self,
        obs: Float[Array, "t o"],
        rewards: Float[Array, "t"],
        last_obs: Float[Array, "o"] | None = None,
    ) -> tuple[Float[Array, "t"], Float[Array, "t"]]:
        def tail_sum_step(v, xi):
            v = xi + self.discount * self.gae_lambda * v
            return v, v

        V = eqx.filter_vmap(self.value)(obs)
        V_last = self.value(last_obs) if last_obs is not None else jnp.zeros(())
        V_next = jnp.roll(V, -1).at[-1].set(V_last)
        delta = rewards - V + self.discount * V_next
        _, A = jax.lax.scan(tail_sum_step, V_last, delta, reverse=True)
        V = A + V
        return V, A

    def loss(
        self,
        obs: Float[Array, "o"],
        a: Float[Array, "u"],
        log_p: Float[Array, ""],
        A: Float[Array, ""],
        V: Float[Array, ""],
    ) -> Float[Array, ""]:
        # foward pass of the policy networks
        _, new_log_p, new_entropy = self(obs, a=a)
        new_V = self.value(obs)

        # surrogate loss
        ratio_pi = jnp.exp(new_log_p - log_p)
        ratio_pi_clip = ratio_pi.clip(1 - self.clip_range_pi, 1 + self.clip_range_pi)
        loss_policy = -jnp.minimum(A * ratio_pi, A * ratio_pi_clip)

        # value function loss
        new_V = new_V.clip(V * (1 - self.clip_range_vf), V * (1 + self.clip_range_vf))
        loss_value_function = (new_V - V) ** 2

        # entropy loss
        loss_entropy = -self.entropy_weight * new_entropy
        return loss_policy + loss_value_function + loss_entropy
