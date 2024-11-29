from typing import Callable
from jaxtyping import Float, Array, Key
from jax.scipy.stats import multivariate_normal
import optax
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.random as jr
from flax import nnx
from env import Simulator, Rollout


class Logger(dict):
    def log(self, **kwargs):
        for key, value in kwargs.items():
            if key not in self:
                self[key] = []
            self[key].append(value)

    def to_array(self):
        return {key: jnp.array(value) for key, value in self.items()}


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


class Qnet(nnx.Module):
    def __init__(self, obs_dim: int, a_dim: int, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.Q = MLP(obs_dim + a_dim, 1, rngs=rngs)

    def __call__(self, obs: Float[Array, "o"], a: Float[Array, "a"]):
        return self.Q(jnp.concatenate([obs, a], axis=-1))


class GaussianPolicy(nnx.Module):
    def __init__(self, obs_dim: int, a_dim: int, *, rngs: nnx.Rngs, action_range=None):
        self.rngs = rngs
        self.mu = MLP(obs_dim, a_dim, rngs=rngs)
        self.std = jnp.ones((a_dim,)) * noise
        self.action_range = action_range

    def __call__(self, obs: Float[Array, "o"]):
        mu = self.mu(obs)
        std = self.std.value
        if self.action_range is not None:
            amin, amax = self.action_range
            mu = jnp.clip(mu, amin, amax)
            std = jnp.clip(std, 1e-8, (amax - amin) / 2)
        cov = jnp.diag(std**2)
        return mu, cov

    def sample(self, obs: Float[Array, "o"]):
        mu, cov = self(obs)
        a = jr.multivariate_normal(self.rngs(), mu, cov)
        log_p = jnp.array(multivariate_normal.logpdf(a, mu, cov))
        return a, log_p


class Trainer(nnx.Module):
    def __init__(
        self,
        discount: float = 0.99,
        polyak_tau: float = 0.9,
        smoothing: float = 0.005,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.discount = discount
        self.polyak_tau = polyak_tau
        self.smoothing = smoothing

    def polyak_update(self, target: nnx.Module, source: nnx.Module):
        weighted_avg = lambda t, s: self.polyak_tau * t + (1 - self.polyak_tau) * s
        return jax.tree_map(weighted_avg, target, source)

    def train(
        self,
        env: Simulator,
        epochs: int,
        episode_length: int,
        lr: float = 1e-3,
        pool_size: int = 256,
        action_range=None,
    ):
        self.Q1 = Qnet(env.obs_dim + env.a_dim, 1, rngs=self.rngs)
        self.Q2 = Qnet(env.obs_dim + env.a_dim, 1, rngs=self.rngs)
        self.policy = GaussianPolicy(
            env.obs_dim, env.a_dim, rngs=self.rngs, action_range=action_range
        )
        optimizer_q1 = nnx.Optimizer(self.Q1, optax.adam(lr))
        optimizer_q2 = nnx.Optimizer(self.Q2, optax.adam(lr))
        optimizer_policy = nnx.Optimizer(self.policy, optax.adam(lr))

        self.Q1_target = nnx.clone(self.Q1)
        self.Q2_target = nnx.clone(self.Q2)
        self.policy_target = nnx.clone(self.policy)

        logger = Logger()
        for _ in (pbar := tqdm(range(epochs))):
            rollouts = self.get_rollouts(env, pool_size, episode_length)
            logger.log(reward=rollouts.r)

            for _ in range(10):
                loss_q1, loss_q2 = self.update_q_values(
                    optimizer_q1, optimizer_q2, rollouts
                )
                logger.log(loss_q1=loss_q1, loss_q2=loss_q2)
            loss_pi = self.update_policy(optimizer_policy, rollouts)
            logger.log(loss_pi=loss_pi)

            self.Q1_target = self.polyak_update(self.Q1_target, self.Q1)
            self.Q2_target = self.polyak_update(self.Q2_target, self.Q2)
            self.policy_target = self.polyak_update(self.policy_target, self.policy)

            pbar.set_postfix(
                loss_pi=loss_pi.mean().item(),
                loss_q=(loss_q1 + loss_q2).mean().item(),
                reward=rollouts.r.mean().item(),
            )
        return logger.to_array()

    @nnx.jit
    def update_policy(self, optimizer: nnx.Optimizer, steps: Rollout):
        @nnx.value_and_grad(has_aux=True)
        def losses(policy):
            a = jax.vmap(policy.mu(steps.obs))
            q = jax.vmap(self.Q1)(steps.obs, a).squeeze(-1)
            loss = -q
            return loss.mean(), loss

        (_, loss), grads = losses(optimizer.model)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def update_q_values(
        self,
        optimizer_q1: nnx.Optimizer,
        optimizer_q2: nnx.Optimizer,
        steps: Rollout,
    ):
        # Compute targets
        a_next = jax.vmap(self.policy_target.sample)(steps.next_obs)
        q1_next = jax.vmap(self.Q1_target)(steps.next_obs, a_next).squeeze(-1)
        q2_next = jax.vmap(self.Q2_target)(steps.next_obs, a_next).squeeze(-1)
        q_target = steps.r + self.discount * jnp.minimum(q1_next, q2_next)

        @nnx.value_and_grad(has_aux=True)
        def loss(Q_fn):
            q = jax.vmap(Q_fn)(steps.obs, steps.a).squeeze(-1)
            loss = (q - q_target) ** 2
            return loss.mean(), loss

        (_, q1_loss), grads = loss(optimizer_q1.model)
        optimizer_q1.update(grads)
        (_, q2_loss), grads = loss(optimizer_q2.model)
        optimizer_q2.update(grads)
        return q1_loss, q2_loss

    @nnx.jit(static_argnames=("pool_size", "episode_length"))
    def get_rollouts(self, env: Simulator, pool_size: int, episode_length: int):
        def make_pool(m: nnx.Module):
            graph, rng, par, var = nnx.split(m, nnx.RngState, nnx.Param, nnx.Variable)
            var = jax.tree_map(lambda x: jnp.stack([x] * pool_size), var)
            return nnx.merge(graph, rng, par, var)

        map_axes = nnx.StateAxes({nnx.Param: None, (nnx.RngState, nnx.Variable): 0})

        @nnx.split_rngs(splits=pool_size)
        @nnx.vmap(in_axes=(map_axes, map_axes))
        def rollouts(env, policy):
            rollout, outs = env.rollout(policy, episode_length)
            return rollout

        return rollouts(make_pool(env), make_pool(self.policy))
