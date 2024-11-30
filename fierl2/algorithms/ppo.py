from math import prod
from utils import *
from tqdm import tqdm
from jax.scipy.stats import multivariate_normal
from flax import nnx
import optax


class GaussianPolicy(nnx.Module):
    def __init__(self, obs_dim: int, a_dim: int, correlations=False, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.mu = MLP(obs_dim, a_dim, rngs=rngs)
        self.std = nnx.Param(jnp.eye(a_dim) if correlations else jnp.ones((a_dim,)))

    def __call__(self, obs: Float[Array, "o"]):
        mu = self.mu(obs)
        if self.std.ndim == 1:
            cov = jnp.diag(self.std**2 + 1e-8)
        elif self.std.ndim == 2:
            cov = self.std @ self.std.T + jnp.eye(mu.shape[-1]) * 1e-8
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


class Trainer(nnx.Module):
    def __init__(
        self,
        env,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_pi: float = 0.1,
        clip_vf: float = 1.0,
        entropy_weight: float = 0.0001,
        normalize_advantages: bool = True,
        correlations: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_pi = clip_pi
        self.clip_vf = clip_vf
        self.entropy_weight = entropy_weight
        self.normalize_advantages = normalize_advantages

        self.rngs = rngs
        self.env = env
        self.value = MLP(env.obs_dim, 1, rngs=self.rngs)
        self.policy = GaussianPolicy(
            env.obs_dim, env.a_dim, correlations, rngs=self.rngs
        )

    def train(
        self,
        epochs: int,
        episode_length: int,
        pool_size: int = 256,
        lr: float = 3e-4,
        wd: float = 1e-4,
    ):
        optimizer_pi = nnx.Optimizer(self.policy, optax.adamw(lr, weight_decay=wd))
        optimizer_vf = nnx.Optimizer(self.value, optax.adamw(lr, weight_decay=wd))

        logger = Logger()
        for i in (pbar := tqdm(range(epochs))):
            rollouts = self.get_rollouts(pool_size, episode_length)
            logger.log(reward=rollouts.r, cost=rollouts.c)

            V, A = self.estimate_V_A(rollouts)
            if self.normalize_advantages:
                A = (A - A.mean()) / (A.std() + 1e-8)
            logger.log(V=V, A=A)

            for _ in range(10):
                loss_pi = self.policy_optimization_step(optimizer_pi, rollouts, A)
                loss_vf = self.value_optimization_step(optimizer_vf, rollouts, V)
                logger.log(loss_vf=loss_vf, loss_pi=loss_pi)

            pbar.set_postfix(
                r=rollouts.r.mean().item(),
                c=rollouts.c.mean().item(),
                V=V.mean().item(),
            )
        return logger.to_array()

    @nnx.jit(static_argnames=("pool_size", "episode_length"))
    def get_rollouts(self, pool_size: int, episode_length: int):
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

        steps = rollouts(make_pool(self.env), make_pool(self.policy))
        return steps

    @nnx.jit
    @nnx.vmap(in_axes=(None, 0))
    def estimate_V_A(self, rollouts: Rollout):
        obs = jnp.concat([rollouts.obs, rollouts.next_obs[-1:]], axis=0)
        V = jax.vmap(self.value)(obs).squeeze(-1)
        delta = (rollouts.r - rollouts.c) - V[:-1] + self.discount * V[1:]
        _, A = get_returns(delta, self.discount * self.gae_lambda)
        V = V[:-1] + A
        return V, A

    @nnx.jit
    def policy_optimization_step(
        self, optimizer: nnx.Optimizer, rollouts: Rollout, A: Float[Array, "n t"]
    ):
        @nnx.value_and_grad
        def loss(policy):
            # surrogate loss
            log_p, entropy = jax.vmap(jax.vmap(policy.eval))(rollouts.obs, rollouts.a)
            ratio = jnp.exp(log_p - rollouts.log_p)
            ratio_clip = 1 + tanh_clip(ratio - 1, self.clip_pi)
            loss = -jnp.minimum(A * ratio, A * ratio_clip)
            # entropy regularization
            loss = loss - self.entropy_weight * entropy
            return loss.mean()

        loss, grads = loss(self.policy)
        optimizer.update(grads)
        return loss

    @nnx.jit
    def value_optimization_step(
        self, optimizer: nnx.Optimizer, rollouts: Rollout, V: Float[Array, "n t"]
    ):
        @nnx.value_and_grad
        def loss(vf):
            V_pred = jax.vmap(jax.vmap(vf))(rollouts.obs).squeeze(-1)
            delta = tanh_clip(V_pred - V, self.clip_vf)
            return optax.l2_loss(delta).mean()

        loss, grads = loss(self.value)
        optimizer.update(grads)
        return loss
