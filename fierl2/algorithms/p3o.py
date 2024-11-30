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
        max_cost: float = 0.0,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_pi: float = 0.1,
        clip_vf: float = 1.0,
        clip_cvf: float = 1.0,
        cost_loss_weight: float = 10.0,
        entropy_weight: float = 0.0001,
        normalize_advantages: bool = True,
        correlations: bool = False,
        *,
        rngs: nnx.Rngs,
    ):
        self.max_cost = max_cost
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_pi = clip_pi
        self.clip_vf = clip_vf
        self.clip_cvf = clip_cvf
        self.cost_loss_weight = cost_loss_weight
        self.entropy_weight = entropy_weight
        self.normalize_advantages = normalize_advantages

        self.rngs = rngs
        self.env = env
        self.value_fn = MLP(env.obs_dim, 1, rngs=self.rngs)
        self.cvalue_fn = MLP(env.obs_dim, 1, rngs=self.rngs)
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
        optimizer_vf = nnx.Optimizer(self.value_fn, optax.adamw(lr, weight_decay=wd))
        optimizer_cvf = nnx.Optimizer(self.cvalue_fn, optax.adamw(lr, weight_decay=wd))

        logger = Logger()
        for i in (pbar := tqdm(range(epochs))):
            rollouts = self.get_rollouts(pool_size, episode_length)
            logger.log(reward=rollouts.r, cost=rollouts.c)

            V, C, A, Ac = self.estimate_V_C_A(rollouts)
            if self.normalize_advantages:
                A = (A - A.mean()) / (A.std() + 1e-8)
                Ac = (Ac - Ac.mean()) / (Ac.std() + 1e-8)
            logger.log(V=V, C=C, A=A, Ac=Ac)

            for _ in range(10):
                loss_pi, loss_cpi = self.policy_optimization_step(
                    optimizer_pi, rollouts, A, Ac
                )
                loss_vf, loss_cvf = self.value_optimization_step(
                    optimizer_vf, optimizer_cvf, rollouts.obs, V, C
                )
                logger.log(
                    loss_vf=loss_vf,
                    loss_cvf=loss_cvf,
                    loss_pi=loss_pi,
                    loss_cpi=loss_cpi,
                )

            pbar.set_postfix(
                reward=rollouts.r.mean().item(),
                cost=rollouts.c.mean().item(),
                V=V[:, 0].mean().item(),
                C=C[:, 0].mean().item(),
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
    def estimate_V_C_A(self, rollouts: Rollout):
        obs = jnp.concat([rollouts.obs, rollouts.next_obs[-1:]], axis=0)
        V = jax.vmap(self.value_fn)(obs).squeeze(-1)
        C = jax.vmap(self.cvalue_fn)(obs).squeeze(-1)
        delta_r = rollouts.r - V[:-1] + self.discount * V[1:]
        delta_c = rollouts.c - C[:-1] + self.discount * C[1:]

        _, A = get_returns(delta_r, self.discount * self.gae_lambda)
        _, Ac = get_returns(delta_c, self.discount * self.gae_lambda)
        V = V[:-1] + A
        C = C[:-1] + Ac
        return V, C, A, Ac

    @nnx.jit
    def policy_optimization_step(
        self,
        optimizer: nnx.Optimizer,
        rollouts: Rollout,
        A: Float[Array, "n t"],
        Ac: Float[Array, "n t"],
    ):
        @nnx.value_and_grad(has_aux=True)
        def loss(policy):
            log_p, entropy = jax.vmap(jax.vmap(policy.eval))(rollouts.obs, rollouts.a)
            ratio = jnp.exp(log_p - rollouts.log_p)
            ratio_clip = 1 + tanh_clip(ratio - 1, self.clip_pi)
            # surrogate loss
            loss_pi = -jnp.minimum(A * ratio, A * ratio_clip)
            loss_pi = jnp.mean(loss_pi - self.entropy_weight * entropy)

            # cost surrogate loss
            C, _ = jax.vmap(lambda c: get_returns(c, self.discount))(rollouts.c)
            avg_violation = (1 - self.discount) * (self.max_cost - C.mean())
            loss_c = jnp.maximum(Ac * ratio, Ac * ratio_clip)
            loss_c = jnp.clip(loss_c, min=avg_violation).mean()

            loss = loss_pi + self.cost_loss_weight * loss_c
            return loss, (loss_pi, loss_c)

        (loss, (loss_pi, loss_c)), grads = loss(optimizer.model)
        optimizer.update(grads)
        return loss_pi, loss_c

    @nnx.jit
    def value_optimization_step(
        self,
        optimizer_vf: nnx.Optimizer,
        optimizer_cvf: nnx.Optimizer,
        obs: Float[Array, "n t o"],
        V: Float[Array, "n t"],
        C: Float[Array, "n t"],
    ):
        @nnx.value_and_grad
        def loss(vf, target):
            V = jax.vmap(jax.vmap(vf))(obs).squeeze(-1)
            delta = tanh_clip(V - target, self.clip_vf)
            return optax.l2_loss(delta).mean()

        loss_vf, grads = loss(self.value_fn, V)
        optimizer_vf.update(grads)
        loss_cvf, grads = loss(self.cvalue_fn, C)
        optimizer_cvf.update(grads)
        return loss_vf, loss_cvf
