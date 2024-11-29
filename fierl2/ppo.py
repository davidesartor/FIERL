from utils import *
from tqdm import tqdm
from jax.scipy.stats import multivariate_normal
from flax import nnx
import optax


class MLP(nnx.Sequential):
    def __init__(self, in_dim, out_dim, hidden_dim=64, *, rngs: nnx.Rngs):
        super().__init__(
            nnx.Linear(in_dim, hidden_dim, rngs=rngs),
            nnx.gelu,
            nnx.RMSNorm(hidden_dim, rngs=rngs),
            nnx.Linear(hidden_dim, hidden_dim, rngs=rngs),
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
    def __init__(self, obs_dim: int, a_dim: int, *, rngs: nnx.Rngs):
        self.rngs = rngs
        self.mu = MLP(obs_dim, a_dim, rngs=rngs)
        self.std_diag = nnx.Param(jnp.ones((a_dim,)))
        self.std_offdiag = MLP(obs_dim, a_dim**2, rngs=rngs)

    def __call__(self, obs: Float[Array, "o"]):
        mu = self.mu(obs)
        # corr = self.std_offdiag(obs)
        # corr = corr.reshape(mu.shape[-1], mu.shape[-1])
        cov = jnp.diag(self.std_diag**2)  # + corr @ corr.T
        return mu, cov

    def sample(self, obs: Float[Array, "o"]):
        mu, cov = self(obs)
        a = jr.multivariate_normal(self.rngs(), mu, cov)
        return a

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
        clip_pi: float = 0.3,
        clip_vf: float = 1.0,
        entropy_weight: float = 0.0,
        normalize_advantages: bool = True,
        *,
        rngs: nnx.Rngs,
    ):
        self.rngs = rngs
        self.env = env
        self.value = MLP(env.obs_dim, 1, rngs=self.rngs)
        self.policy = GaussianPolicy(env.obs_dim, env.a_dim, rngs=self.rngs)

        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_pi = clip_pi
        self.clip_vf = clip_vf
        self.entropy_weight = entropy_weight
        self.normalize_advantages = normalize_advantages

    def train(
        self,
        epochs: int,
        episode_length: int,
        lr: float = 1e-4,
        wd: float = 1e-3,
        pool_size: int = 256,
    ):
        self.optimizer_pi = nnx.Optimizer(self.policy, optax.adamw(lr, weight_decay=wd))
        self.optimizer_vf = nnx.Optimizer(self.value, optax.adamw(lr, weight_decay=wd))

        logger = Logger()
        for i in (pbar := tqdm(range(epochs))):
            steps, log_p = self.get_rollouts(pool_size, episode_length)
            V0, V_target, A = self.estimate_V_A(steps.obs, steps.r, steps.next_obs)
            if self.normalize_advantages:
                A = (A - A.mean()) / (A.std() + 1e-8)

            for _ in range(10):
                loss_pi = self.policy_optimization_step(steps.obs, steps.a, log_p, A)
                loss_vf = self.value_optimization_step(steps.obs, V0, V_target)
                logger.log(loss_vf=loss_vf, loss_pi=loss_pi)

            logger.log(reward=steps.r, V=V0, A=A, loss_pi=loss_pi, loss_vf=loss_vf)
            pbar.set_postfix(
                V0=V0.mean().item(),
                loss_pi=loss_pi.mean().item(),
                loss_vf=loss_vf.mean().item(),
                reward=steps.r.mean().item(),
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
            return rollout, outs

        steps, outs = rollouts(make_pool(self.env), make_pool(self.policy))
        log_p, entropy = jax.vmap(jax.vmap(self.policy.eval))(steps.obs, steps.a)
        return steps, log_p

    @nnx.jit
    @nnx.vmap(in_axes=(None, 0, 0, 0))
    def estimate_V_A(
        self,
        obs: Float[Array, "n t o"],
        rewards: Float[Array, "n t"],
        next_obs: Float[Array, "n t o"],
    ):
        def tail_sum_step(v, xi):
            v = xi + self.discount * self.gae_lambda * v
            return v, v

        obs = jnp.concat([obs, next_obs[-1:]], axis=0)
        V = jax.vmap(self.value)(obs).squeeze(-1)
        delta = rewards - V[:-1] + self.discount * V[1:]
        _, A = jax.lax.scan(tail_sum_step, jnp.zeros(()), delta, reverse=True)
        V_target = V[:-1] + A
        return V[:-1], V_target, A

    @nnx.jit
    def policy_optimization_step(
        self,
        obs: Float[Array, "n t o"],
        a: Float[Array, "n t a"],
        log_p: Float[Array, "n t"],
        A: Float[Array, "n t"],
    ):
        @nnx.value_and_grad
        def loss(policy):
            # surrogate loss
            new_log_p, entropy = jax.vmap(jax.vmap(policy.eval))(obs, a)
            ratio = jnp.exp(new_log_p - log_p)
            ratio_clip = jnp.clip(ratio, 1 + self.clip_pi, 1 - self.clip_pi)
            loss = -jnp.minimum(A * ratio, A * ratio_clip)
            # entropy regularization
            loss = loss - self.entropy_weight * entropy
            return loss.mean()

        loss, grads = loss(self.policy)
        self.optimizer_pi.update(grads)
        return loss

    @nnx.jit
    def value_optimization_step(
        self,
        obs: Float[Array, "n t o"],
        V0: Float[Array, "n t"],
        V_target: Float[Array, "n t"],
    ):
        @nnx.value_and_grad
        def loss(vf):
            V = jax.vmap(jax.vmap(vf))(obs).squeeze(-1)
            # V = V0 + tanh_clip(V - V0, self.clip_vf)
            # delta = tanh_clip(V - V_target, self.clip_vf)
            delta = V - V_target
            return optax.l2_loss(delta).mean()

        loss, grads = loss(self.value)
        self.optimizer_vf.update(grads)
        return loss
