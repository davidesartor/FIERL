from tqdm import tqdm
from flax import nnx
import optax
from utils import *


class Trainer(nnx.Module):
    def __init__(
        self,
        env,
        max_cost: float = 0.0,
        discount: float = 0.9,
        gae_lambda: float = 0.9,
        clip_pi: float = 0.3,
        clip_vf: float = 1.0,
        cost_weight: float = 20.0,
        entropy_weight: float = 0.0001,
        normalize_advantages: bool = True,
        policy_params: dict = {},
        *,
        rngs: nnx.Rngs,
    ):
        self.max_cost = max_cost
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_pi = clip_pi
        self.clip_vf = clip_vf
        self.cost_weight = cost_weight
        self.entropy_weight = entropy_weight
        self.normalize_advantages = normalize_advantages

        self.rngs = rngs
        self.env = env
        self.value_fn = MLP(env.obs_dim, 1, rngs=self.rngs)
        self.cvalue_fn = MLP(env.obs_dim, 1, rngs=self.rngs)
        self.policy = GaussianPolicy(
            env.obs_dim, env.a_dim, **policy_params, rngs=self.rngs
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
            V, A, Vc, Ac, violation = self.estimate_returns_and_advantages(rollouts)
            logger.log(reward=rollouts.r, cost=rollouts.c, V=V, Vc=Vc, A=A, Ac=Ac)

            for _ in range(10):
                loss_pi, loss_cpi = self.policy_optimization_step(
                    optimizer_pi, rollouts, A, Ac, violation
                )
                loss_vf, loss_cvf = self.value_optimization_step(
                    optimizer_vf, optimizer_cvf, rollouts.obs, V, Vc
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
                Vc=Vc[:, 0].mean().item(),
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
    def estimate_returns_and_advantages(self, rollouts, last_is_done=True):
        @nnx.vmap(in_axes=(None, 0, 0))
        def gae_estimate(vf, obs, signal):
            V = jax.vmap(vf)(obs).squeeze(-1)
            if last_is_done:
                V = V.at[-1].set(0.0)
            delta = signal - V[:-1] + self.discount * V[1:]
            A = get_returns(delta, self.discount * self.gae_lambda)
            V = V[:-1] + A
            return V, A

        obs = jnp.concat([rollouts.obs, rollouts.next_obs[:, -1:]], axis=1)
        V, A = gae_estimate(self.value_fn, obs, rollouts.r)
        Vc, Ac = gae_estimate(self.cvalue_fn, obs, rollouts.c)
        violation = (1 - self.discount) * (Vc[:, 0].mean() - self.max_cost)

        if self.normalize_advantages:
            # DO NOT CENTER COST ADVANTAGES!!! (it changes the clipping threshold)
            A = A / (A.std() + 1e-8)
            # violation = violation / (Ac.std() + 1e-8)
            # Ac = Ac / (Ac.std() + 1e-8)
        return V, A, Vc, Ac, violation

    @nnx.jit
    def policy_optimization_step(
        self,
        optimizer: nnx.Optimizer,
        rollouts: Rollout,
        A: Float[Array, "n t"],
        Ac: Float[Array, "n t"],
        violation: Float[Array, ""],
    ):
        @nnx.value_and_grad(has_aux=True)
        def loss(policy):
            log_p, entropy = jax.vmap(jax.vmap(policy.eval))(rollouts.obs, rollouts.a)
            ratio = jnp.exp(log_p - rollouts.log_p)
            ratio_clip = 1 + tanh_clip(ratio - 1, self.clip_pi)

            # cost surrogate loss
            gained_c = jnp.maximum(Ac * ratio, Ac * ratio_clip).mean()
            loss_c = self.cost_weight * jnp.clip(violation + gained_c, min=0.0)

            # surrogate loss (why not clip this when the cost is violated ??)
            loss_pi = -jnp.minimum(A * ratio, A * ratio_clip).mean()

            # entropy regularization
            loss_ent = -self.entropy_weight * entropy.mean()
            return (loss_pi + loss_c + loss_ent), (loss_pi, loss_c)

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
        Vc: Float[Array, "n t"],
    ):
        @nnx.value_and_grad
        def loss(vf, target):
            pred = jax.vmap(jax.vmap(vf))(obs).squeeze(-1)
            # use huber loss instead of mse + clipping
            return optax.huber_loss(pred, target, self.clip_vf).mean()

        loss_vf, grads_vf = loss(self.value_fn, V)
        optimizer_vf.update(grads_vf)
        loss_cvf, grads_cvf = loss(self.cvalue_fn, Vc)
        optimizer_cvf.update(grads_cvf)
        return loss_vf, loss_cvf
