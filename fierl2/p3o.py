from tqdm import tqdm
from flax import nnx
import optax
from utils import *


class Trainer(nnx.Module):
    def __init__(
        self,
        env,
        max_cost: float = 0.0,
        pool_size: int = 256,
        discount: float = 0.99,
        gae_lambda: float = 0.95,
        clip_pi: float = 0.3,
        clip_vf: float = 10.0,
        min_cost_weight: float = 0.0,
        max_cost_weight: float = 20.0,
        min_entropy_weight: float = 1e-8,
        max_entropy_weight: float = 1e-3,
        normalize_advantages: bool = True,
        stationary: bool = False,
        policy_hidden_dim: int = 16,
        value_hidden_dim: int = 32,
        *,
        rngs: nnx.Rngs | None = None,
    ):
        self.max_cost = max_cost
        self.pool_size = pool_size
        self.discount = discount
        self.gae_lambda = gae_lambda
        self.clip_pi = clip_pi
        self.clip_vf = clip_vf
        self.normalize_advantages = normalize_advantages
        self.min_cost_weight = min_cost_weight
        self.max_cost_weight = max_cost_weight
        self.min_entropy_weight = min_entropy_weight
        self.max_entropy_weight = max_entropy_weight

        self.rngs = rngs if rngs is not None else env.rngs
        self.logger = Logger()
        self.env = env
        self.value_fn = MLP(env.obs_dim, value_hidden_dim, 1, rngs=self.rngs)
        self.cvalue_fn = MLP(env.obs_dim, value_hidden_dim, 1, rngs=self.rngs)
        self.policy = GaussianPolicy(
            env.obs_dim, env.a_dim, policy_hidden_dim, stationary, rngs=self.rngs
        )

        # adjust_init_variance
        for i in range(10):
            rollouts, outs = get_rollouts(self.env, self.policy, self.pool_size)
            Jc = jax.vmap(lambda costs: get_returns(costs, self.discount))(rollouts.c)
            if Jc[:, 0].mean() < self.max_cost:
                break
            self.policy.std /= 2
        self.policy.std *= 2
        self.untrained_policy = nnx.clone(self.policy)

    def train(
        self,
        epochs: int,
        policy_lr: float = 1e-4,
        policy_steps: int = 10,
        value_lr: float = 1e-3,
        value_steps: int = 10,
    ):
        self.logger = Logger()
        optimizer_vf = nnx.Optimizer(self.value_fn, optax.adamw(value_lr))
        optimizer_cvf = nnx.Optimizer(self.cvalue_fn, optax.adamw(value_lr))
        optimizer_pi = nnx.Optimizer(self.policy, optax.adamw(policy_lr))

        temperatures = (1 + jnp.cos(jnp.pi * jnp.linspace(0.0, 1.0, epochs))) / 2
        for i, t in enumerate((pbar := tqdm(temperatures))):
            # set cost and entropy weights
            cost_w = t * self.min_cost_weight + (1 - t) * self.max_cost_weight
            entropy_w = jnp.exp(
                t * jnp.log(self.max_entropy_weight)
                + (1 - t) * jnp.log(self.min_entropy_weight)
            )

            # collect rollouts
            rollouts, outs = get_rollouts(self.env, self.policy, self.pool_size)
            self.logger.log(reward=rollouts.r, cost=rollouts.c)

            # estimate returns and advantages
            V, A, Vc, Ac, violation = self.estimate_returns_and_advantages(rollouts)
            self.logger.log(V=V, Vc=Vc, A=A, Ac=Ac, violation=violation)
            pbar.set_postfix(V=V[:, 0].mean(), Vc=Vc[:, 0].mean())

            # value optimization step
            for _ in range(value_steps):
                loss_vf, loss_cvf = self.value_optimization_step(
                    optimizer_vf, optimizer_cvf, rollouts.obs, V, Vc
                )
                self.logger.log(loss_vf=loss_vf, loss_cvf=loss_cvf)

            # policy optimization step
            for _ in range(policy_steps):
                loss_pi, loss_cpi, loss_ent = self.policy_optimization_step(
                    optimizer_pi, rollouts, A, Ac, violation, cost_w, entropy_w
                )
                self.logger.log(loss_pi=loss_pi, loss_cpi=loss_cpi, loss_ent=loss_ent)
        self.logger = self.logger.to_array()

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
            A = (A - A.mean()) / (A.std() + 1e-8)
            violation = violation / (Ac.std() + 1e-8)
            Ac = Ac / (Ac.std() + 1e-8)
        return V, A, Vc, Ac, violation

    @nnx.jit
    def policy_optimization_step(
        self,
        optimizer: nnx.Optimizer,
        rollouts: Rollout,
        A: Float[Array, "n t"],
        Ac: Float[Array, "n t"],
        violation: Float[Array, ""],
        cost_weight: Float[Array, ""],
        entropy_weight: Float[Array, ""],
    ):
        def average(x: Float[Array, "n t"]):
            discount = (1 - self.discount) * self.discount ** jnp.arange(x.shape[-1])
            return jnp.mean(jnp.sum(x * discount, axis=-1))

        @nnx.value_and_grad(has_aux=True)
        def loss(policy):
            log_p, entropy = jax.vmap(jax.vmap(policy.eval))(rollouts.obs, rollouts.a)
            ratio = jnp.exp(log_p - rollouts.log_p)
            ratio_clip = 1 + tanh_clip(ratio - 1, self.clip_pi)

            # cost surrogate loss
            gained_c = average(jnp.maximum(Ac * ratio, Ac * ratio_clip)) - average(Ac)
            loss_c = cost_weight * jnp.clip(violation + gained_c, min=0.0)

            # surrogate loss (why not clip this when the cost is violated ??)
            loss_pi = average(-jnp.minimum(A * ratio, A * ratio_clip))

            # entropy regularization
            loss_ent = -entropy_weight * entropy.mean()
            return (loss_pi + loss_c + loss_ent), (loss_pi, loss_c, loss_ent)

        (loss, (loss_pi, loss_cpi, loss_ent)), grads = loss(optimizer.model)
        optimizer.update(grads)
        return loss_pi, loss_cpi, loss_ent

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

    def dump(self, name):
        import pickle
        import os

        with open(os.getcwd() + f"/runs/{name}.pkl", "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load_from(name):
        import pickle
        import os

        with open(os.getcwd() + f"/runs/{name}.pkl", "rb") as f:
            return pickle.load(f)
