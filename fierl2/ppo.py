from functools import partial
from jaxtyping import Array, Float
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from tqdm import tqdm


class Policy(eqx.Module):
    mean: Float[Array, "u"]
    cov: Float[Array, "u u"]

    def __call__(self, *, rng=None):
        if rng is None:
            return self.mean, self.log_prob(self.mean)
        a = jr.multivariate_normal(rng, self.mean, self.cov)
        return a, self.log_prob(a)

    def entropy(self):
        return 0.5 * jnp.log(jnp.linalg.det(2 * jnp.pi * jnp.e * self.cov))

    def log_prob(self, a: Array) -> Float[Array, "1"]:
        return jax.scipy.stats.multivariate_normal.logpdf(a, self.mean, self.cov)  # type: ignore


class PPO(eqx.Module):
    mlp_mu: eqx.nn.MLP
    mlp_cov: eqx.nn.MLP
    mlp_V: eqx.nn.MLP
    discount: float = eqx.field(static=True, default=0.99)
    gae_lambda: float = eqx.field(static=True, default=0.95)
    clip_range_pi: float = eqx.field(static=True, default=0.1)
    clip_range_vf: float = eqx.field(static=True, default=0.1)

    def __init__(self, x_dim, u_dim, hidden=32, depth=2, *, rng):
        rng_mu, rng_cov, rng_V = jr.split(rng, 3)
        in_size = x_dim + x_dim**2
        self.mlp_V = eqx.nn.MLP(in_size, 1, hidden, depth, jax.nn.gelu, key=rng_V)
        self.mlp_mu = eqx.nn.MLP(in_size, u_dim, hidden, depth, jax.nn.gelu, key=rng_mu)
        self.mlp_cov = eqx.nn.MLP(
            in_size, 2 * u_dim, hidden, depth, jax.nn.gelu, key=rng_cov
        )

    def __call__(self, state):
        x = jnp.concat([state.x.flatten(), state.P.flatten()])
        mu = self.mlp_mu(x)
        sigma_diag, sigma_corr = jnp.split(self.mlp_cov(x), 2, axis=-1)
        cov = jnp.diag(jnp.clip(sigma_diag**2, max=1.0) + 1e-8)
        return Policy(mean=mu, cov=cov)

    def value(self, state) -> Float[Array, "1"]:
        x = jnp.concat([state.x.flatten(), state.P.flatten()])
        V = self.mlp_V(x).squeeze()
        return V

    def estimate_V_and_A_gae(
        self, states, rewards: Float[Array, "t"], last_state=None
    ) -> tuple[Array, Array]:
        def tail_sum_step(v, xi):
            v = xi + self.discount * self.gae_lambda * v
            return v, v

        V = eqx.filter_vmap(self.value)(states)
        V_last = self.value(last_state) if last_state is not None else jnp.zeros(())
        V_next = jnp.roll(V, -1).at[-1].set(V_last)
        delta = rewards - V + self.discount * V_next

        _, A_est = jax.lax.scan(tail_sum_step, V_last, delta, reverse=True)
        V_est = A_est + V
        return V_est, A_est

    def loss(
        self,
        state,
        a: Array,
        log_p: Array,
        V: Array,
        A: Array,
        entropy_weight: float = 0.0001,
    ):
        pi = self(state)
        V_est = self.value(state)

        # surrogate loss
        ratio_pi = jnp.exp(pi.log_prob(a) - log_p)
        ratio_pi_clip = ratio_pi.clip(1 - self.clip_range_pi, 1 + self.clip_range_pi)
        loss_policy = -jnp.minimum(A * ratio_pi, A * ratio_pi_clip)

        # value function loss
        V_est = V_est.clip(V * (1 - self.clip_range_vf), V * (1 + self.clip_range_vf))
        loss_value_function = (V_est - V) ** 2

        # entropy loss
        loss_entropy = -entropy_weight * pi.entropy()
        return loss_policy + loss_value_function + loss_entropy


def optimize(sim):
    def returns(rewards):
        step = lambda v, xi: (xi + sim.discount * v, None)
        scan = lambda xi: jax.lax.scan(step, 0.0, xi, reverse=True)[0]
        for _ in range(rewards.ndim - 1):
            scan = jax.vmap(scan)
        return jnp.mean(scan(rewards))

    optimizer = optax.adam(1e-4)
    opt_state = optimizer.init(eqx.filter(sim.policy, eqx.is_array))

    @eqx.filter_jit
    def step(sim, opt_state, *, rng):
        rngs = jr.split(rng, 64)
        final, rollouts, rewards = jax.vmap(lambda k: sim.rollout(rng=k))(rngs)

        states, last_state = rollouts.state.kf, final.kf
        V, A = jax.vmap(sim.policy.estimate_V_and_A_gae)(states, rewards, last_state)
        A = (A - A.mean()) / (A.std() + 1e-8)

        @eqx.filter_value_and_grad
        def loss_fn(policy):
            batch_loss = jax.vmap(jax.vmap(policy.loss))
            a, log_p = rollouts.a, rollouts.log_p
            return batch_loss(states, a, log_p, V, A).mean()

        policy = sim.policy
        for _ in range(10):
            loss, grads = loss_fn(policy)
            updates, opt_state = optimizer.update(
                grads, opt_state, eqx.filter(policy, eqx.is_array)
            )
            policy = eqx.apply_updates(policy, updates)

        sim = eqx.tree_at(lambda sim: sim.policy, sim, policy)
        return sim, opt_state, loss, rewards, V.mean()

    log_loss, log_rew, log_rew_std = [], [], []
    steps = 300
    for i, rng in enumerate(pbar := tqdm(jr.split(jr.key(0), steps))):
        sim, opt_state, loss, rew, V = step(sim, opt_state, rng=rng)
        log_loss.append(loss.item())
        log_rew.append(jax.vmap(returns)(rew).mean().item())
        log_rew_std.append(jax.vmap(returns)(rew).std().item())
        pbar.set_postfix(
            loss=log_loss[-1], rew=log_rew[-1], rew_std=log_rew_std[-1], V=V.item()
        )
    log_loss, log_rew, log_rew_std = map(jnp.array, (log_loss, log_rew, log_rew_std))
    return sim, log_loss, log_rew, log_rew_std
