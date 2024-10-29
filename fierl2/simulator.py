from dataclasses import InitVar
from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax
from jax import Array
from systems import DSSM

KeyArray = Array
SqrMatrix = Array


class Gaussian(NamedTuple):
    mean: Array
    cov: Array

    def __call__(self, *, key: KeyArray, **kwargs) -> Array:
        return jr.multivariate_normal(key, self.mean, self.cov, **kwargs)

    def log_prob(self, x: Array):
        return jnp.array(
            jax.scipy.stats.multivariate_normal.logpdf(x, self.mean, self.cov)
        )

    @property
    def entropy(self):
        pdf_norm_factor = jnp.linalg.det(2 * jnp.pi * self.cov)
        return 0.5 * (jnp.log(pdf_norm_factor) + self.mean.shape[-1])


class KalmanFilter(eqx.Module):
    system: DSSM
    Q: SqrMatrix
    R: SqrMatrix
    Je: SqrMatrix

    def reset_est(
        self, mean=jnp.zeros(()), cov=jnp.ones(()), *, key: KeyArray
    ) -> Gaussian:
        return Gaussian(
            mean=mean * jnp.ones(self.system.x_dim),
            cov=cov * jnp.eye(self.system.x_dim),
        )

    def update_est(self, est: Gaussian, u: Array, y: Array) -> Gaussian:
        A, B, C, D, dx, dy = self.system.linearize(est.mean, u)
        # a posteriori update
        K = est.cov @ C.T @ jnp.linalg.inv(C @ est.cov @ C.T + self.R)
        y_pred = C @ est.mean + D @ u + dy
        est = Gaussian(
            mean=est.mean + K @ (y - y_pred),
            cov=est.cov - K @ C @ est.cov,
        )
        # a priori update
        est = Gaussian(
            mean=A @ est.mean + B @ u + dx,
            cov=A @ est.cov @ A.T + self.Q,
        )
        return est


class MPC(eqx.Module):
    system: DSSM
    horizon: int
    Jy: SqrMatrix
    Ju: SqrMatrix
    Jx: SqrMatrix
    discount = 1.0

    def cal_cost_matrices(self, horizon=None):
        discount = jnp.diag(self.discount ** jnp.arange(horizon or self.horizon))
        return tuple(map(lambda J: jnp.kron(discount, J), (self.Jy, self.Ju, self.Jx)))

    def windows(self, reference: Array) -> Array:
        ref = jnp.pad(reference, ((0, self.horizon), (0, 0)), mode="edge")
        windows = [ref[i : i + self.horizon] for i in range(len(ref) - self.horizon)]
        return jnp.stack(windows)

    def reset_ut(self, ut=jnp.zeros(()), *, key: KeyArray) -> Array:
        return ut * jnp.ones((self.horizon, self.system.u_dim))

    def update_ut(
        self, ut: Array, x0: Array, y_ref: Array, u_ref: Array, x_ref: Array
    ) -> Array:
        assert self.horizon == len(ut) == len(y_ref) == len(u_ref) == len(x_ref)
        ut = jnp.roll(ut, -1, axis=0).at[-1].set(ut[-1])
        A, B, C, D, dx, dy = self.system.linearize_on_trajectory(x0, ut)
        Jy, Ju, Jx = self.cal_cost_matrices()

        return jnp.linalg.solve(
            a=D.T @ Jy @ D + Ju + B.T @ Jx @ B,
            b=(
                D.T @ Jy @ (y_ref.flatten() - C @ x0 - dy)
                + Ju @ u_ref.flatten()
                + B.T @ Jx @ (x_ref.flatten() - A @ x0 - dx)
            ),
        ).reshape(ut.shape)


class SimStep(NamedTuple):
    x: Array
    est: Gaussian
    ut: Array
    u: Array
    a: Array
    log_p: Array
    next_x: Array
    next_est: Gaussian
    y: Array


class PPOPolicy(eqx.Module):
    mlp_pi: eqx.nn.MLP
    mlp_V: eqx.nn.MLP
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_range_pi: float = 0.1
    clip_range_vf: float = 0.1
    value_loss_weight: float = 0.5
    entropy_loss_weight: float = 0.001

    def __init__(self, x_dim, u_dim, horizon, hidden, depth, *, key: KeyArray):
        key_pi, key_V = jr.split(key)
        kwargs: dict = dict(
            in_size=x_dim + x_dim**2 + u_dim * horizon,
            width_size=hidden,
            depth=depth,
            activation=jax.nn.gelu,
        )
        self.mlp_pi = eqx.nn.MLP(out_size=2 * u_dim, key=key_pi, **kwargs)
        self.mlp_V = eqx.nn.MLP(out_size=1, key=key_V, **kwargs)

    def flatten_state(self, state: tuple[Gaussian, Array]) -> Array:
        (mu, cov), u = state
        return jnp.concat([mu.flatten(), cov.flatten(), u.flatten()])

    def __call__(self, state: tuple[Gaussian, Array]):
        x = self.mlp_pi(self.flatten_state(state))
        mean, std = jnp.split(x, 2, axis=-1)
        pi = Gaussian(mean=mean, cov=jnp.diag(jax.nn.sigmoid(std)))
        return pi

    def value(self, state: tuple[Gaussian, Array]):
        V = self.mlp_V(self.flatten_state(state)).squeeze()
        return V

    def sample(self, state: tuple[Gaussian, Array], *, key: KeyArray | None = None):
        pi = self(state)
        a = pi.mean if key is None else pi(key=key)
        return a, pi.log_prob(a)

    def estimate_V_and_A_gae(
        self, states: tuple[Gaussian, Array], rewards: Array, V_last=jnp.zeros(())
    ) -> tuple[Array, Array]:
        def tail_sum_step(v, xi):
            v = xi + self.gamma * self.gae_lambda * v
            return v, v

        V = eqx.filter_vmap(self.value)(states)
        V_next = jnp.roll(V, -1).at[-1].set(V_last)
        delta = rewards - V + self.gamma * V_next

        _, A_est = jax.lax.scan(tail_sum_step, V_last, delta, reverse=True)
        V_est = A_est + V
        return V_est, A_est

    def loss(
        self,
        state: tuple[Gaussian, Array],
        a: Array,
        log_p: Array,
        V_est: Array,
        A_est: Array,
    ):
        pi, V = self(state), self.value(state)
        ratio_pi = jnp.exp(pi.log_prob(a) - log_p)

        # loss functions
        dpi, dvf = self.clip_range_pi, self.clip_range_vf
        loss_value_function = self.value_loss_weight * (V_est - V).clip(-dvf, dvf) ** 2
        loss_entropy = self.entropy_loss_weight * jnp.mean(-pi.entropy)
        loss_policy = -jnp.minimum(
            A_est * ratio_pi, A_est * ratio_pi.clip(1 - dpi, 1 + dpi)
        )
        return loss_policy + loss_value_function + loss_entropy


class Simulator(eqx.Module):
    system: DSSM
    mpc: MPC
    observer: KalmanFilter
    policy: PPOPolicy

    @property
    def cost_matrices(self):
        return self.mpc.Jy, self.mpc.Ju, self.mpc.Jx, self.observer.Je

    def __init__(
        self,
        system: DSSM,
        mpc_horizon: int,
        Jy=jnp.ones(()),
        Ju=jnp.zeros(()) + 1e-8,
        Jx=jnp.zeros(()),
        Je=jnp.ones(()),
        Q=jnp.ones(()),
        R=jnp.ones(()),
        mlp_hidden=128,
        mlp_depth=2,
        *,
        key: KeyArray,
    ):
        self.system = system
        Jy = Jy * jnp.eye(self.system.y_dim)
        Ju = Ju * jnp.eye(self.system.u_dim)
        Jx = Jx * jnp.eye(self.system.x_dim)
        self.mpc = MPC(self.system, mpc_horizon, Jy=Jy, Ju=Ju, Jx=Jx)
        Q = Q * jnp.diag(jnp.ones(self.system.x_dim))
        R = R * jnp.eye(self.system.y_dim)
        Je = Je * jnp.eye(self.system.x_dim)
        self.observer = KalmanFilter(self.system, Q, R, Je=Je)
        self.policy = PPOPolicy(
            self.system.x_dim,
            self.system.u_dim,
            mpc_horizon,
            mlp_hidden,
            mlp_depth,
            key=key,
        )

    def get_reward(self, step, y_ref, u_ref, x_ref):
        Jy, Ju, Jx, Je = self.cost_matrices
        norm = lambda x, J: x.T @ J @ x
        cost_y = norm(step.y - y_ref, Jy)
        cost_u = norm(step.u + step.a - u_ref, Ju)
        cost_x = norm(step.x - x_ref, Jx)
        cost_e1 = norm(step.est.mean - step.x, Je)
        cost_e2 = jnp.trace(Je @ step.est.cov, axis1=-1, axis2=-2)
        reward = -(cost_y + cost_u + cost_x + cost_e1 + cost_e2)
        return reward

    @eqx.filter_jit
    def rollout(
        self,
        y_ref: Array,
        u_ref: Array,
        x_ref: Array,
        *,
        key: KeyArray,
        use_policy: bool = True,
    ):
        # prepare inputs
        assert len(y_ref) == len(u_ref) == len(x_ref)
        key_x0, key_est0, key_ut0, key_steps = jr.split(key, 4)
        keys_steps = jr.split(key_steps, len(y_ref))
        inputs = (keys_steps, *map(self.mpc.windows, (y_ref, u_ref, x_ref)))

        # initial state
        x = self.system.reset_state(key=key_x0)
        x0_nominal = self.system.reset_state()
        est = self.observer.reset_est(mean=x0_nominal, key=key_est0)
        ut = self.mpc.reset_ut(key=key_ut0)
        carry = x, est, ut

        # simulation loop
        def sim_step(carry, inputs):
            x, est, ut = carry
            key, y_ref, u_ref, x_ref = inputs

            key_step, key_act = jr.split(key)
            ut = self.mpc.update_ut(ut, est.mean, y_ref, u_ref, x_ref)
            u = ut[0]

            a, log_p = self.policy.sample((est, ut), key=key_act)
            if not use_policy:
                a = 0.0 * a

            x_next, y = self.system.step(x, u + a, key=key_step)
            est_next = self.observer.update_est(est, u + a, y)

            out = SimStep(x, est, ut, u, a, log_p, x_next, est_next, y)
            return (x_next, est_next, ut), out

        (x, est, ut), rollout = jax.lax.scan(sim_step, carry, inputs)
        rewards = eqx.filter_vmap(self.get_reward)(rollout, y_ref, u_ref, x_ref)
        V, A = self.policy.estimate_V_and_A_gae(
            (rollout.est, rollout.ut), rewards, V_last=self.policy.value((est, ut))
        )
        return rollout, rewards, V, A
