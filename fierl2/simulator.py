from dataclasses import InitVar
from typing import NamedTuple
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
from jax import Array
from systems import DSSM, FaultyDSSM
from mpc import MPC
from kalman import KalmanFilter, Gaussian

KeyArray = Array
SqrMatrix = Array


class SimState(NamedTuple):
    x: Array
    est: Gaussian
    ut: Array


class SimStep(NamedTuple):
    x: Array
    est: Gaussian
    ut: Array
    u: Array
    y: Array


class Simulator(eqx.Module):
    system: FaultyDSSM = eqx.field(static=True)
    mpc_horizon: int = eqx.field(static=True)
    Jy: SqrMatrix = eqx.field(static=True, default=1.0)
    Ju: SqrMatrix = eqx.field(static=True, default=1e-8)
    Jx: SqrMatrix = eqx.field(static=True, default=0.0)
    Je: SqrMatrix = eqx.field(static=True, default=1.0)
    Q: SqrMatrix = eqx.field(static=True, default=1.0)
    R: SqrMatrix = eqx.field(static=True, default=1.0)

    mpc: MPC = eqx.field(init=False, static=True)
    observer: KalmanFilter = eqx.field(init=False, static=True)

    def __post_init__(self):
        self.Jy = self.Jy * jnp.eye(self.system.y_dim)
        self.Ju = self.Ju * jnp.eye(self.system.u_dim)
        self.Jx = self.Jx * jnp.eye(self.system.x_dim)
        self.Je = self.Je * jnp.eye(self.system.x_dim)
        self.mpc = MPC(
            self.system.step, self.mpc_horizon, Jy=self.Jy, Ju=self.Ju, Jx=self.Jx
        )
        self.Q = self.Q * jnp.diag(jnp.ones(self.system.x_dim))
        self.R = self.R * jnp.eye(self.system.y_dim)
        self.observer = KalmanFilter(self.system.step, self.Q, self.R)

    def get_reward(self, step, y_ref, u_ref, x_ref):
        norm = lambda x, J: x.T @ J @ x
        cost_y = norm(step.y - y_ref, Jy)
        cost_u = norm(step.u + step.a - u_ref, Ju)
        cost_x = norm(step.x - x_ref, Jx)
        cost_e = norm(step.est.mean - step.x, Je)
        cost_e_std = jnp.trace(Je @ step.est.cov, axis1=-1, axis2=-2)
        reward = -(cost_y + cost_u + cost_x + cost_e + cost_e_std)
        return reward

    @eqx.filter_jit
    def rollout(
        self,
        y_ref: Array,
        u_ref: Array,
        x_ref: Array,
        *,
        key: KeyArray,
        t_fault: int | float = 0.1,
        use_policy: bool = True,
        det_policy: bool = False,
    ):
        def prepare_inputs(key, y_ref, u_ref, x_ref):
            assert len(y_ref) == len(u_ref) == len(x_ref)
            keys_steps = jr.split(key, len(y_ref))
            inputs = (keys_steps, *map(self.mpc.windows, (y_ref, u_ref, x_ref)))
            return inputs

        def init_sim_state(key):
            key_x0, key_est0, key_ut0 = jr.split(key, 3)
            x0_nominal = self.system.reset_state()
            x = x0_nominal  # self.system.reset_state(key=key_x0)
            est = self.observer.reset_est(mean=x0_nominal, key=key_est0)
            ut = self.mpc.reset_ut(key=key_ut0)
            return x, est, ut

        def sim_step(carry, inputs):
            x, est, ut = carry
            key, y_ref, u_ref, x_ref = inputs

            key_step, key_act = jr.split(key)
            ut = self.mpc.update_ut(ut, est.mean, y_ref, u_ref, x_ref)
            u = ut[0]

            if det_policy:
                key_act = None
            a, log_p = self.policy.sample((est, ut), key=key_act)
            if not use_policy:
                a = 0.0 * a

            x_next, y = self.system.step(x, u + a, key=key_step)
            est_next = self.observer.update_est(est, u + a, y)

            out = SimStep(x, est, ut, u, a, log_p, x_next, est_next, y)
            return (x_next, est_next, ut), out

        key_init, key_steps1, key_steps2 = jr.split(key, 3)
        t_fault = t_fault if t_fault > 1 else int(t_fault * len(y_ref))
        (x, est, ut) = init_sim_state(key_init)
        inputs1 = prepare_inputs(
            key_steps1, y_ref[:t_fault], u_ref[:t_fault], x_ref[:t_fault]
        )
        inputs2 = prepare_inputs(
            key_steps2, y_ref[t_fault:], u_ref[t_fault:], x_ref[t_fault:]
        )

        (x, est, ut), rollout1 = jax.lax.scan(sim_step, (x, est, ut), inputs1)
        x = self.system.reset_fault(x, key=key_steps2)
        (x, est, ut), rollout2 = jax.lax.scan(sim_step, (x, est, ut), inputs2)

        rollout = jax.tree_map(lambda x, y: jnp.concat([x, y]), rollout1, rollout2)

        rewards = eqx.filter_vmap(self.get_reward)(rollout, y_ref, u_ref, x_ref)
        V, A = self.policy.estimate_V_and_A_gae(
            (rollout.est, rollout.ut), rewards, V_last=self.policy.value((est, ut))
        )
        return rollout, rewards, V, A
