from typing import NamedTuple, Self
from jaxtyping import Array, Float, Key
from tqdm import tqdm
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx
import optax


from utils import Module, RESET
from systems import FDSSM
from observers import KalmanFilter
from controllers import MPC
from ppo import PPO


class StepOut(NamedTuple):
    obs: Float[Array, "..."]
    u: Float[Array, "u"]
    a: Float[Array, "u"]
    log_p: Float[Array, ""]
    y: Float[Array, "y"]
    r: Float[Array, ""]


class Simulator(Module):
    t: int = eqx.field(init=False, default_factory=RESET)
    sys: FDSSM
    mpc: MPC
    kf: KalmanFilter
    policy: PPO

    Jy: Float[Array, "y y"] = eqx.field(static=True)
    Ju: Float[Array, "u u"] = eqx.field(static=True)
    Je: Float[Array, "z+x z+x"] = eqx.field(static=True)
    ref_y: Float[Array, "t y"] = eqx.field(static=True)
    ref_u: Float[Array, "t u"] = eqx.field(static=True)

    t_aux_on: float = eqx.field(static=True, default=0.5)
    t_fault: float = eqx.field(static=True, default=0.2)

    def reset(self, *, rng: Key | None):
        rng_1, rng_2, rng_3 = (None,) * 3 if rng is None else jr.split(rng, 3)
        return self.replace(
            t=0,
            sys=self.sys.reset(rng=rng_1),
            kf=self.kf.reset(rng=rng_2),
            mpc=self.mpc.reset(rng=rng_3),
        )

    def observable(self) -> Float[Array, "(z+x)+(z+x)**2"]:
        # CAREFUL: this is called before the controller state is updated
        # so it will not include the planned control action at the current time
        # keep this in mind if you want to include the controller state
        P = self.kf.P.reshape((*self.kf.x.shape[:-1], -1))
        return jnp.concat([self.kf.x, P], axis=-1)

    def rewards(self, u: Float[Array, "u"], a: Float[Array, "u"], y: Float[Array, "y"]):
        norm = lambda x, J: jnp.einsum("i, ij, j", x, J, x)
        cost_y = norm(y - self.ref_y[self.t], self.Jy)
        cost_u = norm(u + a - self.ref_u[self.t], self.Ju)

        x_aug = jnp.concat([self.sys.z, self.sys.x])
        cost_e = 0.0 * norm(self.kf.x - x_aug, self.Je)
        cost_e_std = jnp.trace(self.kf.P @ self.Je)
        return -(cost_y + cost_u + cost_e + cost_e_std)

    def step(self, *, rng: Key) -> tuple[Self, StepOut]:
        # get the aux control from the policy
        rng_step, rng_act = jr.split(rng)
        obs = self.observable()
        a, log_p, _ = self.policy(obs, rng=rng_act)
        a, log_p = jax.lax.cond(
            self.t > self.t_aux_on * len(self.ref_y),
            lambda: (a, log_p),
            lambda: (jnp.zeros_like(a), jnp.zeros(())),
        )

        # get the nominal control
        ref_y = jax.lax.dynamic_slice_in_dim(self.ref_y, self.t, self.mpc.horizon)
        ref_u = jax.lax.dynamic_slice_in_dim(self.ref_u, self.t, self.mpc.horizon)
        mpc = self.mpc.update(x=self.kf(), ref_y=ref_y, ref_u=ref_u)
        u = mpc()

        # step the system and update the observer
        sys, y = self.sys.step(u=u + a, rng=rng_step)
        kf = self.kf.update(u=u + a, y=y)
        return (
            self.replace(t=self.t + 1, sys=sys, kf=kf, mpc=mpc),
            StepOut(obs=obs, u=u, a=a, log_p=log_p, y=y, r=self.rewards(u, a, y)),
        )

    @eqx.filter_jit
    def rollout(self, *, rng: Key):
        def scan_fn(state, rng):
            newstate, steps = state.step(rng=rng)
            return newstate, (state, steps)

        rng_init, rng_fault, rng_steps1, rng_steps2 = jr.split(rng, 4)
        T1 = int(self.t_fault * len(self.ref_y))
        T2 = len(self.ref_y) - T1
        rng_steps1 = jr.split(rng_steps1, T1)
        rng_steps2 = jr.split(rng_steps2, T2)

        # phase 1: no fault
        self = self.reset(rng=rng_init)
        nominal_z = self.sys.reset(rng=None).z
        self = self.replace(sys=self.sys.replace(z=nominal_z))
        self, steps1 = jax.lax.scan(scan_fn, self, rng_steps1)

        # phase 3: add abrupt fault
        faulty_z = self.sys.reset(rng=rng_fault).z
        self = self.replace(sys=self.sys.replace(z=faulty_z))
        self, steps2 = jax.lax.scan(scan_fn, self, rng_steps2)

        # aggregate and compute rewards
        states, outs = jax.tree.map(lambda *xs: jnp.concatenate(xs), steps1, steps2)
        return self, states, outs

    def optimize(self, steps, lr=1e-4, batch=256, normalize_advantages=True):
        @eqx.filter_jit
        def collect_rollouts(sim: Self, rng: Key):
            get_rollouts = jax.vmap(lambda k: sim.rollout(rng=k))
            last_state, states, outs = get_rollouts(jr.split(rng, batch))
            last_obs = last_state.observable()
            A, V = jax.vmap(sim.policy.gen_advantage_estimation)(
                outs.obs, outs.r, last_obs
            )
            if normalize_advantages:
                A = (A - A.mean()) / (A.std() + 1e-8)
            return outs, A, V

        @eqx.filter_jit
        def update_policy(policy, opt_state, outs, A, V):
            @eqx.filter_value_and_grad
            def loss_fn(policy):
                o, a, p = outs.obs, outs.a, outs.log_p
                loss = jax.vmap(jax.vmap(policy.loss))(o, a, p, A, V)
                return loss.mean()

            def scan_fn(carry, _):
                policy, opt_state = carry
                loss, grads = loss_fn(policy)
                updates, opt_state = optimizer.update(
                    grads, opt_state, eqx.filter(policy, eqx.is_array)
                )
                policy = eqx.apply_updates(policy, updates)
                return (policy, opt_state), loss

            (policy, opt_state), losses = jax.lax.scan(
                scan_fn, (policy, opt_state), length=10
            )
            return policy, opt_state, losses

        log_loss, log_rew = [], []
        optimizer = optax.adam(lr)
        opt_state = optimizer.init(eqx.filter(self.policy, eqx.is_array))
        for i, rng in enumerate(pbar := tqdm(jr.split(jr.key(0), steps))):
            outs, A, V = collect_rollouts(self, rng=rng)
            new_policy, opt_state, losses = update_policy(
                self.policy, opt_state, outs, A, V
            )
            self = self.replace(policy=new_policy)
            log_loss.append(losses.mean().item())
            log_rew.append(outs.r.mean())
            pbar.set_postfix(loss=log_loss[-1], rew=log_rew[-1], V=V.mean())
        log_loss, log_rew = map(jnp.array, (log_loss, log_rew))
        return self, log_loss, log_rew

    @staticmethod
    def plot(sim, outs: StepOut):
        import matplotlib.pyplot as plt

        def plot(v, name: str, ref=None, est=None):
            t = list(range(len(v)))
            if est is not None:
                mean, cov = est
                plt.plot(mean, label="est", color="tab:orange")
                plt.fill_between(
                    t, mean - cov, mean + cov, alpha=0.5, color="tab:orange"
                )
            plt.plot(v, label=name)
            if ref is not None:
                plt.plot(ref, "k:", label="ref")
            Tf, Ta = int(len(t) * sim.t_fault), int(len(t) * sim.t_aux_on)
            plt.vlines([Tf, Ta], *plt.ylim(), colors=["r", "g"])
            plt.legend()
            plt.grid(True)

        x = sim.sys.x
        z = sim.sys.z
        est_mean = sim.kf.x
        est_cov = sim.kf.P

        plt.figure(figsize=(20, 10))
        for i in range(z.shape[-1]):
            plt.subplot(z.shape[-1], 4, 4 * i + 1)
            plot(v=z[:, i], name=f"$z_{i}$", est=(est_mean[:, i], est_cov[:, i, i]))
        for i in range(x.shape[-1]):
            plt.subplot(x.shape[-1], 4, 4 * (x.shape[-1] - i - 1) + 2)
            est = est_mean[:, -i - 1]
            cov = est_cov[:, -i - 1, -i - 1] ** 0.5
            plot(v=x[:, -i - 1], name=f"$x_{x.shape[-1]-i}$", est=(est, cov))
        for i in range(outs.u.shape[-1]):
            plt.subplot(outs.u.shape[-1], 4, 4 * i + 3)
            plot(v=outs.u[:, i], name=f"$u_{i}$")
            plot(v=outs.a[:, i], name=f"$a_{i}$", ref=sim.ref_u[:, i])
        for i in range(outs.y.shape[-1]):
            plt.subplot(outs.y.shape[-1], 4, 4 * i + 4)
            plot(v=outs.y[:, i], name=f"$y_{i}$", ref=sim.ref_y[:, i])
        plt.show()
