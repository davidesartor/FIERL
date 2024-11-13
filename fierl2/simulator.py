from functools import partial
from typing import NamedTuple, Self
from jaxtyping import Array, Float, Key
import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


from utils import Module, RESET
from systems import FDSSM
from observers import KalmanFilter
from controllers import MPC
from ppo import PPO, Policy


class StepOut(NamedTuple):
    u: Float[Array, "t u"]
    a: Float[Array, "t u"]
    log_p: Float[Array, "t"]
    y: Float[Array, "t y"]


class Simulator(Module):
    t: int = eqx.field(init=False, default_factory=RESET)
    sys: FDSSM
    mpc: MPC
    kf: KalmanFilter
    # policy: PPO

    Jy: Float[Array, "y y"] = eqx.field(static=True)
    Ju: Float[Array, "u u"] = eqx.field(static=True)
    Je: Float[Array, "z+x z+x"] = eqx.field(static=True)
    ref_y: Float[Array, "t y"] = eqx.field(static=True)
    ref_u: Float[Array, "t u"] = eqx.field(static=True)

    discount: float = 0.9
    t_fault: float = 0.2

    def reset(self, *, rng: Key | None):
        rng_1, rng_2, rng_3 = (None,) * 3 if rng is None else jr.split(rng, 3)
        return self.replace(
            t=0,
            sys=self.sys.reset(rng=rng_1),
            kf=self.kf.reset(rng=rng_2),
            mpc=self.mpc.reset(rng=rng_3),
        )

    def step(self, *, rng: Key, use_aux=True) -> tuple[Self, StepOut]:
        rng_step, rng_act = jr.split(rng)

        # get the control
        ref_y = jax.lax.dynamic_slice_in_dim(self.ref_y, self.t, self.mpc.horizon)
        ref_u = jax.lax.dynamic_slice_in_dim(self.ref_u, self.t, self.mpc.horizon)
        mpc = self.mpc.update(x=self.kf(), ref_y=ref_y, ref_u=ref_u)
        u = mpc()

        # a, log_p = self.policy(state.kf)(rng=rng_act)
        # if not policy:
        #     a = jnp.zeros_like(a)
        a = jnp.zeros_like(u)
        log_p = jnp.zeros(())

        # step the system and update the observer
        sys, y = self.sys.step(u=u + a, rng=rng_step)
        kf = self.kf.update(u=u + a, y=y)
        self = self.replace(t=self.t + 1, sys=sys, kf=kf, mpc=mpc)

        return self, StepOut(u=u, a=a, log_p=log_p, y=y)

    def rewards(self, steps: StepOut):
        norm = lambda x, J: jnp.einsum("ti, ij, tj -> t", x, J, x)
        cost_y = norm(steps.y - self.ref_y, self.Jy)
        cost_u = norm(steps.u + steps.a - self.ref_u, self.Ju)

        x_aug = jnp.concat([steps.state.z, steps.state.x], axis=-1)
        cost_e = norm(steps.state.kf.x - x_aug, self.Je)
        cost_e_std = jnp.trace(steps.state.kf.P @ self.Je, axis1=-1, axis2=-2)
        return -(cost_y + cost_u + cost_e + cost_e_std)

    @eqx.filter_jit
    def rollout(self, *, rng: Key, use_aux=True):
        def scan_fn(state, rng, use_aux):
            state, steps = state.step(rng=rng, use_aux=use_aux)
            return state, (state, steps)

        rng_init, rng_fault, rng_steps1, rng_steps2 = jr.split(rng, 4)
        sim = self.reset(rng=rng_init)
        nominal_z = self.sys.reset(rng=None).z
        faulty_z = self.sys.reset(rng=rng_fault).z
        T1 = int(self.t_fault * len(self.ref_y))
        T2 = len(self.ref_y) - T1

        # fault free phase
        sim = sim.replace(sys=sim.sys.replace(z=nominal_z))
        sim, steps1 = jax.lax.scan(
            partial(scan_fn, use_aux=False), sim, jr.split(rng_steps1, T1)
        )

        # fault phase
        sim = sim.replace(sys=sim.sys.replace(z=faulty_z))
        sim, steps2 = jax.lax.scan(
            partial(scan_fn, use_aux=False), sim, jr.split(rng_steps2, T2)
        )
        steps = jax.tree.map(lambda *xs: jnp.concatenate(xs), steps1, steps2)
        return steps

        # rewards = self.get_reward(steps_out, self.ref_y, self.ref_u)
        return state, steps_out, rewards

    def plot(self, rollout):
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
            plt.vlines(int(len(t) * self.t_fault), *plt.ylim(), color="r")
            plt.legend()
            plt.grid(True)

        state, out = rollout
        x = state.sys.x
        z = state.sys.z
        est_mean = state.kf.x
        est_cov = state.kf.P

        plt.figure(figsize=(20, 10))
        for i in range(z.shape[-1]):
            plt.subplot(z.shape[-1], 4, 4 * i + 1)
            plot(v=z[:, i], name=f"$z_{i}$", est=(est_mean[:, i], est_cov[:, i, i]))
        for i in range(x.shape[-1]):
            plt.subplot(x.shape[-1], 4, 4 * (x.shape[-1] - i - 1) + 2)
            est = est_mean[:, -i - 1]
            cov = est_cov[:, -i - 1, -i - 1]
            plot(v=x[:, -i - 1], name=f"$x_{x.shape[-1]-i}$", est=(est, cov))
        for i in range(out.u.shape[-1]):
            plt.subplot(out.u.shape[-1], 4, 4 * i + 3)
            plot(v=out.u[:, i], name=f"$u_{i}$")
            plot(v=out.a[:, i], name=f"$a_{i}$", ref=self.ref_u[:, i])
        for i in range(out.y.shape[-1]):
            plt.subplot(out.y.shape[-1], 4, 4 * i + 4)
            plot(v=out.y[:, i], name=f"$y_{i}$", ref=self.ref_y[:, i])
        plt.show()
