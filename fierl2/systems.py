import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx


class DSSM(eqx.Module):
    x_dim: eqx.AbstractVar[int]
    u_dim: eqx.AbstractVar[int]
    y_dim: eqx.AbstractVar[int]
    dt: eqx.AbstractVar[float]

    def step(self, x, u, *, key=None) -> tuple[jax.Array, jax.Array]:
        raise NotImplementedError

    def reset_state(self, *, key=None):
        raise NotImplementedError

    def trajectory(self, x0, ut, *, key=None):
        def scan_fn(x, input):
            u, k = input
            k = None if key is None else k
            x, y = self.step(x, u, key=k)
            return x, (x, y)

        inputs = ut, jr.split(key or jr.key(0), len(ut))
        _, (xt, yt) = jax.lax.scan(scan_fn, x0, inputs)
        return xt, yt

    def linearize(self, x, u):
        (A, B), (C, D) = jax.jacobian(self.step, argnums=(0, 1))(x, u)
        dx, dy = self.step(x, u)
        dx = dx - A @ x - B @ u
        dy = dy - C @ x - D @ u
        return A, B, C, D, dx, dy

    def linearize_on_trajectory(self, x0, ut):
        (A, B), (C, D) = jax.jacobian(self.trajectory, argnums=(0, 1))(x0, ut)
        A = A.reshape(self.x_dim * len(ut), self.x_dim)
        B = B.reshape(self.x_dim * len(ut), self.u_dim * len(ut))
        C = C.reshape(self.y_dim * len(ut), self.x_dim)
        D = D.reshape(self.y_dim * len(ut), self.u_dim * len(ut))

        dx, dy = self.trajectory(x0, ut)
        dx = dx.flatten() - A @ x0 - B @ ut.flatten()
        dy = dy.flatten() - C @ x0 - D @ ut.flatten()
        return A, B, C, D, dx, dy


class FaultyDSSM(DSSM):
    z_dim: eqx.AbstractVar[int]

    def __post_init__(self):
        self.x_dim += self.z_dim

    def split_state(self, x) -> tuple[jax.Array, jax.Array]:
        x, z = jnp.split(x, (self.x_dim - self.z_dim,), axis=-1)
        return x, z


class ToyExample(FaultyDSSM):
    x_dim: int = 2
    u_dim: int = 2
    y_dim: int = 1
    z_dim: int = eqx.field(init=False)

    flow_coef: float = 0.5
    input_coef: float = 1.0
    output_coef: float = 1.0

    x_noise_std: float = 1e-1
    y_noise_std: float = 1e-1
    dt: float = 1.0

    def __post_init__(self):
        self.z_dim = self.u_dim
        super().__post_init__()

    def reset_state(self, *, key=None):
        x = jnp.zeros(self.x_dim - self.z_dim)
        z = jnp.ones(self.z_dim)
        if key is not None:
            key_idx, key_val = jr.split(key)
            z = z.at[jr.choice(key_idx, self.z_dim)].set(jr.uniform(key_val))
        return jnp.concatenate([x, z], axis=-1)

    def step(self, x, u, *, key=None):
        x, z = self.split_state(x)
        inflow = self.input_coef * (u * z).sum()
        mixflows = self.flow_coef * x[:-1]
        outflow = self.output_coef * x[-1]

        x = x.at[0].add(inflow)
        x = x.at[:-1].add(-mixflows)
        x = x.at[1:].add(mixflows)
        x = x.at[-1].add(-outflow)

        y = outflow * jnp.ones(self.y_dim)

        if key is not None:
            kx, ky = jr.split(key)
            x = x + jr.normal(kx, x.shape) * self.x_noise_std
            y = y + jr.normal(ky, y.shape) * self.y_noise_std

        x = jnp.concat([x, z], axis=-1)
        return x, y


class Car(DSSM):
    mass: float = 1.0
    steer: float = 1.0
    integration_steps: int = 10

    x_dim = 4
    u_dim = 2
    y_dim = 2

    dt = 1.0

    def step(self, x, u, *, key=None):
        def dx(x, u):
            p1, p2, th, v = jnp.split(x, 4, axis=-1)
            a, w = jnp.split(u, 2, axis=-1)
            dp1 = v * jnp.cos(th)
            dp2 = v * jnp.sin(th)
            dth = w * self.steer
            dv = a / self.mass
            return jnp.concatenate([dp1, dp2, dth, dv], axis=-1)

        mini_step = lambda i, x: x + dx(x, u) * self.dt / self.integration_steps
        x = jax.lax.fori_loop(0, self.integration_steps, mini_step, x, unroll=True)
        y = x[..., :2]
        return x, y


class RoboArm(DSSM):
    """
    'Sensor bias fault isolation in a class of nonlinear systems'
    Xiaodong Zhang; T. Parisini; M.M. Polycarpou
    https://ieeexplore.ieee.org/abstract/document/1406131
    """

    mass: float = 4.0
    center_of_mass: float = 0.5
    link_inertia: float = 2.0
    rotor_inertia: float = 1.0
    link_friction: float = 0.5
    motor_friction: float = 1.0
    elastic_constant: float = 2.0
    gravity: float = 9.81

    integration_steps: int = 10

    x_dim = 4
    u_dim = 1
    y_dim = 2

    dt = 1.0

    def step(self, x, u, *, key=None):
        def dx(x, u):
            th1, th2, w1, w2 = jnp.split(x, 4, axis=-1)
            dth1, dth2 = w1, w2
            dw1 = (k * (th2 - th1) - Fl * w1 - m * g * l * jnp.sin(th1)) / Jl
            dw2 = (k * (th1 - th2) - Fm * w2 + u) / Jm
            return jnp.concatenate([dth1, dth2, dw1, dw2], axis=-1)

        Jl, Jm = self.link_inertia, self.rotor_inertia
        Fl, Fm = self.link_friction, self.motor_friction
        k, l = self.elastic_constant, self.center_of_mass
        m, g = self.mass, self.gravity

        mini_step = lambda i, x: x + dx(x, u) * self.dt / self.integration_steps
        x = jax.lax.fori_loop(0, self.integration_steps, mini_step, x, unroll=True)
        y = x[..., :2]
        return x, y


class PointSatellite(DSSM):
    """
    'A Geometric Approach to Nonlinear Fault Detection and Isolation'
    Claudio De Persis and Alberto Isidori
    https://www.sciencedirect.com/science/article/pii/S1474667017373627
    """

    mass: float = 1.0
    theta1: float = 1.0
    theta2: float = 1.0
    integration_steps: int = 10

    x_dim = 4
    u_dim = 2
    y_dim = 3

    dt = 1.0

    def init_state(self, *, key=None):
        r, phi = 10.0, 0.0
        w = jnp.sqrt(self.theta1 / (r**3))
        v = self.theta2 * self.mass / (2 * w)
        return jnp.array([r, phi, v, w])

    def step(self, x, u, *, key=None):
        def dx(x, u):
            r, phi, v, w = jnp.split(x, 4, axis=-1)
            ur, uw = jnp.split(u, 2, axis=-1)
            dr, dphi = v, w
            dv = r * w**2 - self.theta1 / (r**2) + self.theta2 * ur
            dw = (self.theta2 * self.mass - 2 * v * w + self.theta2 * uw) / r
            return jnp.concatenate([dr, dphi, dv, dw], axis=-1)

        mini_step = lambda i, x: x + dx(x, u) * self.dt / self.integration_steps
        x = jax.lax.fori_loop(0, self.integration_steps, mini_step, x, unroll=True)
        y = x[..., :2]
        return x, y
