import jax
import jax.numpy as jnp
from flax import struct
from scipy.signal import lti


class DSSM(struct.PyTreeNode):
    A: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    W_x: jax.Array
    W_y: jax.Array
    dt: float

    @property
    def state_dim(self):
        return self.A.shape[0]

    @property
    def input_dim(self):
        return self.B.shape[1]

    @property
    def output_dim(self):
        return self.C.shape[0]

    @classmethod
    def from_discrete(
        cls,
        A: jax.Array,
        B: jax.Array,
        C: jax.Array,
        D: jax.Array | None,
        W_x: jax.Array | None,
        W_y: jax.Array | None,
        dt: float,
    ):
        if D is None:
            D = jnp.zeros((C.shape[0], B.shape[1]))
        if W_x is None:
            W_x = jnp.zeros((A.shape[0], 1))
        if W_y is None:
            W_y = jnp.zeros((C.shape[0], 1))
        return cls(A, B, C, D, W_x, W_y, dt)

    @classmethod
    def from_continuos(
        cls,
        A: jax.Array,
        B: jax.Array,
        C: jax.Array,
        D: jax.Array | None,
        W_x: jax.Array | None,
        W_y: jax.Array | None,
        dt: float = 1.0,
    ):
        D = D if D is not None else jnp.zeros((C.shape[0], B.shape[1]))
        continuos_ssm = lti(A, B, C, D).to_discrete(dt)
        A_d = jnp.array(continuos_ssm.A)  # type: ignore
        B_d = jnp.array(continuos_ssm.B)  # type: ignore
        C_d = jnp.array(continuos_ssm.C)  # type: ignore
        D_d = jnp.array(continuos_ssm.D)  # type: ignore
        if W_x is None:
            W_x = jnp.zeros((A.shape[0], 1))
        if W_y is None:
            W_y = jnp.zeros((C.shape[0], 1))
        return cls(A_d, B_d, C_d, D_d, W_x * dt, W_y, dt)

    def step(self, x: jax.Array, rng_key: jax.Array, u: jax.Array):
        w = jax.random.normal(rng_key, (self.W_x.shape[1],))
        y = self.C @ x + self.D @ u + self.W_y * w
        new_x = self.A @ x + self.B @ u + self.W_x * w
        return new_x, y

    def simulate(self, x0: jax.Array, rng_key: jax.Array, u_t: jax.Array):
        def scan_fn(x, input):
            rng, u = input
            new_x, y = self.step(x, rng, u)
            return new_x, (x, y)

        rng_steps = jax.random.split(rng_key, u_t.shape[0])
        x_final, (x_t, y_t) = jax.lax.scan(scan_fn, x0, (rng_steps, u_t))
        return x_final, x_t, y_t


class ThreeTank(DSSM):
    @classmethod
    def make(
        cls,
        dt=0.1,
        noise_std_x=1e-3,
        noise_std_y=1e-3,
        h_target=(0.489, 0.2332, 0.3611),
        outflow_coeff=(0.46, 0.48, 0.58),
        tank_cross_section=1.54e-2,
        connector_cross_section=5e-5,
        gravitational_constant=9.81,
    ):
        g = gravitational_constant
        sn, sa = connector_cross_section, tank_cross_section
        az1, az2, az3 = outflow_coeff
        hstar1, hstar2, hstar3 = h_target

        k = g * sn / (sa * jnp.sqrt(2 * g))
        a1 = az1 * k / jnp.sqrt(hstar1 - hstar3)
        a2 = az3 * k / jnp.sqrt(hstar3 - hstar2)
        a3 = az2 * k / jnp.sqrt(hstar2)

        A = jnp.array([[-a1, 0.0, a1], [0.0, -a2 - a3, a2], [a1, a2, -a1 - a2]])
        B = jnp.array([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]]) / sa
        C = jnp.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        D = None

        noise_std_x = noise_std_x * jnp.ones(3)
        noise_std_y = noise_std_y * jnp.ones(2)
        return cls.from_continuos(A, B, C, D, noise_std_x, noise_std_y, dt=dt)
