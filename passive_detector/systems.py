import jax
import jax.numpy as jnp
from flax import struct
from scipy.signal import lti


class DSSM(struct.PyTreeNode):
    A: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    noise_std_x: jax.Array
    noise_std_y: jax.Array
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
    def from_continuos(
        cls,
        A: jax.Array,
        B: jax.Array,
        C: jax.Array,
        D: jax.Array | None,
        noise_std_x: jax.Array,
        noise_std_y: jax.Array,
        dt: float = 1.0,
    ):
        D = D or jnp.zeros((C.shape[0], B.shape[1]))
        continuos_ssm = lti(A, B, C, D).to_discrete(dt)
        A_d = jnp.array(continuos_ssm.A)  # type: ignore
        B_d = jnp.array(continuos_ssm.B)  # type: ignore
        C_d = jnp.array(continuos_ssm.C)  # type: ignore
        D_d = jnp.array(continuos_ssm.D)  # type: ignore
        return cls(A_d, B_d, C_d, D_d, noise_std_x * jnp.sqrt(dt), noise_std_y, dt)

    def step(self, x: jax.Array, rng_key: jax.Array, u: jax.Array):
        rng_noise_x, rng_noise_y = jax.random.split(rng_key)
        w_x = jax.random.normal(rng_noise_x, (self.state_dim,))
        w_y = jax.random.normal(rng_noise_y, (self.output_dim,))
        y = self.C @ x + self.D @ u + self.noise_std_y * w_y
        new_x = self.A @ x + self.B @ u + self.noise_std_x * w_x
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
