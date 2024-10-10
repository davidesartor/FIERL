import jax
import jax.numpy as jnp
import jax.random as jr
import equinox as eqx

from scipy.signal import lti


class DSSM(eqx.Module):
    A: jax.Array
    B: jax.Array
    C: jax.Array
    D: jax.Array
    Wx: jax.Array
    Wy: jax.Array
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

    def check_shapes(self):
        assert self.A.shape == (self.state_dim, self.state_dim)
        assert self.B.shape == (self.state_dim, self.input_dim)
        assert self.C.shape == (self.output_dim, self.state_dim)
        assert self.D.shape == (self.output_dim, self.input_dim)
        assert self.Wx.shape[0] == self.state_dim
        assert self.Wy.shape[0] == self.output_dim

    @classmethod
    def from_discrete(
        cls,
        A: jax.Array,
        B: jax.Array,
        C: jax.Array,
        D: jax.Array | None = None,
        Wx: jax.Array | None = None,
        Wy: jax.Array | None = None,
        dt: float = 1.0,
    ):
        D = jnp.zeros((C.shape[0], B.shape[1])) if D is None else D
        Wx = jnp.zeros((A.shape[0], 1)) if Wx is None else Wx
        Wy = jnp.zeros((C.shape[0], 1)) if Wy is None else Wy
        model = cls(A, B, C, D, Wx, Wy, dt)
        model.check_shapes()
        return model

    @classmethod
    def from_continuos(
        cls,
        A: jax.Array,
        B: jax.Array,
        C: jax.Array,
        D: jax.Array | None = None,
        Wx: jax.Array | None = None,
        Wy: jax.Array | None = None,
        dt: float = 1.0,
    ):
        D = D if D is not None else jnp.zeros((C.shape[0], B.shape[1]))
        continuos_ssm = lti(A, B, C, D).to_discrete(dt)
        A_d = jnp.array(continuos_ssm.A)  # type: ignore
        B_d = jnp.array(continuos_ssm.B)  # type: ignore
        C_d = jnp.array(continuos_ssm.C)  # type: ignore
        D_d = jnp.array(continuos_ssm.D)  # type: ignore
        return cls.from_discrete(A_d, B_d, C_d, D_d, Wx, Wy, dt)

    def step(self, x: jax.Array, u: jax.Array, *, key):
        key_x, key_y = jr.split(key)
        wx = self.Wx @ jr.normal(key_x, (self.Wx.shape[-1],))
        wy = self.Wy @ jr.normal(key_y, (self.Wy.shape[-1],))

        y = self.C @ x + self.D @ u + wy
        new_x = self.A @ x + self.B @ u + wx
        return new_x, y

    def simulate(self, x0: jax.Array, u_t: jax.Array, *, key):
        def scan_step(x, input):
            key, u = input
            new_x, y = self.step(x, u, key=key)
            return new_x, (x, y)

        keys = jax.random.split(key, len(u_t))
        x_final, (x_t, y_t) = jax.lax.scan(scan_step, x0, (keys, u_t))
        return x_final, x_t, y_t


class ToyExample(DSSM):
    @classmethod
    def make(cls, alpha=0.1, beta=0.1, gamma=0.1, noise_std_x=1e0, noise_std_y=1e0):
        A = [
            [1 - alpha, 0.0, gamma],
            [alpha, 1 - beta, 0.0],
            [0.0, 0.0, 1.0],
        ]
        B = [
            [1.0, 1.0],
            [0.0, 0.0],
            [0.0, 0.0],
        ]
        C = [[0.0, 1.0, 0.0]]
        A, B, C, D = jnp.array(A), jnp.array(B), jnp.array(C), None
        Wx = noise_std_x * jnp.eye(3).at[-1, -1].set(0.0)
        Wy = noise_std_y * jnp.eye(1)
        return cls.from_discrete(A, B, C, D, Wx, Wy, dt=1.0)


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

        A = [
            [-a1, 0.0, a1],
            [0.0, -a2 - a3, a2],
            [a1, a2, -a1 - a2],
        ]
        B = [
            [1 / sa, 0.0],
            [0.0, 1 / sa],
            [0.0, 0.0],
        ]
        C = [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
        ]
        A, B, C, D = jnp.array(A), jnp.array(B), jnp.array(C), None
        Wx = noise_std_x * jnp.eye(3)
        Wy = noise_std_y * jnp.eye(2)
        return cls.from_continuos(A, B, C, D, Wx, Wy, dt)
