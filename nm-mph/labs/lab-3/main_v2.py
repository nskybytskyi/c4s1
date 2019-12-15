#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt


def solve_tridiagonal(a: np.array, b: np.array, c: np.array, d: np.array) -> np.array:
    n = len(d)
    for i in range(1, n):
        m = a[i - 1] / b[i - 1]
        b[i] = b[i] - m * c[i - 1] 
        d[i] = d[i] - m * d[i - 1]

    x = b
    x[-1] = d[-1] / b[-1]

    for i in range(n - 2, -1, -1):
        x[i] = (d[i] - c[i] * x[i + 1]) / b[i]

    return x


class PhysicalMaterial:
    def __init__(self, lambda_: float, c_: float, rho_: float, gamma_: float) -> None:
        self._lambda, self._c, self._rho, self._gamma = lambda_, c_, rho_, gamma_

    def __repr__(self) -> str:
        return f"PhysicalMaterial(lambda_={self._lambda}, c_={self._c}, rho_={self._rho}, gamma_={self._gamma})"


class PhysicalObject:
    def __init__(self, material_: PhysicalMaterial, r_: float, m_: int, u_0_: float) -> None:
        self._material, self._r, self._m, self._u_0 = material_, r_, m_, u_0_

    def __repr__(self) -> str:
        return f"PhysicalObject(material_={self._material}, r_={self._r}, m_={self._m}, u_0_={self._u_0})"

    @property
    def _lambda(self):
        return self._material._lambda

    @property
    def _c(self):
        return self._material._c

    @property
    def _rho(self):
        return self._material._rho

    @property
    def _gamma(self):
        return self._material._gamma


class PhysicalEnvironment:
    def __init__(self, u_env_: float) -> None:
        self._u_env = u_env_

    def __repr__(self) -> str:
        return f"PhysicalEnvironment(u_env_={self._u_env})"


class TimeRescaler:
    def __init__(self, physical_object: PhysicalObject) -> None:
        self._a_squared = physical_object._lambda / (physical_object._c * physical_object._rho)
        self._r_squared = physical_object._r ** 2

    def t_to_t1(self, t: float) -> float:
        return self._a_squared * t / self._r_squared

    def t1_to_t(self, t1: float) -> float:
        return self._r_squared * t1 / self._a_squared


class TemperatureRescaler:
    def __init__(self, physical_object: PhysicalObject, physical_environment: PhysicalEnvironment) -> None:
        self._u_0, self._u_env = physical_object._u_0, physical_environment._u_env

    def u_to_v(self, u: float) -> float:
        return (u - self._u_0) / (self._u_env - self._u_0)

    def v_to_u(self, v: float) -> float:
        return v * (self._u_env - self._u_0) + self._u_0


class RadiusRescaler:
    def __init__(self, physical_object: PhysicalObject) -> None:
        self._r = physical_object._r

    def x_to_x1(self, x: float) -> float:
        return x / self._r

    def x1_to_x(self, x1: float) -> float:
        return x1 * self._r


class Plotter:
    def __init__(self, t_scaler: TimeRescaler, u_scaler: TemperatureRescaler, x_scaler: RadiusRescaler) -> None:
        self._t_scaler, self._u_scaler, self._x_scaler = t_scaler, u_scaler, x_scaler

    def plot(self, t1: float, x1s: np.array, vs: np.array, *args) -> None:
        xs = np.array([self._x_scaler.x1_to_x(x1) for x1 in x1s])
        us = np.array([self._u_scaler.v_to_u(v) for v in vs])
        t = self._t_scaler.t1_to_t(t1)

        plt.figure(figsize=(10, 20))
        plt.grid(True)
        plt.plot(xs, us, 'r', label=f'$u(x, {t:.2f})$')
        plt.xlabel("$x$")
        plt.ylabel("$u$")
        plt.legend(loc="best")

        if "show" in args:
            plt.show()
        if "save" in args:
            plt.savefig(f"{t:.2f}.png", bbox_inches='tight')


class Solver:
    def __init__(self, physical_object: PhysicalObject, physical_environment: PhysicalEnvironment) -> None:
        self._gamma_1 = physical_object._gamma * physical_object._r / physical_object._lambda
        self._lambda = physical_object._lambda

        self._N = 1000
        self._h = 1 / self._N
        self._tau = 100
        self._sigma = .5

        self._t1 = 0
        self._x1s = np.linspace(0, 1, self._N + 1)
        self._vs = np.zeros(self._N + 1)

    def _a(self, i: int) -> float:
        if i == 0:
            return 0.0
        if i == self._N:
            return self._lambda * self._sigma / self._h
        else:
            return -self._sigma / self._h ** 2

    def _b(self, i: int) -> float:
        if i == 0:
            return -self._lambda * self._sigma / self._h \
                -self._gamma_1 * self._sigma \
                -self._h * self._lambda / (2 * self._tau)
        elif i == self._N:
            return -self._lambda * self._sigma / self._h \
                -self._gamma_1 * self._sigma \
                -self._h * self._lambda / (2 * self._tau)
        else:
            return 1 / self._tau + 2 * self._sigma / (self._h ** 2)

    def _c(self, i: int) -> float:
        if i == 0:
            return self._lambda * self._sigma / self._h
        if i == self._N:
            return 0.0
        else:
            return -self._sigma / self._h ** 2

    def _d(self, i: int) -> float:
        if i == 0:
            return -self._lambda * (1 - self._sigma) * (self._vs[1] - self._vs[0]) / self._h \
                + (1 - self._sigma) * self._gamma_1 * self._vs[0] \
                -self._h * self._lambda / (2 * self._tau) * self._vs[0]
        elif i == self._N:
            return self._lambda * (1 - self._sigma) * (self._vs[self._N] - self._vs[self._N - 1]) / self._h \
                + (1 - self._sigma) * self._gamma_1 * self._vs[self._N] \
                -self._h * self._lambda / (2 * self._tau) * self._vs[self._N]
        else:
            return self._vs[i] / self._tau \
                + (1 - self._sigma) * (self._vs[i + 1] - 2 * self._vs[i] + self._vs[i - 1]) / (self._h ** 2)

    def step(self) -> None:
        self._vs = solve_tridiagonal(
            a=np.array([self._a(i) for i in range(self._N + 1)]),
            b=np.array([self._b(i) for i in range(self._N + 1)]),
            c=np.array([self._c(i) for i in range(self._N + 1)]),
            d=np.array([self._d(i) for i in range(self._N + 1)])
        )

        self._t1 += self._tau


if __name__ == "__main__":
    aluminum = PhysicalMaterial(lambda_=220, c_=890, rho_=2_700, gamma_=300)
    aluminum_ball = PhysicalObject(material_=aluminum, r_=0.01, m_=2, u_0_=273)
    environment = PhysicalEnvironment(u_env_=573)
    t_scaler = TimeRescaler(physical_object=aluminum_ball)
    u_scaler = TemperatureRescaler(physical_object=aluminum_ball, physical_environment=environment)
    x_scaler = RadiusRescaler(physical_object=aluminum_ball)
    solver = Solver(physical_object=aluminum_ball, physical_environment=environment)
    plotter = Plotter(t_scaler=t_scaler, u_scaler=u_scaler, x_scaler=x_scaler)

    for _ in range(100):
        solver.step()
        plotter.plot(solver._t1, solver._x1s, solver._vs, "show")
