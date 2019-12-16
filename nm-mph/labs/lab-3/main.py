#!/usr/bin/env python
import numpy as np
import typing as tp
import matplotlib.pyplot as plt


class PhysicalMaterial:
    def __init__(self, λ: float, c: float, ρ: float, γ: float) -> None:
        self.λ, self.c, self.ρ, self.γ = λ, c, ρ, γ


class PhysicalObject:
    def __init__(self, material: PhysicalMaterial, r: float, m: int, u_0: float) -> None:
        self.material, self.r, self._m, self.u_0 = material, r, m, u_0

    @property
    def λ(self):
        return self.material.λ

    @property
    def c(self):
        return self.material.c

    @property
    def ρ(self):
        return self.material.ρ

    @property
    def γ(self):
        return self.material.γ

    @property
    def a(self):
        return np.sqrt(self.λ / (self.c * self.ρ))


class PhysicalEnvironment:
    def __init__(self, u_env: float) -> None:
        self.u_env = u_env


class TimeRescaler:
    def __init__(self, a: float, r: float) -> None:
        self.a, self.r = a, r

    def t_to_t1(self, t: float) -> float:
        return self.a**2 * t / (self.r**2)

    def t1_to_t(self, t1: float) -> float:
        return self.r**2 * t1 / (self.a**2)


class TemperatureRescaler:
    def __init__(self, u_0: float, u_env: float) -> None:
        self.u_0, self.u_env = u_0, u_env

    def u_to_v(self, u: float) -> float:
        return (u - self.u_0) / (self.u_env - self.u_0)

    def v_to_u(self, v: float) -> float:
        return v * (self.u_env - self.u_0) + self.u_0


class RadiusRescaler:
    def __init__(self, r: float) -> None:
        self.r = r

    def x_to_x1(self, x: float) -> float:
        return x / self.r

    def x1_to_x(self, x1: float) -> float:
        return x1 * self.r


class Plotter:
    def __init__(self) -> None:
        plt.figure(figsize=(20, 10))
        plt.grid(True)
        plt.xlabel("$x$")
        plt.ylabel("$u$")

    def plot(self, t: float, xs: np.array, us: np.array) -> None:
        plt.plot(xs, us, label=f'$u(x, {t:.2f})$')

    def show(self):
        plt.legend(loc="best")
        plt.show()
    
    def save(self):
        plt.legend(loc="best")
        plt.savefig("plots.png", bbox_inches='tight')


class Solver:
    def __init__(self, physical_object: PhysicalObject, physical_environment: PhysicalEnvironment) -> None:
        self.γ1 = physical_object.γ * physical_object.r / physical_object.λ  # slower
        self.k = physical_object.a

        self.N = 100
        self.h = 1 / self.N
        self.τ = self.h**2 / (2 * self.k)  # faster
        self.σ = .5 - self.h**2 / (12 * self.τ)

        self.τ_h = self.τ / self.h
        self.τ_h2 = self.τ / self.h**2

        self.t1 = 0
        self.x1s = np.linspace(0, 1, self.N + 1)
        self.vs = np.zeros(self.N + 1)

    def bar_p(self, i: int) -> float:
        if i == 0:
            return 0.0
        elif i != self.N + 1:
            return ((self.x1s[i - 1] + self.x1s[i]) / 2)**2 * self.k
        else:
            return 0.0

    def bar_x2(self, i: int) -> float:
        if i == 0:
            return (self.x1s[i + 1]**3 - self.x1s[i]**3) / (3 * self.h)
        elif i != self.N:
            return (self.x1s[i + 1]**3 - self.x1s[i - 1]**3) / (6 * self.h)
        else:
            return (self.x1s[i]**3 - self.x1s[i - 1]**3) / (3 * self.h)

    def a(self, i: int) -> float:
        if i == 0:
            return 0.0
        elif i != self.N:
            return self.σ * self.τ_h2 * self.bar_p(i)
        else:
            return self.σ * self.τ_h2 * self.bar_p(i)

    def b(self, i: int) -> float:
        if i == 0:
            return -self.bar_x2(i) / 2 - self.c(i)
        elif i != self.N:
            return -self.bar_x2(i) - self.a(i) - self.c(i)
        else:
            return -self.σ * self.τ_h * self.γ1 * self.x1s[i]**2 - self.bar_x2(i) / 2 - self.a(i)

    def c(self, i: int) -> float:
        if i == 0:
            return self.σ * self.τ_h2 * self.bar_p(i + 1)
        elif i != self.N:
            return self.σ * self.τ_h2 * self.bar_p(i + 1)
        else:
            return 0.0

    def d(self, i: int) -> float:
        if i == 0:
            return -self.bar_x2(i) * self.vs[i] / 2 \
                -(1 - self.σ) * self.τ_h2 * self.bar_p(i + 1) * (self.vs[i + 1] - self.vs[i])
        elif i != self.N:
            return -self.bar_x2(i) * self.vs[i] - (1 - self.σ) * self.τ_h2 \
                * (self.bar_p(i + 1) * (self.vs[i + 1] - self.vs[i]) - self.bar_p(i) * (self.vs[i] - self.vs[i - 1])) 
        else:
            return (1 - self.σ) * self.τ_h * self.γ1 * self.x1s[i]**2 * self.vs[i] \
                -self.τ_h * self.γ1 * self.x1s[i]**2 - self.bar_x2(i) * self.vs[i] / 2 \
                + (1 - self.σ) * self.τ_h2 * self.bar_p(i) * (self.vs[i] - self.vs[i - 1])

    def step(self) -> None:
        lhs = np.zeros((self.N + 1, self.N + 1))
        rhs = np.zeros(self.N + 1)

        lhs[0, 0] = self.b(0)
        lhs[0, 1] = self.c(0)
        rhs[0] = self.d(0)

        for i in range(1, self.N):
            lhs[i, i - 1] = self.a(i)
            lhs[i, i] = self.b(i)
            lhs[i, i + 1] = self.c(i)
            rhs[i] = self.d(i)

        lhs[self.N, self.N - 1] = self.a(self.N)
        lhs[self.N, self.N] = self.b(self.N)
        rhs[self.N] = self.d(self.N)

        self.vs = np.linalg.solve(lhs, rhs)

        self.t1 += self.τ


class Iterater:
    def __init__(self, solver: Solver, plotter: Plotter, t_scaler: TimeRescaler,
            u_scaler: TemperatureRescaler, x_scaler: RadiusRescaler) -> None:
        self.solver, self.plotter = solver, plotter
        self.t_scaler, self.u_scaler, self.x_scaler = t_scaler, u_scaler, x_scaler

    def iterate(self, count_outer: int, count_inner: int) -> None:
        for _ in range(count_outer):
            self.solver.step()

            if _ % count_inner == count_inner - 1:
                t = self.t_scaler.t1_to_t(self.solver.t1)
                xs = np.array([self.x_scaler.x1_to_x(x1) for x1 in self.solver.x1s])
                us = np.array([self.u_scaler.v_to_u(v) for v in self.solver.vs])

                self.plotter.plot(t, xs, us)

                if us[1] > 373:
                    break

        # self.plotter.show()
        self.plotter.save()


if __name__ == "__main__":

    aluminum    = PhysicalMaterial(λ=220, c=890, ρ=2_700, γ=300)
    sphere      = PhysicalObject(material=aluminum, r=0.01, m=2, u_0=273)
    environment = PhysicalEnvironment(u_env=573)

    t_scaler = TimeRescaler(a=sphere.a, r=sphere.r)
    u_scaler = TemperatureRescaler(u_0=sphere.u_0, u_env=environment.u_env)
    x_scaler = RadiusRescaler(r=sphere.r)

    solver   = Solver(physical_object=sphere, physical_environment=environment)
    plotter  = Plotter()
    iterater = Iterater(solver=solver, plotter=plotter, t_scaler=t_scaler, u_scaler=u_scaler, x_scaler=x_scaler)

    iterater.iterate(count_outer=100_000, count_inner=351)  # 1402 # 701 # 351 # 175
