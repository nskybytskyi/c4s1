#!/usr/bin/env python
import scipy.integrate as integrate
import matplotlib.pyplot as plt
import typing as tp
import numpy as np


class SinLinFunc:
    def __init__(self, c_1: float, c_2: float, c_3: float) -> None:
        self._c_1, self._c_2, self._c_3 = c_1, c_2, c_3

    def __call__(self, x: float) -> float:
        return self._c_1 * np.sin(self._c_2 * x) + self._c_3


class PhiFunc:
    def __init__(self, xi: float, h: float) -> None:
        self._xi, self._h = xi, h

    def __call__(self, x: float) -> float:
        if self._xi - self._h <= x <= self._xi:
            return (x - (self._xi - self._h)) / self._h
        elif self._xi <= x <= self._xi + self._h:
            return ((self._xi + self._h) - x) / self._h
        else:
            return 0.0


class PhiFactory:
    def __init__(self, a: float, b: float, n: int, h: float, x: np.array) -> None:
        self._a, self._b, self._n, self._h, self._x = a, b, n, h, x

    def __getitem__(self, i: int) -> PhiFunc:
        return PhiFunc(self._x[i], self._h)


if __name__ == '__main__':
    a, b = 0.0, 4.0
    alpha_1, alpha_2 = 4.0, 2.0
    k, q, u_true = SinLinFunc(2.0, 3.0, 1.0), SinLinFunc(0.0, 2.0, 3.0), SinLinFunc(2.0, 2.0, 1.0)

    mu_1, mu_2 = 0.0, 6.0

    def f(x: float) -> float:
        return 14 * np.sin(2 * x) - 4 * np.cos(x) - 20 * np.cos(5 * x) + 3

    for n in (80,):
        xs, h = np.linspace(a, b, n + 1), (b - a) / n
        phi = PhiFactory(a, b, n, h, xs)

        # region A
        A = np.zeros((n + 1, n + 1))

        A[0, 0] = integrate.quad(lambda x: k(x) / h**2 + q(x) * phi[0](x)**2, xs[0], xs[1])[0] + alpha_1

        for i in range(1, n + 1):
            A[i, i - 1] = integrate.quad(lambda x: -k(x) / h**2 + q(x) * phi[i](x) * phi[i - 1](x), xs[i - 1], xs[i])[0]

        for i in range(1, n):
            A[i, i] = integrate.quad(lambda x: k(x) / h**2 + q(x) * phi[i](x)**2, xs[i - 1], xs[i + 1])[0]

        for i in range(n):
            A[i, i + 1] = integrate.quad(lambda x: -k(x) / h**2 + q(x) * phi[i](x) * phi[i + 1](x), xs[i], xs[i + 1])[0]

        A[n, n] = integrate.quad(lambda x: k(x) / h**2 + q(x) * phi[n](x)**2, xs[n - 1], xs[n])[0] + alpha_2
        # endregion

        # region B
        B = np.zeros((n + 1,))

        B[0] = integrate.quad(lambda x: f(x) * phi[0](x), xs[0], xs[1])[0] + mu_1
        
        for i in range(1, n):
            B[i] = integrate.quad(lambda x: f(x) * phi[i](x), xs[i - 1], xs[i + 1])[0]

        B[n] = integrate.quad(lambda x: f(x) * phi[n](x), xs[n - 1], xs[n])[0] + mu_2
        # endregion

        # region solution
        C = np.linalg.solve(A, B)

        def u_n(x: float) -> float:
            s = 0.0
            for i in range(n + 1):
                s += C[i] * phi[i](x)
            return s
        # endregion

        print(f"n = {n}, error = {integrate.quad(lambda x: (u_n(x) - u_true(x))**2, a, b)[0] / (b - a)}")
        
        print("-------------------------------------------")
        print("x_i  | Справжній  | Наближений | Відхилення")
        print("-------------------------------------------")

        for xi in xs:
            print(f"{xi:.2f} | {u_true(xi):10.7f} | {u_n(xi):10.7f} | {u_true(xi) - u_n(xi):10.7f}")

        print("------------------------------------------")

        if input() == 'n':
            break

        # region plotting
        xs = np.linspace(a, b, 16 * n + 1)
        u_ns = np.array([u_n(x) for x in xs])
        u_trues = np.array([u_true(x) for x in xs])
        plt.figure(figsize=(20,20))
        plt.title(f"Finite elements method", fontsize=40)
        plt.plot(xs, u_ns, 'k-', label="$u_{" + str(n) + "}(x)$")
        plt.plot(xs, u_trues, 'r--', label="True u(x)")
        plt.xlabel('$x$', fontsize=40)
        plt.ylabel('$u$', fontsize=40)
        plt.legend(loc='best', fontsize=40)
        plt.grid(True)
        plt.show()
        # plt.savefig(f'fem_{n}.png', bbox_inches='tight')
        # endregion

        if input() == 'n':
            break
