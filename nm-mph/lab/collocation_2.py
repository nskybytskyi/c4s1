#!/usr/bin/env python
import sympy
import numpy as np
import typing as tp
import matplotlib.pyplot as plt

SympyExpression = tp.Any  # TODO: change to actual value
SympySymbol = tp.Any  # TODO: change to actual value
SympyExpressionFactory = tp.Callable[[SympySymbol], SympyExpression]
SympyExpressionOrFactory = tp.Union[SympyExpressionFactory, SympyExpression]
FloatFunction = tp.Callable[[float], float]


class DifferentialOperator:
    def __init__(self, k: SympyExpressionOrFactory, p: SympyExpressionOrFactory, q: SympyExpressionOrFactory) -> None:
        self.__x: SympySymbol = sympy.Symbol('x')
        self.__k: SympyExpression = k(self.__x) if type(k) == SympyExpressionFactory else k
        self.__p: SympyExpression = p(self.__x) if type(p) == SympyExpressionFactory else p
        self.__q: SympyExpression = q(self.__x) if type(q) == SympyExpressionFactory else q

    def __call__(self, u: SympyExpressionOrFactory, x: tp.Optional[SympySymbol]=None) -> SympyExpression:
        if x is not None:
            self.__k = self.__k.subs(self.__x, x)
            self.__p = self.__p.subs(self.__x, x)
            self.__q = self.__q.subs(self.__x, x)
            self.__x = x

        u: SympyExpression = u(self.__x) if isinstance(u, SympyExpressionFactory) else u
        
        return -(self.__k * u.diff(x)).diff(x) + self.__p * u.diff(x) + self.__q * u


class ProjectionFunction:
    def __init__(self, a: float, b: float, n: int, i: int) -> None:
        self.__a, self.__b, self.__n, self.__i = a, b, n, i

    def __call__(self, f: SympyExpressionFactory, x: SympySymbol) -> float:
        return f(x).evalf(subs={x: self.__a + (self.__b - self.__a) * self.__i / (self.__n - 1)})


class ProjectionSystem:
    def __init__(self, a: float, b: float, n: int) -> None:
        self.__a, self.__b, self.__n = a, b, n

    def __getitem__(self, i: int) -> ProjectionFunction:
        return ProjectionFunction(self.__a, self.__b, self.__n, i)


class CoordinateExpressionFactory:
    def __init__(self, a: float, b: float, k: SympyExpressionFactory, alpha_1: float, alpha_2: float,
                 mu_1: float, mu_2: float, i: int) -> None:
        self.__i, self.__a, self.__b, self.__k, self.__alpha_1, self.__alpha_2, self.__mu_1, self.__mu_2 = \
            i, a, b, k, alpha_1, alpha_2, mu_1, mu_2
        self.__c, self.__d, self.__C, self.__D = None, None, None, None

    def __call__(self, x: SympySymbol) -> SympyExpression:
        if self.__i == 0:
            if self.__C is None:
                self.__C = (self.__alpha_2 * self.__mu_1 - self.__mu_2 * self.__alpha_1) / \
                    (self.__alpha_2 * (self.__alpha_1 * self.__a - self.__k(x).evalf(subs={x: self.__a})) -
                        self.__alpha_1 * (self.__k(x).evalf(subs={x: self.__b}) + self.__alpha_2 * self.__b))
            assert self.__C is not None, "Not Reachable"
            if self.__D is None:
                self.__D = (self.__mu_1 + self.__k(x).evalf(subs={x: self.__a}) * self.__C) / self.__alpha_1 - \
                    self.__C * self.__a
            return self.__C * x + self.__D
        elif self.__i == 1:
            if self.__c is None:
                self.__c = self.__b + (self.__k(x).evalf(subs={x: self.__b}) * (self.__b - self.__a)) / \
                    (2 * self.__k(x).evalf(subs={x: self.__b}) + self.__alpha_2 * (self.__b - self.__a))
            return (x - self.__a) ** 2 * (x - self.__c)
        elif self.__i == 2:
            if self.__d is None:
                self.__d = self.__a - (self.__k(x).evalf(subs={x: self.__a}) * (self.__b - self.__a)) / \
                    (2 * self.__k(x).evalf(subs={x: self.__a}) + self.__alpha_1 * (self.__b - self.__a))
            return (self.__b - x) ** 2 * (x - self.__d)
        else:
            return (x - self.__a) ** 2 * (self.__b - x) ** self.__i


class CoordinateSystemFactory:
    def __init__(self, a: float, b: float, k: FloatFunction, alpha_1: float, alpha_2: float,
                 mu_1: float, mu_2: float) -> None:
        self.__a, self.__b, self.__k, self.__alpha_1, self.__alpha_2, self.__mu_1, self.__mu_2 = \
            a, b, k, alpha_1, alpha_2, mu_1, mu_2

    def __getitem__(self, i: int) -> CoordinateExpressionFactory:
        return CoordinateExpressionFactory(self.__a, self.__b, self.__k,
            self.__alpha_1, self.__alpha_2, self.__mu_1, self.__mu_2, i)


def collocation_method(A: DifferentialOperator, f: SympyExpressionFactory, phi: CoordinateSystemFactory,
                       psi: ProjectionSystem, x: SympySymbol, n: int) -> tp.List[float]:
    return [1., ] + np.linalg.solve(
        np.matrix(
            [[psi[i](lambda x: A(phi[j](x)), x) for j in range(1, n + 1)] for i in range(n)], dtype=float
        ),
        np.array([psi[i](lambda x: f(x) - A(phi[0](x)), x) for i in range(n)], dtype=float)
    ).tolist()


def graph(points, u_true: SympyExpressionFactory, u: SympyExpressionFactory, **kwargs) -> None:
    x: SympySymbol = sympy.Symbol('x')
    u_true_np, u_np = sympy.lambdify(x, u_true(x), "numpy"), sympy.lambdify(x, u(x), "numpy")

    plt.plot(points, u_true_np(points), 'k-', label="True u(points)")
    if "u_label" in kwargs:
        plt.plot(points, u_np(points), 'b-', label=kwargs["u_label"])
    else:
        plt.plot(points, u_np(points), 'b-')

    plt.xlabel('$points$')
    plt.ylabel('$u$')
    if "title" in kwargs:
        plt.title(kwargs["title"])
    plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    x: SympySymbol = sympy.Symbol('x')

    a, b = 0, 4
    alpha_1, alpha_2 = 4, 2

    m_1, m_2, m_3 = 2, 2, 1
    u_true: SympyExpressionFactory = lambda x: m_1 * sympy.sin(m_2 * x) + m_3

    k_1, k_2, k_3 = 2, 3, 1
    k: SympyExpressionFactory = lambda x: k_1 * sympy.sin(k_2 * x) + k_3

    mu_1: float = (alpha_1 * u_true(x) - k(x) * u_true(x).diff(x)).evalf(subs={x: a}, chop=True)
    mu_2: float = (alpha_2 * u_true(x) + k(x) * u_true(x).diff(x)).evalf(subs={x: b}, chop=True)

    p_1, p_2, p_3 = 2, 1, 1
    p: SympyExpressionFactory = lambda x: p_1 * sympy.cos(p_2 * x) + p_3

    q_1, q_2, q_3 = 0, 2, 3
    q: SympyExpressionFactory = lambda x: q_1 * sympy.sin(q_2 * x) + q_3

    A = DifferentialOperator(k, p, q)
    f: SympyExpressionFactory = lambda x: A(u_true(x)).simplify()

    n = 8

    phi: CoordinateSystemFactory = CoordinateSystemFactory(a, b, k, alpha_1, alpha_2, mu_1, mu_2)

    psi: ProjectionSystem = ProjectionSystem(a, b, n)

    c: tp.List[float] = collocation_method(A, f, phi, psi, x, n)
    u: SympyExpressionFactory = lambda x: sum(c[i] * phi[i](x) for i in range(n + 1))

    graph(np.linspace(a, b, 50 + 1), u_true, u, title="Collocation method", u_label="$u_{%i}(x)$" % n)

    # delta = sympy.integrate((u_true(x) - u(x))**2, (x, a, b)) / (b - a)
    # print(f"Collocation-{n} delta = {delta}")
