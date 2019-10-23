#!/usr/bin/env python
import sympy
import numpy as np
import typing as tp
from scipy import integrate
import matplotlib.pyplot as plt

SympyExpression = tp.Any  # TODO: change to actual value
SympySymbol = tp.Any  # TODO: change to actual value


class DifferentialOperator:
    def __init__(self, k: SympyExpression, p: SympyExpression, q: SympyExpression) -> None:
        self.__k, self.__p, self.__q = k, p, q

    def __call__(self, u: SympyExpression, x: SympySymbol) -> SympyExpression:
        return -(self.__k * u.diff(x)).diff(x) + self.__p * u.diff(x) + self.__q * u


class PsiFunction:
    def __init__(self, a: float, b: float, n: int, i: int) -> None:
        self.__a, self.__b, self.__n, self.__i = a, b, n, i

    def __call__(self, f: SympyExpression, x: SympySymbol) -> float:
        return f.evalf(subs={x: self.__a + (self.__b - self.__a) * self.__i / (self.__n - 1)})


def collocation_method(a: DifferentialOperator, f: SympyExpression, phi: tp.List[SympyExpression],
        psi: tp.List[tp.Callable[[SympyExpression], float]], x: SympySymbol, n: int) -> tp.List[float]:
    return np.linalg.solve(np.matrix([[psi[i](a(phi[j], x), x) for j in range(1, n + 1)] for i in range(n)], dtype=float),
        np.array([psi[i](f - a(phi[0], x), x) for i in range(n)], dtype=float)).tolist()


class PhiExpression:
    def __init__(self, _x, _a, _b, k, _alpha_1, _alpha_2):
        """
        :param _x: variable in which functions will be created, sympy.Symbol
        :param _a: left endpoint of a segment, float
        :param _b: right endpoint of a segment, float
        :param k: function
        :param _alpha_1: float
        :param _alpha_2: float
        """
        self.__x, self.__a, self.__b, = _x, _a, _b
        self.__c = _b + (k(_b) * (_b - _a)) / (2 * k(_b) + _alpha_2 * (_b - _a))
        self.__d = _a - (k(_a) * (_b - _a)) / (2 * k(_a) + _alpha_1 * (_b - _a))

    def __call__(self, i):
        """
        :param i: non-negative integer
        :return: expression
        """
        if i == 0:
            return ((self.__x - self.__a) / (self.__b - self.__a)) ** 2 * \
                ((self.__x - self.__c) / (self.__b - self.__a))
        elif i == 1:
            return ((self.__b - self.__x) / (self.__b - self.__a)) ** 2 * \
                ((self.__x - self.__d) / (self.__b - self.__a))
        else:
            return ((self.__x - self.__a) / (self.__b - self.__a)) ** 2 * \
                ((self.__b - self.__x) / (self.__b - self.__a)) ** i
        # elif i & 1:
        #     return ((self.__x - self.__a) / (self.__b - self.__a)) ** (1 + i // 2) * \
        #         ((self.__b - self.__x) / (self.__b - self.__a)) ** (2 + i // 2)
        # else:
        #     return ((self.__x - self.__a) / (self.__b - self.__a)) ** (1 + i // 2) * \
        #         ((self.__b - self.__x) / (self.__b - self.__a)) ** (1 + i // 2)


def graph(points, u_true, u, **kwargs) -> None:
    """
    :param points: points in which to evaluate functions
    :param u_true: true solution
    :param u: approximate solution
    :param kwargs: optional title and labels for plot
    """
    plt.plot(points, u_true(points), 'k-', label="True u(points)")
    if "u_label" in kwargs:
        plt.plot(points, u(points), 'b-', label=kwargs["u_label"])
    else:
        plt.plot(points, u(points), 'b-')
    plt.xlabel('$points$')
    plt.ylabel('$u$')
    if "title" in kwargs:
        plt.title(kwargs["title"])
    plt.legend(loc='best')
    plt.grid()
    plt.show()


if __name__ == '__main__':
    a, b = 0, 4
    alpha_1, alpha_2 = 4, 2

    x = sympy.Symbol('x')

    m_1, m_2, m_3 = 2, 2, 1
    u_true_expression = m_1 * sympy.sin(m_2 * x) + m_3
    u_true_function = sympy.lambdify(x, u_true_expression, 'numpy')

    du_true_expression = u_true_expression.diff(x)
    du_true_function = sympy.lambdify(x, du_true_expression, 'numpy')

    k_1, k_2, k_3 = 2, 3, 1
    k_expression = k_1 * sympy.sin(k_2 * x) + k_3
    k_function = sympy.lambdify(x, k_expression, 'numpy')

    mu_1 = -k_function(a) * du_true_function(a) + alpha_1 * u_true_function(a)
    mu_2 = k_function(b) * du_true_function(b) + alpha_2 * u_true_function(b)

    p_1, p_2, p_3 = 2, 1, 1
    p_expression = p_1 * sympy.cos(p_2 * x) + p_3
    p_function = sympy.lambdify(x, p_expression, 'numpy')

    q_1, q_2, q_3 = 0, 2, 3
    q_expression = q_1 * sympy.sin(q_2 * x) + q_3
    q_function = sympy.lambdify(x, q_expression, 'numpy')

    A_operator = DifferentialOperator(k_expression, p_expression, q_expression)

    f_expression = A_operator(u_true_expression, x).simplify()
    f_function = sympy.lambdify(x, f_expression, 'numpy')

    C = (alpha_2 * mu_1 - mu_2 * alpha_1) / \
        (alpha_2 * (alpha_1 * a - k_function(a)) - alpha_1 * (k_function(b) + alpha_2 * b))
    D = (mu_1 + k_function(a) * C) / alpha_1 - C * a
    phi_0_expression = C * x + D

    n = 20

    phi_expression = PhiExpression(x, a, b, k_function, alpha_1, alpha_2)
    u_expression_coefficients = collocation_method(A_operator, f_expression,
        [phi_0_expression,] + [phi_expression(i) for i in range(n)], [PsiFunction(a, b, n, i) for i in range(n)], x, n)
    u_expression = sum(u_expression_coefficients[i] * phi_expression(i) for i in range(n)) + phi_0_expression
    u_function = sympy.lambdify(x, u_expression, 'numpy')

    graph(np.linspace(a, b, 50 + 1), u_true_function, u_function,
          title="Collocation method", u_label="Collocation method: $u_{%i}(x)$" % n)

    delta = integrate.quad(lambda x: (u_true_function(x) - u_function(x))**2, a, b)[0] / (b - a)
    print(f"Collocation-{n} delta = {delta}")

    n = 20

    # scalar_product = ScalarProductExpression(x, a, b)
    scalar_product = ScalarProductFunction(a, b)

    # u_expression_coefficients = ritz_method_expressions(A_operator, f_expression, phi_expression, scalar_product)
    u_expression_coefficients = ritz_method_functions(A_operator, f_expression, phi_expression, scalar_product)
    u_expression = sum(u_expression_coefficients[i] * phi_expression(i) for i in range(n)) + phi_0_expression
    u_function = sympy.lambdify(x, u_expression, 'numpy')

    graph(np.linspace(a, b, 50 + 1), u_true_function, u_function,
          title="Ritz method", u_label="Ritz method: $u_{%i}(x)$" % n)

    delta = integrate.quad(lambda x: (u_true_function(x) - u_function(x))**2, a, b)[0] / (b - a)
    print(f"Ritz-{n} delta = {delta}")
