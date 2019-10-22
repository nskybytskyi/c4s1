#!/usr/bin/env python
import matplotlib.pyplot as plt
import numpy as np
import sympy
from scipy import integrate


def A_operator(u_expression):
    return -(k_expression * u_expression.diff(x)).diff(x) + \
           p_expression * u_expression.diff(x) + q_expression * u_expression


def collocation_method(A_operator, f_expression, phi_expressions, PsiFunctionFactory):
    f_function_modified = sympy.lambdify(x, f_expression - A_operator(phi_0_expression), 'numpy')
    A_phi_functions = [sympy.lambdify(x, A_operator(phi_expressions(j)), 'numpy') for j in range(n)]
    lhs_matrix = np.matrix([[PsiFunctionFactory(i)(A_phi_functions[j]) for j in range(n)] for i in range(n)])
    rhs_vector = np.matrix([[PsiFunctionFactory(i)(f_function_modified)] for i in range(n)])
    return np.linalg.solve(lhs_matrix, rhs_vector)


def phi_expressions(i):
    if i == 0:
        return (x - a) ** 2 * (x - c)
    elif i == 1:
        return (b - x) ** 2 * (x - d)
    else:
        return (x - a) ** 2 * (b - x) ** i


class PsiFunctionFactory:
    def __init__(self, i):
        self.__i = i

    def __call__(self, f_function):
        return f_function(a + self.__i * (b - a) / (n - 1))


def graph(x, u_true, u, **kwargs):
    plt.plot(x, u_true(x), 'k-', label="True u(x)")
    if "u_label" in kwargs:
        plt.plot(x, u(x), 'b-', label=kwargs["u_label"])
    else:
        plt.plot(x, u(x), 'b-')
    plt.xlabel('$x$')
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

    f_expression = A_operator(u_true_expression).simplify()
    f_function = sympy.lambdify(x, f_expression, 'numpy')

    C = (alpha_2 * mu_1 - mu_2 * alpha_1) / \
        (alpha_2 * (alpha_1 * a - k_function(a)) - alpha_1 * (k_function(b) + alpha_2 * b))
    D = (mu_1 + k_function(a) * C) / alpha_1 - C * a
    phi_0_expression = C * x + D

    c = b + (k_function(b) * (b - a)) / (2 * k_function(b) + alpha_2 * (b - a))
    d = a + (k_function(a) * (b - a)) / (2 * k_function(a) + alpha_1 * (b - a))

    n = 14

    u_expression_coefficients = collocation_method(A_operator, f_expression, phi_expressions, PsiFunctionFactory)
    u_expression = sum(u_expression_coefficients[i] * phi_expressions(i) for i in range(n)) + phi_0_expression
    u_function = sympy.lambdify(x, u_expression, 'numpy')

    graph(np.linspace(a, b, 50 + 1), u_true_function, u_function,
          title="Collocation method", u_label="Collocation method: $u_{%i}(x)$" % n)

    print(f"delta = {integrate.quad(lambda x: (u_true_function(x) - u_function(x))**2, a, b)}")
