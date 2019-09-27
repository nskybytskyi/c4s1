#!/usr/bin/env python
import numpy as np
import typing as tp
from trigonometry import LinearTrigonometricFunction as LTF, DifferentialOperator as D


class Solver:
    def __init__(self, a: float, b: float, alpha: tp.Tuple[float, float], k: LTF, p: LTF, q: LTF,
            mu: tp.Optional[tp.Tuple[float, float]]=None, f: tp.Optional[tp.Callable[[float], float]]=None,
            true_u: tp.Optional[LTF]=None):
        self.__a, self.__b = a, b
        self.__alpha = alpha
        self.__k = k
        self.__p = p
        self.__q = q
        assert (mu is not None and f is not None) or true_u is not None, "problem under-defined"
        if mu is not None and f is not None:
            self.__mu = mu
            self.__f = f
        else:  # m is not None
            self.__true_u = true_u
            self.__mu = (
                -k(a) * D(true_u)(a) + alpha[0] * u(a),
                k(b) * D(true_u)(b) + alpha[1] * u(b)
            )
            f = -D(k * D(true_u)) + p * D(u) + q * u

    @property
    def k(self) -> LTF:
        return self.__k
            
    @property
    def p(self) -> LTF:
        return self.__p
            
    @property
    def q(self) -> LTF:
        return self.__q
            
    @property
    def true_u(self) -> LTF:
        return self.__true_u
            
    @property
    def mu(self) -> tp.Tuple[float, float]:
        return self.__mu
            
    @property
    def f(self) -> LTF:
        return self.__f

    def A(f: LTF) -> LTF:
        return -D(self.__k * D(f)) + self.__p * D(f) + self.__q * f


if __name__ == "__main__":
    k = LTF(k_0=1, sine_ks=[(2, 3)], cosine_ks=[])
    p = LTF(k_0=1, cosine_ks=[(2, 1)], sine_ks=[])
    q = LTF(k_0=3, sine_ks=[], cosine_ks=[])
    true_u = LTF(k_0=1, sine_ks=[(2, 2)], cosine_ks=[])
    solver = Solver(a=0, b=4, alpha=(4, 2), k=k, p=p, q=q, true_u=true_u)
    print(f"k(x) = {solver.k}")
    print(f"p(x) = {solver.p}")
    print(f"q(x) = {solver.q}")
    print(f"u(x) = {solver.true_u}")
    print(f"dk/dx(x) = {D(solver.k)}")
    print(f"du/dx(x) = {D(solver.true_u)}")
    print(f"d²u/dx²(x) = {D(solver.true_u, 2)}")
    print(f"mu = {solver.mu}")
    print(f"f(x) = {solver.f}")
    print(f"u(x) = {solver.true_u}")
    print(f"Au(x) = {solver.A(solver.true_u)}")

    # D = [[psi[j](A(phi[i]) for j in range(n)] for i in range(n)]
