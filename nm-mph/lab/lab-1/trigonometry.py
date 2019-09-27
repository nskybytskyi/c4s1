#!/usr/bin/env python
import copy
import unittest
from collections import defaultdict
import numpy as np
import typing as tp


class _SineFunction:
    """ k_0 sin(k_1 x) """
    def __init__(self, k: tp.Tuple[float, float]):
        assert k[0] != 0, "function is constant zero"
        assert k[1] != 0, "function is constant zero"
        self.__k = k
        if self.__k[1] < 0:
            self.__k = -self.__k[0], -self.__k[1]

    def __call__(self, x: float) -> float:
        return self.__k[0] * np.sin(self.__k[1] * x)

    def __repr__(self) -> str:
        if self.__k[0] == 1:
            if self.__k[1] == 1:
                return "sin(x)"
            elif self.__k[1] == -1:
                return "sin(-x)"
            else:
                return f"sin({self.__k[1]} x)"
        elif self.__k[0] == -1:
            if self.__k[1] == 1:
                return "-sin(x)"
            elif self.__k[1] == -1:
                return "-sin(-x)"
            else:
                return f"-sin({self.__k[1]} x)"
        else:
            if self.__k[1] == 1:
                return f"{self.__k[0]} sin(x)"
            elif self.__k[1] == -1:
                return f"{self.__k[0]} sin(-x)"
            else:
                return f"{self.__k[0]} sin({self.__k[1]} x)"

    def __add__(self, other):
        """
        :param other: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        return _LinearTrigonometricFunctionFactory(self) + other

    def __sub__(self, other):
        """
        :param other: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        return _LinearTrigonometricFunctionFactory(self) - other

    def __neg__(self):
        """ :return: SineFunction """
        return _SineFunction((-self.k[0], self.k[1]))

    def __mul__(self, other):
        """
        :param other: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        return _LinearTrigonometricFunctionFactory(self) * other

    @property
    def k(self) -> tp.Tuple[float, float]:
        return self.__k


class _CosineFunction:
    """ k_0 cos(k_1 x) """
    def __init__(self, k: tp.Tuple[float, float]):
        assert k[0] != 0, "function is constant zero"
        assert k[1] != 0, "function is constant one"
        self.__k = k
        if self.__k[1] < 0:
            self.__k = self.__k[0], -self.__k[1]

    def __call__(self, x: float) -> float:
        return self.__k[0] * np.cos(self.__k[1] * x)

    def __repr__(self) -> str:
        if self.__k[0] == 1:
            if self.__k[1] == 1:
                return "cos(x)"
            elif self.__k[1] == -1:
                return "cos(-x)"
            else:
                return f"cos({self.__k[1]} x)"
        elif self.__k[0] == -1:
            if self.__k[1] == 1:
                return "-cos(x)"
            elif self.__k[1] == -1:
                return "-cos(-x)"
            else:
                return f"-cos({self.__k[1]} x)"
        else:
            if self.__k[1] == 1:
                return f"{self.__k[0]} cos(x)"
            elif self.__k[1] == -1:
                return f"{self.__k[0]} cos(-x)"
            else:
                return f"{self.__k[0]} cos({self.__k[1]} x)"

    def __add__(self, other):
        """
        :param other: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        return _LinearTrigonometricFunctionFactory(self) + other

    def __sub__(self, other):
        """
        :param other: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        return _LinearTrigonometricFunctionFactory(self) - other

    def __neg__(self):
        """ :return: CosineFunction """
        return _CosineFunction((-self.k[0], self.k[1]))

    def __mul__(self, other):
        """
        :param other: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        return _LinearTrigonometricFunctionFactory(self) * other

    @property
    def k(self) -> tp.Tuple[float, float]:
        return self.__k


class LinearTrigonometricFunction:
    """ k_0 + sum_i _SineFunction_i(x) + sum_j _CosineFunction_j(x) """
    def __init__(self, k_0: float, sine_ks: tp.List[tp.Tuple[float, float]],
                 cosine_ks: tp.List[tp.Tuple[float, float]]):
        self.__k_0 = k_0
        sine_ks_d: tp.Dict[float, float] = defaultdict(float)
        for sine_k in sine_ks:
            if sine_k[1] < 0:
                sine_ks_d[-sine_k[1]] -= sine_k[0]
            else:
                sine_ks_d[sine_k[1]] += sine_k[0]
        self.__sine_ks = [(v, k) for k, v in sorted(sine_ks_d.items()) if v != 0]
        cosine_ks_d: tp.Dict[float, float] = defaultdict(float)
        for cosine_k in cosine_ks:
            if cosine_k[1] < 0:
                cosine_ks_d[-cosine_k[1]] += cosine_k[0]
            else:
                cosine_ks_d[cosine_k[1]] += cosine_k[0]
        self.__cosine_ks = [(v, k) for k, v in sorted(cosine_ks_d.items()) if v != 0]
        self.__sine_fs = [_SineFunction(sine_ks) for sine_ks in self.__sine_ks]
        self.__cosine_fs = [_CosineFunction(cosine_ks) for cosine_ks in self.__cosine_ks]

    def __call__(self, x: float) -> float:
        return self.__k_0 + np.sum(sine_function(x) for sine_function in self.__sine_fs) + \
                            np.sum(cosine_function(x) for cosine_function in self.__cosine_fs)

    def __repr__(self) -> str:
        if self.__k_0 != 0:
            return ' + '.join([str(self.__k_0), ] + [repr(sine_function) for sine_function in self.__sine_fs] +
                              [repr(cosine_function) for cosine_function in self.__cosine_fs]).replace(' + -', ' - ')
        else:
            return ' + '.join([repr(sine_function) for sine_function in self.__sine_fs] +
                              [repr(cosine_function) for cosine_function in self.__cosine_fs]).replace(' + -', ' - ')

    @property
    def k_0(self) -> float:
        return self.__k_0

    @property
    def sine_ks(self) -> tp.List[tp.Tuple[float, float]]:
        return self.__sine_ks

    @property
    def cosine_ks(self) -> tp.List[tp.Tuple[float, float]]:
        return self.__cosine_ks

    @property
    def sine_fs(self):
        """  :return: tp.List[_SineFunction] """
        return self.__sine_fs

    @property
    def cosine_fs(self):
        """  :return: tp.List[_CosineFunction] """
        return self.__cosine_fs

    def __add__(self, other_):
        """
        :param other_: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        other = _LinearTrigonometricFunctionFactory(other_)
        return LinearTrigonometricFunction(
            k_0=self.k_0 + other.k_0,
            sine_ks=self.__sine_ks + other.sine_ks,
            cosine_ks=self.__cosine_ks + other.cosine_ks
        )

    def __sub__(self, other_):
        """
        :param other_: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        other = _LinearTrigonometricFunctionFactory(other_)
        return LinearTrigonometricFunction(
            k_0=self.k_0 - other.k_0,
            sine_ks=self.__sine_ks + [(-sine_k[0], sine_k[1]) for sine_k in other.sine_ks],
            cosine_ks=self.__cosine_ks + [(-cosine_k[0], cosine_k[1]) for cosine_k in other.cosine_ks]
        )

    def __neg__(self):
        """ :return: LinearTrigonometricFunction """
        return LinearTrigonometricFunction(
            k_0=-self.k_0,
            sine_ks=[(-sine_k[0], sine_k[1]) for sine_k in self.sine_ks],
            cosine_ks=[(-cosine_k[0], cosine_k[1]) for cosine_k in self.cosine_ks]
        )

    def __mul__(self, other):
        """
        :param other_: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
        :return: LinearTrigonometricFunction
        """
        return _FunctionsProduct([self, other])


def _LinearTrigonometricFunctionFactory(
        function: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction]) -> LinearTrigonometricFunction:
    """ in case you (or I) need to construct LinearTrigonometricFunction from anther trigonometric function """
    if isinstance(function, _SineFunction):
        return LinearTrigonometricFunction(k_0=0, sine_ks=[function.k], cosine_ks=[])
    elif isinstance(function, _CosineFunction):
        return LinearTrigonometricFunction(k_0=0, cosine_ks=[function.k], sine_ks=[])
    else:
        return copy.deepcopy(function)


def DifferentialOperator(function_: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
                         order: int = 1) -> tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction]:
    """ f -> D^order f """
    function = copy.deepcopy(function_)
    for i in range(order):
        if isinstance(function, _SineFunction):
            function = _CosineFunction((function.k[0] * function.k[1], function.k[1]))
        elif isinstance(function, _CosineFunction):
            function = _SineFunction((-function.k[0] * function.k[1], function.k[1]))
        else:
            function = LinearTrigonometricFunction(
                k_0=0, cosine_ks=[(sine_k[0] * sine_k[1], sine_k[1]) for sine_k in function.sine_ks],
                sine_ks=[(-cosine_k[0] * cosine_k[1], cosine_k[1]) for cosine_k in function.cosine_ks]
            )
    return function


# TODO(nsk): add polynomials to LTF representation to create IntegralOperator
def Integrate(function_: tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction],
              a: float, b: float) -> float:
    """ f -> D^order f """
    function = copy.deepcopy(function_)
    if isinstance(function, _SineFunction):
        function = _CosineFunction((-function.k[0] / function.k[1], function.k[1]))
        return function(b) - function(a)
    elif isinstance(function, _CosineFunction):
        function = _SineFunction((function.k[0] / function.k[1], function.k[1]))
        return function(b) - function(a)
    else:
        function = LinearTrigonometricFunction(
            k_0=function.k_0, cosine_ks=[(-sine_k[0] / sine_k[1], sine_k[1]) for sine_k in function.sine_ks],
            sine_ks=[(cosine_k[0] / cosine_k[1], cosine_k[1]) for cosine_k in function.cosine_ks]
        )
        return function(b) - function(a) + (b - a) * function.k_0


def _FunctionsProduct(functions__: tp.List[tp.Union[_SineFunction, _CosineFunction, LinearTrigonometricFunction]]) \
        -> LinearTrigonometricFunction:
    """ LinearTrigonometricFunctions -> prod_k LinearTrigonometricFunction_k """
    functions_ = [_LinearTrigonometricFunctionFactory(function__) for function__ in functions__]
    function = functions_[0]
    for function_ in functions_[1:]:
        function = LinearTrigonometricFunction(
            k_0=np.sum(
                [function.k_0 * function_.k_0, ] +
                [cosine_f.k[0] * cosine_f_.k[0] / 2
                    for cosine_f in function.cosine_fs for cosine_f_ in function_.cosine_fs
                    if cosine_f.k[1] == cosine_f_.k[1]] +
                [cosine_f.k[0] * cosine_f_.k[0] / 2
                    for cosine_f in function.cosine_fs for cosine_f_ in function_.cosine_fs
                    if cosine_f.k[1] + cosine_f_.k[1] == 0] +
                [sine_f.k[0] * sine_f_.k[0] / 2
                    for sine_f in function.sine_fs for sine_f_ in function_.sine_fs
                    if sine_f.k[1] == sine_f_.k[1]] +
                [-sine_f.k[0] * sine_f_.k[0] / 2
                    for sine_f in function.sine_fs for sine_f_ in function_.sine_fs
                    if sine_f.k[1] + sine_f_.k[1] == 0]
            ),
            cosine_ks=list(
                [(cosine_f.k[0] * cosine_f_.k[0] / 2, cosine_f.k[1] - cosine_f_.k[1])
                    for cosine_f in function.cosine_fs for cosine_f_ in function_.cosine_fs
                    if cosine_f.k[1] != cosine_f_.k[1]] +
                [(cosine_f.k[0] * cosine_f_.k[0] / 2, cosine_f.k[1] + cosine_f_.k[1])
                    for cosine_f in function.cosine_fs for cosine_f_ in function_.cosine_fs
                    if cosine_f.k[1] + cosine_f_.k[1] != 0] +
                [(sine_f.k[0] * sine_f_.k[0] / 2, sine_f.k[1] - sine_f_.k[1])
                    for sine_f in function.sine_fs for sine_f_ in function_.sine_fs
                    if sine_f.k[1] != sine_f_.k[1]] +
                [(-sine_f.k[0] * sine_f_.k[0] / 2, sine_f.k[1] + sine_f_.k[1])
                    for sine_f in function.sine_fs for sine_f_ in function_.sine_fs
                    if sine_f.k[1] + sine_f_.k[1] != 0] +
                [(function.k_0 * cosine_f_.k[0], cosine_f_.k[1])
                    for cosine_f_ in function_.cosine_fs
                    if function.k_0 != 0] +
                [(function_.k_0 * cosine_f.k[0], cosine_f.k[1])
                    for cosine_f in function.cosine_fs
                    if function_.k_0 != 0]
            ),
            sine_ks=list(
                [(sine_f.k[0] * cosine_f_.k[0] / 2, sine_f.k[1] + cosine_f_.k[1])
                    for sine_f in function.sine_fs for cosine_f_ in function_.cosine_fs
                    if sine_f.k[1] + cosine_f_.k[1] != 0] +
                [(sine_f.k[0] * cosine_f_.k[0] / 2, sine_f.k[1] - cosine_f_.k[1])
                    for sine_f in function.sine_fs for cosine_f_ in function_.cosine_fs
                    if sine_f.k[1] != cosine_f_.k[1]] +
                [(cosine_f.k[0] * sine_f_.k[0] / 2, cosine_f.k[1] + sine_f_.k[1])
                    for cosine_f in function.cosine_fs for sine_f_ in function_.sine_fs
                    if cosine_f.k[1] + sine_f_.k[1] != 0] +
                [(-cosine_f.k[0] * sine_f_.k[0] / 2, cosine_f.k[1] - sine_f_.k[1])
                    for cosine_f in function.cosine_fs for sine_f_ in function_.sine_fs
                    if cosine_f.k[1] != sine_f_.k[1]] +
                [(function.k_0 * sine_f_.k[0], sine_f_.k[1])
                    for sine_f_ in function_.sine_fs
                    if function.k_0 != 0] +
                [(function_.k_0 * sine_f.k[0], sine_f.k[1])
                    for sine_f in function.sine_fs
                    if function_.k_0 != 0]
            )
        )
    return function


class TestLTF(unittest.TestCase):
    def setUp(self):
        self.s = _SineFunction(k=(1, 1))
        self.c = _CosineFunction(k=(1, 1))
        self.ltf = LinearTrigonometricFunction(k_0=1, sine_ks=[(1, 2), (2, 1)], cosine_ks=[(3, -1), (-1, 3)])

    def test_add(self):
        for f1 in [self.s, self.c, self.ltf]:
            for f2 in [self.s, self.c, self.ltf]:
                with self.subTest(f1=f1, f2=f2):
                    for x in np.arange(0, 1.01, .01):
                        self.assertAlmostEqual(f1(x) + f2(x), (f1 + f2)(x), msg="Addition is broken")

    def test_sub(self):
        for f1 in [self.s, self.c, self.ltf]:
            for f2 in [self.s, self.c, self.ltf]:
                with self.subTest(f1=f1, f2=f2):
                    for x in np.arange(0, 1.01, .01):
                        self.assertAlmostEqual(f1(x) - f2(x), (f1 - f2)(x), msg="Substraction is broken")

    def test_neg(self):
        for f in [self.s, self.c, self.ltf]:
            with self.subTest(f=f):
                for x in np.arange(0, 1.01, .01):
                    self.assertAlmostEqual(-f(x), (-f)(x), msg="Negation is broken")

    def test_mul(self):
        for f1 in [self.s, self.c, self.ltf]:
            for f2 in [self.s, self.c, self.ltf]:
                with self.subTest(f1=f1, f2=f2):
                    for x in np.arange(0, 1.01, .01):
                        self.assertAlmostEqual(f1(x) * f2(x), (f1 * f2)(x), msg="Multiplication is broken")

    def test_call(self):
        with self.subTest(f=self.s):
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(self.s(x), np.sin(x), msg="Call is broken")
        with self.subTest(f=self.c):
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(self.c(x), np.cos(x), msg="Call is broken")
        with self.subTest(f=self.ltf):
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(
                    self.ltf(x),
                    1 + np.sin(2*x) + 2*np.sin(x) + 3*np.cos(-x) - np.cos(3*x),
                    msg="Call is broken"
                )

    def test_repr(self):
        with self.subTest(f=self.s):
            self.assertEqual(repr(self.s), "sin(x)", msg="Repr is broken")
        with self.subTest(f=self.c):
            self.assertEqual(repr(self.c), "cos(x)", msg="Repr is broken")
        with self.subTest(f=self.ltf):
            self.assertEqual(repr(self.ltf), "1 + 2.0 sin(x) + sin(2 x) + 3.0 cos(x) - cos(3 x)", msg="Repr is broken")

    def test_derivative(self):
        with self.subTest(f=self.s):
            ds = DifferentialOperator(self.s)
            self.assertEqual(repr(ds), "cos(x)", msg="Derivative is broken")
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(ds(x), np.cos(x), msg="Call derivative is broken")
            d2s = DifferentialOperator(self.s, 2)
            self.assertEqual(repr(d2s), "-sin(x)", msg="2nd derivative is broken")
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(d2s(x), -np.sin(x), msg="Call 2nd derivative is broken")
        with self.subTest(f=self.c):
            dc = DifferentialOperator(self.c)
            self.assertEqual(repr(dc), "-sin(x)", msg="Derivative is broken")
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(dc(x), -np.sin(x), msg="Call derivative is broken")
            d2c = DifferentialOperator(self.c, 2)
            self.assertEqual(repr(d2c), "-cos(x)", msg="2nd derivative is broken")
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(d2c(x), -np.cos(x), msg="Call 2nd derivative is broken")
        with self.subTest(f=self.ltf):
            dltf = DifferentialOperator(self.ltf)
            self.assertEqual(
                repr(dltf),
                "-3.0 sin(x) + 3.0 sin(3 x) + 2.0 cos(x) + 2.0 cos(2 x)",
                msg="Derivative is broken"
            )
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(
                    dltf(x),
                    3*np.sin(-x) + 3*np.sin(3*x) + 2*np.cos(2*x) + 2*np.cos(x),
                    msg="Call derivative is broken"
                )
            d2ltf = DifferentialOperator(self.ltf, 2)
            self.assertEqual(
                repr(d2ltf),
                "-2.0 sin(x) - 4.0 sin(2 x) - 3.0 cos(x) + 9.0 cos(3 x)",
                msg="2nd derivative is broken"
            )
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(
                    d2ltf(x),
                    -4*np.sin(2*x) - 2*np.sin(x) - 3*np.cos(-x) + 9*np.cos(3*x),
                    msg="Call 2nd derivative is broken"
                )

    def test_integrate(self):
        with self.subTest(f=self.s):
            ds = DifferentialOperator(self.s)
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(Integrate(ds, 0, x), self.s(x) - self.s(0), msg="Integrate is broken")
            d2s = DifferentialOperator(self.s, 2)
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(Integrate(d2s, 0, x), ds(x) - ds(0), msg="Integrate is broken")
        with self.subTest(f=self.c):
            dc = DifferentialOperator(self.c)
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(Integrate(dc, 0, x), self.c(x) - self.c(0), msg="Integrate is broken")
            d2c = DifferentialOperator(self.c, 2)
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(Integrate(d2c, 0, x), dc(x) - dc(0), msg="Integrate is broken")
        with self.subTest(f=self.ltf):
            dltf = DifferentialOperator(self.ltf)
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(Integrate(dltf, 0, x), self.ltf(x) - self.ltf(0), msg="Integrate is broken")
            d2ltf = DifferentialOperator(self.ltf, 2)
            for x in np.arange(0, 1.01, .01):
                self.assertAlmostEqual(Integrate(d2ltf, 0, x), dltf(x) - dltf(0), msg="Integrate is broken")


if __name__ == "__main__":
    unittest.main()
