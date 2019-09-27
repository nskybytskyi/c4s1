#!/usr/bin/env python
import copy
import unittest
from collections import defaultdict
import math
import typing as tp


class _PrimitiveFunction:
    def __init__(self, k:   tp.Tuple[float, float]):
        self.__k = k

    def __add__(self, other):
        """ 
        :param other: tp.Union[_PrimitiveFunction, Function]
        :return: Function
        """
        return Function([self]) + other

    def __sub__(self, other):
        """ 
        :param other: tp.Union[_PrimitiveFunction, Function]
        :return: Function
        """
        return Function([self]) - other

    def __neg__(self):
        """ :return: _PrimitiveFunction """
        return _PrimitiveFunction(k=(-self.__k[0], self.__k[1]))

    def __mul__(self, other):
        return Function([self]) * other

    @property
    def k(self) -> tp.Tuple[float, float]:
        return self.__k


class Function:
    def __init__(self, primitive_functions: tp.List[primitiveFunction]):
        self.__primitive_functions = primitive_functions

    def __call__(self, x: float) -> float:
        return sum([])

    def __repr__(self) -> str:
        return f"{self.__k[0]} x^{self.__k[1]}"

    def __add__(self, other):
        return Function(self) + other

    def __sub__(self, other):
        return SimpleFunction(self) - other

    def __neg__(self):
        return _MonomialFunction(k=(-self.__k[0], self.__k[1]))

    def __mul__(self, other):
        return SimpleFunction(self) * other

    @property
    def k(self) -> tp.Tuple[float, float]:
        return self.__k

        
class _SinFunction(_PrimitiveFunction):
    """ k_0 sin(k_1 x) """
    def __init__(self, k: tp.Tuple[float, float]):
        super(_SineFunction, self).__init__()

    def __call__(self, x: float) -> float:
        return self.__k[0] * math.sin(self.__k[1] * x)

    def __repr__(self) -> str:
        return f"{self.__k[0]} sin({self.__k[1]} x)"


class _CosFunction(_PrimitiveFunction):
    """ k_0 cos(k_1 x) """
    def __init__(self, k: tp.Tuple[float, float]):
        super(_CosFunction, self).__init__()

    def __call__(self, x: float) -> float:
        return self.__k[0] * math.cos(self.__k[1] * x)

    def __repr__(self) -> str:
        return f"{self.__k[0]} cos({self.__k[1]} x)"


class _PowFunction(_PrimitiveFunction):
    """ k_0 x^{k_1} """
    def __init__(self, k: tp.Tuple[float, float]):
        super(_PowFunction, self).__init__()

    def __call__(self, x: float) -> float:
        return self.__k[0] * math.pow(x, self.__k[1])

    def __repr__(self) -> str:
        return f"{self.__k[0]} x^{self.__k[1]}"


class _ExpFunction(_PrimitiveFunction):
    """ k_0 e^{k_1 x} """
    def __init__(self, k: tp.Tuple[float, float]):
        super(_ExpFunction, self).__init__()

    def __call__(self, x: float) -> float:
        return self.__k[0] * math.exp(self.__k[1] * x)

    def __repr__(self) -> str:
        return f"{self.__k[0]} e^({self.__k[1]} x)"


class _LogFunction(_PrimitiveFunction):
    """ k_0 e^{k_1 x} """
    def __init__(self, k: 0tp.Tuple[float, float]):
        super(_LogFunction, self).__init__()

    def __call__(self, x: float) -> float:
        return self.__k[0] * math.log(self.__k[1] * x)

    def __repr__(self) -> str:
        return f"{self.__k[0]} ln({self.__k[1]} x)"


if __name__ == "__main__":
    unittest.main()
