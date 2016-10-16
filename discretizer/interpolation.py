from __future__ import print_function, division

import itertools
import numpy as np
import sympy


def _follow_path(expr, path):
    res = expr
    for i in np.arange(len(path)):
        res = res.args[path[i]]
    return res

def _interchange(expr, sub, path):
    res = sub
    for i in np.arange(len(path)):
        temp = _follow_path(expr, path[:-(i+1)])
        args = list(temp.args)
        args[path[len(path)-i-1]] = res
        res = temp.func(*tuple(args))
    return res

def _interpolate_Function(expr):
    path = 'None'
    factor = 'None'
    change = False
    summand_0 = 'None'
    summand_1 = 'None'
    res = expr
    for i in np.arange(len(expr.args)):
        argument = sympy.expand(expr.args[i])
        if argument.func == sympy.Add:
            for j in np.arange(len(argument.args)):
                summand = argument.args[j]
                if summand.func == sympy.Mul:
                    for k in np.arange(len(summand.args)):
                        temp = 0
                        if summand.args[k] == sympy.Symbol('a'):
                            temp = sympy.Mul(sympy.Mul(*summand.args[:k]),
                                             sympy.Mul(*summand.args[k+1:]))
                            #print(temp)
                        if not temp == int(temp):
                            #print('found one')
                            factor = (temp)
                            path = np.array([i, j, k])
    if not factor == 'None':
        change = True

        sign = np.sign(factor)
        offsets = np.array([int(factor), sign * (int(sign * factor) + 1)])
        weights = 1/np.abs(offsets - factor)
        weights = weights/np.sum(weights)

        res = (  weights[0] * _interchange(expr, offsets[0] * sympy.Symbol('a'), path[:-1])
               + weights[1] * _interchange(expr, offsets[1] * sympy.Symbol('a'), path[:-1]))

    return sympy.expand(res), change


def _interpolate_expression(expr):
    change = False
    expr = sympy.expand(expr)
    res = expr

    if isinstance(expr, sympy.Function):# and not change:
        path = np.array([])
        temp, change = _interpolate_Function(_follow_path(expr, path))
        res = _interchange(expr, temp, path)

    for i in np.arange(len(expr.args)):
        path = np.array([i])
        if isinstance(_follow_path(expr, path), sympy.Function) and not change:
            temp, change = _interpolate_Function(_follow_path(expr, path))
            res = _interchange(expr, temp, path)

        for j in np.arange(len(expr.args[i].args)):
            path = np.array([i, j])
            if isinstance(_follow_path(expr, path), sympy.Function) and not change:
                temp, change = _interpolate_Function(_follow_path(expr, path))
                res = _interchange(expr, temp, path)

    if change:
        res = _interpolate_expression(res)

    return sympy.expand(res)


def _interpolate(expression):
    if not isinstance(expression, sympy.Matrix):
        return _interpolate_expression(expression)

    shape = expression.shape
    interpolated = sympy.zeros(*shape)
    for i,j in itertools.product(range(shape[0]), repeat=2):
        interpolated[i,j] = _interpolate_expression(expression[i, j])

    return interpolated


def interpolate_tb_hamiltonian(tb_hamiltonian):
    """Interpolate tight binding hamiltonian.

    This function perform linear interpolation to provide onsite and hoppings
    depending only on parameters values at sites positions.
    """
    interpolated = {}
    for key, val in tb_hamiltonian.items():
        interpolated[key] = _interpolate(val)
    return interpolated
