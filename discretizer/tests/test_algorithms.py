from __future__ import print_function, division

import sympy
import discretizer
from discretizer.algorithms import read_coordinates
from discretizer.algorithms import split_factors
from discretizer.algorithms import wavefunction_name
from discretizer.algorithms import derivate
from discretizer.algorithms import _discretize_summand

from nose.tools import raises
from nose.tools import assert_raises
import numpy as np

kx, ky, kz = sympy.symbols('k_x k_y k_z', commutative=False)
x, y, z = sympy.symbols('x y z', commutative=False)
ax, ay, az = sympy.symbols('a_x a_y a_z')

wf =  sympy.Function(wavefunction_name)
Psi = sympy.Function(wavefunction_name)(x, y, z)
A, B = sympy.symbols('A B', commutative=False)

ns = {'A': A, 'B': B, 'a_x': ax, 'a_y': ay, 'az': az, 'x': x, 'y': y, 'z': z}

def test_read_coordinates():
    test = {
        kx**2                         : {'x'},
        kx**2 + ky**2                 : {'x', 'y'},
        kx**2 + ky**2 + kz**2         : {'x', 'y', 'z'},
        ky**2 + kz**2                 : {'y', 'z'},
        kz**2                         : {'z'},
        kx * A(x,y) * kx              : {'x', 'y'},
        kx**2 + kz * B(y)             : {'x', 'y', 'z'},
    }
    for inp, out in test.items():
        got = read_coordinates(inp)
        assert got == out,\
            "Should be: split_factors({})=={}. Not {}".format(inp, out, got)            

def test_split_factors_1():
    test = {
        kz * Psi                      : (1, kz, Psi),
        A * kx**2 * Psi               : (A * kx, kx, Psi),
        A * kx**2 * ky * Psi          : (A * kx**2, ky, Psi),
        ky * A * kx * B * Psi         : (ky * A, kx, B * Psi),
        kx                            : (1, kx, 1),
        kx**2                         : (kx, kx, 1),
        A                             : (1, 1, A),
        A**2                          : (1, 1, A**2),
        kx*A**2                       : (1, kx, A**2),
        kx**2*A**2                    : (kx, kx, A**2),
        A(x, y, z)                    : (1, 1, A(x, y, z)),
        Psi                           : (1, 1, Psi),
        np.int(5)                     : (1, 1, np.int(5)),
        np.float(5)                   : (1, 1, np.float(5)),
        sympy.Integer(5)              : (1, 1, sympy.Integer(5)),
        sympy.Float(5)                : (1, 1, sympy.Float(5)),
        1                             : (1, 1, 1),
        1.0                           : (1, 1, 1.0),
        5                             : (1, 1, 5),
        5.0                           : (1, 1, 5.0),
    }

    for inp, out in test.items():
        got = split_factors(inp, discrete_coordinates={'x', 'y', 'z'})
        assert  got == out,\
            "Should be: split_factors({})=={}. Not {}".format(inp, out, got)


@raises(AssertionError)
def test_split_factors_2():
    split_factors(A+B, discrete_coordinates={'x', 'y', 'z'})


def test_derivate_1():
    test = {
        (A(x), kx): '-I*(-A(-a_x + x)/(2*a_x) + A(a_x + x)/(2*a_x))',
        (A(x), ky): '0',
        (A(x)*B, kx): '-I*(-A(-a_x + x)*B/(2*a_x) + A(a_x + x)*B/(2*a_x))',
        (A(x) + B(x), kx): '-I*(-A(-a_x + x)/(2*a_x) + A(a_x + x)/(2*a_x) - B(-a_x + x)/(2*a_x) + B(a_x + x)/(2*a_x))',
        (A, kx): '0',
        (5, kx): '0',
        (A(x) * B(x), kx): '-I*(-A(-a_x + x)*B(-a_x + x)/(2*a_x) + A(a_x + x)*B(a_x + x)/(2*a_x))',
        (Psi, ky): '-I*(-Psi(x, -a_y + y, z)/(2*a_y) + Psi(x, a_y + y, z)/(2*a_y))',
    }

    for inp, out in test.items():
        got = (derivate(*inp))
        out = sympy.sympify(out, locals=ns)
        assert  sympy.simplify(sympy.expand(got - out)) == 0,\
            "Should be: derivate({0[0]}, {0[1]})=={1}. Not {2}".format(inp, out, got)


@raises(TypeError)
def test_derivate_2():
    derivate(A(x), kx**2)


@raises(ValueError)
def test_derivate_3():
    derivate(A(x), sympy.Symbol('A'))


def test_discretize_summand_1():
    test = {
        kx * A(x): '-I*(-A(-a_x + x)/(2*a_x) + A(a_x + x)/(2*a_x))',
        kx * Psi: '-I*(-Psi(-a_x + x, y, z)/(2*a_x) + Psi(a_x + x, y, z)/(2*a_x))',
        kx**2 * Psi: 'Psi(x, y, z)/(2*a_x**2) - Psi(-2*a_x + x, y, z)/(4*a_x**2) - Psi(2*a_x + x, y, z)/(4*a_x**2)',
        kx * A(x) * kx * Psi: 'A(-a_x + x)*Psi(x, y, z)/(4*a_x**2) - A(-a_x + x)*Psi(-2*a_x + x, y, z)/(4*a_x**2) + A(a_x + x)*Psi(x, y, z)/(4*a_x**2) - A(a_x + x)*Psi(2*a_x + x, y, z)/(4*a_x**2)',
    }

    for inp, out in test.items():
        got = _discretize_summand(inp, discrete_coordinates={'x', 'y', 'z'})
        out = sympy.sympify(out, locals=ns)
        assert  sympy.simplify(sympy.expand(got - out)) == 0,\
            "Should be: _discretize_summand({})=={}. Not {}".format(inp, out, got)

@raises(AssertionError)
def test_discretize_summand_2():
    _discretize_summand(kx*A(x)+ B(x), discrete_coordinates={'x', 'y', 'z'})
