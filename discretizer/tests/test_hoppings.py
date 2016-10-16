from __future__ import print_function, division

import sympy
import discretizer
from discretizer.algorithms import wavefunction_name
from discretizer.algorithms import read_hopping_from_wf
from discretizer.algorithms import extract_hoppings

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


def test_read_hoppings_from_wf_1():
    offsets = [(0,0,0), (1,0,0), (0,1,0), (0,0,1), (1,1,1), (1,2,3)]
    test = {}

    for offset in offsets:
        nx, ny, nz = offset
        key = Psi.subs({x: x + nx*ax, y: y + ny * ay, z: z + nz * az})
        test[key] = offset

    for inp, out in test.items():
        got = read_hopping_from_wf(inp)
        assert got == out,\
            "Should be: read_hopping_from_wf({}) == {}. Not {}".format(inp, out, got)


def test_read_hoppings_from_wf_2():
    test = {
        wf(x, y, z): (0,0,0),
        wf(x, y): (0, 0),
        wf(x, z): (0, 0),
        wf(x+ax, y-2*ay): (1, -2),
        wf(x, z+3*az): (0, 3),
        wf(y, z): (0, 0),
        wf(y, z+az): (0, 1),
    }

    for inp, out in test.items():
        got = read_hopping_from_wf(inp)
        assert got == out,\
            "Should be: read_hopping_from_wf({}) == {}. Not {}".format(inp, out, got)


def test_test_read_hoppings_from_wf_ValueError():
    tests = {
        wf(x+ay, ay),
        wf(y, x),
        wf(x, x),
        wf(x, ax),
        wf(y, A),
    }
    for inp in tests:
        assert_raises(ValueError, read_hopping_from_wf, inp)


def test_test_read_hoppings_from_wf_TypeError():
    tests = {
        wf(x,y,z) + A,
        A(x,y),
        5*Psi,
        Psi+2,
        A,
    }
    for inp in tests:
        assert_raises(TypeError, read_hopping_from_wf, inp)



def test_extract_hoppings():
    discrete_coordinates = {'x', 'y', 'z'}
    tests = [
        {
            'test_inp': '-I*(-Psi(-a_x + x, y)/(2*a_x) + Psi(a_x + x, y)/(2*a_x))',
            'test_out': {
                (1, 0): '-I/(2*a)',
                (-1, 0): 'I/(2*a)',
            },
        },
        {
            'test_inp': 'Psi(x, y)/(2*a_x**2) - Psi(-2*a_x + x, y)/(4*a_x**2) - Psi(2*a_x + x, y)/(4*a_x**2)',
            'test_out': {
                (1, 0): '-1/a**2',
                (0, 0): '2/a**2',
                (-1, 0): '-1/a**2',
            },
        },
        {
            'test_inp': 'A(-a_x + x)*Psi(x, y)/(4*a_x**2) - A(-a_x + x)*Psi(-2*a_x + x, y)/(4*a_x**2) + A(a_x + x)*Psi(x, y)/(4*a_x**2) - A(a_x + x)*Psi(2*a_x + x, y)/(4*a_x**2)',
            'test_out': {
                (1, 0): '-A(a/2 + x)/a**2',
                (0, 0): 'A(-a/2 + x)/a**2 + A(a/2 + x)/a**2',
                (-1, 0): '-A(-a/2 + x)/a**2',
            },
        },
        {
            'test_inp': 'I*A(x, y)*Psi(x, -a_y + y, z)/(4*a_x**2*a_y) - I*A(x, y)*Psi(x, a_y + y, z)/(4*a_x**2*a_y) - I*A(-2*a_x + x, y)*Psi(-2*a_x + x, -a_y + y, z)/(8*a_x**2*a_y) + I*A(-2*a_x + x, y)*Psi(-2*a_x + x, a_y + y, z)/(8*a_x**2*a_y) - I*A(2*a_x + x, y)*Psi(2*a_x + x, -a_y + y, z)/(8*a_x**2*a_y) + I*A(2*a_x + x, y)*Psi(2*a_x + x, a_y + y, z)/(8*a_x**2*a_y)',
            'test_out': {
                (-1, 1, 0): 'I*A(-a + x, y)/(2*a**3)',
                (1, 1, 0): 'I*A(a + x, y)/(2*a**3)',
                (0, -1, 0): 'I*A(x, y)/a**3',
                (0, 1, 0): '-I*A(x, y)/a**3',
                (1, -1, 0): '-I*A(a + x, y)/(2*a**3)',
                (-1, -1, 0): '-I*A(-a + x, y)/(2*a**3)',
            },
        },
    ]

    for test in tests:
        test_inp = test['test_inp']
        test_out = test['test_out']
        inp = sympy.sympify(test_inp, locals=ns)
        result = extract_hoppings(inp, discrete_coordinates)
        for key, got in result.items():
            out = sympy.sympify(test_out[key], locals=ns)
            assert sympy.simplify(sympy.expand(got - out)) == 0, \
                "Should be: extract_hoppings({})=={}. Not {}".format(inp, test_out, result)
