# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import pytest
import tinyarray as ta

from sympy.physics.matrices import msigma
from sympy.physics.quantum import TensorProduct
import sympy

from kwant.continuum._common import position_operators, momentum_operators
from kwant.continuum._common import make_commutative, sympify
from kwant.continuum._common import monomials
from kwant.continuum._common import lambdify


com_A, com_B, com_C = sympy.symbols('A B C')
fA, fB, fC = sympy.symbols('A B C', cls=sympy.Function)
x_op, y_op, z_op = position_operators
kx, ky, kz = momentum_operators


@pytest.mark.parametrize('input_expr, output_expr', [
    ('k_x * A(x) * k_x', kx * fA(x_op) * kx),
    ('[[k_x * A(x) * k_x]]', sympy.Matrix([kx * fA(x_op) * kx])),
    ('k_x * sigma_y + k_y * sigma_x', kx * msigma(2) + ky * msigma(1)),
    ('[[k_x*A(x)*k_x, B(x, y)*k_x], [k_x*B(x, y), C*k_y**2]]',
     sympy.Matrix([[kx*fA(x_op)*kx, fB(x_op, y_op)*kx],
                   [kx*fB(x_op, y_op), com_C*ky**2]])),
    ('kron(sigma_x, sigma_y)', TensorProduct(msigma(1), msigma(2))),
    ('identity(2)', sympy.eye(2)),
    ('eye(2)', sympy.eye(2)),
    ('1 * sigma_x + 2 * sigma_y + 3 * sigma_z',
     msigma(1) + 2 * msigma(2) + 3 * msigma(3))
])
def test_sympify(input_expr, output_expr):
    assert sympify(input_expr) == output_expr
    assert sympify(sympify(input_expr)) == output_expr


@pytest.mark.parametrize('input_expr, output_expr, subs', [
    ('k_x', kx + ky, {'k_x': 'k_x + k_y'}),
    ('x', x_op + y_op, {'x': 'x + y'}),
    ('A', com_A + com_B, {'A': 'A + B'}),
    ('A', com_A + fB(x_op), {'A': 'A + B(x)'}),
    ('A', msigma(2), {'A': "[[0, -1j], [1j, 0]]"}),
])
def test_sympify_substitutions(input_expr, output_expr, subs):
    assert sympify(input_expr, locals=subs) == output_expr

    subs = {k: sympify(v) for k, v in subs.items()}
    assert sympify(input_expr, locals=subs) == output_expr


@pytest.mark.parametrize('subs', [
    {1: 2},
    {'1': 2},
    {'for': 33},
    {'x': 1, '1j': 7}
])
def test_sympify_invalid_substitutions(subs):
    with pytest.raises(ValueError):
        sympify('x + y', locals=subs)


@pytest.mark.parametrize('input_expr, output_expr, subs', [
    ('A + k_x**2 * eye(2)',
        kx**2 * sympy.eye(2) + msigma(2),
        {'A': "[[0, -1j], [1j, 0]]"}),
    ('A + k_x**2 * identity(2)',
        kx**2 * sympy.eye(2) + msigma(2),
        {'A': "[[0, -1j], [1j, 0]]"})

])
def test_sympify_mix_symbol_and_matrx(input_expr, output_expr, subs):
    assert sympify(input_expr, locals=subs) == output_expr

    subs = {k: sympify(v) for k, v in subs.items()}
    assert sympify(input_expr, locals=subs) == output_expr


A, B, x = sympy.symbols('A B x', commutative=False)
com_x, com_y = sympy.symbols('x y')

expr1 = x*A*x + x**2 * A * x + B*x**2

matr_com = sympy.Matrix([[expr1, expr1+A*x], [0, -expr1]])
res_mat = sympy.Matrix([[com_x**3*A + com_x**2*A + com_x**2*B, com_x**3*A + com_x**2*A + com_x**2*B + com_x*A],
                        [0, -com_x**3*A - com_x**2*A - com_x**2*B]])


def test_make_commutative():
    assert make_commutative(expr1, com_x).expand() == (make_commutative(expr1, x)).expand()
    assert make_commutative(expr1, com_x).expand() == (com_x**3*A + com_x**2*A + com_x**2*B).expand()
    assert make_commutative(matr_com, com_x).expand() == (res_mat).expand()


matr_monomials = sympify("[[x+y, a*x**2 + b*y], [y, x]]")
x, y, z = position_operators
a, b = sympy.symbols('a, b')
fA, fB = sympy.symbols('A B', cls=sympy.Function)

@pytest.mark.parametrize('expr, gens, output', [
    (x * fA(x) * x + x**2 * a, None, {x**2: fA(x), a*x**2: 1}),
    (x * fA(x) * x + x**2 * a, [x], {x**2: fA(x) + a}),
    (x**2, [x], {x**2: 1}),
    (2 * x + 3 * x**2, [x], {x: 2, x**2: 3}),
    (2 * x + 3 * x**2, 'x', {x: 2, x**2: 3}),
    (a * x**2 + 2 * b * x**2, 'x', {x**2: a + 2 * b}),
    (x**2 * (a + 2 * b) , 'x', {x**2: a + 2 * b}),
    (2 * x * y  + 3 * y * x, 'xy', {x*y: 2, y*x: 3}),
    (2 * x * a + 3 * b, 'ab', {a: 2*x, b: 3}),
    (matr_monomials, None, {
        x: sympy.Matrix([[1, 0], [0, 1]]),
        b*y: sympy.Matrix([[0, 1], [0, 0]]),
        a*x**2: sympy.Matrix([[0, 1], [0, 0]]),
        y: sympy.Matrix([[1, 0], [1, 0]])
    }),
    (matr_monomials, [x], {
        x: sympy.Matrix([[1, 0], [0, 1]]),
        1: sympy.Matrix([[y, b*y], [y, 0]]),
        x**2: sympy.Matrix([[0, a], [0, 0]])
    }),
    (matr_monomials, [x, y], {
        x: sympy.Matrix([[1, 0], [0, 1]]),
        x**2: sympy.Matrix([[0, a], [0, 0]]),
        y: sympy.Matrix([[1, b], [1, 0]])
    }),
])
def test_monomials(expr, gens, output):
    assert monomials(expr, gens) == output



@pytest.mark.parametrize("e, should_be, kwargs", [
    ("x+y", lambda x, y: x+y, dict(x=1, y=2)),
    ("1", lambda: 1, dict()),
    ("f(x)", lambda f, x: f(x), dict(f=lambda x: x, x=2)),
    (sympify("f(x)"), lambda f, x: f(x), dict(f=lambda x: x, x=2)),
    ("[[f(x)]]", lambda f, x: ta.array(f(x)), dict(f=lambda x: x, x=2)),
])
def test_lambdify(e, should_be, kwargs):
    e = lambdify(e)
    assert e(**kwargs) == should_be(**kwargs)


@pytest.mark.parametrize("e, kwargs", [
    ("x + y", dict(x=1, y=3, z=5)),
    (sympify("x+y+z"), dict(x=1, y=3, z=5)),
])
def test_lambdify_substitutions(e, kwargs):
    should_be = lambda x, y, z: x + y + z
    subs = {'y': 'y + z'}

    if not isinstance(e, str):
        # warns that 'locals' are not used if 'e' is
        # a sympy expression already
        with pytest.warns(RuntimeWarning):
            e = lambdify(e, locals=subs)
    else:
        e = lambdify(e, locals=subs)

    assert e(**kwargs) == should_be(**kwargs)
