# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from functools import reduce
from operator import mul

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
x_op, y_op, z_op = position_operators
kx, ky, kz = momentum_operators


@pytest.mark.parametrize('input_expr, output_expr', [
    ('k_x * A(x) * k_x', kx * com_A(x_op) * kx),
    ('[[k_x * A(x) * k_x]]', sympy.Matrix([kx * com_A(x_op) * kx])),
    ('k_x * sigma_y + k_y * sigma_x', kx * msigma(2) + ky * msigma(1)),
    ('[[k_x*A(x)*k_x, B(x, y)*k_x], [k_x*B(x, y), C*k_y**2]]',
     sympy.Matrix([[kx*com_A(x_op)*kx, com_B(x_op, y_op)*kx],
                   [kx*com_B(x_op, y_op), com_C*ky**2]])),
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
    ('A', com_A + com_B(x_op), {'A': 'A + B(x)'}),
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


A, B, non_x = sympy.symbols('A B x', commutative=False)
x, y = sympy.symbols('x y')

expr1 = non_x*A*non_x + x**2 * A * x + B*non_x**2

matr = sympy.Matrix([[expr1, expr1+A*non_x], [0, -expr1]])
res_mat = sympy.Matrix([[x**3*A + x**2*A + x**2*B, x**3*A + x**2*A + x**2*B + x*A],
                        [0, -x**3*A - x**2*A - x**2*B]])


def test_make_commutative():
    assert make_commutative(expr1, x) == make_commutative(expr1, non_x)
    assert make_commutative(expr1, x) == x**3*A + x**2*A + x**2*B
    assert make_commutative(matr, x) == res_mat


expr2 = non_x*A*non_x + x**2 * A*2 * x + B*non_x/2 + non_x*B/2 + x + A + non_x + x/A


def test_monomials():
    f, g, a, b = sympy.symbols('f g a b')

    assert monomials(expr2, gens='x') == {x**3: 2*A, 1: A, x: 2 + A**(-1) + B, x**2: A}
    assert monomials(expr1, gens='x') == {x**2: A + B, x**3: A}
    assert monomials(x, gens='x') == {x: 1}
    assert monomials(x**2, gens='x') == {x**2: 1}
    assert monomials(x**2 + x, gens='x') == {x: 1, x**2: 1}
    assert monomials(x**2 + x + A**2, gens='x') == {x: 1, x**2: 1, 1: A**2}
    assert monomials(x * f(a, b), gens='x') == {x: f(a, b)}

    expr = x * f(a) + y * g(b)
    out = {y: g(b), x: f(a)}
    assert monomials(expr, gens=('x', 'y')) == out

    expr = 1 + x + A*x + 2*x + x**2 + A*x**2 + non_x*A*non_x
    out = {1: 1, x: 3 + A, x**2: 2 * A + 1}
    assert monomials(expr, gens='x') == out

    expr = 1 + x * (3 + A) + x**2 * (1 + A)
    out = {1: 1, x: 3 + A, x**2: 1 * A + 1}
    assert monomials(expr, gens='x') == out

    with pytest.raises(ValueError):
        monomials(f(x), gens=[x])

    with pytest.raises(ValueError):
        monomials(f(a), gens='a')




def test_matrix_monomials():
    out = {
        x**2: sympy.Matrix([[A + B,  A + B],[0, -A - B]]),
        x: sympy.Matrix([[0, A], [0, 0]]),
        x**3: sympy.Matrix([[A,  A], [0, -A]]),
    }
    mons = monomials(matr, gens=[x])
    assert mons == out


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
