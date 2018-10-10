# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import inspect
import warnings
from functools import wraps

import numpy as np
import pytest

import sympy

from ..discretizer import discretize
from ..discretizer import discretize_symbolic
from ..discretizer import build_discretized
from ..discretizer import  _wf

from ...lattice import Monatomic


def swallows_extra_kwargs(f):
    sig = inspect.signature(f)
    pars = sig.parameters
    if any(i.kind is inspect.Parameter.VAR_KEYWORD for i in pars.values()):
        return f

    names = {name for name, value in pars.items() if
             value.kind not in (inspect.Parameter.VAR_POSITIONAL,
                                inspect.Parameter.POSITIONAL_ONLY)}

    @wraps(f)
    def wrapped(*args, **kwargs):
        bound = sig.bind(*args, **{name: value for name, value
                                   in kwargs.items() if name in names})
        return f(*bound.args, **bound.kwargs)

    return wrapped


I = sympy.I
kx, ky, kz = sympy.symbols('k_x k_y k_z', commutative=False)
x, y, z = sympy.symbols('x y z', commutative=False)
ax, ay, az = sympy.symbols('a_x a_y a_z')
a = sympy.symbols('a')

wf = _wf
Psi = wf(x, y, z)
A, B = sympy.symbols('A B', commutative=False)
fA, fB = sympy.symbols('A B', cls=sympy.Function)

ns = {'A': A, 'B': B, 'a_x': ax, 'a_y': ay, 'az': az, 'x': x, 'y': y, 'z': z}


@pytest.mark.parametrize('commutative', [True, False])
def test_reading_coordinates(commutative):
    kx, ky, kz = sympy.symbols('k_x k_y k_z', commutative=commutative)

    test = {
        kx**2                         : ['x'],
        kx**2 + ky**2                 : ['x', 'y'],
        kx**2 + ky**2 + kz**2         : ['x', 'y', 'z'],
        ky**2 + kz**2                 : ['y', 'z'],
        kz**2                         : ['z'],
        kx * fA(x, y) * kx              : ['x'],
        kx**2 + kz * fB(y)             : ['x', 'z'],
    }
    for inp, out in test.items():
        ham, got = discretize_symbolic(inp)
        assert got == out


def test_reading_coordinates_matrix():
    test = [
        (sympy.Matrix([sympy.sympify('k_x**2')])      , ['x']),
        (sympy.Matrix([kx**2])                        , ['x']),
        (sympy.Matrix([kx**2 + ky**2])                , ['x', 'y']),
        (sympy.Matrix([kx**2 + ky**2 + kz**2])        , ['x', 'y', 'z']),
        (sympy.Matrix([ky**2 + kz**2])                , ['y', 'z']),
        (sympy.Matrix([kz**2])                        , ['z']),
        (sympy.Matrix([kx * fA(x, y) * kx])            , ['x']),
        (sympy.Matrix([kx**2 + kz * fB(y)])            , ['x', 'z']),
    ]
    for inp, out in test:
        ham, got = discretize_symbolic(inp)
        assert got == out


def test_reading_different_matrix_types():
    test = [
        (sympy.MutableMatrix([kx**2])                    , ['x']),
        (sympy.ImmutableMatrix([kx**2])                  , ['x']),
        (sympy.MutableDenseMatrix([kx**2])               , ['x']),
        (sympy.ImmutableDenseMatrix([kx**2])             , ['x']),
    ]
    for inp, out in test:
        ham, got = discretize_symbolic(inp)
        assert got == out,\
            "Should be: _split_factors({})=={}. Not {}".format(inp, out, got)


@pytest.mark.parametrize('commutative', [True, False])
def test_simple_derivations(commutative):
    kx, ky, kz = sympy.symbols('k_x k_y k_z', commutative=commutative)
    test = {
        kx**2                   : {(0,): 2/ax**2, (1,): -1/ax**2},
        kx**2 + 1               : {(0,): 1 + 2/ax**2, (1,): -1/ax**2},
        kx**2 + 2               : {(0,): 2 + 2/ax**2, (1,): -1/ax**2},
        kx**2 + 1.0             : {(0,): 1.0 + 2/ax**2, (1,): -1/ax**2},
        kx**2 + 2.0             : {(0,): 2.0 + 2/ax**2, (1,): -1/ax**2},
        kx**2 + ky**2           : {(0, 1): -1/ay**2, (0, 0): 2/ax**2 + 2/ay**2,
                                   (1, 0): -1/ax**2},
        kx**2 + ky**2 + kz**2   : {(1, 0, 0): -1/ax**2, (0, 0, 1): -1/az**2,
                                   (0, 0, 0): 2/ax**2 + 2/ay**2 + 2/az**2,
                                   (0, 1, 0): -1/ay**2},
        ky**2 + kz**2           : {(0, 1): -1/az**2, (0, 0): 2/ay**2 + 2/az**2,
                                   (1, 0): -1/ay**2},
        kz**2                   : {(0,): 2/az**2, (1,): -1/az**2},
    }
    non_commutative_test = {
        kx * fA(x, y) * kx       : {(1, ): -fA(ax/2 + x, y)/ax**2,
                                  (0, ): fA(-ax/2 + x, y)/ax**2 + fA(ax/2 + x, y)/ax**2},
        kx**2 + kz * fB(y)       : {(1, 0): -1/ax**2, (0, 1): -I*fB(y)/(2*az),
                                   (0, 0): 2/ax**2},
        kx * fA(x)               : {(0,): 0, (1,): -I*fA(ax + x)/(2*ax)},
        ky * fA(x)               : {(1,): -I*fA(x)/(2*ay), (0,): 0},
        kx * fA(x) * B           : {(0,): 0, (1,): -I*fA(ax + x)*B/(2*ax)},
        5 * kx                  : {(0,): 0, (1,): -5*I/(2*ax)},
        kx * (fA(x) + fB(x))      : {(0,): 0,
                                   (1,): -I*fA(ax + x)/(2*ax) - I*fB(ax + x)/(2*ax)},
    }

    if not commutative:
        test.update(non_commutative_test)

    for inp, out in test.items():
        got, _ = discretize_symbolic(inp)
        assert got == out

    for inp, out in test.items():
        got, _ = discretize_symbolic(str(inp), locals=ns)
        assert got == out


@pytest.mark.parametrize('e_to_subs, e, subs', [
    ('A * k_x', '(A + B) * k_x', {'A': 'A + B'}),
    ('k_x', 'k_x + k_y', {'k_x': 'k_x + k_y'}),
    ('k_x**2 + V', 'k_x**2 + V + V_0', {'V': 'V + V_0'}),
    ('k_x**2 + A + C', 'k_x**2 + B + 5', {'A': 'B + 5', 'C': 0}),
    ('x + y + z', '1 + 3 + 5', {'x': 1, 'y': 3, 'z': 5}),
])
def test_simple_derivations_with_subs(e_to_subs, e, subs):
    one = discretize_symbolic(e_to_subs, 'xyz', locals=subs)
    two = discretize_symbolic(e, 'xyz')
    assert one == two


def test_simple_derivations_matrix():
    test = {
        kx**2                   : {(0,): 2/ax**2, (1,): -1/ax**2},
        kx**2 + ky**2           : {(0, 1): -1/ay**2, (0, 0): 2/ax**2 + 2/ay**2,
                                   (1, 0): -1/ax**2},
        kx**2 + ky**2 + kz**2   : {(1, 0, 0): -1/ax**2, (0, 0, 1): -1/az**2,
                                   (0, 0, 0): 2/ax**2 + 2/ay**2 + 2/az**2,
                                   (0, 1, 0): -1/ay**2},
        ky**2 + kz**2           : {(0, 1): -1/az**2, (0, 0): 2/ay**2 + 2/az**2,
                                   (1, 0): -1/ay**2},
        kz**2                   : {(0,): 2/az**2, (1,): -1/az**2},

        kx * fA(x, y) * kx       : {(1, ): -fA(ax/2 + x, y)/ax**2,
                                  (0, ): fA(-ax/2 + x, y)/ax**2 + fA(ax/2 + x, y)/ax**2},
        kx**2 + kz * fB(y)       : {(1, 0): -1/ax**2, (0, 1): -I*fB(y)/(2*az),
                                   (0, 0): 2/ax**2},
        kx * fA(x)               : {(0,): 0, (1,): -I*fA(ax + x)/(2*ax)},
        ky * fA(x)               : {(1,): -I*fA(x)/(2*ay), (0,): 0},
        kx * fA(x) * B           : {(0,): 0, (1,): -I*fA(ax + x)*B/(2*ax)},
        5 * kx                  : {(0,): 0, (1,): -5*I/(2*ax)},
        kx * (fA(x) + fB(x))      : {(0,): 0,
                                   (1,): -I*fA(ax + x)/(2*ax) - I*fB(ax + x)/(2*ax)},
   }

    new_test = []
    for inp, out in test.items():
        new_out = {}
        for k, v in out.items():
            new_out[k] = sympy.Matrix([v])
        new_test.append((sympy.Matrix([inp]), new_out))

    for inp, out in new_test:
        got, _ = discretize_symbolic(inp)
        assert got == out

    for inp, out in new_test:
        got, _ = discretize_symbolic(str(inp), locals=ns)
        assert got == out

    for inp, out in new_test:
        got, _ = discretize_symbolic(str(inp).replace('Matrix', ''), locals=ns)
        assert got == out



def test_integer_float_input():
    test = {
        0: {(0, 0, 0): 0},
        1: {(0, 0, 0): 1},
        5: {(0, 0, 0): 5},
    }

    for inp, out in test.items():
        got, _ = discretize_symbolic(int(inp), 'xyz')
        assert got == out

        got, _ = discretize_symbolic(float(inp), 'xyz')
        assert got == out

    # let's test in matrix version too
    new_test = []
    for inp, out in test.items():
        new_out = {}
        for k, v in out.items():
            new_out[k] = sympy.Matrix([v])
        new_test.append((inp, new_out))

    for inp, out in new_test:
        got, _ = discretize_symbolic(sympy.Matrix([int(inp)]), 'xyz')
        assert got == out

        got, _ = discretize_symbolic(sympy.Matrix([float(inp)]), 'xyz')
        assert got == out


def test_different_discrete_coordinates():
    test = [
        (
            'xyz', {
                (1, 0, 0): -1/ax**2, (0, 0, 1): -1/az**2,
                (0, 0, 0): 2/ax**2 + 2/ay**2 + 2/az**2, (0, 1, 0): -1/ay**2
            }
        ),
        (
            'xy', {
                (0, 1): -1/ay**2,
                (1, 0): -1/ax**2,
                (0, 0): kz**2 + 2/ax**2 + 2/ay**2
            }
        ),
        (
            'xz', {
                (0, 1): -1/az**2,
                (1, 0): -1/ax**2,
                (0, 0): ky**2 + 2/ax**2 + 2/az**2
            }
        ),
        (
            'yz', {
                (0, 1): -1/az**2,
                (1, 0): -1/ay**2,
                (0, 0): kx**2 + 2/ay**2 + 2/az**2
            }
        ),
        (
            'x', {
                (0,): ky**2 + kz**2 + 2/ax**2, (1,): -1/ax**2
            }
        ),
        (
            'y', {
                (0,): kx**2 + kz**2 + 2/ay**2, (1,): -1/ay**2
            }
        ),
        (
            'z', {
                (0,): ky**2 + kx**2 + 2/az**2, (1,): -1/az**2
            }
        ),
    ]
    for inp, out in test:
        got, _ = discretize_symbolic(kx**2 + ky**2 + kz**2, inp)
        assert got == out

    # let's test in matrix version too
    new_test = []
    for inp, out in test:
        new_out = {}
        for k, v in out.items():
            new_out[k] = sympy.Matrix([v])
        new_test.append((inp, new_out))

    for inp, out in new_test:
        got, _ = discretize_symbolic(sympy.Matrix([kx**2 + ky**2 + kz**2]), inp)
        assert got == out


def test_non_expended_input():
    symbolic, coords = discretize_symbolic(kx * (kx + fA(x)))
    desired = {
        (0,): 2/ax**2,
        (1,): -I*fA(ax + x)/(2*ax) - 1/ax**2
    }
    assert symbolic == desired


def test_matrix_with_zeros():
    Matrix = sympy.Matrix
    symbolic, _ = discretize_symbolic("[[k_x*A(x)*k_x, 0], [0, k_x*A(x)*k_x]]")
    output = {
        (0,):  Matrix([[fA(-ax/2 + x)/ax**2 + fA(ax/2 + x)/ax**2, 0], [0, fA(-ax/2 + x)/ax**2 + fA(ax/2 + x)/ax**2]]),
        (1,):  Matrix([[-fA(ax/2 + x)/ax**2, 0], [0, -fA(ax/2 + x)/ax**2]]),
        }
    assert symbolic == output


def test_numeric_functions_basic_symbolic():
    for i in [0, 1, 3, 5]:
        builder = discretize(i, 'x')
        lat = next(iter(builder.sites()))[0]
        assert builder.lattice is lat
        assert builder[lat(0)] == i

        p = dict(t=i)

        tb = {(0,): sympy.sympify("2*t"), (1,): sympy.sympify('-t')}
        builder = build_discretized(tb, 'x', grid=1)
        lat = next(iter(builder.sites()))[0]
        assert 2*p['t'] == builder[lat(0)](None, **p)
        assert -p['t'] == builder[lat(1), lat(0)](None, None, **p)

        tb = {(0,): sympy.sympify("0"), (1,): sympy.sympify('-1j * t')}
        builder = build_discretized(tb, 'x', grid=1)
        lat = next(iter(builder.sites()))[0]
        assert -1j * p['t'] == builder[lat(0), lat(1)](None, None, **p)
        assert +1j * p['t'] == builder[lat(1), lat(0)](None, None, **p)


@pytest.mark.parametrize('commutative', [True, False])
def test_numeric_function_coords_from_site(commutative):
    tb = {(0,): sympy.symbols('x', commutative=commutative)}
    builder = build_discretized(tb, 'x')

    lat = next(iter(builder.sites()))[0]
    onsite = builder[lat(0)]
    assert (onsite(lat(0)) == 0 and onsite(lat(1)) == 1)


def test_numeric_functions_not_discrete_coords():
    builder = discretize('k_y + y', 'x')
    lat = next(iter(builder.sites()))[0]
    onsite = builder[lat(0)]

    assert onsite(None, k_y=2, y=1) == 2 + 1


@pytest.mark.parametrize('ham, val, params', [
    ("pi", np.pi, {}),
    ("A + pi", 1 + np.pi, {"A": 1}),
    ("A + B(pi)", 1 + np.pi, {"A": 1, "B": lambda x: x}),
    ("A + I", 1 + 1j, {"A": 1}),
    ("A + 1j", 1 + 1j, {"A": 1}),
    ("A + B(I)", 1 + 1j, {"A": 1, "B": lambda x: x}),
    ("A + B(1j)", 1 + 1j, {"A": 1, "B": lambda x: x}),
    ("exp(1j * pi)", np.exp(1j*np.pi), {"exp": np.exp}),
    (sympy.exp(sympy.sympify("1j * pi * A")), np.exp(1j*np.pi),
        {"exp": np.exp, "A": 1}),
])
def test_numeric_functions_advanced(ham, val, params):
    builder = discretize(ham, 'x')
    lat = next(iter(builder.sites()))[0]
    onsite = builder[lat(0)]
    try:
        assert np.allclose(onsite(None, **params), val)
    except TypeError:
        assert np.allclose(onsite, val)


def test_numeric_functions_basic_string():
    for i in [0, 1, 3, 5]:
        builder = discretize(i, 'x')
        lat = next(iter(builder.sites()))[0]
        assert builder[lat(0)] == i

        p = dict(t=i)

        tb = {(0,): "2*t", (1,): "-t"}
        builder = build_discretized(tb, 'x', grid=1)
        lat = next(iter(builder.sites()))[0]
        assert 2*p['t'] == builder[lat(0)](None, **p)
        assert -p['t'] == builder[lat(1), lat(0)](None, None, **p)

        tb = {(0,): "0", (1,): "-1j * t"}
        builder = build_discretized(tb, 'x', grid=1)
        lat = next(iter(builder.sites()))[0]
        assert -1j * p['t'] == builder[lat(0), lat(1)](None, None, **p)
        assert +1j * p['t'] == builder[lat(1), lat(0)](None, None, **p)

        tb = {(0,): "0", (-1,): "+1j * t"}
        builder = build_discretized(tb, 'x', grid=1)
        lat = next(iter(builder.sites()))[0]
        assert -1j * p['t'] == builder[lat(0), lat(1)](None, None, **p)
        assert +1j * p['t'] == builder[lat(1), lat(0)](None, None, **p)


@pytest.mark.parametrize('e_to_subs, e, subs', [
    ('A * k_x + A', '(A + B) * k_x + A + B', {'A': 'A + B'}),
])
def test_numeric_functions_with_subs(e_to_subs, e, subs):
    p = {'A': 1, 'B': 2}
    builder_direct = discretize(e)
    builder_subs = discretize(e_to_subs, locals=subs)

    lat = next(iter(builder_direct.sites()))[0]
    assert builder_direct[lat(0)](None, **p) == builder_subs[lat(0)](None, **p)

    hop_direct = builder_direct[lat(0), lat(1)](None, None, **p)
    hop_subs = builder_subs[lat(0), lat(1)](None, None, **p)
    assert hop_direct == hop_subs


def test_onsite_hopping_function_name():
    template = str(discretize('A * k_x'))
    assert 'def hopping' in template


def test_numeric_functions_advance():
    hams = [
        kx**2,
        kx**2 + x,
        fA(x),
        kx*fA(x)*kx,
        sympy.Matrix([[kx * fA(x) * kx, fA(x)*kx], [kx*fA(x), fA(x)+B]]),
        kx**2 + B * x,
        'k_x**2 + sin(x)',
        B ** 0.5 * kx**2,
        B ** (1/2) * kx**2,
        sympy.sqrt(B) * kx**2,

    ]
    for hamiltonian in hams:
        for a in [1, 2, 5]:
            for func in [lambda x: x, lambda x: x**2, lambda x: x**3]:
                symbolic, coords = discretize_symbolic(hamiltonian, 'x')
                builder = build_discretized(symbolic, coords, grid=a)
                lat = next(iter(builder.sites()))[0]

                p = dict(A=func, B=5, sin=np.sin)

                # test onsite
                v = symbolic.pop((0,)).subs({sympy.symbols('a_x'): a, B: p['B']})
                f_sym = sympy.lambdify(['A', 'x'], v)
                f_num = builder[lat(0)]

                if callable(f_num):
                    f_num = swallows_extra_kwargs(f_num)
                    for n in range(-100, 100, 10):
                        assert np.allclose(f_sym(func, a*n), f_num(lat(n), **p))
                else:
                    for n in range(-100, 100, 10):
                        assert np.allclose(f_sym(func, a*n), f_num)


                # test hoppings
                for k, v in symbolic.items():
                    v = v.subs({sympy.symbols('a_x'): a, B: p['B']})
                    f_sym = sympy.lambdify(['A', 'x'], v)
                    f_num = builder[lat(0), lat(k[0])]

                    if callable(f_num):
                        f_num = swallows_extra_kwargs(f_num)
                        for n in range(10):
                            lhs = f_sym(func, a * n)
                            rhs = f_num(lat(n), lat(n+k[0]), **p)
                            assert np.allclose(lhs, rhs)
                    else:
                        for n in range(10):
                            lhs = f_sym(fA, a * n)
                            rhs = f_num
                            assert np.allclose(lhs, rhs)


def test_numeric_functions_with_parameter():

    hamiltonian = kx**2 + fA(B, x)

    for a in [1, 2, 5]:
        for func in [lambda c, x: x+c, lambda c, x: x**2 + c]:
            symbolic, coords = discretize_symbolic(hamiltonian, 'x')
            builder = build_discretized(symbolic, coords, grid=a)
            lat = next(iter(builder.sites()))[0]

            p = dict(A=func, B=5)

            # test onsite
            v = symbolic.pop((0,)).subs({sympy.symbols('a_x'): a, B: p['B']})
            f_sym = sympy.lambdify(['A', 'x'], v)

            f_num = builder[lat(0)]
            if callable(f_num):
                f_num = swallows_extra_kwargs(f_num)

            for n in range(10):
                s = lat(n)
                xi = a * n
                if callable(f_num):
                    assert np.allclose(f_sym(func, xi), f_num(s, **p))
                else:
                    assert np.allclose(f_sym(func, xi), f_num)

            # test hoppings
            for k, v in symbolic.items():
                v = v.subs({sympy.symbols('a_x'): a, B: p['B']})
                f_sym = sympy.lambdify(['A', 'x'], v)
                f_num = builder[lat(0), lat(k[0])]

                if callable(f_num):
                    f_num = swallows_extra_kwargs(f_num)

                for n in range(10):
                    s = lat(n)
                    xi = a * n

                    lhs = f_sym(func, xi)
                    if callable(f_num):
                        rhs = f_num(lat(n), lat(n+k[0]), **p)
                    else:
                        rhs = f_num

                    assert np.allclose(lhs, rhs)


###### test grid parameter
@pytest.mark.parametrize('ham, grid_spacing, grid', [
    ('k_x', None, Monatomic([[1, ]], norbs=1)),
    ('k_x * sigma_z', None, Monatomic([[1, ]], norbs=2)),
    ('k_x', 0.5, Monatomic([[0.5, ]], norbs=1)),
    ('k_x**2 + k_y**2', 2, Monatomic([[2, 0], [0, 2]], norbs=1)),
])
def test_grid(ham, grid_spacing, grid):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        t1 = discretize(ham, grid_spacing=grid_spacing)
        t2 = discretize(ham, grid=grid_spacing)
        t3 = discretize(ham, grid=grid)
    assert t1.lattice == t2.lattice == t3.lattice


@pytest.mark.parametrize('ham, grid_offset, offset, norbs', [
    ('k_x', None, 0, None),
    ('k_x', None, 0, 1),
    ('k_x * eye(2)', None, 0, 2),
    ('k_x', (0,), 0, None),
    ('k_x', (1,), 1, None),
    ('k_x + k_y', None, (0, 0), None),
    ('k_x + k_y', (0, 0), (0, 0), None),
    ('k_x + k_y', (1, 2), (1, 2), None),
])
def test_grid_input(ham, grid_offset, offset, norbs):
    # build appriopriate grid
    if isinstance(offset, int):
        prim_vecs = [[1, ]]
    else:
        prim_vecs = np.eye(len(offset))
    grid = Monatomic(prim_vecs, offset=grid_offset, norbs=norbs)

    tmp = discretize(ham, grid=grid)
    assert np.allclose(tmp.lattice.offset, offset)
    assert tmp.lattice.norbs == norbs

    tb_ham, coords = discretize_symbolic(ham)
    tmp = build_discretized(
        tb_ham, coords, grid=grid
    )
    assert np.allclose(tmp.lattice.offset, offset)
    assert tmp.lattice.norbs == norbs


def test_grid_offset_passed_to_functions():
    V = lambda x: x
    grid = Monatomic([[1, ]], offset=[0.5, ])
    tb = discretize('V(x)', 'x', grid=grid)
    onsite = tb[tb.lattice(0)]
    bools = [np.allclose(onsite(tb.lattice(i), V), V(tb.lattice(i).pos))
             for i in [0, 1, 5]]
    assert all(bools)


@pytest.mark.parametrize("ham, coords, grid", [
    ("k_x", None, Monatomic([[1, 0]])),
    ("k_x", 'xy', Monatomic([[1, 0]])),
    ("k_x", None, Monatomic([[1, ]], norbs=2)),
    ("k_x * eye(2)", None, Monatomic([[1, ]], norbs=1)),
    ("k_x+k_y", None, Monatomic([[1, 0], [1, 1]])),
])
def test_grid_constraints(ham, coords, grid):
    with pytest.raises(ValueError):
        discretize(ham, coords, grid=grid)


@pytest.mark.parametrize('name', ['1', '1a', '-a', '+a', 'while', 'for'])
def test_check_symbol_names(name):
    with pytest.raises(ValueError):
        discretize(sympy.Symbol(name), 'x')


def test_rectangular_grid():
    lat = Monatomic([[1, 0], [0, 2]])

    tb = discretize("V(x, y)", 'xy', grid=lat)
    assert np.allclose(tb[lat(0, 0)](lat(1, 0), lambda x, y: x), 1)
    assert np.allclose(tb[lat(0, 0)](lat(0, 1), lambda x, y: y), 2)

    tb = discretize('k_x**2 + k_y**2', grid=lat)
    assert np.allclose(tb[lat(0, 0), lat(1, 0)], -1)
    assert np.allclose(tb[lat(0, 0), lat(0, 1)], -1/4)
