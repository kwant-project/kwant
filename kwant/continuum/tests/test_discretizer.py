import sympy
import numpy as np

from .._common import sympify
from ..discretizer import discretize
from ..discretizer import discretize_symbolic
from ..discretizer import build_discretized
from ..discretizer import  _wf

import inspect
from functools import wraps
import pytest


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

wf =  _wf
Psi = wf(x, y, z)
A, B = sympy.symbols('A B', commutative=False)

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
        kx * A(x,y) * kx              : ['x'],
        kx**2 + kz * B(y)             : ['x', 'z'],
    }
    for inp, out in test.items():
        ham, got = discretize_symbolic(inp)
        assert got == out


def test_reading_coordinates_matrix():
    test = [
        (sympy.Matrix([sympy.sympify('k_x**2')])        , ['x']),
        (sympy.Matrix([kx**2])                        , ['x']),
        (sympy.Matrix([kx**2 + ky**2])                , ['x', 'y']),
        (sympy.Matrix([kx**2 + ky**2 + kz**2])        , ['x', 'y', 'z']),
        (sympy.Matrix([ky**2 + kz**2])                , ['y', 'z']),
        (sympy.Matrix([kz**2])                        , ['z']),
        (sympy.Matrix([kx * A(x,y) * kx])             , ['x']),
        (sympy.Matrix([kx**2 + kz * B(y)])            , ['x', 'z']),
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
        kx**2                   : {(0,): 2/a**2, (1,): -1/a**2},
        kx**2 + ky**2           : {(0, 1): -1/a**2, (0, 0): 4/a**2,
                                   (1, 0): -1/a**2},
        kx**2 + ky**2 + kz**2   : {(1, 0, 0): -1/a**2, (0, 0, 1): -1/a**2,
                                   (0, 0, 0): 6/a**2, (0, 1, 0): -1/a**2},
        ky**2 + kz**2           : {(0, 1): -1/a**2, (0, 0): 4/a**2,
                                   (1, 0): -1/a**2},
        kz**2                   : {(0,): 2/a**2, (1,): -1/a**2},
    }
    non_commutative_test = {
        kx * A(x,y) * kx        : {(1, ): -A(a/2 + x, y)/a**2,
                                  (0, ): A(-a/2 + x, y)/a**2 + A(a/2 + x, y)/a**2},
        kx**2 + kz * B(y)       : {(1, 0): -1/a**2, (0, 1): -I*B(y)/(2*a),
                                   (0, 0): 2/a**2},
        kx * A(x)               : {(0,): 0, (1,): -I*A(a + x)/(2*a)},
        ky * A(x)               : {(1,): -I*A(x)/(2*a), (0,): 0},
        kx * A(x) * B           : {(0,): 0, (1,): -I*A(a + x)*B/(2*a)},
        5 * kx                  : {(0,): 0, (1,): -5*I/(2*a)},
        kx * (A(x) + B(x))      : {(0,): 0,
                                   (1,): -I*A(a + x)/(2*a) - I*B(a + x)/(2*a)},
    }

    if not commutative:
        test.update(non_commutative_test)

    for inp, out in test.items():
        got, _ = discretize_symbolic(inp)
        assert got == out

    for inp, out in test.items():
        got, _ = discretize_symbolic(str(inp), substitutions=ns)
        assert got == out


@pytest.mark.parametrize('e_to_subs, e, subs', [
    ('k_x', 'k_x + k_y', {'k_x': 'k_x + k_y'}),
    ('k_x**2 + V', 'k_x**2 + V + V_0', {'V': 'V + V_0'}),
    ('k_x**2 + A + C', 'k_x**2 + B + 5', {'A': 'B + 5', 'C': 0}),
    ('x + y + z', '1 + 3 + 5', {'x': 1, 'y': 3, 'z': 5})
    ])
def test_simple_derivations_with_subs(e_to_subs, e, subs):
    # check with strings
    one = discretize_symbolic(e_to_subs, 'xyz', substitutions=subs)
    two = discretize_symbolic(e, 'xyz')
    assert one == two

    # check with sympy objects
    one = discretize_symbolic(sympify(e_to_subs), 'xyz', substitutions=subs)
    two = discretize_symbolic(sympify(e), 'xyz')
    assert one == two


def test_simple_derivations_matrix():
    test = {
        kx**2                   : {(0,): 2/a**2, (1,): -1/a**2},
        kx**2 + ky**2           : {(0, 1): -1/a**2, (0, 0): 4/a**2,
                                   (1, 0): -1/a**2},
        kx**2 + ky**2 + kz**2   : {(1, 0, 0): -1/a**2, (0, 0, 1): -1/a**2,
                                   (0, 0, 0): 6/a**2, (0, 1, 0): -1/a**2},
        ky**2 + kz**2           : {(0, 1): -1/a**2, (0, 0): 4/a**2,
                                   (1, 0): -1/a**2},
        kz**2                   : {(0,): 2/a**2, (1,): -1/a**2},
        kx * A(x,y) * kx        : {(1, ): -A(a/2 + x, y)/a**2,
                                  (0, ): A(-a/2 + x, y)/a**2 + A(a/2 + x, y)/a**2},
        kx**2 + kz * B(y)       : {(1, 0): -1/a**2, (0, 1): -I*B(y)/(2*a),
                                   (0, 0): 2/a**2},
        kx * A(x)               : {(0,): 0, (1,): -I*A(a + x)/(2*a)},
        ky * A(x)               : {(1,): -I*A(x)/(2*a), (0,): 0},
        kx * A(x) * B           : {(0,): 0, (1,): -I*A(a + x)*B/(2*a)},
        5 * kx                  : {(0,): 0, (1,): -5*I/(2*a)},
        kx * (A(x) + B(x))      : {(0,): 0,
                                   (1,): -I*A(a + x)/(2*a) - I*B(a + x)/(2*a)},
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
        got, _ = discretize_symbolic(str(inp), substitutions=ns)
        assert got == out

    for inp, out in new_test:
        got, _ = discretize_symbolic(str(inp).replace('Matrix', ''), substitutions=ns)
        assert got == out



def test_integer_float_input():
    test = {
        0      : {(0,0,0): 0},
        1      : {(0,0,0): 1},
        5      : {(0,0,0): 5},
    }

    for inp, out in test.items():
        got, _ = discretize_symbolic(int(inp), {'x', 'y', 'z'})
        assert got == out

        got, _ = discretize_symbolic(float(inp), {'x', 'y', 'z'})
        assert got == out

    # let's test in matrix version too
    new_test = []
    for inp, out in test.items():
        new_out = {}
        for k, v in out.items():
            new_out[k] = sympy.Matrix([v])
        new_test.append((inp, new_out))

    for inp, out in new_test:
        got, _ = discretize_symbolic(sympy.Matrix([int(inp)]), {'x', 'y', 'z'})
        assert got == out

        got, _ = discretize_symbolic(sympy.Matrix([float(inp)]), {'x', 'y', 'z'})
        assert got == out


def test_different_discrete_coordinates():
    test = [
        (
            {'x', 'y', 'z'}, {
                (1, 0, 0): -1/a**2, (0, 0, 1): -1/a**2,
                (0, 0, 0): 6/a**2, (0, 1, 0): -1/a**2
            }
        ),
        (
            {'x', 'y'}, {
                (0, 1): -1/a**2,
                (1, 0): -1/a**2,
                (0, 0): kz**2 + 4/a**2
            }
        ),
        (
            {'x', 'z'}, {
                (0, 1): -1/a**2,
                (1, 0): -1/a**2,
                (0, 0): ky**2 + 4/a**2
            }
        ),
        (
            {'y', 'z'}, {
                (0, 1): -1/a**2,
                (1, 0): -1/a**2,
                (0, 0): kx**2 + 4/a**2
            }
        ),
        (
            {'x'}, {
                (0,): ky**2 + kz**2 + 2/a**2, (1,): -1/a**2
            }
        ),
        (
            {'y'}, {
                (0,): kx**2 + kz**2 + 2/a**2, (1,): -1/a**2
            }
        ),
        (
            {'z'}, {
                (0,): ky**2 + kx**2 + 2/a**2, (1,): -1/a**2
            }
        ) ,
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
    symbolic, coords = discretize_symbolic(kx * (kx + A(x)))
    desired = {
        (0,): 2/a**2,
        (1,): -I*A(a + x)/(2*a) - 1/a**2
    }
    assert symbolic == desired



def test_matrix_with_zeros():
    Matrix = sympy.Matrix
    symbolic, _ = discretize_symbolic("[[k_x*A(x)*k_x, 0], [0, k_x*A(x)*k_x]]")
    output = {
        (0,) :  Matrix([[A(-a/2 + x)/a**2 + A(a/2 + x)/a**2, 0], [0, A(-a/2 + x)/a**2 + A(a/2 + x)/a**2]]),
        (1,) :  Matrix([[-A(a/2 + x)/a**2, 0], [0, -A(a/2 + x)/a**2]]),
        }
    assert symbolic == output


def test_numeric_functions_basic_symbolic():
    for i in [0, 1, 3, 5]:
        builder = discretize(i, {'x'})
        lat = next(iter(builder.sites()))[0]
        assert builder[lat(0)] == i

        p = dict(t=i)

        tb = {(0,): sympy.sympify("2*t"), (1,): sympy.sympify('-t')}
        builder = build_discretized(tb, {'x'}, lattice_constant=1)
        lat = next(iter(builder.sites()))[0]
        assert 2*p['t'] == builder[lat(0)](None, **p)
        assert -p['t'] == builder[lat(1), lat(0)](None, None, **p)

        tb = {(0,): sympy.sympify("0"), (1,): sympy.sympify('-1j * t')}
        builder = build_discretized(tb, {'x'}, lattice_constant=1)
        lat = next(iter(builder.sites()))[0]
        assert -1j * p['t'] == builder[lat(0), lat(1)](None, None, **p)
        assert +1j * p['t'] == builder[lat(1), lat(0)](None, None, **p)


def test_numeric_functions_not_discrete_coords():
    builder = discretize('k_y + y', 'x')
    lat = next(iter(builder.sites()))[0]
    onsite = builder[lat(0)]

    assert onsite(None, k_y=2, y=1) == 2 + 1


def test_numeric_functions_with_pi():
    # Two cases because once it is casted
    # to complex, one there is a function created

    builder = discretize('A + pi', 'x')
    lat = next(iter(builder.sites()))[0]
    onsite = builder[lat(0)]
    assert onsite(None, A=1) == 1 + np.pi


    builder = discretize('pi', 'x')
    lat = next(iter(builder.sites()))[0]
    onsite = builder[lat(0)]
    assert onsite == np.pi


def test_numeric_functions_basic_string():
    for i in [0, 1, 3, 5]:
        builder = discretize(i, {'x'})
        lat = next(iter(builder.sites()))[0]
        assert builder[lat(0)] == i

        p = dict(t=i)

        tb = {(0,): "2*t", (1,): "-t"}
        builder = build_discretized(tb, {'x'}, lattice_constant=1)
        lat = next(iter(builder.sites()))[0]
        assert 2*p['t'] == builder[lat(0)](None, **p)
        assert -p['t'] == builder[lat(1), lat(0)](None, None, **p)

        tb = {(0,): "0", (1,): "-1j * t"}
        builder = build_discretized(tb, {'x'}, lattice_constant=1)
        lat = next(iter(builder.sites()))[0]
        assert -1j * p['t'] == builder[lat(0), lat(1)](None, None, **p)
        assert +1j * p['t'] == builder[lat(1), lat(0)](None, None, **p)

        tb = {(0,): "0", (-1,): "+1j * t"}
        builder = build_discretized(tb, {'x'}, lattice_constant=1)
        lat = next(iter(builder.sites()))[0]
        assert -1j * p['t'] == builder[lat(0), lat(1)](None, None, **p)
        assert +1j * p['t'] == builder[lat(1), lat(0)](None, None, **p)


def test_numeric_functions_advance():
    hams = [
        kx**2,
        kx**2 + x,
        A(x),
        kx*A(x)*kx,
        sympy.Matrix([[kx * A(x) * kx, A(x)*kx], [kx*A(x), A(x)+B]]),
        kx**2 + B * x,
        'k_x**2 + sin(x)',
        B ** 0.5 * kx**2,
        B ** (1/2) * kx**2,
        sympy.sqrt(B) * kx**2,

    ]
    for hamiltonian in hams:
        for a in [1, 2, 5]:
            for fA in [lambda x: x, lambda x: x**2, lambda x: x**3]:
                symbolic, coords = discretize_symbolic(hamiltonian, {'x'})
                builder = build_discretized(symbolic, coords, lattice_constant=a)
                lat = next(iter(builder.sites()))[0]

                p = dict(A=fA, B=5, sin=np.sin)

                # test onsite
                v = symbolic.pop((0,)).subs({sympy.symbols('a'): a, B: p['B']})
                f_sym = sympy.lambdify(['A', 'x'], v)
                f_num = builder[lat(0)]

                if callable(f_num):
                    f_num = swallows_extra_kwargs(f_num)
                    for n in range(-100, 100, 10):
                        assert np.allclose(f_sym(fA, a*n), f_num(lat(n), **p))
                else:
                    for n in range(-100, 100, 10):
                        assert np.allclose(f_sym(fA, a*n), f_num)


                # test hoppings
                for k, v in symbolic.items():
                    v = v.subs({sympy.symbols('a'): a, B: p['B']})
                    f_sym = sympy.lambdify(['A', 'x'], v)
                    f_num = builder[lat(0), lat(k[0])]

                    if callable(f_num):
                        f_num = swallows_extra_kwargs(f_num)
                        for n in range(10):
                            lhs = f_sym(fA, a * n)
                            rhs = f_num(lat(n), lat(n+k[0]), **p)
                            assert np.allclose(lhs, rhs)
                    else:
                        for n in range(10):
                            lhs = f_sym(fA, a * n)
                            rhs = f_num
                            assert np.allclose(lhs, rhs)


def test_numeric_functions_with_parameter():
    hams = [
        kx**2 + A(B, x)
    ]
    for hamiltonian in hams:
        for a in [1, 2, 5]:
            for fA in [lambda c, x: x+c, lambda c, x: x**2 + c]:
                symbolic, coords = discretize_symbolic(hamiltonian, {'x'})
                builder = build_discretized(symbolic, coords, lattice_constant=a)
                lat = next(iter(builder.sites()))[0]

                p = dict(A=fA, B=5)

                # test onsite
                v = symbolic.pop((0,)).subs({sympy.symbols('a'): a, B: p['B']})
                f_sym = sympy.lambdify(['A', 'x'], v)

                f_num = builder[lat(0)]
                if callable(f_num):
                    f_num = swallows_extra_kwargs(f_num)


                for n in range(10):
                    s = lat(n)
                    xi = a * n
                    if callable(f_num):
                        assert np.allclose(f_sym(fA, xi), f_num(s, **p))
                    else:
                        assert np.allclose(f_sym(fA, xi), f_num)


                # test hoppings
                for k, v in symbolic.items():
                    v = v.subs({sympy.symbols('a'): a, B: p['B']})
                    f_sym = sympy.lambdify(['A', 'x'], v)
                    f_num = builder[lat(0), lat(k[0])]

                    if callable(f_num):
                        f_num = swallows_extra_kwargs(f_num)

                    for n in range(10):
                        s = lat(n)
                        xi = a * n

                        lhs = f_sym(fA, xi)
                        if callable(f_num):
                            rhs = f_num(lat(n), lat(n+k[0]), **p)
                        else:
                            rhs = f_num

                        assert np.allclose(lhs, rhs)


def test_basic_verbose(capsys): # or use "capfd" for fd-level
    discretize('k_x * A(x) * k_x', verbose=True)
    out, err = capsys.readouterr()
    assert "Discrete coordinates set to" in out
    assert "Function generated for (0,)" in out


def test_that_verbose_covers_all_hoppings(capsys):
    discretize('k_x**2 + k_y**2 + k_x*k_y', verbose=True)
    out, err = capsys.readouterr()

    for tag in [(0, 1), (0, 0), (1, -1), (1, 1)]:
        assert "Function generated for {}".format(tag) in out


def test_verbose_cache(capsys):
    discretize('[[k_x * A(x) * k_x]]', verbose=True)
    out, err = capsys.readouterr()
    assert '_cache_0' in out


def test_no_output_when_verbose_false(capsys):
    discretize('[[k_x * A(x) * k_x]]', verbose=False)
    out, err = capsys.readouterr()
    assert out == ''
