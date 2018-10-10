# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from keyword import iskeyword
from collections import defaultdict
import itertools
import warnings

import numpy as np
import tinyarray as ta

import sympy
from sympy.utilities.lambdify import lambdastr
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.precedence import precedence
from sympy.core.function import AppliedUndef

from .. import builder, lattice
from .. import KwantDeprecationWarning
from .._common import reraise_warnings
from ._common import (sympify, gcd, position_operators, momentum_operators,
                      monomials)


__all__ = ['discretize']


_wf = sympy.Function('_internal_unique_name', commutative=False)
_momentum_operators = {s.name: s for s in momentum_operators}
_position_operators = {s.name: s for s in position_operators}
_displacements = {s: sympy.Symbol('a_{}'.format(s)) for s in 'xyz'}


class _DiscretizedBuilder(builder.Builder):
    """A builder that is made from a discretized model and knows how to
    pretty-print itself."""

    def __init__(self, coords, lattice, symmetry=None, **kwargs):
        super().__init__(symmetry, **kwargs)
        self.lattice = lattice
        self._coords = coords

    def __str__(self):
        result = []

        sv = list(s for s in self.site_value_pairs())
        if len(sv) != 1:
            raise ValueError("Cannot pretty-print _DiscretizedBuilder: "
                             "must contain a single site.")
        site, site_value = sv[0]
        if any(e != 0 for e in site.tag):
            raise ValueError("Cannot pretty-print _DiscretizedBuilder: "
                             "site must be located at origin.")

        result.extend(["# Discrete coordinates: ",
                       " ".join(self._coords),
                       "\n\n"])

        for key, val in itertools.chain(self.site_value_pairs(),
                                        self.hopping_value_pairs()):
            if isinstance(key, builder.Site):
                result.append("# Onsite element:\n")
            else:
                a, b = key
                assert a is site
                result.extend(["# Hopping from ",
                               str(tuple(b.tag)),
                               ":\n"])
            result.append(val._source if callable(val) else repr(val))
            result.append('\n\n')

        result.pop()

        return "".join(result)

    # Enable human-readable rendering in Jupyter notebooks:
    def _repr_html_(self):
        return self.__str__()


################ Interface functions


def discretize(hamiltonian, coords=None, *, grid=None, locals=None,
               grid_spacing=None):
    """Construct a tight-binding model from a continuum Hamiltonian.

    If necessary, the given Hamiltonian is sympified using
    `kwant.continuum.sympify`.  It is then discretized symbolically and turned
    into a `~kwant.builder.Builder` instance that may be used with
    `~kwant.builder.Builder.fill`.

    This is a convenience function that is equivalent to first calling
    `~kwant.continuum.discretize_symbolic` and feeding its result into
    `~kwant.continuum.build_discretized`.

    .. warning::
        This function uses ``eval`` (because it calls ``sympy.sympify``), and
        thus should not be used on unsanitized input.

    Parameters
    ----------
    hamiltonian : str or SymPy expression
        Symbolic representation of a continuous Hamiltonian.  It is
        converted to a SymPy expression using `kwant.continuum.sympify`.
    coords : sequence of strings, optional
        The coordinates for which momentum operators will be treated as
        differential operators. May contain only "x", "y" and "z" and must be
        sorted.  If not provided, `coords` will be obtained from the input
        Hamiltonian by reading the present coordinates and momentum operators.
    grid : scalar or kwant.lattice.Monatomic instance, optional
        Lattice that will be used as a discretization grid. It must have
        orthogonal primitive vectors. If a scalar value is given, a lattice
        with the appropriate grid spacing will be generated. If not provided,
        a lattice with grid spacing 1 in all directions will be generated.
    locals : dict, optional
        Additional namespace entries for `~kwant.continuum.sympify`.  May be
        used to simplify input of matrices or modify input before proceeding
        further. For example:
        ``locals={'k': 'k_x + I * k_y'}`` or
        ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
    grid_spacing : int or float, optional
        (deprecated) Spacing of the discretization grid. If unset the default
        value will be 1. Cannot be used together with ``grid``.

    Returns
    -------
    model : `~kwant.builder.Builder`
        The translationally symmetric builder that corresponds to the provided
        Hamiltonian.  This builder instance belongs to a subclass of the
        standard builder that may be printed to obtain the source code of the
        value functions.  It also holds the discretization lattice (a
        `~kwant.lattice.Monatomic` instance with lattice constant
        `grid_spacing`) in the ``lattice`` attribute.
    """
    tb, coords = discretize_symbolic(hamiltonian, coords, locals=locals)
    return build_discretized(tb, coords, grid=grid, grid_spacing=grid_spacing)


def discretize_symbolic(hamiltonian, coords=None, *, locals=None):
    """Discretize a continuous Hamiltonian into a tight-binding representation.

    If necessary, the given Hamiltonian is sympified using
    `kwant.continuum.sympify`.  It is then discretized symbolically.

    The two return values may be used directly as the first two arguments for
    `~kwant.continuum.build_discretized`.

    .. warning::
        This function uses ``eval`` (because it calls ``sympy.sympify``), and
        thus should not be used on unsanitized input.

    Parameters
    ----------
    hamiltonian : str or SymPy expression
        Symbolic representation of a continuous Hamiltonian.  It is
        converted to a SymPy expression using `kwant.continuum.sympify`.
    coords : sequence of strings, optional
        The coordinates for which momentum operators will be treated as
        differential operators. May contain only "x", "y" and "z" and must be
        sorted.  If not provided, `coords` will be obtained from the input
        Hamiltonian by reading the present coordinates and momentum operators.
    locals : dict, optional
        Additional namespace entries for `~kwant.continuum.sympify`.  May be
        used to simplify input of matrices or modify input before proceeding
        further. For example:
        ``locals={'k': 'k_x + I * k_y'}`` or
        ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.

    Returns
    -------
    tb_hamiltonian : dict
        Keys are tuples of integers; the offsets of the hoppings ((0, 0, 0) for
        the onsite). Values are symbolic expressions for the hoppings/onsite.
    coords : list of strings
        The coordinates that have been discretized.
    """
    with reraise_warnings():
        hamiltonian = sympify(hamiltonian, locals)

    atoms_names = [s.name for s in hamiltonian.atoms(sympy.Symbol)]
    if any(s in ('a_x', 'a_y', 'a_z') for s in atoms_names):
        raise TypeError("'a_x', 'a_y' and 'a_z' are  symbols used internally "
                        "to represent grid spacings; please use a different "
                        "symbol.")

    hamiltonian = sympy.expand(hamiltonian)
    if coords is None:
        used_momenta = set(_momentum_operators) & set(atoms_names)
        coords = {k[-1] for k in used_momenta}
    else:
        coords = list(coords)
        if coords != sorted(coords):
            raise ValueError("The argument 'coords' must be sorted.")
        if any(c not in 'xyz' for c in coords):
            raise ValueError("The argument 'coords' may only contain "
                             "'x', 'y', or 'z'.")

    coords = sorted(coords)

    if len(coords) == 0:
        raise ValueError("Failed to read any discrete coordinates. This is "
                         "probably due to a lack of momentum operators in "
                         "your input. You can use the 'coords' "
                         "parameter to provide them.")

    onsite_zeros = (0,) * len(coords)

    if not isinstance(hamiltonian, sympy.matrices.MatrixBase):
        hamiltonian = sympy.Matrix([hamiltonian])
        _input_format = 'expression'
    else:
        _input_format = 'matrix'

    shape = hamiltonian.shape
    tb = defaultdict(lambda: sympy.zeros(*shape))
    tb[onsite_zeros] = sympy.zeros(*shape)

    for (i, j), expression in np.ndenumerate(hamiltonian):
        hoppings = _discretize_expression(expression, coords)

        for offset, hop in hoppings.items():
            tb[offset][i, j] += hop

    # do not include Hermitian conjugates of hoppings
    wanted_hoppings = sorted(list(tb))[len(list(tb)) // 2:]
    tb = {k: v for k, v in tb.items() if k in wanted_hoppings}

    if _input_format == 'expression':
        tb = {k: v[0, 0] for k, v in tb.items()}

    return tb, coords


def build_discretized(tb_hamiltonian, coords, *, grid=None, locals=None,
                      grid_spacing=None):
    """Create a template builder from a symbolic tight-binding Hamiltonian.

    The provided symbolic tight-binding Hamiltonian is put on a (hyper) square
    lattice and turned into Python functions.  These functions are used to
    create a `~kwant.builder.Builder` instance that may be used with
    `~kwant.builder.Builder.fill` to construct a system of a desired shape.

    The return values of `~kwant.continuum.discretize_symbolic` may be used
    directly for the first two arguments of this function.

    .. warning::
        This function uses ``eval`` (because it calls ``sympy.sympify``), and
        thus should not be used on unsanitized input.

    Parameters
    ----------
    tb_hamiltonian : dict
        Keys must be the offsets of the hoppings, represented by tuples of
        integers ((0, 0, 0) for onsite). Values must be symbolic expressions
        for the hoppings/onsite or expressions that can by sympified with
        `kwant.continuum.sympify`.
    coords : sequence of strings
        The coordinates for which momentum operators will be treated as
        differential operators. May contain only "x", "y" and "z" and must be
        sorted.
    grid : scalar or kwant.lattice.Monatomic instance, optional
        Lattice that will be used as a discretization grid. It must have
        orthogonal primitive vectors. If a scalar value is given, a lattice
        with the appropriate grid spacing will be generated. If not provided,
        a lattice with grid spacing 1 in all directions will be generated.
    locals : dict, optional
        Additional namespace entries for `~kwant.continuum.sympify`.  May be
        used to simplify input of matrices or modify input before proceeding
        further. For example:
        ``locals={'k': 'k_x + I * k_y'}`` or
        ``locals={'sigma_plus': [[0, 2], [0, 0]]}``.
    grid_spacing : int or float, optional
        (deprecated) Spacing of the discretization grid. If not provided,
        the default value will be 1. Cannot be used together with ``grid``.

    Returns
    -------
    model : `~kwant.builder.Builder`
        The translationally symmetric builder that corresponds to the provided
        Hamiltonian.  This builder instance belongs to a subclass of the
        standard builder that may be printed to obtain the source code of the
        value functions.  It also holds the discretization lattice (a
        `~kwant.lattice.Monatomic` instance with lattice constant
        `grid_spacing`) in the ``lattice`` attribute.
    """
    # check already available constraints (grid will be check later)
    if len(coords) == 0:
        raise ValueError('Discrete coordinates cannot be empty.')

    if grid_spacing is not None:  # TODO: remove when we remove 'grid_spacing'
        warnings.warn('The "grid_spacing" parameter is deprecated. Use '
                      '"grid" instead.', KwantDeprecationWarning, stacklevel=3)
    if grid is None and grid_spacing is None:
        grid = 1  # default case
    elif grid is None:  # TODO: remove when we remove 'grid_spacing'
        grid = grid_spacing
    elif grid_spacing is not None:
        raise ValueError('"grid_spacing" and "grid" are mutually exclusive.')

    coords = list(coords)
    grid_dim = len(coords)

    if coords != sorted(coords):
        raise ValueError("The argument 'coords' must be sorted.")

    # run sympifcation on hamiltonian values
    with reraise_warnings():
        for k, v in tb_hamiltonian.items():
            tb_hamiltonian[k] = sympify(v, locals)

    # generate grid if required, check constraints if provided
    random_element = next(iter(tb_hamiltonian.values()))
    norbs = (1 if isinstance(random_element, sympy.Expr)
             else random_element.shape[0])

    if np.isscalar(grid):
        lat = lattice.Monatomic(grid * np.eye(grid_dim), norbs=norbs)
    else:
        lat = grid

    # check grid constraints
    is_diagonal = lambda m: np.allclose(m, np.diag(np.diagonal(m)))
    if not (lat.prim_vecs.shape[0] == grid_dim and
            is_diagonal(lat.prim_vecs)):
        raise ValueError('"grid" is expected to by an orthogonal lattice '
                         'of dimension matching number of "coords".')

    if (lat.norbs is not None) and (lat.norbs != norbs):
        raise ValueError(
            'Number of lattice orbitals does not match the number '
            'of orbitals in the Hamiltonian.'
        )

    # continue with building the template
    tb = {}
    for n, (offset, hopping) in enumerate(tb_hamiltonian.items()):
        onsite = all(i == 0 for i in offset)

        if onsite:
            name = 'onsite'
        else:
            name = 'hopping_{}'.format(n)

        tb[offset] = _builder_value(hopping, coords, np.diag(lat.prim_vecs),
                                    onsite, name)

    onsite_zeros = (0,) * grid_dim
    onsite = tb.pop(onsite_zeros)
    # 'delta' parameter to HoppingKind is the negative of the 'hopping offset'
    hoppings = {builder.HoppingKind(tuple(-i for i in d), lat): val
                for d, val in tb.items()}

    syst = _DiscretizedBuilder(
        coords, lat, lattice.TranslationalSymmetry(*lat.prim_vecs)
    )
    syst[lat(*onsite_zeros)] = onsite
    for hop, val in hoppings.items():
        syst[hop] = val

    return syst


def _differentiate(expression, coordinate_name):
    """Calculate derivative of an expression for given coordinate.

    Parameters:
    -----------
    expression : sympy.Expr
        Sympy expression containing function to be derivated.
    coordinate_name : string
        Coordinate over which derivative is calculated.

    Returns
    -------
    sympy.Expr
    """
    assert coordinate_name in 'xyz'
    coordinate = _position_operators[coordinate_name]
    h = _displacements[coordinate_name]

    expr1 = expression.subs(coordinate, coordinate + h)
    expr2 = expression.subs(coordinate, coordinate - h)

    return (expr1 - expr2) / (2 * h)


def _discretize_summand(summand, coords):
    """Discretize a product of factors.

    Parameters
    ----------
    summand : sympy.Expr
    coords : sequence of strings
        Must be a subset of ``{'x', 'y', 'z'}``.

    Returns
    -------
    sympy.Expr
    """
    assert not isinstance(summand, sympy.Add), "Input should be one summand."
    momenta = ['k_{}'.format(s) for s in coords]

    factors = reversed(summand.as_ordered_factors())
    result = 1
    for factor in factors:
        symbol, exponent = factor.as_base_exp()
        if isinstance(symbol, sympy.Symbol) and symbol.name in momenta:
            for i in range(exponent):
                coordinate = symbol.name[-1]
                # apply momentum as differential operator '-i d/dx'
                result = -sympy.I * _differentiate(result, coordinate)
        else:
            result = factor * result

    return result


def _discretize_expression(expression, coords):
    """Discretize an expression into a discrete (tight-binding) representation.

    Parameters
    ----------
    expression : sympy.Expr
    coords : sequence of strings
        Must be a subset of ``{'x', 'y', 'z'}``.

    Returns
    -------
    dict
        Keys are tuples of integers; the offsets of the hoppings
        ((0, 0, 0) for the onsite). Values are symbolic expressions
        for the hoppings/onsite.
    """
    def _read_offset(wf):
        # e.g. wf(x + h, y, z + h) -> (1, 0, 1)
        assert wf.func == _wf

        offset = []
        for c, arg in zip(coords, wf.args):
            coefficients = arg.as_coefficients_dict()
            assert coefficients[_position_operators[c]] == 1

            ai = _displacements[c]
            offset.append(coefficients.pop(ai, 0))
        return tuple(offset)

    def _extract_hoppings(expr):
        """Read hoppings and perform shortening operation."""
        expr = sympy.expand(expr)
        summands = [e.as_ordered_factors() for e in expr.as_ordered_terms()]

        offset = [_read_offset(s[-1]) for s in summands]
        coeffs = [sympy.Mul(*s[:-1]) for s in summands]
        offset = np.array(offset, dtype=int)
        # rescale the offsets for each coordinate by their greatest
        # common divisor across the summands. e.g:
        # wf(x+2h) + wf(x+4h) --> wf(x+h) + wf(x+2h) and a_x //= 2
        subs = {}
        for i, xi in enumerate(coords):
            factor = int(gcd(*offset[:, i]))
            if factor < 1:
                continue
            offset[:, i] //= factor
            subs[_displacements[xi]] = _displacements[xi] / factor
        # apply the rescaling to the hoppings
        output = defaultdict(lambda: sympy.Integer(0))
        for n, c in enumerate(coeffs):
            output[tuple(offset[n].tolist())] += c.subs(subs)
        return dict(output)

    # if there are no momenta in the expression, then it is an onsite
    atoms_names = [s.name for s in expression.atoms(sympy.Symbol)]
    if not set(_momentum_operators) & set(atoms_names):
        n = len(coords)
        return {(0,) * n: expression}

    # make sure we have list of summands
    summands = expression.as_ordered_terms()

    # discretize every summand
    coordinates = tuple(_position_operators[s] for s in coords)
    wf = _wf(*coordinates)

    discrete_expression = defaultdict(int)
    for summand in summands:
        summand = _discretize_summand(summand * wf, coords)
        hops = _extract_hoppings(summand)
        for k, v in hops.items():
            discrete_expression[k] += v

    return dict(discrete_expression)


################ string processing

class _NumericPrinter(LambdaPrinter):

    def __init__(self):
        if 'allow_unknown_functions' in LambdaPrinter._default_settings:
            settings = {'allow_unknown_functions': True}
        else:
            # We're on Sympy without "allow_unknown_functions" setting
            settings = {}

        LambdaPrinter.__init__(self, settings=settings)

        self.known_functions = {}
        self.known_constants = {'pi': 'pi', 'Pi': 'pi', 'I': 'I'}

    def _print_ImaginaryUnit(self, expr):
        # prevent sympy from printing 'I' for imaginary unit
        return "1j"

    def _print_Pow(self, expr):
        # copied from sympy's StrPrinter with the code paths
        # to print 'sqrt' removed.
        PREC = precedence(expr)

        if expr.is_commutative and expr.exp is -sympy.S.One:
            return '1/%s' % self.parenthesize(expr.base, PREC)

        e = self.parenthesize(expr.exp, PREC)
        if (self.printmethod == '_sympyrepr' and
            expr.exp.is_Rational and expr.exp.q != 1):
            if e.startswith('(Rational'):
                return '%s**%s' % (self.parenthesize(expr.base, PREC), e[1:-1])
        return '%s**%s' % (self.parenthesize(expr.base, PREC), e)


def _print_sympy(expr):
    return lambdastr((), expr, printer=_NumericPrinter)[len('lambda : '):]


def _return_string(expr, coords):
    """Process a sympy expression into an evaluatable Python return statement.

    Parameters
    ----------
    expr : sympy.Expr

    Returns
    -------
    output : string
        A return string that can be used to assemble a Kwant value function.
    map_func_calls : dict
        mapping of function calls to assigned constants.
    const_symbols : sequance of sympy.Symbol
        All constants that appear in the expression.
    _cache: dict
        mapping of cache symbols to cached matrices.
    """
    _cache = {}
    def cache(x):
        s = sympy.symbols('_cache_{}'.format(len(_cache)))
        _cache[str(s)] = ta.array(x.tolist(), complex)
        return s

    blacklisted = set(coords) | {'site', 'site1', 'site2'}
    const_symbols = {s for s in expr.atoms(sympy.Symbol)
                     if s.name not in blacklisted}

    # functions will be evaluated within the function body and the
    # result assigned to a symbol '_const_<n>', so we replace all
    # function calls by these symbols in the return statement.
    map_func_calls = expr.atoms(AppliedUndef, sympy.Function)
    map_func_calls = {s: sympy.symbols('_const_{}'.format(n))
                      for n, s in enumerate(map_func_calls)}

    expr = expr.subs(map_func_calls)

    if isinstance(expr, sympy.matrices.MatrixBase):
        # express matrix return values in terms of sums of known matrices,
        # which will be assigned to '_cache_n' in the function body.
        mons = monomials(expr, expr.atoms(sympy.Symbol))
        mons = {k: cache(v) for k, v in mons.items()}
        mons = ["{} * {}".format(_print_sympy(k), _print_sympy(v))
                for k, v in mons.items()]
        output = " + ".join(mons)
    else:
        output = _print_sympy(expr)

    return 'return {}'.format(output), map_func_calls, const_symbols, _cache


def _assign_symbols(map_func_calls, coords, onsite):
    """Generate a series of assignments.

    Parameters
    ----------
    map_func_calls : dict
        mapping of function calls to assigned constants.
    coords : sequence of strings
        If left as None coordinates will not be read from a site.
    onsite : bool
        True if function is called for onsite, false for hoppings

    Returns
    -------
    assignments : list of strings
        List of lines used for including in a function.
    """
    lines = []

    if coords:
        site = 'site' if onsite else 'site1'
        args = ', '.join(coords), site
        lines.append('({}, ) = {}.pos'.format(*args))

    for k, v in map_func_calls.items():
        lines.append("{} = {}".format(v, _print_sympy(k)))

    return lines


def _builder_value(expr, coords, grid_spacing, onsite,
                   name='_anonymous_func'):
    """Generate a builder value from a sympy expression.

    Parameters
    ----------
    expr : sympy.Expr or sympy.matrix
        Expr that from which value function will be generated.
    coords : sequence of strings
        List of coodinates present in the system.
    grid_spacing : sequence of scalars
        Lattice spacing of the system in each coordinate.

    Returns
    -------
    `expr` transformed into an object that can be used as a
    `kwant.builder.Builder` value.  Either a numerical value
    (``tinyarray.array`` instance or complex number) or a value function.  In
    the case of a function, the source code is available in its `_source`
    attribute.
    """

    expr = expr.subs({_displacements[c]: grid_spacing[n]
                      for n, c in enumerate(coords)})
    return_string, map_func_calls, const_symbols, _cache = _return_string(
        expr, coords=coords)

    # first check if value function needs to read coordinates
    atoms_names = {s.name for s in expr.atoms(sympy.Symbol)}
    if not set(coords) & atoms_names:
        coords = None

    # constants and functions in the sympy input will be passed
    # as arguments to the value function
    arg_names = set.union({s.name for s in const_symbols},
                          {str(k.func) for k in map_func_calls})

    # check if all argument names are valid python identifiers
    for arg_name in arg_names:
        if not (arg_name.isidentifier() and not iskeyword(arg_name)):
            raise ValueError("Invalid name in used symbols: {}\n"
                             "Names of symbols used in Hamiltonian "
                             "must be valid Python identifiers and "
                             "may not be keywords".format(arg_name))

    arg_names = ', '.join(sorted(arg_names))

    if (not arg_names) and (coords is None):
        # we can just use a constant value instead of a value function
        if isinstance(expr, sympy.MatrixBase):
            return ta.array(expr.tolist(), complex)
        else:
            return complex(expr)

    lines = _assign_symbols(map_func_calls, onsite=onsite, coords=coords)
    lines.append(return_string)

    separator = '\n    '
    # 'site_string' is tightly coupled to the symbols used in '_assign_symbol'
    site_string = 'site' if onsite else 'site1, site2'
    if arg_names:
        header_str = 'def {}({}, {}):'
        header = header_str.format(name, site_string, arg_names)
    else:
        header = 'def {}({}):'.format(name, site_string)
    func_code = separator.join([header] + list(lines))

    # Add "I" to namespace just in case sympy again would miss to replace it
    # with Python's 1j as it was the case with SymPy 1.2 when I was argument
    # of some function.
    namespace = {'pi': np.pi, 'I': 1j}
    namespace.update(_cache)

    source = []
    for k, v in _cache.items():
        source.append("{} = (\n{})\n".format(k, repr(np.array(v))))
    source.append(func_code)

    exec(func_code, namespace)
    f = namespace[name]
    f._source = "".join(source)

    return f
