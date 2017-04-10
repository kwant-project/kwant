# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from collections import defaultdict

import numpy as np
import tinyarray as ta

import sympy
from sympy.utilities.lambdify import lambdastr
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.printing.precedence import precedence
from sympy.core.function import AppliedUndef

from ..builder import Builder, HoppingKind
from ..lattice import Monatomic, TranslationalSymmetry

from ._common import sympify, gcd
from ._common import position_operators, momentum_operators
from ._common import monomials


__all__ = ['discretize']


################ Globals variables and definitions

_wf = sympy.Function('_internal_unique_name', commutative=False)
_momentum_operators = {s.name: s for s in momentum_operators}
_position_operators = {s.name: s for s in position_operators}
_displacements = {s: sympy.Symbol('_internal_a_{}'.format(s)) for s in 'xyz'}


################ Interface functions

def discretize(hamiltonian, discrete_coordinates=None, *, grid_spacing=1,
               substitutions=None, verbose=False):
    """Construct a tight-binding model from a continuum Hamiltonian.

    Parameters
    ----------
    hamiltonian : sympy.Expr or sympy.Matrix, or string
        Symbolic representation of a continous Hamiltonian. When providing
        a sympy expression, it is recommended to use operators
        defined in `~kwant.continuum.momentum_operators` in order to provide
        proper commutation properties. If a string is provided it will be
        converted to a sympy expression using `kwant.continuum.sympify`.
    discrete_coordinates : sequence of strings, or ``None`` (default)
        Set of coordinates for which momentum operators will be treated as
        differential operators. For example ``discrete_coordinates=('x', 'y')``.
        If not provided they will be obtained from the input hamiltonian by
        reading present coordinates and momentum operators. Order of discrete
        coordinates is always lexical, even if provided otherwise.
    grid_spacing : int or float, default: 1
        Grid spacing for the template Builder.
    substitutions : dict, defaults to empty
        A namespace of substitutions to be performed on the input
        ``hamiltonian``. Values can be either strings or ``sympy`` objects.
        It can be used to simplify input of matrices or alternate input before
        proceeding further. For example:
        ``substitutions={'k': 'k_x + I * k_y'}`` or
        ``substitutions={'s_z': [[1, 0], [0, -1]]}``.
    verbose : bool, default: False
        If ``True`` additional information will be printed.

    Returns
    -------
    `~kwant.builder.Builder` with translational symmetry,
    which can be used as a template.
    """

    tb, coords = discretize_symbolic(hamiltonian, discrete_coordinates,
                                substitutions=substitutions, verbose=verbose)

    return build_discretized(tb, coords, grid_spacing=grid_spacing,
                             substitutions=substitutions, verbose=verbose)


def discretize_symbolic(hamiltonian, discrete_coordinates=None, *,
                        substitutions=None, verbose=False):
    """Discretize a continuous Hamiltonian into a tight-binding representation.

    Parameters
    ----------
    hamiltonian : sympy.Expr or sympy.Matrix, or string
        Symbolic representation of a continous Hamiltonian. When providing
        a sympy expression, it is recommended to use operators
        defined in `~kwant.continuum.momentum_operators` in order to provide
        proper commutation properties. If a string is provided it will be
        converted to a sympy expression using `kwant.continuum.sympify`.
    discrete_coordinates : sequence of strings, or ``None`` (default)
        Set of coordinates for which momentum operators will be treated as
        differential operators. For example ``discrete_coordinates=('x', 'y')``.
        If not provided they will be obtained from the input hamiltonian by
        reading present coordinates and momentum operators. Order of discrete
        coordinates is always lexical, even if provided otherwise.
    substitutions : dict, defaults to empty
        A namespace of substitutions to be performed on the input
        ``hamiltonian``. Values can be either strings or ``sympy`` objects.
        It can be used to simplify input of matrices or alternate input before
        proceeding further. For example:
        ``substitutions={'k': 'k_x + I * k_y'}`` or
        ``substitutions={'s_z': [[1, 0], [0, -1]]}``.
    verbose : bool, default: False
        If ``True`` additional information will be printed.

    Returns
    -------
    (discrete_hamiltonian, discrete_coordinates)
        discrete_hamiltonian: dict
            Keys are tuples of integers; the offsets of the hoppings
            ((0, 0, 0) for the onsite). Values are symbolic expressions
            for the hoppings/onsite.
        discrete_coordinates : sequence of strings
            The coordinates that have been discretized.
    """
    hamiltonian = sympify(hamiltonian, substitutions)

    atoms_names = [s.name for s in hamiltonian.atoms(sympy.Symbol)]
    if any( s == 'a' for s in atoms_names):
        raise TypeError("'a' is a symbol used internally to represent "
                        "grid spacing; please use a different symbol.")

    hamiltonian = sympy.expand(hamiltonian)
    if discrete_coordinates is None:
        used_momenta = set(_momentum_operators) & set(atoms_names)
        discrete_coordinates = {k[-1] for k in used_momenta}
    else:
        discrete_coordinates = set(discrete_coordinates)
        if not discrete_coordinates <= set('xyz'):
            msg = "Discrete coordinates must only contain 'x', 'y', or 'z'."
            raise ValueError(msg)

    discrete_coordinates = sorted(discrete_coordinates)

    if len(discrete_coordinates) == 0:
        raise ValueError("Failed to read any discrete coordinates. This is "
                         "probably due to a lack of momentum operators in "
                         "your input. You can use the 'discrete_coordinates'"
                         "parameter to provide them.")

    if verbose:
        print('Discrete coordinates set to: ',
              discrete_coordinates, end='\n\n')

    onsite_zeros = (0,) * len(discrete_coordinates)

    if not isinstance(hamiltonian, sympy.matrices.MatrixBase):
        hamiltonian = sympy.Matrix([hamiltonian])
        _input_format = 'expression'
    else:
        _input_format = 'matrix'

    shape = hamiltonian.shape
    tb = defaultdict(lambda: sympy.zeros(*shape))
    tb[onsite_zeros] = sympy.zeros(*shape)

    for (i, j), expression in np.ndenumerate(hamiltonian):
        hoppings = _discretize_expression(expression, discrete_coordinates)

        for offset, hop in hoppings.items():
            tb[offset][i, j] += hop

    # do not include Hermitian conjugates of hoppings
    wanted_hoppings = sorted(list(tb))[len(list(tb)) // 2:]
    tb = {k: v for k, v in tb.items() if k in wanted_hoppings}

    if _input_format == 'expression':
        tb = {k: v[0, 0] for k, v in tb.items()}

    return tb, discrete_coordinates


def build_discretized(tb_hamiltonian, discrete_coordinates, *,
                      grid_spacing=1, substitutions=None, verbose=False):
    """Create a template Builder from a symbolic tight-binding Hamiltonian.

    Parameters
    ----------
    tb_hamiltonian : dict
        Keys are tuples of integers: the offsets of the hoppings
        ((0, 0, 0) for the onsite). Values are symbolic expressions
        for the hoppings/onsite or expressions that can by sympified using
        `kwant.continuum.sympify`.
    discrete_coordinates : sequence of strings
        Set of coordinates for which momentum operators will be treated as
        differential operators. For example ``discrete_coordinates=('x', 'y')``.
    grid_spacing : int or float, default: 1
        Grid spacing for the template Builder.
    substitutions : dict, defaults to empty
        A namespace of substitutions to be performed on the values of input
        ``tb_hamiltonian``. Values can be either strings or ``sympy`` objects.
        It can be used to simplify input of matrices or alternate input before
        proceeding further. For example:
        ``substitutions={'k': 'k_x + I * k_y'}`` or
        ``substitutions={'s_z': [[1, 0], [0, -1]]}``.
    verbose : bool, default: False
        If ``True`` additional information will be printed.

    Returns
    -------
    `~kwant.builder.Builder` with translational symmetry,
    which can be used as a template.
    """
    if len(discrete_coordinates) == 0:
        raise ValueError('Discrete coordinates cannot be empty.')

    for k, v in tb_hamiltonian.items():
        tb_hamiltonian[k] = sympify(v, substitutions)

    discrete_coordinates = sorted(discrete_coordinates)

    tb = {}
    first = True
    for n, (offset, hopping) in enumerate(tb_hamiltonian.items()):
        if verbose:
            if first:
                first = False
            else:
                print('\n')
            print("Function generated for {}:".format(offset))

        onsite = all(i == 0 for i in offset)

        if onsite:
            name = 'onsite'
        else:
            name = 'hopping_{}'.format(n)

        tb[offset] = _value_function(hopping, discrete_coordinates,
                                     grid_spacing, onsite, name,
                                     verbose=verbose)

    dim = len(discrete_coordinates)
    onsite_zeros = (0,) * dim

    prim_vecs = grid_spacing * np.eye(dim)
    random_element = next(iter(tb_hamiltonian.values()))
    norbs = (1 if isinstance(random_element, sympy.Expr)
             else random_element.shape[0])
    lattice = Monatomic(prim_vecs, norbs=norbs)

    onsite = tb.pop(onsite_zeros)
    # 'delta' parameter to HoppingKind is the negative of the 'hopping offset'
    hoppings = {HoppingKind(tuple(-i for i in d), lattice): val
                for d, val in tb.items()}

    syst = Builder(TranslationalSymmetry(*prim_vecs))
    syst[lattice(*onsite_zeros)] = onsite
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


def _discretize_summand(summand, discrete_coordinates):
    """Discretize a product of factors.

    Parameters
    ----------
    summand : sympy.Expr
    discrete_coordinates : sequence of strings
        Must be a subset of ``{'x', 'y', 'z'}``.

    Returns
    -------
    sympy.Expr
    """
    assert not isinstance(summand, sympy.Add), "Input should be one summand."
    momenta = ['k_{}'.format(s) for s in discrete_coordinates]

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

    return sympy.expand(result)


def _discretize_expression(expression, discrete_coordinates):
    """Discretize an expression into a discrete (tight-binding) representation.

    Parameters
    ----------
    expression : sympy.Expr
    discrete_coordinates : sequence of strings
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
        for c, arg in zip(discrete_coordinates, wf.args):
            coefficients = arg.as_coefficients_dict()
            assert coefficients[_position_operators[c]] == 1

            ai = _displacements[c]
            offset.append(coefficients.pop(ai, 0))
        return tuple(offset)

    def _extract_hoppings(expr):
        """Read hoppings and perform shortening operation."""
        expr = sympy.expand(expr)
        summands = expr.args if expr.func == sympy.Add else [expr]

        offset = [_read_offset(s.args[-1]) for s in summands]
        coeffs = [sympy.Mul(*s.args[:-1]) for s in summands]
        offset = np.array(offset, dtype=int)
        # rescale the offsets for each coordinate by their greatest
        # common divisor across the summands. e.g:
        # wf(x+2h) + wf(x+4h) --> wf(x+h) + wf(x+2h) and a_x //= 2
        subs = {}
        for i, xi in enumerate(discrete_coordinates):
            factor = int(gcd(*offset[:, i]))
            if factor < 1:
                continue
            offset[:, i] //= factor
            subs[_displacements[xi]] = sympy.symbols('a') / factor
        # apply the rescaling to the hoppings
        output = defaultdict(lambda: sympy.Integer(0))
        for n, c in enumerate(coeffs):
            output[tuple(offset[n].tolist())] += c.subs(subs)
        return dict(output)

    # if there are no momenta in the expression, then it is an onsite
    atoms_names = [s.name for s in expression.atoms(sympy.Symbol)]
    if not set(_momentum_operators) & set(atoms_names):
        n = len(discrete_coordinates)
        return {(0,) * n: expression}

    # make sure we have list of summands
    summands = expression.as_ordered_terms()

    # discretize every summand
    coordinates = tuple(_position_operators[s] for s in discrete_coordinates)
    wf = _wf(*coordinates)

    discrete_expression = defaultdict(int)
    for summand in summands:
        summand = _discretize_summand(summand * wf, discrete_coordinates)
        hops = _extract_hoppings(summand)
        for k, v in hops.items():
            discrete_expression[k] += v

    return dict(discrete_expression)


################ string processing

class _NumericPrinter(LambdaPrinter):

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


def _return_string(expr, discrete_coordinates):
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

    blacklisted = set(discrete_coordinates) | {'site', 'site1', 'site2'}
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
        mons = monomials(expr, *expr.atoms(sympy.Symbol))
        mons = {k: cache(v) for k, v in mons.items()}
        mons = ["{} * {}".format(_print_sympy(k), _print_sympy(v))
                for k, v in mons.items()]
        output = " + ".join(mons)
    else:
        output = _print_sympy(expr)

    return 'return {}'.format(output), map_func_calls, const_symbols, _cache


def _assign_symbols(map_func_calls, grid_spacing,
                    discrete_coordinates, onsite):
    """Generate a series of assignments.

    Parameters
    ----------
    map_func_calls : dict
        mapping of function calls to assigned constants.
    grid_spacing : int or float
        Used to get site.pos from site.tag
    discrete_coordinates : sequence of strings
        If left as None coordinates will not be read from a site.
    onsite : bool
        True if function is called for onsite, false for hoppings

    Returns
    -------
    assignments : list of strings
        List of lines used for including in a function.
    """
    lines = []

    if discrete_coordinates:
        site = 'site' if onsite else 'site1'
        args = ', '.join(discrete_coordinates), str(grid_spacing), site
        lines.append('({}, ) = {} * {}.tag'.format(*args))

    for k, v in map_func_calls.items():
        lines.append("{} = {}".format(v, _print_sympy(k)))

    return lines


def _value_function(expr, discrete_coordinates, grid_spacing, onsite,
                    name='_anonymous_func', verbose=False):
    """Generate a numeric function from a sympy expression.

    Parameters
    ----------
    expr : sympy.Expr or sympy.matrix
        Expr that from which value function will be generated.
    discrete_coordinates : sequence of strings
        List of coodinates present in the system.
    grid_spacing : int or float
        Lattice spacing of the system
    verbose : bool, default: False
        If True, the function body is printed.

    Returns
    -------
    numerical function that can be used with Kwant.
    """

    expr = expr.subs({sympy.Symbol('a'): grid_spacing})
    return_string, map_func_calls, const_symbols, _cache = \
        _return_string(expr, discrete_coordinates=discrete_coordinates)

    # first check if value function needs to read coordinates
    atoms_names = {s.name for s in expr.atoms(sympy.Symbol)}
    if not set(discrete_coordinates) & atoms_names:
        discrete_coordinates = None

    # constants and functions in the sympy input will be passed
    # as keyword-only arguments to the value function
    required_kwargs = set.union({s.name for s in const_symbols},
                                {str(k.func) for k in map_func_calls})
    required_kwargs = ', '.join(sorted(required_kwargs))

    if (not required_kwargs) and (discrete_coordinates is None):
        # we can just use a constant value instead of a value function
        if isinstance(expr, sympy.MatrixBase):
            output = ta.array(expr.tolist(), complex)
        else:
            output = complex(expr)

        if verbose:
            print("\n{}".format(output))

        return output

    lines = _assign_symbols(map_func_calls, onsite=onsite,
                            grid_spacing=grid_spacing,
                            discrete_coordinates=discrete_coordinates)

    lines.append(return_string)

    separator = '\n' + 4 * ' '
    # 'site_string' is tightly coupled to the symbols used in '_assign_symbol'
    site_string = 'site' if onsite else 'site1, site2'
    if required_kwargs:
        header_str = 'def {}({}, *, {}):'
        header = header_str.format(name, site_string, required_kwargs)
    else:
        header = 'def {}({}):'.format(name, site_string)
    func_code = separator.join([header] + list(lines))

    namespace = {'pi': np.pi}
    namespace.update(_cache)

    if verbose:
        for k, v in _cache.items():
            print("\n{} = (\n{})".format(k, repr(np.array(v))))
        print('\n' + func_code)

    exec(func_code, namespace)
    f = namespace[name]

    return f
