# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import keyword
from collections import defaultdict

import numpy as np

import sympy
import sympy.abc
import sympy.physics.quantum
from sympy.core.function import AppliedUndef
from sympy.core.sympify import converter
from sympy.core.core import all_classes as sympy_classes
from sympy.physics.matrices import msigma as _msigma

import warnings

from .._common import reraise_warnings

# TODO: remove when sympy correctly includes MutableDenseMatrix (lol).
sympy_classes = set(sympy_classes) | {sympy.MutableDenseMatrix}

momentum_operators = sympy.symbols('k_x k_y k_z', commutative=False)
position_operators = sympy.symbols('x y z', commutative=False)

pauli = [sympy.eye(2), _msigma(1), _msigma(2), _msigma(3)]

extra_ns = sympy.abc._clash.copy()
extra_ns.update({s.name: s for s in momentum_operators})
extra_ns.update({s.name: s for s in position_operators})
extra_ns.update({'kron': sympy.physics.quantum.TensorProduct,
                 'eye': sympy.eye, 'identity': sympy.eye})
extra_ns.update({'sigma_{}'.format(c): p for c, p in zip('0xyz', pauli)})


# workaroud for https://github.com/sympy/sympy/issues/12060
del extra_ns['I']
del extra_ns['pi']


################  Helpers to handle sympy

def lambdify(expr, locals=None):
    """Return a callable object for computing a continuum Hamiltonian.

    .. warning::
        This function uses ``eval`` (because it calls ``sympy.sympify``), and
        thus should not be used on unsanitized input.

    If necessary, the given expression is sympified using
    `kwant.continuum.sympify`.  It is then converted into a callable object.

    Parameters
    ----------
    expr : str or SymPy expression
        Expression to be converted into a callable object
    locals : dict or ``None`` (default)
        Additional definitions for `~kwant.continuum.sympify`.

    Examples
    --------
    >>> f = lambdify('a + b', locals={'b': 'b + c'})
    >>> f(1, 3, 5)
    9

    >>> ns = {'sigma_plus': [[0, 2], [0, 0]]}
    >>> f = lambdify('k_x**2 * sigma_plus', ns)
    >>> f(0.25)
    array([[ 0.   ,  0.125],
           [ 0.   ,  0.   ]])
    """
    with reraise_warnings(level=4):
        expr = sympify(expr, locals)

    args = [s.name for s in expr.atoms(sympy.Symbol)]
    args += [str(f.func) for f in expr.atoms(AppliedUndef, sympy.Function)]

    return sympy.lambdify(sorted(args), expr)


def sympify(expr, locals=None):
    """Sympify object using special rules for Hamiltonians.

    If `'expr`` is already a type that SymPy understands, it will do nothing
    but return that value. Note that ``locals`` will not be used in this
    situation.

    Otherwise, it is sympified by ``sympy.sympify`` with a modified namespace
    such that

    * the position operators "x", "y" or "z" and momentum operators "k_x",
      "k_y", and "k_z" do not commute,
    * all single-letter identifiers and names of Greek letters (e.g. "pi" or
      "gamma") are treated as symbols,
    * "kron" corresponds to ``sympy.physics.quantum.TensorProduct``, and
      "identity" to ``sympy.eye``,
    * "sigma_0", "sigma_x", "sigma_y", "sigma_z" are the Pauli matrices.

    In addition, Python list literals are interpreted as SymPy matrices.

    .. warning::
        This function uses ``eval`` (because it calls ``sympy.sympify``), and
        thus should not be used on unsanitized input.

    Parameters
    ----------
    expr : str or SymPy expression
        Expression to be converted to a SymPy object.
    locals : dict or ``None`` (default)
        Additional entries for the namespace under which `expr` is sympified.
        The keys must be valid Python variable names.  The values may be
        strings, since they are all are sent through `continuum.sympify`
        themselves before use.  (Note that this is a difference to how
        ``sympy.sympify`` behaves.)

        .. note::
            When a value of `locals` is already a SymPy object, it is used
            as-is, and the caller is responsible to set the commutativity of
            its symbols appropriately.  This possible source of errors is
            demonstrated in the last example below.

    Returns
    -------
    result : SymPy object

    Examples
    --------
        >>> sympify('k_x * A(x) * k_x + V(x)')
        k_x*A(x)*k_x + V(x)     # as valid sympy object

        >>> sympify('k_x**2 + V', locals={'V': 'V_0 + V(x)'})
        k_x**2 + V(x) + V_0

        >>> ns = {'sigma_plus': [[0, 2], [0, 0]]}
        >>> sympify('k_x**2 * sigma_plus', ns)
        Matrix([
        [0, 2*k_x**2],
        [0,        0]])

        >>> sympify('k_x * A(c) * k_x', locals={'c': 'x'})
        k_x*A(x)*k_x
        >>> sympify('k_x * A(c) * k_x', locals={'c': sympy.Symbol('x')})
        A(x)*k_x**2

    """
    stored_value = None

    # if ``expr`` is already a ``sympy`` object we may terminate a code path
    if isinstance(expr, tuple(sympy_classes)):
        if locals:
            warnings.warn('Input expression is already SymPy object: '
                          '"locals" will not be used.',
                          RuntimeWarning,
                          stacklevel=2)

        # We assume that all present functions, like "sin", "cos", will be
        # provided by user during the final evaluation through "params".
        # Therefore we make sure they are defined as AppliedUndef, not built-in
        # sympy types.
        subs = {r: sympy.Function(str(r.func))(*r.args)
                for r in expr.atoms(sympy.Function)}

        return expr.subs(subs)

    # if ``expr`` is not a "sympy" then we proceed with sympifying process
    if locals is None:
        locals = {}

    for k in locals:
        if (not isinstance(k, str)
            or not k.isidentifier() or keyword.iskeyword(k)):
            raise ValueError(
                "Invalid key in 'locals': {}\nKeys must be "
                "identifiers and may not be keywords".format(repr(k)))

    # sympify values of locals before updating it with extra_ns
    # Cast numpy array values in locals to sympy matrices to make sure they have
    # correct format
    locals = {k: (sympy.Matrix(v) if isinstance(v, np.ndarray) else sympify(v))
              for k, v in locals.items()}

    for k, v in extra_ns.items():
        locals.setdefault(k, v)
    try:
        stored_value = converter.pop(list, None)
        converter[list] = lambda x: sympy.Matrix(x)
        hamiltonian = sympy.sympify(expr, locals=locals)
        # if input is for example ``[[k_x * A(x) * k_x]]`` after the first
        # sympify we are getting list of sympy objects, so we call sympify
        # second time to obtain ``sympy`` matrices.
        hamiltonian = sympy.sympify(hamiltonian)
    finally:
        if stored_value is not None:
            converter[list] = stored_value
        else:
            del converter[list]

    return hamiltonian


def make_commutative(expr, *symbols):
    """Make sure that specified symbols are defined as commutative.

    Parameters
    ----------
    expr: sympy.Expr or sympy.Matrix
    symbols: sequace of symbols
        Set of symbols that are requiered to be commutative. It doesn't matter
        of symbol is provided as commutative or not.

    Returns
    -------
    input expression with all specified symbols changed to commutative.
    """
    symbols = [sympy.Symbol(s.name, commutative=False) for s in symbols]
    expr = expr.subs({s: sympy.Symbol(s.name) for s in symbols})
    return expr


def monomials(expr, gens=None):
    """Parse ``expr`` into monomials in the symbols in ``gens``.

    Parameters
    ----------
    expr: sympy.Expr or sympy.Matrix
        Sympy expression to be parsed into monomials.
    gens: sequence of sympy.Symbol objects or strings (optional)
        Generators of monomials. If unset it will default to all
        symbols used in ``expr``.

    Returns
    -------
    dictionary (generator: monomial)

    Examples
    --------
        >>> expr = kwant.continuum.sympify("A * (x**2 + y) + B * x + C")
        >>> monomials(expr, gens=('x', 'y'))
        {1: C, x: B, x**2: A, y: A}
    """
    if gens is None:
        gens = expr.atoms(sympy.Symbol)
    else:
        gens = [sympify(g) for g in gens]

    if not isinstance(expr, sympy.MatrixBase):
        return _expression_monomials(expr, gens)
    else:
        output = defaultdict(lambda: sympy.zeros(*expr.shape))
        for (i, j), e in np.ndenumerate(expr):
            mons = _expression_monomials(e, gens)
            for key, val in mons.items():
                output[key][i, j] += val
        return dict(output)


def _expression_monomials(expr, gens):
    """Parse ``expr`` into monomials in the symbols in ``gens``.

    Parameters
    ----------
    expr: sympy.Expr
        Sympy expr to be parsed.
    gens: sequence of sympy.Symbol
        Generators of monomials.

    Returns
    -------
    dictionary (generator: monomial)
    """
    expr = sympy.expand(expr)
    output = defaultdict(lambda: sympy.Integer(0))
    for summand in expr.as_ordered_terms():
        key = []
        val = []
        for factor in summand.as_ordered_factors():
            symbol, exponent = factor.as_base_exp()
            if symbol in gens:
                key.append(factor)
            else:
                val.append(factor)
        output[sympy.Mul(*key)] += sympy.Mul(*val)

    return dict(output)


################ general help functions

def gcd(*args):
    if len(args) == 1:
        return args[0]

    L = list(args)

    while len(L) > 1:
        a = L[len(L) - 2]
        b = L[len(L) - 1]
        L = L[:len(L) - 2]

        while a:
            a, b = b%a, a

        L.append(b)

    return abs(b)
