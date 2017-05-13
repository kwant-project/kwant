# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import functools
import inspect
from collections import defaultdict
from operator import mul

import numpy as np

import sympy
import sympy.abc
import sympy.physics.quantum
from sympy.core.function import AppliedUndef
from sympy.core.sympify import converter
from sympy.physics.matrices import msigma as _msigma


momentum_operators = sympy.symbols('k_x k_y k_z', commutative=False)
position_operators = sympy.symbols('x y z', commutative=False)

pauli = [sympy.eye(2), _msigma(1), _msigma(2), _msigma(3)]

default_subs = sympy.abc._clash.copy()
default_subs.update({s.name: s for s in momentum_operators})
default_subs.update({s.name: s for s in position_operators})
default_subs.update({'kron': sympy.physics.quantum.TensorProduct,
                     'eye': sympy.eye, 'identity': sympy.eye})
default_subs.update({'sigma_{}'.format(c): p for c, p in zip('0xyz', pauli)})


# workaroud for https://github.com/sympy/sympy/issues/12060
del default_subs['I']
del default_subs['pi']


################  Helpers to handle sympy

def lambdify(expr, subs=None):
    """Return a callable object for computing a continuum Hamiltonian.

    .. warning::
        This function uses ``eval`` (because it calls ``sympy.sympify``), and
        thus should not be used on unsanitized input.

    Parameters
    ----------
    expr : str or SymPy expression
        Expression to be converted into a callable object
    subs : dict or ``None`` (default)
        Namespace of substitutions to be performed on `expr`.

    Example:
    --------
        >>> f = lambdify('a + b', subs={'b': 'b + c'})
        >>> f(1, 3, 5)
        9

        >>> subs = {'sigma_plus': [[0, 2], [0, 0]]}
        >>> f = lambdify('k_x**2 * sigma_plus', subs)
        >>> f(0.25)
        array([[ 0.   ,  0.125],
               [ 0.   ,  0.   ]])
    """
    expr = sympify(expr, subs)

    args = [s.name for s in expr.atoms(sympy.Symbol)]
    args += [str(f.func) for f in expr.atoms(AppliedUndef, sympy.Function)]

    return sympy.lambdify(sorted(args), expr)


def sympify(expr, subs=None):
    """Sympify object using special rules for Hamiltonians.

    If ``expr`` is a string, all keys in ``subs`` must be strings as well. In a
    first step, all values of ``subs`` are sympified.  Then, `subs` is used as
    the ``locals`` argument in an internal call to ``sympy.sympify``.  The
    ``locals`` namespace is pre-populated such that

    * the position operators "x", "y" or "z" and momentum operators "k_x",
      "k_y", and "k_z" do not commute,
    * all single-letter identifiers and names of greek letters (e.g. "pi" or
      "gamma") are treated as symbols,
    * "kron" corresponds to ``sympy.physics.quantum.TensorProduct``, and
      "identity" to ``sympy.eye``,
    * "sigma_0", "sigma_x", "sigma_y", "sigma_z" are the Pauli matrices.

    In addition, Python list literals are interpreted as SymPy matrices.

    .. warning::
        This function uses ``eval`` (because it calls ``sympy.sympify``), and
        thus should not be used on unsanitized input.

    If `expr` is already a SymPy expression or a SymPy matrix, both the keys
    and the values of `subs` are sympified (with the above special rules in
    force) and used to to perform substitution using the ``.subs`` method of
    `expr`.

    .. note::
        Any (part of) argument to this function that gets sympified (for
        example the values of `subs`) may be already a SymPy object.  In this
        case it is left unchanged.  In particular, the commutativity of its
        terms is not altered.  This possibly confusing effect is demonstrated
        in the last example below.

    Parameters
    ----------
    expr : str or SymPy expression
        Expression to be converted to a SymPy object.
    subs : dict or ``None`` (default)
        Namespace of substitutions to be performed on the input `expr`.
        If `expr` is a string, the keys must be strings as well.  Otherwise
        they may be any sympifyable object. The values must be sympifyable
        objects.

    Returns
    -------
    result : SymPy object

    Examples
    --------
        >>> sympify('k_x * A(x) * k_x + V(x)')
        k_x*A(x)*k_x + V(x)     # as valid sympy object

        >>> sympify('k_x**2 + V', subs={'V': 'V_0 + V(x)'})
        k_x**2 + V(x) + V_0

        >>> subs = {'sigma_plus': [[0, 2], [0, 0]]}
        >>> sympify('k_x**2 * sigma_plus', subs)
        Matrix([
        [0, 2*k_x**2],
        [0,        0]])

        >>> sympify('k_x * A(c) * k_x', subs={'c': 'x'})
        k_x*A(x)*k_x
        >>> sympify('k_x * A(c) * k_x', subs={'c': sympy.Symbol('x')})
        A(x)*k_x**2

    """
    stored_value = None
    sympified_types = (sympy.Expr, sympy.matrices.MatrixBase)
    if subs is None:
        subs = {}

    # if ``expr`` is already a ``sympy`` object we make use of ``subs``
    # and terminate a code path.
    if isinstance(expr, sympified_types):
        subs = {sympify(k): sympify(v) for k, v in subs.items()}
        return expr.subs(subs)

    # if ``expr`` is not a sympified type then we proceed with sympifying
    # process, we expect all keys in ``subs`` to be strings at this moment.
    if not all(isinstance(k, str) for k in subs):
        raise ValueError("If 'expr' is not already a sympy object ",
                         "then keys of 'subs' must be strings.")

    # sympify values of subs before updating it with default_subs
    subs = {k: sympify(v) for k, v in subs.items()}
    for k, v in default_subs.items():
        subs.setdefault(k, v)
    try:
        stored_value = converter.pop(list, None)
        converter[list] = lambda x: sympy.Matrix(x)
        hamiltonian = sympy.sympify(expr, locals=subs)
        # if input is for example ``[[k_x * A(x) * k_x]]`` after the first
        # sympify we are getting list of sympy objects, so we call sympify
        # second time to obtain ``sympy`` matrices.
        hamiltonian = sympy.sympify(hamiltonian)
    finally:
        if stored_value is not None:
            converter[list] = stored_value
        else:
            del converter[list]
    return sympy.expand(hamiltonian)


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
    subs = {s: sympy.Symbol(s.name) for s in symbols}
    expr = expr.subs(subs)
    return expr


def monomials(expr, *gens):
    """Parse ``expr`` into monomials in the symbols in ``gens``.

    Parameters
    ----------
    expr: sympy.Expr or sympy.Matrix
        Input expression that will be parsed into monomials.
    gens: sequence of sympy.Symbol objects
        Generators used to separate input ``expr`` into monomials.

    Returns
    -------
    dictionary (generator: monomial)

    Note
    ----
    All generators will be substituted with its commutative version using
    `kwant.continuum.make_commutative`` function.
    """
    if not isinstance(expr, sympy.MatrixBase):
        return _expression_monomials(expr, *gens)
    else:
        output = defaultdict(lambda: sympy.zeros(*expr.shape))
        for (i, j), e in np.ndenumerate(expr):
            mons = _expression_monomials(e, *gens)
            for key, val in mons.items():
                output[key][i, j] += val
        return dict(output)


def _expression_monomials(expression, *gens):
    """Parse ``expression`` into monomials in the symbols in ``gens``.

    Example
    -------
        >>> expr = A * (x**2 + y) + B * x + C
        >>> _expression_monomials(expr, x, y)
        {1: C, x**2: A, y: A, x: B}
    """
    f_args = [f.args for f in expression.atoms(AppliedUndef, sympy.Function)]
    f_args = [i for s in f_args for i in s]

    if set(gens) & set(f_args):
        raise ValueError('Functions in "expression" cannot contain any of '
                         '"gens" as their argument.')

    expression = make_commutative(expression, *gens)
    gens = [make_commutative(g, g) for g in gens]

    expression = sympy.expand(expression)
    summands = expression.as_ordered_terms()

    output = defaultdict(int)
    for summand in summands:
        key = [sympy.Integer(1)]
        if summand in gens:
            key.append(summand)

        elif isinstance(summand, sympy.Pow):
            if summand.args[0] in gens:
                key.append(summand)

        else:
            for arg in summand.args:
                if arg in gens:
                    key.append(arg)
                if isinstance(arg, sympy.Pow):
                    if arg.args[0] in gens:
                        key.append(arg)

        key = functools.reduce(mul, key)
        val = summand.xreplace({g: sympy.S.One for g in gens})

        ### to not create key
        if val != 0:
            output[key] += val

    new_expression = sum(k * v for k, v in output.items())
    assert sympy.expand(expression) == sympy.expand(new_expression)
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
