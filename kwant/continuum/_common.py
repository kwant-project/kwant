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
from sympy.core.function import AppliedUndef
from sympy.core.sympify import converter
from sympy.abc import _clash
from sympy.physics.matrices import msigma as _msigma
from sympy.physics.quantum import TensorProduct


momentum_operators = sympy.symbols('k_x k_y k_z', commutative=False)
position_operators = sympy.symbols('x y z', commutative=False)

msigma = lambda i: sympy.eye(2) if i == 0 else _msigma(i)
pauli = (msigma(0), msigma(1), msigma(2), msigma(3))

_clash = _clash.copy()
_clash.update({s.name: s for s in momentum_operators})
_clash.update({s.name: s for s in position_operators})
_clash.update({'kron': TensorProduct, 'eye': sympy.eye})
_clash.update({'sigma_{}'.format(c): p for c, p in zip('0xyz', pauli)})


# workaroud for https://github.com/sympy/sympy/issues/12060
del _clash['I']
del _clash['pi']


################  Various helpers to handle sympy


def lambdify(hamiltonian, substitutions=None):
    """Return a callable object for computing continuum Hamiltonian.

    Parameters
    ----------
    hamiltonian : sympy.Expr or sympy.Matrix, or string
        Symbolic representation of a continous Hamiltonian. When providing
        a sympy expression, it is recommended to use operators
        defined in `~kwant.continuum.momentum_operators` in order to provide
        proper commutation properties. If a string is provided it will be
        converted to a sympy expression using `kwant.continuum.sympify`.
    substitutions : dict, defaults to empty
        A namespace of substitutions to be performed on the input
        ``hamiltonian``. It can be used to simplify input of matrices or
        alternate input before proceeding further. Please see examples below.

    Example:
    --------
        >>> f = kwant.continuum.lambdify('a + b', substitutions={'b': 'b + c'})
        >>> f(1, 3, 5)
        9

        >>> subs = {'s_z': [[1, 0], [0, -1]]}
        >>> f = kwant.continuum.lambdify('k_z**2 * s_z', substitutions=subs)
        >>> f(.25)
        array([[ 0.0625,  0.    ],
               [ 0.    , -0.0625]])
    """
    expr = sympify(hamiltonian, substitutions)

    args = [s.name for s in expr.atoms(sympy.Symbol)]
    args += [str(f.func) for f in expr.atoms(AppliedUndef, sympy.Function)]

    f = sympy.lambdify(sorted(args), expr)

    sig = inspect.signature(f)
    pars = list(sig.parameters.values())
    pars = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in pars]
    f.__signature__ = inspect.Signature(pars)
    return f


def sympify(e, substitutions=None):
    """Return sympified object with respect to kwant-specific rules.

    This is a modification of ``sympy.sympify`` to apply kwant-specific rules,
    which includes preservation of proper commutation relations between
    position and momentum operators, and array to matrix casting.

    Parameters
    ----------
    e : arbitrary expression
        An expression that will be converted to a sympy object.
        Momenta must be defined as ``k_i`` where ``i`` stands for ``x``, ``y``,
        or ``z``. All present momenta and coordinates will be interpreted
        as non commutative.
    substitutions : dict, defaults to empty
        A namespace of substitutions to be performed on the input ``e``.
        It works in a similar way to ``locals`` argument in ``sympy.sympify``
        but extends its functionality to i.e. simplify matrix input:
        ``sympify('k_x**2 * s_z', substitutions={'s_z': [[1, 0], [0, -1]]})``.
        Keys should be strings, unless ``e`` is already a sympy object.
        Values can be strings or ``sympy`` objects.

    Example
    -------
        >>> from kwant.continuum import sympify
        >>> sympify('k_x * A(x) * k_x + V(x)')
        k_x*A(x)*k_x + V(x)     # as valid sympy object

        or

        >>> from kwant.continuum import sympify
        >>> sympify('k_x**2 + V', substitutions={'V': 'V_0 + V(x)'})
        k_x**2 + V(x) + V_0
    """
    stored_value = None
    sympified_types = (sympy.Expr, sympy.matrices.MatrixBase)
    if substitutions is None:
        substitutions = {}

    # if ``e`` is already a ``sympy`` object we make use of ``substitutions``
    # and terminate a code path.
    if isinstance(e, sympified_types):
        subs = {sympify(k): sympify(v) for k, v in substitutions.items()}
        return e.subs(subs)

    # if ``e`` is not a sympified type then we proceed with sympifying process,
    # we expect all keys in ``substitutions`` to be strings at this moment.
    if not all(isinstance(k, str) for k in substitutions):
        raise ValueError("If 'e' is not already a sympy object ",
                         "then keys of 'substitutions' must be strings.")

    # sympify values of substitutions before updating it with _clash
    substitutions = {k: (sympify(v) if not isinstance(v, sympified_types)
                         else v)
                     for k, v in substitutions.items()}

    substitutions.update({s: v for s, v in _clash.items()
                          if s not in substitutions})
    try:
        stored_value = converter.pop(list, None)
        converter[list] = lambda x: sympy.Matrix(x)
        hamiltonian = sympy.sympify(e, locals=substitutions)
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
    if expression.atoms(AppliedUndef):
        raise NotImplementedError('Getting monomials of expressions containing '
                                  'functions is not implemented.')

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
        val = summand.xreplace({g: 1 for g in gens})

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
