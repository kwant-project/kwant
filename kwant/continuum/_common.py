import numpy as np

import sympy
from sympy.core.function import AppliedUndef
from sympy.core.sympify import converter
from sympy.abc import _clash

import functools
import inspect

from collections import defaultdict
from operator import mul

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


def lambdify(hamiltonian, *, substitutions=None):
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
        A namespace to be passed to ``kwant.continuum.sympify`` when
        ``hamiltonian`` is a string. Could be used to simplify matrix input:
        ``sympify('k_x**2 * s_z', substitutions={'s_z': [[1, 0], [0, -1]]})``.
    """
    expr = hamiltonian
    if not isinstance(expr, (sympy.Expr, sympy.matrices.MatrixBase)):
        expr = sympify(expr, substitutions)

    args = [s.name for s in expr.atoms(sympy.Symbol)]
    args += [str(f.func) for f in expr.atoms(AppliedUndef, sympy.Function)]

    f = sympy.lambdify(sorted(args), expr)

    sig = inspect.signature(f)
    pars = list(sig.parameters.values())
    pars = [p.replace(kind=inspect.Parameter.KEYWORD_ONLY) for p in pars]
    f.__signature__ = inspect.Signature(pars)
    return f


def sympify(string, substitutions=None, **kwargs):
    """Return sympified object with respect to kwant-specific rules.

    This is a modification of ``sympy.sympify`` to apply kwant-specific rules,
    which includes preservation of proper commutation relations between
    position and momentum operators, and array to matrix casting.

    Parameters
    ----------
    string : string
        String representation of a Hamiltonian. Momenta must be defined as
        ``k_i`` where ``i`` stands for ``x``, ``y``, or ``z``. All present
        momenta and coordinates will be interpreted as non commutative.
    substitutions : dict, defaults to empty
        A namespace to be passed internally to ``sympy.sympify``.
        Could be used to simplify matrix input:
        ``sympify('k_x**2 * s_z', substitutions={'s_z': [[1, 0], [0, -1]]})``.
    **kwargs
        Additional arguments that will be passed to ``sympy.sympify``.

    Example
    -------
        >>> from kwant.continuum import sympify
        >>> hamiltonian = sympify('k_x * A(x) * k_x + V(x)')
    """
    stored_value = None

    if substitutions is None:
        substitutions = {}

    substitutions.update(_clash)

    try:
        stored_value = converter.pop(list, None)
        converter[list] = lambda x: sympy.Matrix(x)
        substitutions = {k: (sympy.sympify(v, locals=substitutions, **kwargs)
                      if isinstance(v, (list, str)) else v)
                  for k, v in substitutions.items()}
        hamiltonian = sympy.sympify(string, locals=substitutions, **kwargs)
        hamiltonian = sympy.sympify(hamiltonian, **kwargs)
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


def expression_monomials(expression, *gens):
    """Parse ``expression`` into monomials in the symbols in ``gens``.

    Example
    -------
        >>> expr = A * (x**2 + y) + B * x + C
        >>> expression_monomials(expr, x, y)
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


def matrix_monomials(matrix, *gens):
    output = defaultdict(lambda: sympy.zeros(*matrix.shape))
    for (i, j), expression in np.ndenumerate(matrix):
        summands = expression_monomials(expression, *gens)
        for key, val in summands.items():
            output[key][i, j] += val

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
