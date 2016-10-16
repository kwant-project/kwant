from __future__ import print_function, division

import sympy
from sympy.utilities.lambdify import lambdastr
from sympy.printing.lambdarepr import LambdaPrinter
from sympy.core.function import AppliedUndef


class NumericPrinter(LambdaPrinter):
    def _print_ImaginaryUnit(self, expr):
        return "1.j"


def offset_to_direction(discrete_hamiltonian, discrete_coordinates):
    """Translate hopping keys from offsets to directions.

    Parameters:
    -----------
    discrete_hamiltonian: dict
        Discretized hamiltonian, key should be an offset of a hopping and value
        corresponds to symbolic hopping.
    discrete_coordinates: set
        Set of discrete coordinates.

    Returns:
    --------
    discrete_hamiltonian: dict
        Discretized hamiltonian, key is a direction of a hopping and value
        corresponds to symbolic hopping.

    Note:
    -----
    Coordinates (x,y,z) in output stands for a position of a source of the
    hopping.
    """
    coordinates = sorted(list(discrete_coordinates))
    coordinates = [sympy.Symbol(s, commutative=False) for s in coordinates]
    a = sympy.Symbol('a')

    onsite_zeros = (0,)*len(discrete_coordinates)
    output = {onsite_zeros: discrete_hamiltonian.pop(onsite_zeros)}
    for offset, hopping in discrete_hamiltonian.items():
        direction = tuple(-c for c in offset)
        subs = {c: c + d*a for c, d in zip(coordinates, direction)}
        output[direction] = hopping.subs(subs)

    return output

# ************ Making kwant functions ***********
def make_return_string(expr):
    """Process a sympy expression into an evaluatable Python return statement.

    Parameters:
    -----------
    expr : sympy.Expr instance

    Returns:
    --------
    output : string
        A return string that can be used to assemble a Kwant value function.
    func_symbols : set of sympy.Symbol instances
        All space dependent functions that appear in the expression.
    const_symbols : set of sympy.Symbol instances
        All constants that appear in the expression.
    """
    func_symbols = {sympy.Symbol(i.func.__name__) for i in
                    expr.atoms(AppliedUndef)}

    free_symbols = {i for i in expr.free_symbols if i.name not in ['x', 'y', 'z']}
    const_symbols = free_symbols - func_symbols

    expr = expr.subs(sympy.I, sympy.Symbol('1.j')) # quick hack
    output = lambdastr((), expr, printer=NumericPrinter)[len('lambda : '):]
    output = output.replace('MutableDenseMatrix', 'np.array')
    output = output.replace('ImmutableMatrix', 'np.array')

    return 'return {}'.format(output), func_symbols, const_symbols


def assign_symbols(func_symbols, const_symbols, discrete_coordinates,
                   onsite=True):
    """Generate a series of assingments defining a set of symbols.

    Parameters:
    -----------
    func_symbols : set of sympy.Symbol instances
        All space dependent functions that appear in the expression.
    const_symbols : set of sympy.Symbol instances
        All constants that appear in the expression.

    Returns:
    --------
    assignments : list of strings
        List of lines used for including in a function.

    Notes:
    where A, B, C are all the free symbols plus the symbols that appear on the
    ------
    The resulting lines begin with a coordinates assignment of form
    `x,y,z = site.pos` when onsite=True, or
    `x,y,z = site2.pos` when onsite=False

    followed by two lines of form
    `A, B, C = p.A, p.B, p.C`
    `f, g, h = p.f, p.g, p.h`
    where A, B, C are symbols representing constants and f, g, h are symbols
    representing functions. Separation of constant and func symbols is probably
    not necessary but I leave it for now, just in case.
    """
    lines = []
    func_names = [i.name for i in func_symbols]
    const_names = [i.name for i in const_symbols]

    if func_names:
        lines.insert(0, ', '.join(func_names) + ' = p.' +
                     ', p.'.join(func_names))

    if const_names:
        lines.insert(0, ', '.join(const_names) + ' = p.' +
                     ', p.'.join(const_names))

    if onsite:
        site = 'site'
    else:
        site = 'site2'

    names = sorted(list(discrete_coordinates))
    lines.insert(0, '({}, ) = {}.pos'.format(', '.join(names), site))

    return lines


def value_function(content, name='_anonymous_func', onsite=True, verbose=False):
    """Generate a Kwant value function from a list of lines containing its body.

    Parameters:
    -----------
    content : list of lines
        Lines forming the body of the function.
    name : string
        Function name (not important).
    onsite : bool
        If True, the function call signature will be `f(site, p)`, otherwise
        `f(site1, site2, p)`.
    verbose : bool
        Whether the function bodies should be printed.

    Returns:
    --------
    f : function
        The function defined in a separated namespace.
    """
    if not content[-1].startswith('return'):
        raise ValueError('The function does not end with a return statement')

    separator = '\n' + 4 * ' '
    site_string = 'site' if onsite else 'site1, site2'
    header = 'def {0}({1}, p):'.format(name, site_string)
    func_code = separator.join([header] + list(content))

    namespace = {}
    if verbose:
        print(func_code)
    exec("from __future__ import division", namespace)
    exec("import numpy as np", namespace)
    exec("from numpy import *", namespace)
    exec(func_code, namespace)
    return namespace[name]


def make_kwant_functions(discrete_hamiltonian, discrete_coordinates,
                         verbose=False):
    """Transform discrete hamiltonian into valid kwant functions.

    Parameters:
    -----------
    discrete_hamiltonian: dict
        dict in which key is offset of hopping ((0, 0, 0) for onsite)
        and value is corresponding symbolic hopping (onsite).

    verbose : bool
        Whether the function bodies should be printed.

    discrete_coordinates : tuple/list
        List of discrete coordinates. Must corresponds to offsets in
        discrete_hamiltonian keys.

    Note:
    -----

    """
    dim = len(discrete_coordinates)
    if not all(len(i)==dim for i in list(discrete_hamiltonian.keys())):
        raise ValueError("Dimension of offsets and discrete_coordinates" +
                         "do not match.")

    functions = {}
    for offset, hopping in discrete_hamiltonian.items():
        onsite = True if all(i == 0 for i in offset) else False
        return_string, func_symbols, const_symbols = make_return_string(hopping)
        lines = assign_symbols(func_symbols, const_symbols, onsite=onsite,
                               discrete_coordinates=discrete_coordinates)
        lines.append(return_string)

        if verbose:
            print("Function generated for {}:".format(offset))
            f = value_function(lines, verbose=verbose, onsite=onsite)
            print()
        else:
            f = value_function(lines, verbose=verbose, onsite=onsite)

        functions[offset] = f

    return functions
