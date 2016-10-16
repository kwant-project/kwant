from __future__ import print_function, division

import itertools
import sympy
import numpy as np
from collections import defaultdict

from .postprocessing import make_kwant_functions
from .postprocessing import offset_to_direction

from .interpolation import interpolate_tb_hamiltonian

try:
    # normal situation
    from kwant.lattice import Monatomic
except ImportError:
    # probably run on gitlab-ci
    pass

# ************************** Some globals *********************************
wavefunction_name = 'Psi'


# **************** Operation on sympy expressions **************************
def read_coordinates(expression):
    """Read coordinates used in expression.

    This function is used if ``discrete_coordinates`` are not provided by user.

    Parameters:
    -----------
    expression : sympy.Expr or sympy.Matrix instance

    Returns:
    --------
    discrete_coordinates : set of strings
    """
    discrete_coordinates = set()
    for a in expression.atoms(sympy.Symbol):
        if a.name in ['k_x', 'k_y', 'k_z']:
            discrete_coordinates.add(a.name.split('_')[1])

        if a.name in ['x', 'y', 'z']:
            discrete_coordinates.add(a.name)

    return discrete_coordinates

def split_factors(expression, discrete_coordinates):
    """ Split symbolic `expression` for a discretization step.

    Parameters:
    -----------
    expression : sympy.Expr instance
        The expression to be split. It should represents single summand.

    Output:
    -------
    lhs : sympy.Expr instance
        Part of expression standing to the left from operators
        that acts in current discretization step.

    operators: sympy.Expr instance
        Operator that perform discretization in current step.

    rhs : sympy.Expr instance
        Part of expression that is derivated in current step.

    Raises:
    -------
    AssertionError
        if input `expression` is of type ``sympy.Add``
    """
    assert not isinstance(expression, sympy.Add), \
        'Input expression must not be sympy.Add. It should be a single summand.'

    momentum_names = ['k_{}'.format(s) for s in discrete_coordinates]
    momentum_operators = sympy.symbols(momentum_names, commutative=False)
    momentum_operators += sympy.symbols(momentum_names, commutative=True)

    output = {'rhs': [1], 'operator': [1], 'lhs': [1]}

    if isinstance(expression, sympy.Pow):
        base, exponent = expression.args
        if base in momentum_operators:
            output['operator'].append(base)
            output['lhs'].append(sympy.Pow(base, exponent-1))
        else:
            output['rhs'].append(expression)


    elif isinstance(expression, (int, float, sympy.Integer, sympy.Float)):
        output['rhs'].append(expression)

    elif isinstance(expression, (sympy.Symbol, sympy.Function)):
        if expression in momentum_operators:
            output['operator'].append(expression)
        else:
            output['rhs'].append(expression)

    elif isinstance(expression, sympy.Mul):
        iterator = iter(expression.args[::-1])
        for factor in iterator:
            if factor in momentum_operators:
                output['operator'].append(factor)
                break
            elif factor.func == sympy.Pow and factor.args[0] in momentum_operators:
                base, exponent = factor.args
                output['operator'].append(base)
                output['lhs'].append(sympy.Pow(base, exponent-1))
                break
            else:
                output['rhs'].append(factor)

        for factor in iterator:
            output['lhs'].append(factor)

    output = tuple(sympy.Mul(*output[key][::-1])
                   for key in ['lhs', 'operator', 'rhs'])
    return output


def derivate(expression, operator):
    """ Calculate derivate of expression for given momentum operator:

    Parameters:
    -----------
    expression : sympy.Expr instance
        Sympy expression containing functions to to be derivated.
    operator : sympy.Symbol
        Sympy symbol representing momentum operator.

    Returns:
    --------
    output : sympy.Expr instance
        Derivated input expression.
    """
    if not isinstance(operator, sympy.Symbol):
        raise TypeError("Input operator '{}' is not type sympy.Symbol.")

    if operator.name not in ['k_x', 'k_y', 'k_z']:
        raise ValueError("Input operator '{}' unkown.".format(operator))

    if isinstance(expression, (int, float, sympy.Symbol)):
        return 0
    else:
        coordinate_name = operator.name.split('_')[1]
        ct = sympy.Symbol(coordinate_name, commutative=True)
        cf = sympy.Symbol(coordinate_name, commutative=False)
        h = sympy.Symbol('a_'+coordinate_name)

        expr1 = expression.subs({ct: ct + h, cf: cf + h})
        expr2 = expression.subs({ct: ct - h, cf: cf - h})
        output = (expr1 - expr2) / 2 / h
        return -sympy.I * sympy.expand(output)


def _discretize_summand(summand, discrete_coordinates):
    """ Discretize one summand. """
    assert not isinstance(summand, sympy.Add), "Input should be one summand."

    def do_stuff(expr):
        """ Derivate expr recursively. """
        expr = sympy.expand(expr)

        if isinstance(expr, sympy.Add):
            return do_stuff(expr.args[-1]) + do_stuff(sympy.Add(*expr.args[:-1]))

        lhs, operator, rhs = split_factors(expr, discrete_coordinates)
        if rhs == 1 and operator != 1:
            return 0
        elif operator == 1:
            return lhs*rhs
        elif lhs == 1:
            return derivate(rhs, operator)
        else:
            return do_stuff(lhs*derivate(rhs, operator))

    return do_stuff(summand)


def _discretize_expression(expression, discrete_coordinates):
    """ Discretize continous `expression` into discrete tb representation.

    Parameters:
    -----------
    expression : sympy.Expr instance
        The expression to be discretized.

    Returns:
    --------
    discrete_expression: dict
        dict in which key is offset of hopping ((0, 0, 0) for onsite)
        and value is corresponding symbolic hopping (onsite).

    Note:
    -----
    Recursive derivation implemented in _discretize_summand is applied
    on every summand. Shortening is applied before return on output.
    """
    if isinstance(expression, (int, float, sympy.Integer, sympy.Float)):
        n = len(discrete_coordinates)
        return {(tuple(0 for i in range(n))): expression}

    if not isinstance(expression, sympy.Expr):
        raise TypeError('Input expression should be a valid sympy expression.')

    coordinates_names = sorted(list(discrete_coordinates))
    coordinates = [sympy.Symbol(c, commutative=False) for c in coordinates_names]
    wf = sympy.Function(wavefunction_name)(*coordinates)

    if wf in expression.atoms(sympy.Function):
        raise ValueError("Input expression must not contain {}.".format(wf))

    expression = sympy.expand(expression*wf)

    # make sure we have list of summands
    summands = expression.args if expression.func == sympy.Add else [expression]

    # discretize every summand
    outputs = []
    for summand in summands:
        out = _discretize_summand(summand, discrete_coordinates)
        out = extract_hoppings(out, discrete_coordinates)
        outputs.append(out)

    # gather together
    discrete_expression = defaultdict(int)
    for summand in outputs:
        for k, v in summand.items():
                discrete_expression[k] += v

    return dict(discrete_expression)


def discretize(hamiltonian, discrete_coordinates):
    """ Discretize continous `expression` into discrete tb representation.

    Parameters:
    -----------
    hamiltonian : sympy.Expr or sympy.Matrix instance
        The expression for the Hamiltonian.

    Returns:
    --------
    discrete_hamiltonian: dict
        dict in which key is offset of hopping ((0, 0, 0) for onsite)
        and value is corresponding symbolic hopping (onsite).

    Note:
    -----
    Recursive derivation implemented in _discretize_summand is applied
    on every summand. Shortening is applied before return on output.
    """
    if not isinstance(hamiltonian, sympy.matrices.MatrixBase):
        onsite_zeros = (0,)*len(discrete_coordinates)
        discrete_hamiltonian = {onsite_zeros: sympy.Integer(0)}
        hoppings = _discretize_expression(hamiltonian, discrete_coordinates)
        discrete_hamiltonian.update(hoppings)
        return discrete_hamiltonian

    shape = hamiltonian.shape

    discrete_hamiltonian = defaultdict(lambda: sympy.zeros(*shape))
    for i,j in itertools.product(range(shape[0]), range(shape[1])):
        expression = hamiltonian[i, j]
        hoppings = _discretize_expression(expression, discrete_coordinates)

        for offset, hop in hoppings.items():
            discrete_hamiltonian[offset][i,j] += hop
    return discrete_hamiltonian


# ****** extracting hoppings ***********
def read_hopping_from_wf(wf):
    """Read offset of a wave function in respect to (x,y,z).

    Parameters:
    ----------
    wf : sympy.function.AppliedUndef instance
        Function representing correct wave function used in discretizer.
        Should be created using global `wavefunction_name`.

    Returns:
    --------
    offset : tuple
        tuple of integers or floats that represent offset in respect to (x,y,z).

    Raises:
    -------
    ValueError
        If arguments of wf are repeated / do not stand for valid coordinates or
        lattice constants / order of dimensions is not lexical.
    TypeError:
        If wf is not of type sympy.function.AppliedUndef or its name does not
        corresponds to global 'wavefunction_name'.
    """
    if not isinstance(wf, sympy.function.AppliedUndef):
        raise TypeError('Input should be of type sympy.function.AppliedUndef.')

    if not wf.func.__name__ == wavefunction_name:
        msg = 'Input should be function that represents wavefunction in module.'
        raise TypeError(msg)

    # Check if input is consistent and in lexical order.
    # These are more checks for internal usage.
    coordinates_names = ['x', 'y', 'z']
    lattice_const_names = ['a_x', 'a_y', 'a_z']
    arg_coords = []
    for arg in wf.args:
        names = [s.name for s in arg.atoms(sympy.Symbol)]
        ind = -1
        for s in names:
            if not any(s in coordinates_names for s in names):
                raise ValueError("Wave function argument '{}' is incorrect.".format(s))
            if s not in coordinates_names and s not in lattice_const_names:
                raise ValueError("Wave function argument '{}' is incorrect.".format(s))
            if s in lattice_const_names:
                s = s.split('_')[1]
            tmp = coordinates_names.index(s)
            if tmp in arg_coords:
                msg = "Wave function '{}' arguments are inconsistent."
                raise ValueError(msg.format(wf))
            if ind != -1:
                if tmp != ind:
                    msg = "Wave function '{}' arguments are inconsistent."
                    raise ValueError(msg.format(wf))
            else:
                ind = tmp
        arg_coords.append(ind)
    if arg_coords != sorted(arg_coords):
        msg = "Coordinates of wave function '{}' are not in lexical order."
        raise ValueError(msg.format(wf))

    # function real body
    offset = []
    for argument in wf.args:
        temp = sympy.expand(argument)
        if temp in sympy.symbols('x y z', commutative = False):
            offset.append(0)
        elif temp.func == sympy.Add:
            for arg_summands in temp.args:
                if arg_summands.func == sympy.Mul:
                    if len(arg_summands.args) > 2:
                        print('More than two factors in an argument of wf')
                    if not arg_summands.args[0] in sympy.symbols('a_x a_y a_z'):
                        offset.append(arg_summands.args[0])
                    else:
                        offset.append(arg_summands.args[1])
                elif arg_summands in sympy.symbols('a_x a_y a_z'):
                    offset.append(1)
        else:
            print('Argument of \wf is neither a sum nor a single space variable.')
    return tuple(offset)


def extract_hoppings(expression, discrete_coordinates):
    """Extract hopping and perform shortening operation. """

    # make sure we have list of summands
    expression = sympy.expand(expression)
    summands = expression.args if expression.func == sympy.Add else [expression]

    hoppings = defaultdict(int)
    for summand in summands:
        if summand.func.__name__ == wavefunction_name:
            hoppings[read_hopping_from_wf(summand)] += 1
        else:
            for i in range(len(summand.args)):
                if summand.args[i].func.__name__ == wavefunction_name:
                    index = i
            if index < len(summand.args) - 1:
                print('Psi is not in the very end of the term. Output will be wrong!')
            hoppings[read_hopping_from_wf(summand.args[-1])] += sympy.Mul(*summand.args[:-1])

    # START: shortenig
    discrete_coordinates = sorted(list(discrete_coordinates))
    tmps = ['a_{}'.format(s) for s in discrete_coordinates]
    lattice_constants = sympy.symbols(tmps)
    a = sympy.Symbol('a')

    # make a list of all hopping kinds we have to consider during the shortening
    hops_kinds = np.array(list(hoppings))
    # find the longest hopping range in each direction
    longest_ranges = [np.max(hops_kinds[:,i]) for i in range(len(hops_kinds[0,:]))]
    # define an array in which we are going to store by which factor we
    # can shorten the hoppings in each direction
    shortening_factors = np.ones_like(longest_ranges)
    # Loop over the direction and each potential shortening factor.
    # Inside the loop test whether the hopping distances are actually
    # multiples of the potential shortening factor.
    for dim in np.arange(len(longest_ranges)):
        for factor in np.arange(longest_ranges[dim])+1:
            modulos = np.mod(hops_kinds[:, dim], factor)
            if np.sum(modulos) < 0.1:
                shortening_factors[dim] = factor
    # Apply the shortening factors on the hopping.
    short_hopping = {}
    for hopping_kind in hoppings.keys():
        short_hopping_kind = tuple(np.array(hopping_kind) / shortening_factors)

        for i in short_hopping_kind:
            if isinstance(i, float):
                assert i.is_integer()
        short_hopping_kind = tuple(int(i) for i in short_hopping_kind)

        short_hopping[short_hopping_kind] = hoppings[hopping_kind]
        for lat_const, factor in zip(lattice_constants, shortening_factors):
            factor = int(factor)
            subs = {lat_const: lat_const/factor}
            short_hopping[short_hopping_kind] = short_hopping[short_hopping_kind].subs(subs)

    # We don't need separate a_x, a_y and a_z anymore.
    for key, val in short_hopping.items():
        short_hopping[key] = val.subs({i: a for i in sympy.symbols('a_x a_y a_z')})

    return short_hopping
