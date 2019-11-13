# Copyright 2011-2019 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from keyword import iskeyword
import functools
import operator
import collections

from math import sqrt
import tinyarray as ta
import numpy as np
import sympy

import kwant.lattice
import kwant.builder

import kwant.continuum
import kwant.continuum._common
import kwant.continuum.discretizer
from kwant.continuum import momentum_operators, position_operators

__all__ = ["to_landau_basis", "discretize_landau"]

coordinate_vectors = dict(zip("xyz", np.eye(3)))

ladder_lower, ladder_raise = sympy.symbols(r"a a^\dagger", commutative=False)


def to_landau_basis(hamiltonian, momenta=None):
    r"""Replace two momenta by Landau level ladder operators.

    Replaces:

        k_0 -> sqrt(B/2) * (a + a^\dagger)
        k_1 -> 1j * sqrt(B/2) * (a - a^\dagger)

    Parameters
    ----------
    hamiltonian : str or sympy expression
        The Hamiltonian to which to apply the Landau level transformation.
    momenta : sequence of str (optional)
        The momenta to replace with Landau level ladder operators. If not
        provided, 'k_x' and 'k_y' are used

    Returns
    -------
    hamiltonian : sympy expression
    momenta : sequence of sympy atoms
        The momentum operators that have been replaced by ladder operators.
    normal_coordinate : sympy atom
        The remaining position coordinate. May or may not be present
        in 'hamiltonian'.
    """
    hamiltonian = kwant.continuum.sympify(hamiltonian)
    momenta = _normalize_momenta(momenta)
    normal_coordinate = _find_normal_coordinate(hamiltonian, momenta)

    # Substitute ladder operators for Landau level momenta
    B = sympy.symbols("B")
    hamiltonian = hamiltonian.subs(
        {
            momenta[0]: sympy.sqrt(abs(B) / 2) * (ladder_raise + ladder_lower),
            momenta[1]: sympy.I
            * sympy.sqrt(abs(B) / 2)
            * (ladder_lower - ladder_raise),
        }
    )

    return hamiltonian, momenta, normal_coordinate


def discretize_landau(hamiltonian, N, momenta=None, grid_spacing=1):
    """Discretize a Hamiltonian in a basis of Landau levels.

    Parameters
    ----------
    hamiltonian : str or sympy expression
    N : positive integer
        The number of Landau levels in the basis.
    momenta : sequence of str (optional)
        The momenta defining the plane perpendicular to the magnetic field.
        If not provided, "k_x" and "k_y" are used.
    grid_spacing : float, default: 1
        The grid spacing to use when discretizing the normal direction
        (parallel to the magnetic field).

    Returns
    -------
    builder : `~kwant.builder.Builder`
        'hamiltonian' discretized in a basis of Landau levels in the plane
        defined by 'momenta'. If a third coordinate is present in 'hamiltonian',
        then this also has a translational symmetry in that coordinate direction.
        The builder has a parameter 'B' in addition to any other parameters
        present in the provided 'hamiltonian'.

    Notes
    -----
    The units of magnetic field are :math:`ϕ₀ / 2 π a²` with :math:`ϕ₀ = h/e`
    the magnetic flux quantum and :math:`a` the unit length.
    """

    if N <= 0:
        raise ValueError("N must be positive")

    hamiltonian, momenta, normal_coordinate = to_landau_basis(hamiltonian, momenta)

    # Discretize in normal direction and split terms for onsites/hoppings into
    # monomials in ladder operators.
    tb_hamiltonian, _ = kwant.continuum.discretize_symbolic(
        hamiltonian, coords=[normal_coordinate.name]
    )
    tb_hamiltonian = {
        key: kwant.continuum._common.monomials(value, gens=(ladder_lower, ladder_raise))
        for key, value in tb_hamiltonian.items()
    }

    # Replace ladder operator monomials by tuple of integers:
    # e.g. a^\dagger a^2 -> (+1, -2).
    tb_hamiltonian = {
        outer_key: {
            _ladder_term(inner_key): inner_value
            for inner_key, inner_value in outer_value.items()
        }
        for outer_key, outer_value in tb_hamiltonian.items()
    }

    # Construct map from LandauLattice HoppingKinds to a sequence of pairs
    # that encode the ladder operators and multiplying expression.
    tb_hoppings = collections.defaultdict(list)
    for outer_key, outer_value in tb_hamiltonian.items():
        for inner_key, inner_value in outer_value.items():
            tb_hoppings[(*outer_key, sum(inner_key))] += [(inner_key, inner_value)]
    # Extract the number of orbitals on each site/hopping
    random_element = next(iter(tb_hoppings.values()))[0][1]
    norbs = 1 if isinstance(random_element, sympy.Expr) else random_element.shape[0]
    tb_onsite = tb_hoppings.pop((0, 0), None)

    # Construct Builder
    if _has_coordinate(normal_coordinate, hamiltonian):
        sym = kwant.lattice.TranslationalSymmetry([grid_spacing, 0])
    else:
        sym = kwant.builder.NoSymmetry()
    lat = LandauLattice(grid_spacing, norbs=norbs)
    syst = kwant.Builder(sym)

    # Add onsites
    landau_sites = (lat(0, j) for j in range(N))
    if tb_onsite is None:
        syst[landau_sites] = ta.zeros((norbs, norbs))
    else:
        syst[landau_sites] = _builder_value(
            tb_onsite, normal_coordinate.name, grid_spacing, is_onsite=True
        )

    # Add zero hoppings between adjacent Landau levels.
    # Necessary to be able to use the Landau level builder
    # to populate another builder using builder.fill().
    syst[kwant.builder.HoppingKind((0, 1), lat)] = ta.zeros((norbs, norbs))

    # Add the hoppings from the Hamiltonian
    for hopping, parts in tb_hoppings.items():
        syst[kwant.builder.HoppingKind(hopping, lat)] = _builder_value(
            parts, normal_coordinate.name, grid_spacing, is_onsite=False
        )

    return syst


# This has to subclass lattice so that it will work with TranslationalSymmetry.
class LandauLattice(kwant.lattice.Monatomic):
    """
    A `~kwant.lattice.Monatomic` lattice with a Landau level index per site.

    Site tags (see `~kwant.system.SiteFamily`) are pairs of integers, where
    the first integer describes the real space position and the second the
    Landau level index.

    The real space Bravais lattice is one dimensional, oriented parallel
    to the magnetic field. The Landau level index represents the harmonic
    oscillator basis states used for the Landau quantization in the plane
    normal to the field.

    Parameters
    ----------
    grid_spacing : float
        Real space lattice spacing (parallel to the magnetic field).
    offset : float (optional)
        Displacement of the lattice origin from the real space
        coordinates origin.
    """

    def __init__(self, grid_spacing, offset=None, name="", norbs=None):
        if offset is not None:
            offset = [offset, 0]
        # The second vector and second coordinate do not matter (they are
        # not used in pos())
        super().__init__([[grid_spacing, 0], [0, 1]], offset, name, norbs)

    def pos(self, tag):
        return ta.array((self.prim_vecs[0, 0] * tag[0] + self.offset[0],))

    def landau_index(self, tag):
        return tag[-1]


def _builder_value(terms, normal_coordinate, grid_spacing, is_onsite):
    """Construct an onsite/hopping value function from a list of terms

    Parameters
    ----------
    terms : list
        Each element is a pair (ladder_term, sympy expression/matrix).
        ladder_term is a tuple of integers that encodes a string of
        Landau raising/lowering operators and the sympy expression
        is the rest
    normal_coordinate : str
    grid_spacing : float
        The grid spacing in the normal direction
    is_onsite : bool
        True if we are constructing an onsite value function
    """
    ladder_term_symbols = [sympy.Symbol(_ladder_term_name(lt)) for lt, _ in terms]
    # Construct a single expression from the terms, multiplying each part
    # by the placeholder that represents the prefactor from the ladder operator term.
    (ladder, (_, part)), *rest = zip(ladder_term_symbols, terms)
    expr = ladder * part
    for ladder, (_, part) in rest:
        expr += ladder * part
    expr = expr.subs(
        {kwant.continuum.discretizer._displacements[normal_coordinate]: grid_spacing}
    )
    # Construct the return string and temporary variable names
    # for function calls.
    return_string, map_func_calls, const_symbols, _cache = kwant.continuum.discretizer._return_string(
        expr, coords=[normal_coordinate]
    )

    # Remove the ladder term placeholders, as these are not parameters
    const_symbols = set(const_symbols)
    for ladder_term in ladder_term_symbols:
        const_symbols.discard(ladder_term)

    # Construct the argument part of signature. Arguments
    # consist of any constants and functions in the return string.
    arg_names = set.union(
        {s.name for s in const_symbols}, {str(k.func) for k in map_func_calls}
    )
    arg_names.discard("Abs")  # Abs function is not a parameter
    for arg_name in arg_names:
        if not (arg_name.isidentifier() and not iskeyword(arg_name)):
            raise ValueError(
                "Invalid name in used symbols: {}\n"
                "Names of symbols used in Hamiltonian "
                "must be valid Python identifiers and "
                "may not be keywords".format(arg_name)
            )
    arg_names = ", ".join(sorted(arg_names))
    # Construct site part of the function signature
    site_string = "from_site" if is_onsite else "to_site, from_site"
    # Construct function signature
    if arg_names:
        function_header = "def _({}, {}):".format(site_string, arg_names)
    else:
        function_header = "def _({}):".format(site_string)
    # Construct function body
    function_body = []
    if "B" not in arg_names:
        # B is not a parameter for terms with no ladder operators but we still
        # need something to pass to _evaluate_ladder_term
        function_body.append("B = +1")
    function_body.extend(
        [
            "{}, = from_site.pos".format(normal_coordinate),
            "_ll_index = from_site.family.landau_index(from_site.tag)",
        ]
    )
    # To get the correct hopping if B < 0, we need to set the Hermitian
    # conjugate of the ladder operator matrix element, which swaps the
    # from_site and to_site Landau level indices.
    if not is_onsite:
        function_body.extend(
            ["if B < 0:",
             "    _ll_index = to_site.family.landau_index(to_site.tag)"
            ]
        )
    function_body.extend(
        "{} = _evaluate_ladder_term({}, _ll_index, B)".format(symb.name, lt)
        for symb, (lt, _) in zip(ladder_term_symbols, terms)
    )
    function_body.extend(
        "{} = {}".format(v, kwant.continuum.discretizer._print_sympy(k))
        for k, v in map_func_calls.items()
    )
    function_body.append(return_string)
    func_code = "\n    ".join([function_header] + function_body)
    # Add "I" to namespace just in case sympy again would miss to replace it
    # with Python's 1j as it was the case with SymPy 1.2 when "I" was argument
    # of some function.
    namespace = {
        "pi": np.pi,
        "I": 1j,
        "_evaluate_ladder_term": _evaluate_ladder_term,
        "Abs": abs,
    }
    namespace.update(_cache)
    # Construct full source, including cached arrays
    source = []
    for k, v in _cache.items():
        source.append("{} = (\n{})\n".format(k, repr(np.array(v))))
    source.append(func_code)
    exec(func_code, namespace)
    f = namespace["_"]
    f._source = "".join(source)

    return f


def _ladder_term(operator_string):
    r"""Return a tuple of integers representing a string of ladder operators

    Parameters
    ----------
    operator_string : Sympy expression
        Monomial in ladder operators, e.g. a^\dagger a^2 a^\dagger.

    Returns
    -------
    ladder_term : tuple of int
        e.g. a^\dagger a^2 -> (+1, -2)
    """
    ret = []
    for factor in operator_string.as_ordered_factors():
        ladder_op, exponent = factor.as_base_exp()
        if ladder_op == ladder_lower:
            sign = -1
        elif ladder_op == ladder_raise:
            sign = +1
        else:
            sign = 0
        ret.append(sign * int(exponent))
    return tuple(ret)


def _ladder_term_name(ladder_term):
    """
    Parameters
    ----------
    ladder_term : tuple of int

    Returns
    -------
    ladder_term_name : str
    """

    def ns(i):
        if i >= 0:
            return str(i)
        else:
            return "_" + str(-i)

    return "_ladder_{}".format("_".join(map(ns, ladder_term)))


def _evaluate_ladder_term(ladder_term, n, B):
    r"""Evaluates the prefactor for a ladder operator on a landau level.

    Example: a^\dagger a^2 -> (n - 1) * sqrt(n)

    Parameters
    ----------
    ladder_term : tuple of int
        Represents a string of ladder operators. Positive
        integers represent powers of the raising operator,
        negative integers powers of the lowering operator.
    n : non-negative int
        Landau level index on which to act with ladder_term.
    B : float
        Magnetic field with sign

    Returns
    -------
    ladder_term_prefactor : float
    """
    assert n >= 0
    # For negative B we swap a and a^\dagger.
    if B < 0:
        ladder_term = tuple(-i for i in ladder_term)
    ret = 1
    for m in reversed(ladder_term):
        if m > 0:
            factors = range(n + 1, n + m + 1)
        elif m < 0:
            factors = range(n + m + 1, n + 1)
            if n == 0:
                return 0  # a|0> = 0
        else:
            factors = (1,)
        ret *= sqrt(functools.reduce(operator.mul, factors))
        n += m
    return ret


def _normalize_momenta(momenta=None):
    """Return Landau level momenta as Sympy atoms

    Parameters
    ----------
    momenta : None or pair of int or pair of str
        The momenta to choose. If None then 'k_x' and 'k_y'
        are chosen. If integers, then these are the indices
        of the momenta: 0 → k_x, 1 → k_y, 2 → k_z. If strings,
        then these name the momenta.

    Returns
    -------
    The specified momenta as sympy atoms.
    """

    # Specify which momenta to substitute for the Landau level basis.
    if momenta is None:
        # Use k_x and k_y by default
        momenta = momentum_operators[:2]
    else:
        if len(momenta) != 2:
            raise ValueError("Two momenta must be specified.")

        k_names = [k.name for k in momentum_operators]

        if all([type(i) is int for i in momenta]) and all(
            [i >= 0 and i < 3 for i in momenta]
        ):
            momenta = [momentum_operators[i] for i in momenta]
        elif all([isinstance(momentum, str) for momentum in momenta]) and all(
            [momentum in k_names for momentum in momenta]
        ):
            momenta = [
                momentum_operators[k_names.index(momentum)] for momentum in momenta
            ]
        else:
            raise ValueError("Momenta must all be integers or strings.")

    return tuple(momenta)


def _find_normal_coordinate(hamiltonian, momenta):
    discrete_momentum = next(
        momentum for momentum in momentum_operators if momentum not in momenta
    )
    normal_coordinate = position_operators[momentum_operators.index(discrete_momentum)]
    return normal_coordinate


def _has_coordinate(coord, expr):
    momentum = momentum_operators[position_operators.index(coord)]
    atoms = set(expr.atoms())
    return coord in atoms or momentum in atoms
