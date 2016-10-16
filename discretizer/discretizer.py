from __future__ import print_function, division

import warnings
import numpy as np
import sympy

from .algorithms import read_coordinates
from .algorithms import discretize

from .postprocessing import offset_to_direction
from .postprocessing import make_kwant_functions
from .postprocessing import offset_to_direction

from .interpolation import interpolate_tb_hamiltonian

try:
    # normal situation
    from kwant import Builder
    from kwant import TranslationalSymmetry
    from kwant import HoppingKind
    from kwant.lattice import Monatomic
except ImportError:
    # probably run on gitlab-ci
    pass

class Discretizer(object):
    """Discretize continous Hamiltonian into its tight binding representation.

    This class provides easy and nice interface for passing models to Kwant.

    Parameters:
    -----------
    hamiltonian : sympy.Expr or sympy.Matrix instance
        Symbolic representation of a continous Hamiltonian. Momentum operators
        should be taken from ``discretizer.momentum_operators``.
    discrete_coordinates : set of strings
        Set of coordinates for which momentum operators will be treated as
        differential operators. For example ``discrete_coordinates={'x', 'y'}``.
        If left as a None they will be obtained from the input hamiltonian by
        reading present coordinates and momentum operators.
    interpolate : bool
        If True all space dependent parameters in onsite and hopping will be
        interpolated to depenend only on the values at site positions.
        Default is False.
    both_hoppings_directions : bool
        If True all hoppings will be returned. For example, if set to True, both
        hoppings into (1, 0) and (-1, 0) will be returned. Default is False.
    verbose : bool
        If True additional information will be printed. Default is False.

    Attributes:
    -----------
    symbolic_hamiltonian : dictionary
        Dictionary containing symbolic result of discretization. Key is the
        direction of the hopping (zeros for onsite)
    lattice : kwant.lattice.Monatomic instance
        Lattice to create kwant system. Lattice constant is set to
        lattice_constant value.
    onsite : function
        The value of the onsite Hamiltonian.
    hoppings : dict
        A dictionary with keys being tuples of the lattice hopping, and values
        the corresponding value functions.
    discrete_coordinates : set of strings
        As in input.
    input_hamiltonian : sympy.Expr or sympy.Matrix instance
        The input hamiltonian after preprocessing (substitution of functions).
    """
    def __init__(self, hamiltonian, discrete_coordinates=None,
                 lattice_constant=1, interpolate=False,
                 both_hoppings_directions=False, verbose=False):

        self.input_hamiltonian = hamiltonian

        if discrete_coordinates is None:
            self.discrete_coordinates = read_coordinates(hamiltonian)
        else:
            self.discrete_coordinates = discrete_coordinates

        if verbose:
            print('Discrete coordinates set to: ',
                  sorted(self.discrete_coordinates), end='\n\n')

        # discretization
        if self.discrete_coordinates:
            tb_ham = discretize(hamiltonian, self.discrete_coordinates)
            tb_ham = offset_to_direction(tb_ham, self.discrete_coordinates)
        else:
            tb_ham = {(0,0,0): hamiltonian}
            self.discrete_coordinates = {'x', 'y', 'z'}

        if interpolate:
            tb_ham = interpolate_tb_hamiltonian(tb_ham)

        if not both_hoppings_directions:
            keys = list(tb_ham)
            tb_ham = {k: v for k, v in tb_ham.items()
                              if k in sorted(keys)[len(keys)//2:]}

        self.symbolic_hamiltonian = tb_ham.copy()

        for key, val in tb_ham.items():
            tb_ham[key] = val.subs(sympy.Symbol('a'), lattice_constant)

        # making kwant lattice
        dim = len(self.discrete_coordinates)

        self.lattice = Monatomic(lattice_constant*np.eye(dim).reshape(dim, dim))
        self.lattice_constant = lattice_constant

        # making kwant functions
        tb = make_kwant_functions(tb_ham, self.discrete_coordinates, verbose)
        self.onsite = tb.pop((0,)*len(self.discrete_coordinates))
        self.hoppings = {HoppingKind(d, self.lattice): val
                         for d, val in tb.items()}

    def build(self, shape, start, symmetry=None, periods=None):
        """Build Kwant's system.

        Convienient functions that simplifies building of a Kwant's system.

        Parameters:
        -----------
        shape : function
            A function of real space coordinates that returns a truth value:
            true for coordinates inside the shape, and false otherwise.
        start : 1d array-like
            The real-space origin for the flood-fill algorithm.
        symmetry : 1d array-like
            Deprecated. Please use ```periods=[symmetry]`` instead.
        periods : list of tuples
            If periods are provided a translational invariant system will be
            built. Periods corresponds basically to a translational symmetry
            defined in real space. This vector will be scalled by a lattice
            constant before passing it to ``kwant.TranslationalSymmetry``.
            Examples: ``periods=[(1,0,0)]`` or ``periods=[(1,0), (0,1)]``.
            In second case one will need https://gitlab.kwant-project.org/cwg/wraparound
            in order to finalize system.

        Returns:
        --------
        system : kwant.Builder instance
        """
        if symmetry is not None:
            warnings.warn("\nSymmetry argument is deprecated. " +
                          "Please use ```periods=[symmetry]`` instead.",
                          DeprecationWarning)
            periods = [symmetry]

        if periods is None:
            sys = Builder()
        else:
            vecs = [self.lattice.vec(p) for p in periods]
            sys = Builder(TranslationalSymmetry(*vecs))

        sys[self.lattice.shape(shape, start)] = self.onsite
        for hop, val in self.hoppings.items():
            sys[hop] = val

        return sys
