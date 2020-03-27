# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Low-level interface of systems"""

__all__ = ['System', 'FiniteSystem', 'InfiniteSystem']

import abc
import warnings
from copy import copy
from . import _system
from ._common  import deprecate_args


class System(metaclass=abc.ABCMeta):
    """Abstract general low-level system.

    Attributes
    ----------
    graph : kwant.graph.CGraph
        The system graph.
    site_ranges : None or sorted sequence of triples of integers
        If provided, encodes ranges of sites that have the same number of
        orbitals. Each triple consists of ``(first_site, norbs, orb_offset)``:
        the first site in the range, the number of orbitals on each site in the
        range, and the offset of the first orbital of the first site in the
        range.  In addition, the final triple should have the form
        ``(graph.num_nodes, 0, tot_norbs)`` where ``tot_norbs`` is the
        total number of orbitals in the system.
    parameters : frozenset of strings
        The names of the parameters on which the system depends. This attribute
        is provisional and may be changed in a future version of Kwant

    Notes
    -----
    The sites of the system are indexed by integers ranging from 0 to
    ``self.graph.num_nodes - 1``.

    Optionally, a class derived from ``System`` can provide a method ``pos`` which
    is assumed to return the real-space position of a site given its index.

    Due to the ordering semantics of sequences, and the fact that a given
    ``first_site`` can only appear *at most once* in ``site_ranges``,
    ``site_ranges`` is ordered according to ``first_site``.

    Consecutive elements in ``site_ranges`` are not required to have different
    numbers of orbitals.
    """
    @abc.abstractmethod
    def hamiltonian(self, i, j, *args, params=None):
        """Return the hamiltonian matrix element for sites ``i`` and ``j``.

        If ``i == j``, return the on-site Hamiltonian of site ``i``.

        if ``i != j``, return the hopping between site ``i`` and ``j``.

        Hamiltonians may depend (optionally) on positional and
        keyword arguments.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        pass

    @deprecate_args
    def discrete_symmetry(self, args, *, params=None):
        """Return the discrete symmetry of the system.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        # Avoid the circular import.
        from .physics import DiscreteSymmetry
        return DiscreteSymmetry()


    def __str__(self):
        items = [
            # (format, extractor, skip if info not present)
            ('{} sites', self.graph.num_nodes, False),
            ('{} hoppings', self.graph.num_edges, False),
            ('parameters: {}', tuple(self.parameters), True),
        ]
        # Skip some information when it's not present (parameters)
        details = [fmt.format(info) for fmt, info, skip in items
                   if (info or not skip)]
        details = ', and '.join((', '.join(details[:-1]), details[-1]))
        return '<{} with {}>'.format(self.__class__.__name__, details)


# Add a C-implemented function as an unbound method to class System.
System.hamiltonian_submatrix = _system.hamiltonian_submatrix


class FiniteSystem(System, metaclass=abc.ABCMeta):
    """Abstract finite low-level system, possibly with leads.

    Attributes
    ----------
    leads : sequence of leads
        Each lead has to provide a method ``selfenergy`` that has
        the same signature as `InfiniteSystem.selfenergy` (without the
        ``self`` parameter), and must have property ``parameters``:
        a collection of strings that name the system parameters (
        though this requirement is provisional and may be removed in
        a future version of Kwant).
        It may also provide ``modes`` that has the
        same signature as `InfiniteSystem.modes` (without the ``self``
        parameter).
    lead_interfaces : sequence of sequences of integers
        Each sub-sequence contains the indices of the system sites
        to which the lead is connected.
    lead_paddings : sequence of sequences of integers
        Each sub-sequence contains the indices of the system sites
        that belong to the lead, and therefore have the same onsite as the lead
        sites, and are connected by the same hoppings as the lead sites.
    parameters : frozenset of strings
        The names of the parameters on which the system depends. This does
        not include the parameters for any leads. This attribute
        is provisional and may be changed in a future version of Kwant

    Notes
    -----
    The length of ``leads`` must be equal to the length of ``lead_interfaces``
    and ``lead_paddings``.

    For lead ``n``, the method leads[n].selfenergy must return a square matrix
    whose size is ``sum(len(self.hamiltonian(site, site)) for site in
    self.lead_interfaces[n])``. The output of ``leads[n].modes`` has to be a
    tuple of `~kwant.physics.PropagatingModes`, `~kwant.physics.StabilizedModes`.

    Often, the elements of `leads` will be instances of `InfiniteSystem`.  If
    this is the case for lead ``n``, the sites ``lead_interfaces[n]`` match
    the first ``len(lead_interfaces[n])`` sites of the InfiniteSystem.

    """

    @deprecate_args
    def precalculate(self, energy=0, args=(), leads=None,
                     what='modes', *, params=None):
        """
        Precalculate modes or self-energies in the leads.

        Construct a copy of the system, with the lead modes precalculated,
        which may significantly speed up calculations where only the system
        is changing.

        Parameters
        ----------
        energy : float
            Energy at which the modes or self-energies have to be
            evaluated.
        args : sequence
            Additional parameters required for calculating the Hamiltionians.
            Deprecated in favor of 'params' (and mutually exclusive with it).
        leads : sequence of integers or None
            Numbers of the leads to be precalculated. If ``None``, all are
            precalculated.
        what : 'modes', 'selfenergy', 'all'
            The quantitity to precompute. 'all' will compute both
            modes and self-energies. Defaults to 'modes'.
        params : dict, optional
            Dictionary of parameter names and their values. Mutually exclusive
            with 'args'.

        Returns
        -------
        syst : FiniteSystem
            A copy of the original system with some leads precalculated.

        Notes
        -----
        If the leads are precalculated at certain `energy` or `args` values,
        they might give wrong results if used to solve the system with
        different parameter values. Use this function with caution.
        """

        if what not in ('modes', 'selfenergy', 'all'):
            raise ValueError("Invalid value of argument 'what': "
                             "{0}".format(what))

        result = copy(self)
        if leads is None:
            leads = list(range(len(self.leads)))
        new_leads = []
        for nr, lead in enumerate(self.leads):
            if nr not in leads:
                new_leads.append(lead)
                continue
            modes, selfenergy = None, None
            if what in ('modes', 'all'):
                modes = lead.modes(energy, args, params=params)
            if what in ('selfenergy', 'all'):
                if modes:
                    selfenergy = modes[1].selfenergy()
                else:
                    selfenergy = lead.selfenergy(energy, args, params=params)
            new_leads.append(PrecalculatedLead(modes, selfenergy))
        result.leads = new_leads
        return result

    @deprecate_args
    def validate_symmetries(self, args=(), *, params=None):
        """Check that the Hamiltonian satisfies discrete symmetries.

        Applies `~kwant.physics.DiscreteSymmetry.validate` to the
        Hamiltonian, see its documentation for details on the return
        format.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        symmetries = self.discrete_symmetry(args=args, params=params)
        ham = self.hamiltonian_submatrix(args, sparse=True, params=params)
        return symmetries.validate(ham)


class InfiniteSystem(System, metaclass=abc.ABCMeta):
    """Abstract infinite low-level system.

    An infinite system consists of an infinite series of identical cells.
    Adjacent cells are connected by identical inter-cell hoppings.

    Attributes
    ----------
    cell_size : integer
        The number of sites in a single cell of the system.

    Notes
    -----
    The system graph of an infinite systems contains a single cell, as well as
    the part of the previous cell which is connected to it.  The first
    `cell_size` sites form one complete single cell.  The remaining ``N`` sites
    of the graph (``N`` equals ``graph.num_nodes - cell_size``) belong to the
    previous cell.  They are included so that hoppings between cells can be
    represented.  The N sites of the previous cell correspond to the first
    ``N`` sites of the fully included cell.  When an ``InfiniteSystem`` is used
    as a lead, ``N`` acts also as the number of interface sites to which it
    must be connected.

    The drawing shows three cells of an infinite system.  Each cell consists
    of three sites.  Numbers denote sites which are included into the system
    graph.  Stars denote sites which are not included.  Hoppings are included
    in the graph if and only if they occur between two sites which are part of
    the graph::

            * 2 *
        ... | | | ...
            * 0 3
            |/|/|
            *-1-4

        <-- order of cells

    The numbering of sites in the drawing is one of the two valid ones for that
    infinite system.  The other scheme has the numbers of site 0 and 1
    exchanged, as well as of site 3 and 4.

    """
    @deprecate_args
    def cell_hamiltonian(self, args=(), sparse=False, *, params=None):
        """Hamiltonian of a single cell of the infinite system.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        cell_sites = range(self.cell_size)
        return self.hamiltonian_submatrix(args, cell_sites, cell_sites,
                                          sparse=sparse, params=params)

    @deprecate_args
    def inter_cell_hopping(self, args=(), sparse=False, *, params=None):
        """Hopping Hamiltonian between two cells of the infinite system.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        cell_sites = range(self.cell_size)
        interface_sites = range(self.cell_size, self.graph.num_nodes)
        return self.hamiltonian_submatrix(args, cell_sites, interface_sites,
                                          sparse=sparse, params=params)

    @deprecate_args
    def modes(self, energy=0, args=(), *, params=None):
        """Return mode decomposition of the lead

        See documentation of `~kwant.physics.PropagatingModes` and
        `~kwant.physics.StabilizedModes` for the return format details.

        The wave functions of the returned modes are defined over the
        *unit cell* of the system, which corresponds to the degrees of
        freedom on the first ``cell_sites`` sites of the system
        (recall that infinite systems store first the sites in the unit
        cell, then connected sites in the neighboring unit cell).

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        from . import physics   # Putting this here avoids a circular import.
        ham = self.cell_hamiltonian(args, params=params)
        hop = self.inter_cell_hopping(args, params=params)
        symmetries = self.discrete_symmetry(args, params=params)
        # Check whether each symmetry is broken.
        # If a symmetry is broken, it is ignored in the computation.
        broken = set(symmetries.validate(ham) + symmetries.validate(hop))
        attribute_names = {'Conservation law': 'projectors',
                          'Time reversal': 'time_reversal',
                          'Particle-hole': 'particle-hole',
                          'Chiral': 'chiral'}
        for name in broken:
            warnings.warn('Hamiltonian breaks ' + name +
                          ', ignoring the symmetry in the computation.')
            assert name in attribute_names, 'Inconsistent naming of symmetries'
            setattr(symmetries, attribute_names[name], None)

        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy

        # Particle-hole and chiral symmetries only apply at zero energy.
        if energy:
            symmetries.particle_hole = symmetries.chiral = None
        return physics.modes(ham, hop, discrete_symmetry=symmetries)

    @deprecate_args
    def selfenergy(self, energy=0, args=(), *, params=None):
        """Return self-energy of a lead.

        The returned matrix has the shape (s, s), where s is
        ``sum(len(self.hamiltonian(i, i)) for i in range(self.graph.num_nodes -
        self.cell_size))``.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        from . import physics   # Putting this here avoids a circular import.
        ham = self.cell_hamiltonian(args, params=params)
        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy
        return physics.selfenergy(ham,
                                  self.inter_cell_hopping(args, params=params))

    @deprecate_args
    def validate_symmetries(self, args=(), *, params=None):
        """Check that the Hamiltonian satisfies discrete symmetries.

        Returns `~kwant.physics.DiscreteSymmetry.validate` applied
        to the onsite matrix and the hopping. See its documentation for
        details on the return format.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        symmetries = self.discrete_symmetry(args=args, params=params)
        ham = self.cell_hamiltonian(args=args, sparse=True, params=params)
        hop = self.inter_cell_hopping(args=args, sparse=True, params=params)
        broken = set(symmetries.validate(ham) + symmetries.validate(hop))
        return list(broken)


class PrecalculatedLead:
    def __init__(self, modes=None, selfenergy=None):
        """A general lead defined by its self energy.

        Parameters
        ----------
        modes : (kwant.physics.PropagatingModes, kwant.physics.StabilizedModes)
            Modes of the lead.
        selfenergy : numpy array
            Lead self-energy.

        Notes
        -----
        At least one of ``modes`` and ``selfenergy`` must be provided.
        """
        if modes is None and selfenergy is None:
            raise ValueError("No precalculated values provided.")
        self._modes = modes
        self._selfenergy = selfenergy
        # Modes/Self-energy have already been evaluated, so there
        # is no parametric dependence anymore
        self.parameters = frozenset()

    @deprecate_args
    def modes(self, energy=0, args=(), *, params=None):
        if self._modes is not None:
            return self._modes
        else:
            raise ValueError("No precalculated modes were provided. "
                             "Consider using precalculate() with "
                             "what='modes' or what='all'")

    @deprecate_args
    def selfenergy(self, energy=0, args=(), *, params=None):
        if self._selfenergy is not None:
            return self._selfenergy
        else:
            raise ValueError("No precalculated selfenergy was provided. "
                             "Consider using precalculate() with "
                             "what='selfenergy' or what='all'")
