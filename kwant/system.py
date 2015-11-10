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
import types
from copy import copy
from . import _system


class System(object, metaclass=abc.ABCMeta):
    """Abstract general low-level system.

    Attributes
    ----------
    graph : kwant.graph.CGraph
        The system graph.

    Notes
    -----
    The sites of the system are indexed by integers ranging from 0 to
    ``self.graph.num_nodes - 1``.

    Optionally, a class derived from `System` can provide a method `pos` which
    is assumed to return the real-space position of a site given its index.
    """

    @abc.abstractmethod
    def hamiltonian(self, i, j, *args):
        """Return the hamiltonian matrix element for sites `i` and `j`.

        If ``i == j``, return the on-site Hamiltonian of site `i`.

        if ``i != j``, return the hopping between site `i` and `j`.

        Hamiltonians may depend (optionally) on positional and
        keyword arguments
        """
        pass

# Add a C-implemented function as an unbound method to class System.
System.hamiltonian_submatrix = _system.HamiltonianSubmatrix()


class FiniteSystem(System, metaclass=abc.ABCMeta):
    """Abstract finite low-level system, possibly with leads.

    Attributes
    ----------
    leads : sequence of leads
        Each lead has to provide a method
        ``selfenergy(energy, args)``.
        It may provide ``modes(energy, args)`` as well.
    lead_interfaces : sequence of sequences of integers
        Each sub-sequence contains the indices of the system sites
        to which the lead is connected.

    Notes
    -----
    The length of `leads` must be equal to the length of `lead_interfaces`.

    For lead ``n``, the method leads[n].selfenergy must return a square matrix
    whose size is ``sum(len(self.hamiltonian(site, site)) for site in
    self.lead_interfaces[n])``. The output of ``leads[n].modes`` has to be a
    tuple of `~kwant.physics.PropagatingModes, ~kwant.physics.StabilizedModes`.

    Often, the elements of `leads` will be instances of `InfiniteSystem`.  If
    this is the case for lead ``n``, the sites ``lead_interfaces[n]`` match
    the first ``len(lead_interfaces[n])`` sites of the InfiniteSystem.

    """

    def precalculate(self, energy=0, args=(), leads=None,
                     what='modes'):
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
            Additional parameters required for calculating the Hamiltionians
        leads : sequence of integers or None
            Numbers of the leads to be precalculated. If `None`, all are
            precalculated.
        what : 'modes', 'selfenergy', 'all'
            The quantitity to precompute. 'all' will compute both
            modes and self-energies. Defaults to 'modes'.

        Returns
        -------
        sys : FiniteSystem
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
                modes = lead.modes(energy, args)
            if what in ('selfenergy', 'all'):
                if modes:
                    selfenergy = modes[1].selfenergy()
                else:
                    selfenergy = lead.selfenergy(energy, args)
            new_leads.append(PrecalculatedLead(modes, selfenergy))
        result.leads = new_leads
        return result

class InfiniteSystem(System, metaclass=abc.ABCMeta):
    """
    Abstract infinite low-level system.

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
    `cell_size` sites form one complete single cell.  The remaining `N` sites of
    the graph (`N` equals ``graph.num_nodes - cell_size``) belong to the
    previous cell.  They are included so that hoppings between cells can be
    represented.  The N sites of the previous cell correspond to the first `N`
    sites of the fully included cell.  When an InfiniteSystem is used as a lead,
    `N` acts also as the number of interface sites to which it must be
    connected.

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

    def cell_hamiltonian(self, args=(), sparse=False):
        """Hamiltonian of a single cell of the infinite system."""
        cell_sites = range(self.cell_size)
        return self.hamiltonian_submatrix(args, cell_sites, cell_sites,
                                          sparse=sparse)

    def inter_cell_hopping(self, args=(), sparse=False):
        """Hopping Hamiltonian between two cells of the infinite system."""
        cell_sites = range(self.cell_size)
        interface_sites = range(self.cell_size, self.graph.num_nodes)
        return self.hamiltonian_submatrix(args, cell_sites, interface_sites,
                                          sparse=sparse)

    def modes(self, energy=0, args=()):
        """Return mode decomposition of the lead

        See documentation of `~kwant.physics.PropagatingModes` and
        `~kwant.physics.StabilizedModes` for the return format details.
        """
        from . import physics   # Putting this here avoids a circular import.
        ham = self.cell_hamiltonian(args)
        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy
        return physics.modes(ham, self.inter_cell_hopping(args))

    def selfenergy(self, energy=0, args=()):
        """Return self-energy of a lead.

        The returned matrix has the shape (s, s), where s is
        ``sum(len(self.hamiltonian(i, i)) for i in range(self.graph.num_nodes -
        self.cell_size))``.
        """
        from . import physics   # Putting this here avoids a circular import.
        ham = self.cell_hamiltonian(args)
        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy
        return physics.selfenergy(ham, self.inter_cell_hopping(args))


class PrecalculatedLead(object):
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
        At least one of `modes` and `selfenergy` must be provided.
        """
        if modes is None and selfenergy is None:
            raise ValueError("No precalculated values provided.")
        self._modes = modes
        self._selfenergy = selfenergy

    def modes(self, energy=0, args=()):
        if self._modes is not None:
            return self._modes
        else:
            raise ValueError("No precalculated modes were provided. "
                             "Consider using precalculate() with "
                             "what='modes' or what='all'")

    def selfenergy(self, energy=0, args=()):
        if self._selfenergy is not None:
            return self._selfenergy
        else:
            raise ValueError("No precalculated selfenergy was provided. "
                             "Consider using precalculate() with "
                             "what='selfenergy' or what='all'")
