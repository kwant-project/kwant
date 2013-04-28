# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Low-level interface of systems"""

from __future__ import division
__all__ = ['System', 'FiniteSystem', 'InfiniteSystem']

import abc
import types
import numpy as np
from . import physics, _system


class System(object):
    """Abstract general low-level system.

    Instance Variables
    ------------------
    graph : kwant.graph.CGraph
        The system graph.

    Notes
    -----
    The sites of the system are indexed by integers ranging from 0 to
    ``self.graph.num_nodes - 1``.

    Optionally, a class derived from `System` can provide a method `pos` which
    is assumed to return the real-space position of a site given its index.
    """
    __metaclass__ = abc.ABCMeta

    def num_orbitals(self, site, *args):
        """Return the number of orbitals of a site.

        This is an inefficient general implementation.  It should be
        overridden, if a more efficient way to calculate is available.
        """
        ham = self.hamiltonian(site, site, *args)
        return 1 if np.isscalar(ham) else ham.shape[0]

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
System.hamiltonian_submatrix = types.MethodType(
    _system.hamiltonian_submatrix, None, System)


class FiniteSystem(System):
    """Abstract finite low-level system, possibly with leads.

    Instance Variables
    ------------------
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
    whose size is ``sum(self.num_orbitals(neighbor)`` for neighbor in
    self.lead_interfaces[n])``. The output format for ``leads[n].modes`` has to
    be as described in `~kwant.physics.ModesTuple`.

    Often, the elements of `leads` will be instances of `InfiniteSystem`.  If
    this is the case for lead ``n``, the sites ``lead_interfaces[n]`` match
    the first ``len(lead_interfaces[n])`` sites of the InfiniteSystem.
    """
    __metaclass__ = abc.ABCMeta


class InfiniteSystem(System):
    """
    Abstract infinite low-level system.

    An infinite system consists of an infinite series of identical slices.
    Adjacent slices are connected by identical inter-slice hoppings.

    Instance Variables
    ------------------
    slice_size : integer
        The number of sites in a single slice of the system.

    Notes
    -----
    The system graph of an infinite systems contains a single slice, as well as
    the part of the previous slice which is connected to it.  The first
    `slice_size` sites form one complete single slice.  The remaining `N` sites
    of the graph (`N` equals ``graph.num_nodes - slice_size``) belong to the
    previous slice.  They are included so that hoppings between slices can be
    represented.  The N sites of the previous slice correspond to the first `N`
    sites of the fully included slice.  When an InfiniteSystem is used as a
    lead, `N` acts also as the number of interface sites to which it must be
    connected.

    The drawing shows three slices of an infinite system.  Each slice consists
    of three sites.  Numbers denote sites which are included into the system
    graph.  Stars denote sites which are not included.  Hoppings are included
    in the graph if and only if they occur between two sites which are part of
    the graph::

            * 2 *
        ... | | | ...
            * 0 3
            |/|/|
            *-1-4

        <-- order of slices

    The numbering of sites in the drawing is one of the two valid ones for that
    infinite system.  The other scheme has the numbers of site 0 and 1
    exchanged, as well as of site 3 and 4.
    """
    __metaclass__ = abc.ABCMeta

    def slice_hamiltonian(self, sparse=False, args=()):
        """Hamiltonian of a single slice of the infinite system."""
        slice_sites = xrange(self.slice_size)
        return self.hamiltonian_submatrix(slice_sites, slice_sites,
                                          sparse=sparse, args=args)

    def inter_slice_hopping(self, sparse=False, args=()):
        """Hopping Hamiltonian between two slices of the infinite system."""
        slice_sites = xrange(self.slice_size)
        neighbor_sites = xrange(self.slice_size, self.graph.num_nodes)
        return self.hamiltonian_submatrix(slice_sites, neighbor_sites,
                                          sparse=sparse, args=args)

    def modes(self, energy, args=()):
        """Return mode decomposition of the lead

        See documentation of `~kwant.physics.ModesTuple` for the return
        format details.
        """
        ham = self.slice_hamiltonian(args=args)
        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy
        return physics.modes(ham, self.inter_slice_hopping(args=args))

    def selfenergy(self, energy, args=()):
        """Return self-energy of a lead.

        The returned matrix has the shape (n, n), where n is
        ``sum(self.num_orbitals(i) for i in range(self.slice_size))``.
        """
        ham = self.slice_hamiltonian(args=args)
        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy
        return physics.selfenergy(ham, self.inter_slice_hopping(args=args))
