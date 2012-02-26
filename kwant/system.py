"""Low-level interface of tight binding systems"""

from __future__ import division
__all__ = ['System', 'FiniteSystem', 'InfiniteSystem']

import abc, math
import numpy as np
from scipy import sparse as sp
from itertools import chain
from kwant import physics

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
    is assumed to return the real space position of a site given its index.
    """
    __metaclass__ = abc.ABCMeta

    def num_orbitals(self, site):
        """Return the number of orbitals of a site.

        This is an inefficient general implementation.  It should be
        overridden, if a more efficient way to calculate is available.
        """
        ham = self.hamiltonian(site, site)
        return 1 if np.isscalar(ham) else ham.shape[0]

    @abc.abstractmethod
    def hamiltonian(self, i, j):
        """Return the hamiltonian matrix element for sites `i` and `j`.

        If ``i == j``, return the on-site Hamiltonian of site `i`.

        if ``i != j``, return the hopping between site `i` and `j`.
        """
        pass

    def hamiltonian_submatrix(self, b_sites=None, a_sites=None, sparse=False):
        """Return a submatrix of the system Hamiltonian.

        The returned submatrix contains all the Hamiltonian matrix elements
        from `a_sites` to `b_sites`.  If no value for a_sites or `b_sites` is
        provided, the default is to take all sites of the system in the order
        in which they appear. If sparse is set to `True`, a
        `scipy.sparse.coo_matrix` is returned, otherwise a dense one.
        """
        msg = 'Hopping from site {0} to site {1} does not match the ' +\
              'dimensions of onsite Hamiltonians of these sites.'

        def create_sparse():
            # Calculate the data size.
            num_entries = 0
            for n_i, i in enumerate(a_sites):
                for j in chain((i,), gr.out_neighbors(i)):
                    if j in b_coord:
                        n_j = b_coord[j]
                        num_entries += b_norb[n_j] * a_norb[n_i]

            ij = np.empty((2, num_entries), dtype=int)
            data = np.empty(num_entries, dtype=complex)

            offset = 0
            for n_i, i in enumerate(a_sites):
                for j in chain((i,), gr.out_neighbors(i)):
                    if j in b_coord:
                        n_j = b_coord[j]
                        h = ham(j, i)
                        # The shape check is here to prevent data corruption.
                        shape = (1, 1) if np.isscalar(h) else h.shape
                        if shape != (b_norb[n_j], a_norb[n_i]):
                            raise ValueError(msg.format(i, j))
                        if np.isscalar(h):
                            data[offset] = h
                            ij[0, offset] = n_j
                            ij[1, offset] = n_i
                            offset += 1
                        else:
                            h = np.ravel(h, order='F')
                            coord = slice(offset, offset + h.size)
                            data[coord] = h
                            jtmp = np.arange(b_norb[n_j]) + b_off[n_j]
                            itmp = np.arange(a_norb[n_i]) + a_off[n_i]
                            jtmp, itmp = np.meshgrid(jtmp, itmp)
                            ij[0, coord] = jtmp.ravel()
                            ij[1, coord] = itmp.ravel()
                            offset += h.shape[0]
            return sp.coo_matrix((data, ij), shape=result_shape)

        def create_dense():
            # Shape checks of arrays are performed by numpy upon subblock
            # assignment.
            h_sub = np.zeros(result_shape, dtype='complex')
            for n_i, i in enumerate(a_sites):
                for j in chain((i,), gr.out_neighbors(i)):
                    if j in b_coord:
                        n_j = b_coord[j]
                        try:
                            h_sub[b_off[n_j] : b_off[n_j + 1],
                                    a_off[n_i] : a_off[n_i + 1]] = ham(j, i)
                        except ValueError:
                            raise ValueError(msg.format(i, j))
            return h_sub

        gr = self.graph
        ham = self.hamiltonian
        n = self.graph.num_nodes
        if a_sites is None:
            a_sites = xrange(n)
        if b_sites is None:
            b_sites = xrange(n)
        if not all(site < n for site in a_sites) or \
           not all(site < n for site in b_sites):
            raise KeyError('site number out of range')
        a_norb = np.array([self.num_orbitals(i) for i in a_sites], int)
        a_off = np.zeros(a_norb.shape[0] + 1, int)
        a_off[1 :] = np.cumsum(a_norb)

        b_norb = np.array([self.num_orbitals(i) for i in b_sites], int)
        b_off = np.zeros(b_norb.shape[0] + 1, int)
        b_off[1 :] = np.cumsum(b_norb)
        # Instead of doing a double loop over a_sites and b_sites it is more
        # efficient to check if neighbors of a_sites are in b_sites.
        b_coord = dict((i[1], i[0]) for i in enumerate(b_sites))
        result_shape = (b_off[-1],  a_off[-1])

        if sparse:
            return create_sparse()
        else:
            return create_dense()


class FiniteSystem(System):
    """Abstract finite low-level system, possibly with leads.

    Instance Variables
    ------------------
    leads : sequence of lead objects
        Each lead object has to provide at least a method ``self_energy(energy)``.
    lead_neighbor_seqs : sequence of sequences of integers
        Each sub-sequence contains the indices of the system sites to which the
        lead is connected.

    Notes
    -----
    The length of `leads` must be equal to the length of `lead_neighbor_seqs`.

    For lead ``n``, the method leads[n].self_energy must return a square matrix
    whose size is ``sum(self.num_orbitals(neighbor) for neighbor in
    self.lead_neighbor_seqs[n])``.

    Often, the elements of `leads` will be instances of `InfiniteSystem`.  If
    this is the case for lead ``n``, the sites ``lead_neighbor_seqs[n]`` match
    the first ``len(lead_neighbor_seqs[n])`` sites of the InfiniteSystem.
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
    lead, `N` acts also as the number of neighbors to which it must be
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

    def slice_hamiltonian(self):
        """Hamiltonian of a single slice of the infinite system."""
        slice_sites = xrange(self.slice_size)
        return self.hamiltonian_submatrix(slice_sites, slice_sites)

    def inter_slice_hopping(self):
        """Hopping Hamiltonian between two slices of the infinite system."""
        slice_size = self.slice_size
        slice_sites = xrange(self.slice_size)
        neighbor_sites = xrange(self.slice_size, self.graph.num_nodes)
        return self.hamiltonian_submatrix(slice_sites, neighbor_sites)

    def self_energy(self, energy):
        """Return self-energy of a lead.

        The returned matrix has the shape (n, n), where n is
        ``sum(self.num_orbitals(i) for i in range(self.slice_size))``.
        """
        ham = self.slice_hamiltonian()
        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy
        return physics.self_energy(ham, self.inter_slice_hopping())

    def num_orbitals(self, site):
        """Return the number of orbitals of a site.

        This is an inefficient general implementation.  It should be
        overridden, if a more efficient way to calculate is available.
        """
        if site >= self.slice_size:
            site -= self.slice_size
        ham = self.hamiltonian(site, site)
        return 1 if np.isscalar(ham) else ham.shape[0]

    @property
    def energies(self):
        """
        A callable object which returns the energies at wave vector `k`

        Because the value of this property is a callable object, it can be used
        as if it were a method:

        >>> for k in arange(-pi, pi, 0.1):
        >>>     for e in infsys.energies(k):
        >>>         print k, e

        But it is more efficient to evaluate the property only once:

        >>> energies = infsys.energies
        >>> for k in arange(-pi, pi, 0.1):
        >>>     for e in energies(k):
        >>>         print k, e
        """
        result = Energies()
        result.ham = self.slice_hamiltonian()
        if not np.allclose(result.ham, result.ham.T.conj()):
            raise ValueError('The slice Hamiltonian is not Hermitian.')
        hop = self.inter_slice_hopping()
        result.hop = np.empty(result.ham.shape, dtype=complex)
        result.hop[:, : hop.shape[1]] = hop
        result.hop[:, hop.shape[1] :] = 0
        return result


class Energies():
    def __call__(self, k):
        mat = self.hop * complex(math.cos(k), math.sin(k))
        mat += mat.conjugate().transpose() + self.ham
        return np.sort(np.linalg.eigvalsh(mat))
