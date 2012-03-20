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

    def hamiltonian_submatrix(self, to_sites=None, from_sites=None,
                              sparse=False):
        """Return a submatrix of the system Hamiltonian.

        Parameters
        ----------
        to_sites : sequence of sites or None (default)
        from_sites : sequence of sites or None (default)
        sparse : bool
            Whether to return a sparse or a dense matrix. Defaults to `False`.

        Returns
        -------
        hamiltonian_part : numpy.ndarray or scipy.sparse.coo_matrix
            Submatrix of Hamiltonian of the system.
        to_norb : array of integers
            Numbers of orbitals on each site in to_sites.
        from_norb : array of integers
            Numbers of orbitals on each site in from_sites.

        Notes
        -----
        The returned submatrix contains all the Hamiltonian matrix elements
        from `from_sites` to `to_sites`.  The default for `from_sites` and
        `to_sites` is `None` which means to use all sites of the system in the
        order in which they appear.
        """
        msg = 'Hopping from site {0} to site {1} does not match the ' \
              'dimensions of onsite Hamiltonians of these sites.'

        def make_sparse():
            # Calculate the data size.
            num_entries = 0
            for n_i, i in enumerate(from_sites):
                for j in chain((i,), gr.out_neighbors(i)):
                    if j in to_coord:
                        n_j = to_coord[j]
                        num_entries += to_norb[n_j] * from_norb[n_i]

            ij = np.empty((2, num_entries), dtype=int)
            data = np.empty(num_entries, dtype=complex)

            pos = 0
            for n_i, i in enumerate(from_sites):
                for j in chain((i,), gr.out_neighbors(i)):
                    n_j = to_coord.get(j)
                    if n_j is None:
                        continue
                    h = ham(j, i) if j != i else diag[i]
                    # The shape check is here to prevent data corruption.
                    shape = (1, 1) if np.isscalar(h) else h.shape
                    if shape != (to_norb[n_j], from_norb[n_i]):
                        raise ValueError(msg.format(i, j))
                    if np.isscalar(h):
                        data[pos] = h
                        ij[0, pos] = n_j
                        ij[1, pos] = n_i
                        pos += 1
                    else:
                        h = np.ravel(h, order='F')
                        coord = slice(pos, pos + h.size)
                        data[coord] = h
                        jtmp = np.arange(to_norb[n_j]) + to_off[n_j]
                        itmp = np.arange(from_norb[n_i]) + from_off[n_i]
                        jtmp, itmp = np.meshgrid(jtmp, itmp)
                        ij[0, coord] = jtmp.ravel()
                        ij[1, coord] = itmp.ravel()
                        pos += h.shape[0]
            return sp.coo_matrix((data, ij), shape=result_shape)

        def make_dense():
            # Shape checks of arrays are performed by numpy upon subblock
            # assignment.
            h_sub = np.zeros(result_shape, dtype='complex')
            for n_i, i in enumerate(from_sites):
                for j in chain((i,), gr.out_neighbors(i)):
                    n_j = to_coord.get(j)
                    if n_j is None:
                        continue
                    try:
                        h_sub[to_off[n_j] : to_off[n_j + 1],
                              from_off[n_i] : from_off[n_i + 1]] = \
                              ham(j, i) if j != i else diag[i]
                    except ValueError:
                        raise ValueError(msg.format(i, j))
            return h_sub

        gr = self.graph
        ham = self.hamiltonian
        n = self.graph.num_nodes

        if not ((from_sites is None or
                 all(0 <= site < n for site in from_sites)) and
                (to_sites is None or
                 all(0 <= site < n for site in to_sites))):
            raise IndexError('Site number out of range.')

        # Cache diagonal entries.
        isscalar = np.isscalar
        if from_sites is None or to_sites is None:
            # np.fromiter does not work with dtype=object.
            diag = np.array([ham(i, i) for i in xrange(n)], dtype=object)
            norb = np.fromiter(
                (1 if isscalar(h) else h.shape[0] for h in diag), int, n)
        else:
            if (max(len(from_sites), len(to_sites)) < n // 4 > 4):
                diag = {}
                get = diag.get
            else:
                diag = np.empty(n, dtype=object)
                get = diag.__getitem__

        # Make from_norb and from_off.
        if from_sites is None:
            from_sites = xrange(n)
            from_norb = norb
        else:
            from_norb = np.empty(len(from_sites), dtype=int)
            for n_i, i in enumerate(from_sites):
                h = get(i)
                if h is None:
                    diag[i] = h = ham(i, i)
                from_norb[n_i] = 1 if isscalar(h) else h.shape[0]
        from_off = np.zeros(from_norb.shape[0] + 1, int)
        from_off[1 :] = np.cumsum(from_norb)

        # Make to_norb and to_off.
        if to_sites is None:
            to_sites = xrange(n)
            to_norb = norb
        else:
            to_norb = np.empty(len(to_sites), dtype=int)
            for n_i, i in enumerate(to_sites):
                h = get(i)
                if h is None:
                    diag[i] = h = ham(i, i)
                to_norb[n_i] = 1 if isscalar(h) else h.shape[0]
        to_off = np.zeros(to_norb.shape[0] + 1, int)
        to_off[1 :] = np.cumsum(to_norb)

        # Instead of doing a double loop over from_sites and to_sites it is
        # more efficient to check if neighbors of from_sites are in to_sites.
        to_coord = dict((i[1], i[0]) for i in enumerate(to_sites))
        result_shape = (to_off[-1], from_off[-1])

        return make_sparse() if sparse else make_dense(), to_norb, from_norb


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
        return self.hamiltonian_submatrix(slice_sites, slice_sites)[0]

    def inter_slice_hopping(self):
        """Hopping Hamiltonian between two slices of the infinite system."""
        slice_size = self.slice_size
        slice_sites = xrange(self.slice_size)
        neighbor_sites = xrange(self.slice_size, self.graph.num_nodes)
        return self.hamiltonian_submatrix(slice_sites, neighbor_sites)[0]

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
