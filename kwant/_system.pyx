# Copyright 2011-2019 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

cimport cython
import tinyarray as ta
import numpy as np
from scipy import sparse as sp
from itertools import chain
import types
import bisect

from .graph.core cimport CGraph, gintArraySlice
from .graph.defs cimport gint
from .graph.defs import gint_dtype
from ._common import deprecate_args


### Non-vectorized methods

msg = ('Hopping from site {0} to site {1} does not match the '
       'dimensions of onsite Hamiltonians of these sites.')

@cython.boundscheck(False)
def make_sparse(ham, args, params, CGraph gr, diag,
                gint [:] from_sites, n_by_to_site,
                gint [:] to_norb, gint [:] to_off,
                gint [:] from_norb, gint [:] from_off):
    """For internal use by hamiltonian_submatrix."""
    cdef gintArraySlice nbors
    cdef gint n_fs, fs, n_ts, ts
    cdef gint i, j, num_entries
    cdef complex [:, :] h
    cdef gint [:, :] rows_cols
    cdef complex [:] data
    cdef complex value

    matrix = ta.matrix

    # Calculate the data size.
    num_entries = 0
    for n_fs in range(len(from_sites)):
        fs = from_sites[n_fs]
        if fs in n_by_to_site:
            num_entries += from_norb[n_fs] * from_norb[n_fs]
        nbors = gr.out_neighbors(fs)
        for ts in nbors.data[:nbors.size]:
            if ts in n_by_to_site:
                n_ts = n_by_to_site[ts]
                num_entries += to_norb[n_ts] * from_norb[n_fs]

    rows_cols = np.empty((2, num_entries), gint_dtype)
    data = np.empty(num_entries, complex)

    cdef gint k = 0
    for n_fs in range(len(from_sites)):
        fs = from_sites[n_fs]
        if fs in n_by_to_site:
            n_ts = n_by_to_site[fs]
            h = diag[n_fs]
            if not (h.shape[0] == h.shape[1] == from_norb[n_fs]):
                raise ValueError(msg.format(fs, fs))
            for i in range(h.shape[0]):
                for j in range(h.shape[1]):
                    value = h[i, j]
                    if value != 0:
                        data[k] = value
                        rows_cols[0, k] = i + to_off[n_ts]
                        rows_cols[1, k] = j + from_off[n_fs]
                        k += 1

        nbors = gr.out_neighbors(fs)
        for ts in nbors.data[:nbors.size]:
            if ts not in n_by_to_site:
                continue
            n_ts = n_by_to_site[ts]
            h = matrix(ham(ts, fs, *args, params=params), complex)
            if h.shape[0] != to_norb[n_ts] or h.shape[1] != from_norb[n_fs]:
                raise ValueError(msg.format(fs, ts))
            for i in range(h.shape[0]):
                for j in range(h.shape[1]):
                    value = h[i, j]
                    if value != 0:
                        data[k] = value
                        rows_cols[0, k] = i + to_off[n_ts]
                        rows_cols[1, k] = j + from_off[n_fs]
                        k += 1

    # Hack around a bug in Scipy + Python 3 + memoryviews
    # see https://github.com/scipy/scipy/issues/5123 for details.
    # TODO: remove this once we depend on scipy >= 0.18.
    np_data = np.asarray(data)
    np_rows_cols = np.asarray(rows_cols)
    np_to_off = np.asarray(to_off)
    np_from_off = np.asarray(from_off)

    return sp.coo_matrix((np_data[:k], np_rows_cols[:, :k]),
                         shape=(np_to_off[-1], np_from_off[-1]))


@cython.boundscheck(False)
def make_sparse_full(ham, args, params, CGraph gr, diag,
                     gint [:] to_norb, gint [:] to_off,
                     gint [:] from_norb, gint [:] from_off):
    """For internal use by hamiltonian_submatrix."""
    cdef gintArraySlice nbors
    cdef gint n, fs, ts
    cdef gint i, j, num_entries
    cdef complex [:, :] h
    cdef gint [:, :] rows_cols
    cdef complex [:] data
    cdef complex value

    matrix = ta.matrix
    n = gr.num_nodes

    # Calculate the data size.
    num_entries = 0
    for fs in range(n):
        num_entries += from_norb[fs] * from_norb[fs]
        nbors = gr.out_neighbors(fs)
        for ts in nbors.data[:nbors.size]:
            if fs < ts:
                num_entries += 2 * to_norb[ts] * from_norb[fs]

    rows_cols = np.empty((2, num_entries), gint_dtype)
    data = np.empty(num_entries, complex)

    cdef gint k = 0
    for fs in range(n):
        h = diag[fs]
        if not (h.shape[0] == h.shape[1] == from_norb[fs]):
            raise ValueError(msg.format(fs, fs))
        for i in range(h.shape[0]):
            for j in range(h.shape[1]):
                value = h[i, j]
                if value != 0:
                    data[k] = value
                    rows_cols[0, k] = i + to_off[fs]
                    rows_cols[1, k] = j + from_off[fs]
                    k += 1

        nbors = gr.out_neighbors(fs)
        for ts in nbors.data[:nbors.size]:
            if ts < fs:
                continue
            h = matrix(ham(ts, fs, *args, params=params), complex)
            if h.shape[0] != to_norb[ts] or h.shape[1] != from_norb[fs]:
                raise ValueError(msg.format(fs, ts))
            for i in range(h.shape[0]):
                for j in range(h.shape[1]):
                    value = h[i, j]
                    if value != 0:
                        data[k] = value
                        data[k + 1] = h[i, j].conjugate()
                        rows_cols[1, k + 1] = rows_cols[0, k] = i + to_off[ts]
                        rows_cols[0, k + 1] = rows_cols[1, k] = j + from_off[fs]
                        k += 2

    # hack around a bug in Scipy + Python 3 + memoryviews
    # see https://github.com/scipy/scipy/issues/5123 for details
    # TODO: remove this once we depend on scipy >= 0.18.
    np_data = np.asarray(data)
    np_rows_cols = np.asarray(rows_cols)
    np_to_off = np.asarray(to_off)
    np_from_off = np.asarray(from_off)

    return sp.coo_matrix((np_data[:k], np_rows_cols[:, :k]),
                         shape=(np_to_off[-1], np_from_off[-1]))


@cython.boundscheck(False)
def make_dense(ham, args, params, CGraph gr, diag,
               gint [:] from_sites, n_by_to_site,
               gint [:] to_norb, gint [:] to_off,
               gint [:] from_norb, gint [:] from_off):
    """For internal use by hamiltonian_submatrix."""
    cdef gintArraySlice nbors
    cdef gint n_fs, fs, n_ts, ts
    cdef complex [:, :] h_sub_view
    cdef complex [:, :] h

    matrix = ta.matrix

    h_sub = np.zeros((to_off[-1], from_off[-1]), complex)
    h_sub_view = h_sub
    for n_fs in range(len(from_sites)):
        fs = from_sites[n_fs]
        if fs in n_by_to_site:
            n_ts = n_by_to_site[fs]
            h = diag[n_fs]
            if not (h.shape[0] == h.shape[1] == from_norb[n_fs]):
                raise ValueError(msg.format(fs, fs))
            h_sub_view[to_off[n_ts] : to_off[n_ts + 1],
                       from_off[n_fs] : from_off[n_fs + 1]] = h

        nbors = gr.out_neighbors(fs)
        for ts in nbors.data[:nbors.size]:
            if ts not in n_by_to_site:
                continue
            n_ts = n_by_to_site[ts]
            h = matrix(ham(ts, fs, *args, params=params), complex)
            if h.shape[0] != to_norb[n_ts] or h.shape[1] != from_norb[n_fs]:
                raise ValueError(msg.format(fs, ts))
            h_sub_view[to_off[n_ts] : to_off[n_ts + 1],
                       from_off[n_fs] : from_off[n_fs + 1]] = h
    return h_sub


@cython.boundscheck(False)
def make_dense_full(ham, args, params, CGraph gr, diag,
                    gint [:] to_norb, gint [:] to_off,
                    gint [:] from_norb, gint [:] from_off):
    """For internal use by hamiltonian_submatrix."""
    cdef gintArraySlice nbors
    cdef gint n, fs, ts
    cdef complex [:, :] h_sub_view, h, h_herm

    matrix = ta.matrix
    n = gr.num_nodes

    h_sub = np.zeros((to_off[-1], from_off[-1]), complex)
    h_sub_view = h_sub
    for fs in range(n):
        h = diag[fs]
        if not (h.shape[0] ==  h.shape[1] == from_norb[fs]):
            raise ValueError(msg.format(fs, fs))
        h_sub_view[to_off[fs] : to_off[fs + 1],
                   from_off[fs] : from_off[fs + 1]] = h

        nbors = gr.out_neighbors(fs)
        for ts in nbors.data[:nbors.size]:
            if ts < fs:
                continue
            h = mat = matrix(ham(ts, fs, *args, params=params), complex)
            h_herm = mat.transpose().conjugate()
            if h.shape[0] != to_norb[ts] or h.shape[1] != from_norb[fs]:
                raise ValueError(msg.format(fs, ts))
            h_sub_view[to_off[ts] : to_off[ts + 1],
                       from_off[fs] : from_off[fs + 1]] = h
            h_sub_view[from_off[fs] : from_off[fs + 1],
                       to_off[ts] : to_off[ts + 1]] = h_herm
    return h_sub


@deprecate_args
@cython.binding(True)
@cython.embedsignature(True)
def hamiltonian_submatrix(self, args=(), to_sites=None, from_sites=None,
                          sparse=False, return_norb=False, *, params=None):
    """Return a submatrix of the system Hamiltonian.

    Parameters
    ----------
    args : tuple, defaults to empty
        Positional arguments to pass to the ``hamiltonian`` method. Mutually
        exclusive with 'params'.
    to_sites : sequence of sites or None (default)
    from_sites : sequence of sites or None (default)
    sparse : bool
        Whether to return a sparse or a dense matrix. Defaults to ``False``.
    return_norb : bool
        Whether to return arrays of numbers of orbitals.  Defaults to ``False``.
    params : dict, optional
        Dictionary of parameter names and their values. Mutually exclusive
        with 'args'.

    Returns
    -------
    hamiltonian_part : numpy.ndarray or scipy.sparse.coo_matrix
        Submatrix of Hamiltonian of the system.
    to_norb : array of integers
        Numbers of orbitals on each site in to_sites.  Only returned when
        ``return_norb`` is true.
    from_norb : array of integers
        Numbers of orbitals on each site in from_sites.  Only returned when
        ``return_norb`` is true.

    Notes
    -----
    The returned submatrix contains all the Hamiltonian matrix elements
    from ``from_sites`` to ``to_sites``.  The default for ``from_sites`` and
    ``to_sites`` is ``None`` which means to use all sites of the system in the
    order in which they appear.
    """
    cdef gint [:] to_norb, from_norb
    cdef gint site, n_site, n

    ham = self.hamiltonian
    n = self.graph.num_nodes
    matrix = ta.matrix

    if from_sites is None:
        diag = n * [None]
        from_norb = np.empty(n, gint_dtype)
        for site in range(n):
            diag[site] = h = matrix(ham(site, site, *args, params=params),
                                    complex)
            from_norb[site] = h.shape[0]
    else:
        diag = len(from_sites) * [None]
        from_norb = np.empty(len(from_sites), gint_dtype)
        for n_site, site in enumerate(from_sites):
            if site < 0 or site >= n:
                raise IndexError('Site number out of range.')
            diag[n_site] = h = matrix(ham(site, site, *args, params=params),
                                      complex)
            from_norb[n_site] = h.shape[0]
    from_off = np.empty(from_norb.shape[0] + 1, gint_dtype)
    from_off[0] = 0
    from_off[1 :] = np.cumsum(from_norb)

    if to_sites is from_sites:
        to_norb = from_norb
        to_off = from_off
    else:
        if to_sites is None:
            to_norb = np.empty(n, gint_dtype)
            for site in range(n):
                h = matrix(ham(site, site, *args, params=params), complex)
                to_norb[site] = h.shape[0]
        else:
            to_norb = np.empty(len(to_sites), gint_dtype)
            for n_site, site in enumerate(to_sites):
                if site < 0 or site >= n:
                    raise IndexError('Site number out of range.')
                h = matrix(ham(site, site, *args, params=params), complex)
                to_norb[n_site] = h.shape[0]
        to_off = np.empty(to_norb.shape[0] + 1, gint_dtype)
        to_off[0] = 0
        to_off[1 :] = np.cumsum(to_norb)


    if to_sites is from_sites is None:
        func = make_sparse_full if sparse else make_dense_full
        mat = func(ham, args, params, self.graph, diag, to_norb, to_off,
                   from_norb, from_off)
    else:
        if to_sites is None:
            to_sites = np.arange(n, dtype=gint_dtype)
            n_by_to_site = dict((site, site) for site in to_sites)
        else:
            n_by_to_site = dict((site, n_site)
                                for n_site, site in enumerate(to_sites))

        if from_sites is None:
            from_sites = np.arange(n, dtype=gint_dtype)
        else:
            from_sites = np.asarray(from_sites, gint_dtype)

        func = make_sparse if sparse else make_dense
        mat = func(ham, args, params, self.graph, diag, from_sites,
                   n_by_to_site, to_norb, to_off, from_norb, from_off)
    return (mat, to_norb, from_norb) if return_norb else mat


### Vectorized methods


_shape_error_msg = (
    "The following hoppings have matrix elements of incompatible shape "
    "with other matrix elements: {}"
)


@cython.boundscheck(False)
def _vectorized_make_sparse(subgraphs, hams, long [:] norbs, long [:] orb_offsets,
                 long [:] site_offsets, shape=None):
    ndata = sum(h.shape[0] * h.shape[1] * h.shape[2] for h in hams)

    cdef long [:] rows_view, cols_view
    cdef complex [:] data_view

    rows = np.empty((ndata,), dtype=long)
    cols = np.empty((ndata,), dtype=long)
    data = np.empty((ndata,), dtype=complex)
    rows_view = rows
    cols_view = cols
    data_view = data

    cdef long i, j, k, m, N, M, P, to_off, from_off,\
              ta, fa, to_norbs, from_norbs
    cdef const long [:] ts_offs, fs_offs
    cdef complex [:, :, :] h

    m = 0
    # This outer loop zip() is pure Python, but that's ok, as it
    # has very few entries and the inner loops are fully vectorized
    for ((ta, fa), (ts_offs, fs_offs)), h in zip(subgraphs, hams):
        N = h.shape[0]
        M = h.shape[1]
        P = h.shape[2]

        if norbs[ta] != M or norbs[fa] != P:
            to_sites = site_offsets[ta] + np.array(ts_offs)
            from_sites = site_offsets[fa] + np.array(fs_offs)
            hops = np.array([to_sites, from_sites]).transpose()
            raise ValueError(_shape_error_msg.format(hops))

        for i in range(N):
            to_off = orb_offsets[ta] + norbs[ta] * ts_offs[i]
            from_off = orb_offsets[fa] + norbs[fa] * fs_offs[i]
            for j in range(M):
                for k in range(P):
                    rows_view[m] = to_off + j
                    cols_view[m] = from_off + k
                    data_view[m] = h[i, j, k]
                    m += 1

    if shape is None:
        shape = (orb_offsets[-1], orb_offsets[-1])

    return sp.coo_matrix((data, (rows, cols)), shape=shape)


@cython.boundscheck(False)
def _vectorized_make_dense(subgraphs, hams, long [:] norbs, long [:] orb_offsets,
                long [:] site_offsets, shape=None):
    if shape is None:
        shape = (orb_offsets[-1], orb_offsets[-1])
    mat = np.zeros(shape, dtype=complex)
    cdef complex [:, :] mat_view
    mat_view = mat

    cdef long i, j, k, N, M, P, to_off, from_off,\
              ta, fa, to_norbs, from_norbs
    cdef const long [:] ts_offs, fs_offs
    cdef complex [:, :, :] h

    # This outer loop zip() is pure Python, but that's ok, as it
    # has very few entries and the inner loops are fully vectorized
    for ((ta, fa), (ts_offs, fs_offs)), h in zip(subgraphs, hams):
        N = h.shape[0]
        M = h.shape[1]
        P = h.shape[2]

        if norbs[ta] != M or norbs[fa] != P:
            to_sites = site_offsets[ta] + np.array(ts_offs)
            from_sites = site_offsets[fa] + np.array(fs_offs)
            hops = np.array([to_sites, from_sites]).transpose()
            raise ValueError(_shape_error_msg.format(hops))

        for i in range(N):
            to_off = orb_offsets[ta] + norbs[ta] * ts_offs[i]
            from_off = orb_offsets[fa] + norbs[fa] * fs_offs[i]
            for j in range(M):
                for k in range(P):
                    mat_view[to_off + j, from_off + k] = h[i, j, k]
    return mat


def _expand_norbs(compressed_norbs, site_offsets):
    "Return norbs per site, given norbs per site array."
    norbs = np.empty(site_offsets[-1], int)
    for start, stop, norb in zip(site_offsets, site_offsets[1:],
                                  compressed_norbs):
        norbs[start:stop] = norb
    return norbs


def _reverse_subgraph(subgraph):
    (to_sa, from_sa), (to_off, from_off) = subgraph
    return ((from_sa, to_sa), (from_off, to_off))


@deprecate_args
@cython.binding(True)
def vectorized_hamiltonian_submatrix(self, args=(), sparse=False,
                                     return_norb=False, *, params=None):
    """Return The system Hamiltonian.

    Parameters
    ----------
    args : tuple, defaults to empty
        Positional arguments to pass to ``hamiltonian_term``. Mutually
        exclusive with 'params'.
    sparse : bool
        Whether to return a sparse or a dense matrix. Defaults to ``False``.
    return_norb : bool
        Whether to return arrays of numbers of orbitals.  Defaults to ``False``.
    params : dict, optional
        Dictionary of parameter names and their values. Mutually exclusive
        with 'args'.

    Returns
    -------
    hamiltonian_part : numpy.ndarray or scipy.sparse.coo_matrix
       The Hamiltonian of the system.
    norb : array of integers
        Numbers of orbitals on each site. Only returned when ``return_norb``
        is true.

    Notes
    -----
    Providing positional arguments via 'args' is deprecated,
    instead, provide named parameters as a dictionary via 'params'.
    """

    site_offsets = np.cumsum([0] + [len(arr) for arr in self.site_arrays])

    subgraphs = [
        self.subgraphs[t.subgraph]
        for t in self.terms
    ]
    # Add Hermitian conjugates
    subgraphs += [
        _reverse_subgraph(self.subgraphs[t.subgraph])
        for t in self.terms
        if t.hermitian
    ]

    hams = [
        self.hamiltonian_term(n, args=args, params=params)
        for n, _ in enumerate(self.terms)
    ]
    # Add Hermitian conjugates
    hams += [
        ham.conjugate().transpose(0, 2, 1)
        for ham, t in zip(hams, self.terms)
        if t.hermitian
    ]

    _, norbs, orb_offsets = self.site_ranges.transpose()

    func = _vectorized_make_sparse if sparse else _vectorized_make_dense
    mat = func(subgraphs, hams, norbs, orb_offsets, site_offsets)

    if return_norb:
        return (mat, _expand_norbs(norbs, site_offsets))
    else:
        return mat


@deprecate_args
@cython.binding(True)
def vectorized_cell_hamiltonian(self, args=(), sparse=False, *, params=None):
    """Hamiltonian of a single cell of the infinite system.

    Providing positional arguments via 'args' is deprecated,
    instead, provide named parameters as a dictionary via 'params'.
    """

    site_offsets = np.cumsum([0] + [len(arr) for arr in self.site_arrays])
    # Site array where next cell starts
    next_cell = bisect.bisect(site_offsets, self.cell_size) - 1

    def inside_fd(term):
        return all(s == 0 for s in term.symmetry_element)

    cell_terms = [
        n for n, t in enumerate(self.terms)
        if inside_fd(t)
    ]

    subgraphs = [
        self.subgraphs[self.terms[n].subgraph]
        for n in cell_terms
    ]
    # Add Hermitian conjugates
    subgraphs += [
        _reverse_subgraph(self.subgraphs[self.terms[n].subgraph])
        for n in cell_terms
        if self.terms[n].hermitian
    ]

    hams = [
        self.hamiltonian_term(n, args=args, params=params)
        for n in cell_terms
    ]
    # Add Hermitian conjugates
    hams += [
        ham.conjugate().transpose(0, 2, 1)
        for ham, n in zip(hams, cell_terms)
        if self.terms[n].hermitian
    ]

    _, norbs, orb_offsets = self.site_ranges.transpose()

    shape = (orb_offsets[next_cell], orb_offsets[next_cell])
    func = _vectorized_make_sparse if sparse else _vectorized_make_dense
    mat = func(subgraphs, hams, norbs, orb_offsets, site_offsets, shape=shape)

    return mat


@deprecate_args
@cython.binding(True)
def vectorized_inter_cell_hopping(self, args=(), sparse=False, *, params=None):
    """Hopping Hamiltonian between two cells of the infinite system.

    This method returns a complex matrix that represents the hopping from
    the *interface sites* of unit cell ``n - 1`` to *all* the sites of
    unit cell ``n``. It is therefore generally a *rectangular* matrix of
    shape ``(N_uc, N_iface)`` where ``N_uc`` is the number of orbitals
    in the unit cell, and ``N_iface`` is the number of orbitals on the
    *interface* sites (i.e. the sites with hoppings *to* the next unit cell).

    Providing positional arguments via 'args' is deprecated,
    instead, provide named parameters as a dictionary via 'params'.
    """

    site_offsets = np.cumsum([0] + [len(arr) for arr in self.site_arrays])

    # This method is only meaningful for systems with a 1D translational
    # symmetry, and we use this fact in several places
    assert all(len(t.symmetry_element) == 1 for t in self.terms)

    # Symmetry element -1 means hoppings *from* the *previous*
    # unit cell. These are directly the hoppings we wish to return.
    inter_cell_hopping_terms = [
        n for n, t in enumerate(self.terms)
        if t.symmetry_element[0] == -1
    ]
    # Symmetry element +1 means hoppings *from* the *next* unit cell.
    # These are related by translational symmetry to hoppings *to*
    # the *previous* unit cell. We therefore need the *reverse*
    # (and conjugate) of these hoppings.
    reversed_inter_cell_hopping_terms = [
        n for n, t in enumerate(self.terms)
        if t.symmetry_element[0] == +1
    ]

    inter_cell_hams = [
        self.hamiltonian_term(n, args=args, params=params)
        for n in inter_cell_hopping_terms
    ]
    reversed_inter_cell_hams = [
        self.hamiltonian_term(n, args=args, params=params)
            .conjugate().transpose(0, 2, 1)
        for n in reversed_inter_cell_hopping_terms
    ]

    hams = inter_cell_hams + reversed_inter_cell_hams

    subgraphs = [
        self.subgraphs[self.terms[n].subgraph]
        for n in inter_cell_hopping_terms
    ]
    subgraphs += [
        _reverse_subgraph(self.subgraphs[self.terms[n].subgraph])
        for n in reversed_inter_cell_hopping_terms
    ]

    _, norbs, orb_offsets = self.site_ranges.transpose()

    # SiteArrays containing interface sites appear before SiteArrays
    # containing non-interface sites, so the max of the site array
    # indices that appear in the 'from' site arrays of the inter-cell
    # hoppings allows us to get the number of interface orbitals.
    last_iface_site_array = max(
        from_site_array for (_, from_site_array), _ in subgraphs
    )
    iface_norbs = orb_offsets[last_iface_site_array + 1]
    fd_norbs = orb_offsets[-1]

    # TODO: return a square matrix when we no longer need to maintain
    #       backwards compatibility with unvectorized systems.
    shape = (fd_norbs, iface_norbs)
    func = _vectorized_make_sparse if sparse else _vectorized_make_dense
    mat = func(subgraphs, hams, norbs, orb_offsets, site_offsets, shape=shape)
    return mat
