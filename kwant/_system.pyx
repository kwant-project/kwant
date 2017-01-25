# Copyright 2011-2013 Kwant authors.
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

from .graph.core cimport CGraph, gintArraySlice
from .graph.defs cimport gint
from .graph.defs import gint_dtype

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

# workaround for Cython functions not having __get__ and
# Python 3 getting rid of unbound methods
cdef class HamiltonianSubmatrix:

    def __get__(self, obj, objtype):
        if obj is None:
            return hamiltonian_submatrix
        else:
            return types.MethodType(hamiltonian_submatrix, obj)
