# Copyright 2011-2019 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.
"""Tools for working with operators for acting on wavefunctions."""

__all__ = ['Density', 'Current', 'Source']

import cython
import itertools
import functools as ft
import collections
import warnings

import numpy as np
import tinyarray as ta
from scipy.sparse import coo_matrix

from libc cimport math

cdef extern from "complex.h":
    double cabs(double complex)

from .graph.core cimport EdgeIterator
from .graph.core import DisabledFeatureError, NodeDoesNotExistError
from .graph.defs cimport gint
from .graph.defs import gint_dtype
from .system import (
    is_infinite, is_vectorized, Site, SiteArray, _normalize_matrix_blocks
)
from ._common import (
    UserCodeError, KwantDeprecationWarning, get_parameters, deprecate_args
)


################ Generic Utility functions

@cython.boundscheck(False)
@cython.wraparound(False)
cdef gint _bisect(gint[:] a, gint x):
    "bisect.bisect specialized for searching `site_ranges`"
    cdef gint mid, lo = 0, hi = a.shape[0]
    while lo < hi:
        mid = (lo + hi) // 2
        if x < a[mid]:
            hi = mid
        else:
            lo = mid + 1
    return lo


@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _is_hermitian(
    complex[:, :] a, double atol=1e-300, double rtol=1e-13
) except -1:
    "Return True if 'a' is Hermitian"

    if a.shape[0] != a.shape[1]:
        return False

    # compute max(a)
    cdef double tmp, max_a = 0
    cdef gint i, j, k
    for i in range(a.shape[0]):
        for j in range(a.shape[1]):
            tmp = cabs(a[i, j])
            if tmp > max_a:
                max_a = tmp
    max_a = math.sqrt(max_a)

    cdef double tol = rtol * max_a + atol
    for i in range(a.shape[0]):
        for j in range(i, a.shape[1]):
            tmp = cabs(a[i, j] - a[j, i].conjugate())
            if tmp > tol:
                return False
    return True

@cython.boundscheck(False)
@cython.wraparound(False)
cdef int _is_hermitian_3d(
    complex[:, :, :] a, double atol=1e-300, double rtol=1e-13
) except -1:
    "Return True if 'a' is Hermitian"

    if a.shape[1] != a.shape[2]:
        return False

    # compute max(a)
    cdef double tmp, max_a = 0
    cdef gint i, j, k
    for k in range(a.shape[0]):
        for i in range(a.shape[1]):
            for j in range(a.shape[2]):
                tmp = cabs(a[k, i, j])
                if tmp > max_a:
                    max_a = tmp
    max_a = math.sqrt(max_a)

    cdef double tol = rtol * max_a + atol
    for k in range(a.shape[0]):
        for i in range(a.shape[1]):
            for j in range(i, a.shape[2]):
                tmp = cabs(a[k, i, j] - a[k, j, i].conjugate())
                if tmp > tol:
                    return False
    return True


@cython.boundscheck(False)
@cython.wraparound(False)
cdef _select(gint[:, :] arr, gint[:] indexes):
    ret = np.empty((indexes.shape[0], arr.shape[1]), dtype=gint_dtype)
    cdef gint[:, :] ret_view = ret
    cdef gint i, j

    for i in range(indexes.shape[0]):
        for j in range(arr.shape[1]):
            ret_view[i, j] = arr[indexes[i], j]

    return ret

################ Helper functions

_shape_msg = ('{0} matrix dimensions do not match '
              'the declared number of orbitals')

_herm_msg = ('{0} matrix is not hermitian, use the option '
             '`check_hermiticity=True` if this is intentional.')

cdef int _check_onsite(complex[:, :] M, gint norbs,
                       int check_hermiticity) except -1:
    "Check onsite matrix for correct shape and hermiticity."
    if M.shape[0] != M.shape[1]:
        raise UserCodeError('Onsite matrix is not square')
    if M.shape[0] != norbs:
        raise UserCodeError(_shape_msg.format('Onsite'))
    if check_hermiticity and not _is_hermitian(M):
        raise ValueError(_herm_msg.format('Onsite'))
    return 0


cdef int _check_onsites(complex[:, :, :] M, gint norbs,
                       int check_hermiticity) except -1:
    "Check onsite matrix for correct shape and hermiticity."
    if M.shape[1] != M.shape[2]:
        raise UserCodeError('Onsite matrix is not square')
    if M.shape[1] != norbs:
        raise UserCodeError(_shape_msg.format('Onsite'))
    if check_hermiticity and not _is_hermitian_3d(M):
        raise ValueError(_herm_msg.format('Onsite'))
    return 0


cdef int _check_hams(complex[:, :, :] H, gint to_norbs, gint from_norbs,
                     int check_hermiticity) except -1:
    if H.shape[1] != to_norbs or H.shape[2] != from_norbs:
        raise UserCodeError(_shape_msg.format('Hamiltonian'))
    if check_hermiticity and not _is_hermitian_3d(H):
            raise ValueError(_herm_msg.format('Hamiltonian'))
    return 0


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void _get_orbs(gint[:, :] site_ranges, gint site,
                    gint *start_orb, gint *norbs):
    """Return the first orbital of this site and the number of orbitals"""
    cdef gint run_idx, first_site, norb, orb_offset, orb
    # Calculate the index of the range that contains the site.
    run_idx = _bisect(site_ranges[:, 0], site) - 1
    first_site = site_ranges[run_idx, 0]
    norb = site_ranges[run_idx, 1]
    orb_offset = site_ranges[run_idx, 2]
    # calculate the slice
    start_orb[0] = orb_offset + (site - first_site) * norb
    norbs[0] = norb


@cython.boundscheck(False)
@cython.wraparound(False)
def _get_all_orbs(gint[:, :] where, gint[:, :] site_ranges):
    cdef gint[:, :] offsets = np.empty((where.shape[0], 2), dtype=gint_dtype)
    cdef gint[:, :] norbs = np.empty((where.shape[0], 2), dtype=gint_dtype)

    cdef gint w, a, a_offset, a_norbs, b, b_offset, b_norbs
    for w in range(where.shape[0]):
        a = where[w, 0]
        _get_orbs(site_ranges, a, &a_offset, &a_norbs)
        if where.shape[1] == 1:
            b, b_offset, b_norbs = a, a_offset, a_norbs
        else:
            b = where[w, 1]
            _get_orbs(site_ranges, b, &b_offset, &b_norbs)
        offsets[w, 0] = a_offset
        offsets[w, 1] = b_offset
        norbs[w, 0] = a_norbs
        norbs[w, 1] = b_norbs

    return offsets, norbs


def _get_tot_norbs(syst):
    cdef gint _unused, tot_norbs
    is_infinite_system = is_infinite(syst)
    n_sites = syst.cell_size if is_infinite_system else syst.graph.num_nodes
    _get_orbs(np.asarray(syst.site_ranges, dtype=gint_dtype),
              n_sites, &tot_norbs, &_unused)
    return tot_norbs


def _normalize_site_where(syst, where):
    """Normalize the format of `where` when `where` contains sites.

    If `where` is None, then all sites in the system are returned.
    If it is a general sequence then it is expanded into an array. If `syst`
    is a finalized Builder then `where` may contain `Site` objects,
    otherwise it should contain integers.
    """
    if where is None:
        if is_infinite(syst):
            where = list(range(syst.cell_size))
        else:
            where = list(range(syst.graph.num_nodes))
    elif callable(where):
        try:
            where = [syst.id_by_site[s] for s in filter(where, syst.sites)]
        except AttributeError:
            if is_infinite(syst):
                where = [s for s in range(syst.cell_size) if where(s)]
            else:
                where = [s for s in range(syst.graph.num_nodes) if where(s)]
    else:
        if isinstance(where[0], Site):
            try:
                where = [syst.id_by_site[s] for s in where]
            except AttributeError:
                raise TypeError("'where' contains Sites, but the system is not "
                                "a finalized Builder.")

    where = np.asarray(where, dtype=gint_dtype).reshape(-1, 1)

    if is_infinite(syst) and np.any(where >= syst.cell_size):
        raise ValueError('Only sites in the fundamental domain may be '
                         'specified using `where`.')
    if np.any(np.logical_or(where < 0, where >= syst.graph.num_nodes)):
        raise ValueError('`where` contains sites that are not in the '
                         'system.')

    return where


def _normalize_hopping_where(syst, where):
    """Normalize the format of `where` when `where` contains hoppings.

    If `where` is None, then all hoppings in the system are returned.
    If it is a general iterator then it is expanded into an array. If `syst` is
    a finalized Builder then `where` may contain pairs of `Site` objects,
    otherwise it should contain pairs of integers.
    """
    if where is None:
        # we cannot extract the hoppings in the same order as they are in the
        # graph while simultaneously excluding all inter-cell hoppings
        if is_infinite(syst):
            raise ValueError('`where` must be provided when calculating '
                             'current in an InfiniteSystem.')
        where = list(syst.graph)
    elif callable(where):
        if hasattr(syst, "sites"):
            def idxwhere(hop):
                a, b = hop
                return where(syst.sites[a], syst.sites[b])
            where = list(filter(idxwhere, syst.graph))
        else:
            where = list(filter(lambda h: where(*h), syst.graph))
    else:
        if isinstance(where[0][0], Site):
            try:
                where = list((syst.id_by_site[a], syst.id_by_site[b])
                               for a, b in where)
            except AttributeError:
                raise TypeError("'where' contains Sites, but the system is not "
                                "a finalized Builder.")
        # NOTE: if we ever have operators that contain elements that are
        #       not in the system graph, then we should modify this check
        try:
            error = ValueError('`where` contains hoppings that are not '
                               'in the system.')
            if any(not syst.graph.has_edge(*w) for w in where):
                raise error
        # If where contains: negative integers, or integers > # of sites
        except (NodeDoesNotExistError, DisabledFeatureError):
            raise error

    where = np.asarray(where, dtype=gint_dtype)

    if is_infinite(syst) and np.any(where > syst.cell_size):
        raise ValueError('Only intra-cell hoppings may be specified '
                         'using `where`.')

    return where


## These four classes are here to avoid using closures, as these will
## break pickle support. These are only used inside '_normalize_onsite'.

def _raise_user_error(exc, func, vectorized):
    msg = [
        'Error occurred in user-supplied onsite function "{0}".',
        'Did you remember to vectorize "{0}"?' if vectorized else '',
        'See the upper part of the above backtrace for more information.',
    ]
    msg = '\n'.join(line for line in msg if line).format(func.__name__)
    raise UserCodeError(msg) from exc


class _FunctionalOnsite:

    def __init__(self, onsite, sites, site_ranges):
        self.onsite = onsite
        self.sites = sites
        self.site_ranges = site_ranges

    def __call__(self, site_range, site_offsets, *args):
        sites = self.sites
        offset = self.site_ranges[site_range][0]
        try:
            ret = [self.onsite(sites[offset + i], *args) for i in site_offsets]
        except Exception as exc:
            _raise_user_error(exc, self.onsite, vectorized=False)
        return _normalize_matrix_blocks(ret, len(site_offsets))


class _VectorizedFunctionalOnsite:

    def __init__(self, onsite, site_arrays):
        self.onsite = onsite
        self.site_arrays = site_arrays

    def __call__(self, site_range, site_offsets, *args):
        site_array = self.site_arrays[site_range]
        tags = site_array.tags[site_offsets]
        sites = SiteArray(site_array.family, tags)
        try:
            ret = self.onsite(sites, *args)
        except Exception as exc:
            _raise_user_error(exc, self.onsite, vectorized=True)
        return _normalize_matrix_blocks(ret, len(site_offsets))


class _FunctionalOnsiteNoTransform:

    def __init__(self, onsite, site_ranges):
        self.onsite = onsite
        self.site_ranges = site_ranges

    def __call__(self, site_range, site_offsets, *args):
        site_ids = self.site_ranges[site_range][0] + site_offsets
        try:
            ret = [self.onsite(id, *args) for id in site_ids]
        except Exception as exc:
            _raise_user_error(exc, self.onsite, vectorized=False)
        return _normalize_matrix_blocks(ret, len(site_offsets))


class _DictOnsite:

    def __init__(self, onsite, range_family_map):
        self.onsite = onsite
        self.range_family_map = range_family_map

    def __call__(self, site_range, site_offsets, *args):
        fam = self.range_family_map[site_range]
        ret = [self.onsite[fam]] * len(site_offsets)
        return _normalize_matrix_blocks(ret, len(site_offsets))


def _normalize_onsite(syst, onsite, check_hermiticity):
    """Normalize the format of `onsite`.

    If `onsite` is a function or a mapping (dictionary) then a function
    is returned.
    """
    param_names = ()

    if callable(onsite):
        # make 'onsite' compatible with hamiltonian value functions
        param_names = get_parameters(onsite)[1:]
        if is_vectorized(syst):
            _onsite = _VectorizedFunctionalOnsite(onsite, syst.site_arrays)
        elif hasattr(syst, "sites"):  # probably a non-vectorized finalized Builder
            _onsite = _FunctionalOnsite(onsite, syst.sites, syst.site_ranges)
        else:  # generic old-style system, therefore *not* vectorized.
            _onsite = _FunctionalOnsiteNoTransform(onsite, syst.site_ranges)

    elif isinstance(onsite, collections.abc.Mapping):
        # onsites known; immediately check for correct shape and hermiticity
        for fam, _onsite in onsite.items():
            _onsite = ta.matrix(_onsite, complex)
            _check_onsite(_onsite, fam.norbs, check_hermiticity)

        if is_vectorized(syst):
            range_family_map = [arr.family for arr in syst.site_arrays]
        elif not hasattr(syst, 'sites'):
            raise TypeError('Provide `onsite` as a value or a function for '
                            'systems that are not finalized Builders.')
        else:
            # The last entry in 'site_ranges' is just an end marker, so we remove it
            range_family_map = [syst.sites[r[0]].family for r in syst.site_ranges[:-1]]
        _onsite = _DictOnsite(onsite, range_family_map)

    else:
        # single onsite; immediately check for correct shape and hermiticity
        _onsite = ta.matrix(onsite, complex)
        _check_onsite(_onsite, _onsite.shape[0], check_hermiticity)
        if _onsite.shape[0] == 1:
            # NOTE: this is wasteful when many orbitals per site, but it
            # simplifies the code in `_operate`. If this proves to be a
            # bottleneck, then we can add a code path for scalar onsites
            max_norbs = max(norbs for (_, norbs, _) in syst.site_ranges)
            _onsite = _onsite[0, 0] * ta.identity(max_norbs, complex)
        elif len(set(norbs for _, norbs, _ in syst.site_ranges[:-1])) == 1:
            # we have the same number of orbitals everywhere
            norbs = syst.site_ranges[0][1]
            if _onsite.shape[0] != norbs:
                msg = ('Single `onsite` matrix of shape ({0}, {0}) provided '
                       'but there are {1} orbitals per site in the system')
                raise ValueError(msg.format(_onsite.shape[0], norbs))
        else:
            msg = ('Single `onsite` matrix provided, but there are '
                   'different numbers of orbitals on different sites')
            raise ValueError(msg)

    return _onsite, param_names


def _make_onsite_or_hopping_terms(site_ranges, where):

    terms = dict()

    cdef gint[:] site_offsets_ = np.asarray(site_ranges, dtype=gint_dtype)[:, 0]
    cdef gint i

    if where.shape[1] == 1:  # onsite
        for i in range(where.shape[0]):
            a = _bisect(site_offsets_, where[i, 0]) - 1
            terms.setdefault((a, a), []).append(i)
    else:  # hopping
        for i in range(where.shape[0]):
            a = _bisect(site_offsets_, where[i, 0]) - 1
            b = _bisect(site_offsets_, where[i, 1]) - 1
            terms.setdefault((a, b), []).append(i)
    return [(a, None, b) for a, b in terms.items()]


def _vectorized_make_onsite_terms(syst, where):
    assert is_vectorized(syst)
    assert where.shape[1] == 1
    site_offsets = [r[0] for r in syst.site_ranges]

    terms = {}
    term_by_id = syst._onsite_term_by_site_id
    for i in range(where.shape[0]):
        w = where[i, 0]
        terms.setdefault(term_by_id[w], []).append(i)

    ret = []
    for term_id, which in terms.items():
        term = syst.terms[term_id]
        ((term_sa, _), (term_sites, _)) = syst.subgraphs[term.subgraph]
        term_sites += site_offsets[term_sa]
        which = np.asarray(which, dtype=gint_dtype)
        sites = _select(where, which).reshape(-1)
        selector = np.searchsorted(term_sites, sites)
        ret.append((term_id, selector, which))

    return ret


def _vectorized_make_hopping_terms(syst, where):
    assert is_vectorized(syst)
    assert where.shape[1] == 2
    site_offsets = [r[0] for r in syst.site_ranges]

    terms = {}
    term_by_id = syst._hopping_term_by_edge_id
    for i in range(where.shape[0]):
        a, b = where[i, 0], where[i, 1]
        edge = syst.graph.first_edge_id(a, b)
        terms.setdefault(term_by_id[edge], []).append(i)

    ret = []
    dtype = np.dtype([('f0', int), ('f1', int)])
    for term_id, which in terms.items():
        herm_conj = term_id < 0
        if herm_conj:
            real_term_id = -term_id - 1
        else:
            real_term_id = term_id
        which = np.asarray(which, dtype=gint_dtype)
        # Select out the hoppings and reverse them if we are
        # dealing with Hermitian conjugate hoppings
        hops = _select(where, which)
        if herm_conj:
            hops = hops[:, ::-1]
        # Force contiguous array to handle herm conj case also.
        # Needs to be contiguous to cast to compound dtype
        hops = np.ascontiguousarray(hops, dtype=int)
        hops = hops.view(dtype).reshape(-1)
        term = syst.terms[real_term_id]
        # Get array of pairs
        ((to_sa, from_sa), term_hoppings) = syst.subgraphs[term.subgraph]
        term_hoppings = term_hoppings.transpose() + (site_offsets[to_sa], site_offsets[from_sa])
        term_hoppings = term_hoppings.view(dtype).reshape(-1)

        selector = np.recarray.searchsorted(term_hoppings, hops)

        ret.append((term_id, selector, which))

    return ret


def _make_matrix_elements(evaluate_term, terms):
        matrix_elements = []
        for (term_id, term_selector, which) in terms:
            which = np.asarray(which, dtype=gint_dtype)
            data = evaluate_term(term_id, term_selector, which)
            matrix_elements.append((which, data))
        return matrix_elements


cdef class BlockSparseMatrix:
    """A sparse matrix stored as dense blocks.

    Parameters
    ----------
    where : gint[:, :]
        ``Nx2`` matrix or ``Nx1`` matrix: the arguments ``a``
        and ``b`` to be used when evaluating ``f``. If an
        ``Nx1`` matrix, then ``b=a``.
    block_offsets : gint[:, :]
        The row and column offsets for the start of each block
        in the sparse matrix: ``(row_offset, col_offset)``.
    block_shapes : gint[:, :]
        ``Nx2`` array: the shapes of each block, ``(n_rows, n_cols)``.
    matrix_elements : sequence of pairs (where_indices, data)
        'data' is a 3D complex array; a vector of matrices.
        'where_indices' is a 1D array of indices for 'where';
        'data[i]' should be placed at 'where[where_indices[i]]'.

    Attributes
    ----------
    block_offsets : gint[:, :]
        The row and column offsets for the start of each block
        in the sparse matrix: ``(row_offset, col_offset)``.
    block_shapes : gint[:, :]
        The shape of each block: ``(n_rows, n_cols)``
    data_offsets : gint[:]
        The offsets of the start of each matrix block in `data`.
    data : complex[:]
        The matrix of each block, stored in row-major (C) order.
    """

    @cython.embedsignature
    @cython.boundscheck(False)
    @cython.wraparound(False)
    def __init__(self, gint[:, :] where, gint[:, :] block_offsets,
                  gint[:, :] block_shapes, matrix_elements):
        if (block_offsets.shape[0] != where.shape[0] or
            block_shapes.shape[0] != where.shape[0]):
            raise ValueError('Arrays should be the same length along '
                             'the first axis.')
        self.block_shapes = block_shapes
        self.block_offsets = block_offsets
        self.data_offsets = np.empty(where.shape[0], dtype=gint_dtype)
        ### calculate shapes and data_offsets
        cdef gint w, data_size = 0
        for w in range(where.shape[0]):
            self.data_offsets[w] = data_size
            data_size += block_shapes[w, 0] * block_shapes[w, 1]
        ### Populate data array
        self.data = np.empty((data_size,), dtype=complex)
        cdef complex[:, :, :] data
        cdef gint[:] where_indexes
        cdef gint i, j, k, off, a, b, a_norbs, b_norbs
        for where_indexes, data in matrix_elements:
            for i in range(where_indexes.shape[0]):
                w = where_indexes[i]
                off = self.data_offsets[w]
                a_norbs = self.block_shapes[w, 0]
                b_norbs = self.block_shapes[w, 1]
                # Copy data
                for j in range(a_norbs):
                    for k in range(b_norbs):
                        self.data[off + j * b_norbs + k] = data[i, j, k]

    cdef complex* get(self, gint block_idx):
        return  <complex*> &self.data[0] + self.data_offsets[block_idx]

    def __getstate__(self):
        return tuple(map(np.asarray, (
            self.block_offsets,
            self.block_shapes,
            self.data_offsets,
            self.data
        )))

    def __setstate__(self, state):
        (self.block_offsets,
         self.block_shapes,
         self.data_offsets,
         self.data,
        ) = state


################ Local Observables

# supported operations within the `_operate` method
ctypedef enum operation:
    MAT_ELS
    ACT


cdef class _LocalOperator:
    """Base class for operators defined by an on-site matrix and the
    Hamiltonian.

    This includes "true" local operators, as well as "currents" and "sources".

    Attributes
    ----------
    syst : `~kwant.system.System`
        The system for which this operator is defined. Must have the
        number of orbitals defined for all site families.
    onsite : complex 2D array, or callable
        If a complex array, then the same onsite is used everywhere.
        Otherwise, function that can be called with a single site (integer) and
        extra arguments, and returns the representation of the operator on
        that site. This should return either a scalar or a square matrix of the
        same shape as that returned by the system Hamiltonian evaluated on the
        same site.  The extra arguments must be the same as the extra arguments
        to ``syst.hamiltonian``.
    where : 2D array of `int` or `None`
        where to evaluate the operator. A list of sites for on-site
        operators (accessed like `where[n, 0]`), otherwise a list of pairs
        of sites (accessed like `where[n, 0]` and `where[n, 1]`).
    check_hermiticity : bool
        If True, checks that ``onsite``, as well as any relevant parts
        of the Hamiltonian are hermitian.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `~kwant.operator._LocalOperator.__call__` for details).
    """

    @cython.embedsignature
    def __init__(self, syst, onsite, where, *,
                 check_hermiticity=True, sum=False):
        if syst.site_ranges is None:
            raise ValueError('Number of orbitals not defined.\n'
                             'Declare the number of orbitals using the '
                             '`norbs` keyword argument when constructing '
                             'the site families (lattices).')
        # TODO: Update this when it becomes clear how ND systems will be
        #       implemented.
        if is_vectorized(syst) and is_infinite(syst):
            raise TypeError('Vectorized infinite systems cannot yet be '
                            'used with operators.')

        self.syst = syst
        self.onsite, self._onsite_param_names = _normalize_onsite(
            syst, onsite, check_hermiticity)
        self.check_hermiticity = check_hermiticity
        self.sum = sum
        self._site_ranges = np.asarray(syst.site_ranges, dtype=gint_dtype)
        self.where = where
        self._bound_onsite = None
        self._bound_hamiltonian = None

        # Here we pre-compute the datastructures that will enable us to evaluate
        # the Hamiltonian and onsite functions in a vectorized way. Essentially
        # we group the sites/hoppings in 'where' by what term of 'syst' they are
        # in (for vectorized systems), or by the site family(s) (for
        # non-vectorized systems). If the system is vectorized we store a list
        # of triples:
        #
        #   (term_id, term_selector, which)
        #
        # otherwise
        #
        #   ((to_site_range, from_site_range), None, which)
        #
        # Where:
        #
        # 'term_id' → integer: term index, may be negative (indicates herm. conj.)
        # 'term_selector' → 1D integer array: selects which elements from the
        #                   subgraph of term number 'term_id' should be evaluated.
        # 'which' → 1D integer array: selects which elements of 'where' this
        #           vectorized evaluation corresponds to.
        # 'to/from_site_range' → integer: the site ranges that the elements of
        #                        'where' indexed by 'which' correspond to.
        #
        # Note that all sites/hoppings indicated by 'which' belong to the *same*
        # pair of site families by construction. This is what allows for
        # vectorized evaluation.
        if is_vectorized(syst):
            if self.where.shape[1] == 1:
                self._terms = _vectorized_make_onsite_terms(syst, where)
            else:
                self._terms = _vectorized_make_hopping_terms(syst, where)
        else:
            self._terms = _make_onsite_or_hopping_terms(self._site_ranges, where)

    @cython.embedsignature
    def __call__(self, bra, ket=None, args=(), *, params=None):
        r"""Return the matrix elements of the operator.

        An operator ``A`` can be called like

            >>> A(psi)

        to compute the expectation value :math:`\bra{ψ} A \ket{ψ}`,
        or like

            >>> A(phi, psi)

        to compute the matrix element :math:`\bra{φ} A \ket{ψ}`.

        If ``sum=True`` was provided when constructing the operator, then
        a scalar is returned. If ``sum=False``, then a vector is returned.
        The vector is defined over the sites of the system if the operator
        is a `~kwant.operator.Density`, or over the hoppings if it is a
        `~kwant.operator.Current` or `~kwant.operator.Source`. By default,
        the returned vector is ordered in the same way as the sites
        (for `~kwant.operator.Density`) or hoppings in the graph of the
        system (for `~kwant.operator.Current` or `~kwant.operator.Density`).
        If the keyword parameter ``where`` was provided when constructing
        the operator, then the returned vector is instead defined only over
        the sites or hoppings specified, and is ordered in the same way
        as ``where``.

        Alternatively stated, for an operator :math:`Q_{iαβ}`, ``bra``
        :math:`φ_α` and ``ket`` :math:`ψ_β` this computes
        :math:`q_i = ∑_{αβ} φ^*_α Q_{iαβ} ψ_β` if ``self.sum`` is False,
        otherwise computes :math:`q = ∑_{iαβ} φ^*_α Q_{iαβ} ψ_β`. where
        :math:`i` runs over all sites or hoppings, and
        :math:`α` and :math:`β` run over all the degrees of freedom.

        Parameters
        ----------
        bra, ket : sequence of complex
            Must have the same length as the number of orbitals
            in the system. If only one is provided, both ``bra``
            and ``ket`` are taken as equal.
        args : tuple, optional
            The arguments to pass to the system. Used to evaluate
            the ``onsite`` elements and, possibly, the system Hamiltonian.
            Deprecated in favor of 'params' (and mutually exclusive with it).
        params : dict, optional
            Dictionary of parameter names and their values. Mutually exclusive
            with 'args'.

        Returns
        -------
        `float` if ``check_hermiticity`` is True, and ``ket`` is ``None``,
        otherwise `complex`. If this operator was created with ``sum=True``,
        then a single value is returned, otherwise an array is returned.
        """
        if (self._bound_onsite or self._bound_hamiltonian) and (args or params):
            raise ValueError("Extra arguments are already bound to this "
                             "operator. You should call this operator "
                             "providing neither 'args' nor 'params'.")
        if args:
            # deprecate_args does not play nicely with methods of cdef classes,
            # when used as a decorator, so we manually raise the
            # deprecation warning here.
            deprecate_args()
        if args and params:
            raise TypeError("'args' and 'params' are mutually exclusive.")
        if bra is None:
            raise TypeError('bra must be an array')
        bra = np.asarray(bra, dtype=complex)
        ket = bra if ket is None else np.asarray(ket, dtype=complex)
        tot_norbs = _get_tot_norbs(self.syst)
        if bra.shape != (tot_norbs,):
            msg = 'vector is incorrect shape'
            msg = 'bra ' + msg if ket is None else msg
            raise ValueError(msg)
        elif ket.shape != (tot_norbs,):
            raise ValueError('ket vector is incorrect shape')

        result = np.zeros((self.where.shape[0],), dtype=complex)
        self._operate(out_data=result, bra=bra, ket=ket, args=args,
                      params=params, op=MAT_ELS)
        # if everything is Hermitian then result is real if bra == ket
        if self.check_hermiticity and bra is ket:
            result = result.real
        return np.sum(result) if self.sum else result

    @cython.embedsignature
    def act(self, ket, args=(), *, params=None):
        """Act with the operator on a wavefunction.

        For an operator :math:`Q_{iαβ}` and ``ket`` :math:`ψ_β`
        this computes :math:`∑_{iβ} Q_{iαβ} ψ_β`.

        Parameters
        ----------
        ket : sequence of complex
            Wavefunctions defined over all the orbitals of the system.
        args : tuple
            The extra arguments to the Hamiltonian value functions and
            the operator ``onsite`` function.
            Deprecated in favor of 'params' (and mutually exclusive with it).
        params : dict, optional
            Dictionary of parameter names and their values. Mutually exclusive
            with 'args'.

        Returns
        -------
        Array of `complex`.
        """
        if (self._bound_onsite or self._bound_hamiltonian) and (args or params):
            raise ValueError("Extra arguments are already bound to this "
                             "operator. You should call this operator "
                             "providing neither 'args' nor 'params'.")
        if args:
            # deprecate_args does not play nicely with methods of cdef classes,
            # when used as a decorator, so we manually raise the
            # deprecation warning here.
            deprecate_args()
        if args and params:
            raise TypeError("'args' and 'params' are mutually exclusive.")

        if ket is None:
            raise TypeError('ket must be an array')
        ket = np.asarray(ket, dtype=complex)
        tot_norbs = _get_tot_norbs(self.syst)
        if ket.shape != (tot_norbs,):
            raise ValueError('ket vector is incorrect shape')
        result = np.zeros((tot_norbs,), dtype=np.complex)
        self._operate(out_data=result, bra=None, ket=ket, args=args,
                      params=params, op=ACT)
        return result

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        """Bind the given arguments to this operator.

        Returns a copy of this operator that does not need to be passed extra
        arguments when subsequently called or when using the ``act`` method.

        Providing positional arguments via 'args' is deprecated,
        instead provide named parameters as a dictionary via 'params'.
        """
        if args:
            # deprecate_args does not play nicely with methods of cdef classes,
            # when used as a decorator, so we manually raise the
            # deprecation warning here.
            deprecate_args()
        if args and params:
            raise TypeError("'args' and 'params' are mutually exclusive.")
        # generic creation of new instance
        cls = self.__class__
        q = cls.__new__(cls)
        q.syst = self.syst
        q.onsite = self.onsite
        q._onsite_param_names = self._onsite_param_names
        q.where = self.where
        q.sum = self.sum
        q._site_ranges = self._site_ranges
        q.check_hermiticity = self.check_hermiticity
        if callable(self.onsite):
            q._bound_onsite = self._eval_onsites(args, params)
        # NOTE: subclasses should populate `bound_hamiltonian` if needed
        return q

    def _operate(self, complex[:] out_data, complex[:] bra, complex[:] ket,
                 args, operation op, *, params=None):
        """Do an operation with the operator.

        Parameters
        ----------
        out_data : ndarray
            Output array, zero on entry. On exit should contain the required
            data.  What this means depends on the value of `op`, as does the
            length of the array.
        bra, ket : ndarray
            Wavefunctions defined over all the orbitals of the system.
            If `op` is `ACT` then `bra` is None.
        args : tuple
            The extra arguments to the Hamiltonian value functions and
            the operator ``onsite`` function.
            Deprecated in favor of 'params' (and mutually exclusive with it).
        op : operation
            The operation to perform.
            `MAT_ELS`: calculate matrix elements between `bra` and `ket`
            `ACT`: act on `ket` with the operator
        params : dict, optional
            Dictionary of parameter names and their values. Mutually exclusive
            with 'args'.
        """
        raise NotImplementedError()

    cdef BlockSparseMatrix _eval_onsites(self, args, params):
        """Evaluate the onsite matrices on all elements of `where`"""
        assert callable(self.onsite)
        assert not (args and params)
        check_hermiticity = self.check_hermiticity
        syst = self.syst

        _is_vectorized = is_vectorized(syst)

        if params:
            try:
                args = tuple(params[pn] for pn in self._onsite_param_names)
            except KeyError:
                missing = [p for p in self._onsite_param_names
                           if p not in params]
                msg = ('Operator is missing required arguments: ',
                       ', '.join(map('"{}"'.format, missing)))
                raise TypeError(''.join(msg))

        # Evaluate many onsites at once. See _LocalOperator.__init__
        # for an explanation the parameters.
        def eval_onsite(term_id, term_selector, which):
            if _is_vectorized:
                if term_id >= 0:
                    (sr, _), _ = syst.subgraphs[syst.terms[term_id].subgraph]
                else:
                    (_, sr), _ = syst.subgraphs[syst.terms[-term_id - 1].subgraph]
            else:
                sr, _ = term_id
            start_site, norbs, _ = self.syst.site_ranges[sr]
            # All sites selected by 'which' are part of the same site family.
            site_offsets = _select(self.where, which)[:, 0] - start_site
            data = self.onsite(sr, site_offsets, *args)
            _check_onsites(data, norbs, self.check_hermiticity)
            return data

        matrix_elements = _make_matrix_elements(eval_onsite, self._terms)
        offsets, norbs = _get_all_orbs(self.where, self._site_ranges)
        return  BlockSparseMatrix(self.where, offsets, norbs, matrix_elements)


    cdef BlockSparseMatrix _eval_hamiltonian(self, args, params):
        """Evaluate the Hamiltonian on all elements of `where`."""

        where = self.where
        syst = self.syst
        is_onsite = self.where.shape[1] == 1
        check_hermiticity = self.check_hermiticity

        if is_vectorized(self.syst):

            # Evaluate many Hamiltonian elements at once.
            # See _LocalOperator.__init__ for an explanation the parameters.
            def eval_hamiltonian(term_id, term_selector, which):
                herm_conj = term_id < 0
                assert not is_onsite or (is_onsite and not herm_conj)  # onsite terms are never hermitian conjugate
                if herm_conj:
                    term_id = -term_id - 1
                data = syst.hamiltonian_term(term_id, term_selector,
                                             args=args, params=params)
                if herm_conj:
                    data = data.conjugate().transpose(0, 2, 1)
                # Checks for data consistency
                (to_sr, from_sr), _ = syst.subgraphs[syst.terms[term_id].subgraph]
                to_norbs = syst.site_ranges[to_sr][1]
                from_norbs = syst.site_ranges[from_sr][1]
                if herm_conj:
                    to_norbs, from_norbs = from_norbs, to_norbs
                _check_hams(data, to_norbs, from_norbs, is_onsite and check_hermiticity)

                return data

        else:

            # Evaluate many Hamiltonian elements at once.
            # See _LocalOperator.__init__ for an explanation the parameters.
            def eval_hamiltonian(term_id, term_selector, which):
                if is_onsite:
                    data = [
                        syst.hamiltonian(where[i, 0], where[i, 0], *args, params=params)
                        for i in which
                    ]
                else:
                    data = [
                        syst.hamiltonian(where[i, 0], where[i, 1], *args, params=params)
                        for i in which
                    ]
                data = _normalize_matrix_blocks(data, len(which))
                # Checks for data consistency
                (to_sr, from_sr) = term_id
                to_norbs = syst.site_ranges[to_sr][1]
                from_norbs = syst.site_ranges[from_sr][1]
                _check_hams(data, to_norbs, from_norbs, is_onsite and check_hermiticity)

                return data

        matrix_elements = _make_matrix_elements(eval_hamiltonian, self._terms)
        offsets, norbs = _get_all_orbs(where, self._site_ranges)
        return  BlockSparseMatrix(where, offsets, norbs, matrix_elements)

    def __getstate__(self):
        return (
            (self.check_hermiticity, self.sum),
            (self.syst, self.onsite, self._onsite_param_names),
            tuple(map(np.asarray, (self.where, self._site_ranges))),
            (self._terms,),
            (self._bound_onsite, self._bound_hamiltonian),
        )

    def __setstate__(self, state):
        ((self.check_hermiticity, self.sum),
         (self.syst, self.onsite, self._onsite_param_names),
         (self.where, self._site_ranges),
         (self._terms,),
         (self._bound_onsite, self._bound_hamiltonian),
        ) = state


cdef class Density(_LocalOperator):
    """An operator for calculating general densities.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator.Density.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the operator. If a dict is given, it
        maps from site families to square matrices. If a function is given it
        must take the same arguments as the onsite Hamiltonian functions of the
        system.
    where : sequence of `int` or `~kwant.system.Site`, or callable, optional
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of integers. If a function is provided,
        it should take a single `int` or `~kwant.system.Site` (if ``syst`` is
        a finalized builder) and return True or False.  If not provided, the
        operator will be calculated over all sites in the system.
    check_hermiticity: bool
        Check whether the provided ``onsite`` is Hermitian. If it is not
        Hermitian, then an error will be raised when the operator is
        evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `~kwant.operator.Density.__call__` for details).

    Notes
    -----
    In general, if there is a certain "density" (e.g. charge or spin) that is
    represented by a square matrix :math:`M_i` associated with each site
    :math:`i` then an instance of this class represents the tensor
    :math:`Q_{iαβ}` which is equal to :math:`M_i` when α and β are orbitals on
    site :math:`i`, and zero otherwise.
    """

    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *,
                 check_hermiticity=True, sum=False):
        where = _normalize_site_where(syst, where)
        super().__init__(syst, onsite, where,
                         check_hermiticity=check_hermiticity, sum=sum)

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _operate(self, complex[:] out_data, complex[:] bra, complex[:] ket,
                 args, operation op, *, params=None):
        matrix = ta.matrix
        cdef int unique_onsite = not callable(self.onsite)
        # prepare onsite matrices
        cdef complex[:, :] _tmp_mat
        cdef complex *M_a = NULL
        cdef BlockSparseMatrix M_a_blocks

        if unique_onsite:
            _tmp_mat = self.onsite
            M_a = <complex*> &_tmp_mat[0, 0]
        elif self._bound_onsite:
            M_a_blocks = self._bound_onsite
        else:
            M_a_blocks = self._eval_onsites(args, params)

        # loop-local variables
        cdef gint a, a_s, a_norbs
        cdef gint i, j, w
        cdef complex tmp, bra_conj
        ### loop over sites
        for w in range(self.where.shape[0]):
            ### get the next site, start orbital and number of orbitals
            a = self.where[w, 0]
            _get_orbs(self._site_ranges, a, &a_s, &a_norbs)
            ### get the next onsite matrix, if necessary
            if not unique_onsite:
                M_a = M_a_blocks.get(w)
            ### do the actual calculation
            if op == MAT_ELS:
                tmp = 0
                for i in range(a_norbs):
                    for j in range(a_norbs):
                        tmp += (bra[a_s + i].conjugate() *
                                M_a[i * a_norbs + j] * ket[a_s + j])
                out_data[w] = tmp
            elif op == ACT:
                for i in range(a_norbs):
                    tmp = 0
                    for j in range(a_norbs):
                        tmp += M_a[i * a_norbs + j] * ket[a_s + j]
                    out_data[a_s + i] = out_data[a_s + i] + tmp

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    @cython.embedsignature
    def tocoo(self, args=(), *, params=None):
        """Convert the operator to coordinate format sparse matrix.

        Providing positional arguments via 'args' is deprecated,
        instead provide named parameters as a dictionary via 'params'.
        """
        cdef int blk, blk_size, n_blocks, n, k = 0
        cdef int [:, :] offsets, shapes
        cdef int [:] row, col
        if self._bound_onsite and (args or params):
           raise ValueError("Extra arguments are already bound to this "
                            "operator. You should call this operator "
                            "providing neither 'args' nor 'params'.")
        if args:
            # deprecate_args does not play nicely with methods of cdef classes,
            # when used as a decorator, so we manually raise the
            # deprecation warning here.
            deprecate_args()
        if args and params:
            raise TypeError("'args' and 'params' are mutually exclusive.")

        if not callable(self.onsite):
            offsets = _get_all_orbs(self.where, self._site_ranges)[0]
            n_blocks = len(self.where)
            shapes = np.asarray(np.resize([self.onsite.shape], (n_blocks, 2)),
                                gint_dtype)
            data = np.asarray(self.onsite).flatten()
            data = np.resize(data, [len(data) * n_blocks])
        else:
            if self._bound_onsite is not None:
                onsite_matrix = self._bound_onsite
            else:
                onsite_matrix = self._eval_onsites(args, params)
            data = onsite_matrix.data
            offsets = np.asarray(onsite_matrix.block_offsets)
            shapes = np.asarray(onsite_matrix.block_shapes)

        row = np.empty(len(data), gint_dtype)
        col = np.empty(len(data), gint_dtype)
        for blk in range(len(offsets)):
            blk_size = shapes[blk, 0] * shapes[blk, 1]
            for n in range(blk_size):
                row[k] = offsets[blk, 0] + n // shapes[blk, 1]
                col[k] = offsets[blk, 1] + n % shapes[blk, 1]
                k += 1

        norbs = _get_tot_norbs(self.syst)
        return coo_matrix((np.asarray(data),
                           (np.asarray(row), np.asarray(col))),
                          shape=(norbs, norbs))


cdef class Current(_LocalOperator):
    r"""An operator for calculating general currents.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator.Current.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this current is
        derived. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    where : sequence of pairs of `int` or `~kwant.system.Site`, or callable, optional
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of pairs of integers. If a function is
        provided, it should take a pair of integers or a pair of
        `~kwant.system.Site` (if ``syst`` is a finalized builder) and return
        True or False.  If not provided, the operator will be calculated over
        all hoppings in the system.
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it
        is not Hermitian, then an error will be raised when the
        operator is evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `~kwant.operator.Current.__call__` for details).

    Notes
    -----
    In general, if there is a certain "density" (e.g. charge or spin) that is
    represented by a square matrix :math:`M_i` associated with each site
    :math:`i` and :math:`H_{ij}` is the hopping Hamiltonian from site :math:`j`
    to site `i`, then an instance of this class represents the tensor
    :math:`J_{ijαβ}` which is equal to :math:`i\left[(H_{ij})^† M_i - M_i
    H_{ij}\right]` when α and β are orbitals on sites :math:`i` and :math:`j`
    respectively, and zero otherwise.

    The tensor :math:`J_{ijαβ}` will also be referred to as :math:`Q_{nαβ}`,
    where :math:`n` is the index of hopping :math:`(i, j)` in ``where``.
    """

    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *,
                 check_hermiticity=True, sum=False):
        where = _normalize_hopping_where(syst, where)
        super().__init__(syst, onsite, where,
                         check_hermiticity=check_hermiticity, sum=sum)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        """Bind the given arguments to this operator.

        Returns a copy of this operator that does not need to be passed extra
        arguments when subsequently called or when using the ``act`` method.

        Providing positional arguments via 'args' is deprecated,
        instead provide named parameters as a dictionary via 'params'.
        """
        q = super().bind(args, params=params)
        q._bound_hamiltonian = self._eval_hamiltonian(args, params)
        return q

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _operate(self, complex[:] out_data, complex[:] bra, complex[:] ket,
                 args, operation op, *, params=None):
        # prepare onsite matrices and hamiltonians
        cdef int unique_onsite = not callable(self.onsite)
        cdef complex[:, :] _tmp_mat
        cdef complex *M_a = NULL
        cdef complex *H_ab = NULL
        cdef BlockSparseMatrix M_a_blocks, H_ab_blocks

        if unique_onsite:
            _tmp_mat = self.onsite
            M_a = <complex*> &_tmp_mat[0, 0]
        elif self._bound_onsite:
            M_a_blocks = self._bound_onsite
        else:
            M_a_blocks = self._eval_onsites(args, params)

        if self._bound_hamiltonian:
            H_ab_blocks = self._bound_hamiltonian
        else:
            H_ab_blocks = self._eval_hamiltonian(args, params)

        # main loop
        cdef gint a, a_s, a_norbs, b, b_s, b_norbs
        cdef gint i, j, k, w
        cdef complex tmp
        for w in range(self.where.shape[0]):
            ### get the next hopping's start orbitals and numbers of orbitals
            a_s = H_ab_blocks.block_offsets[w, 0]
            b_s = H_ab_blocks.block_offsets[w, 1]
            a_norbs = H_ab_blocks.block_shapes[w, 0]
            b_norbs = H_ab_blocks.block_shapes[w, 1]
            ### get the next onsite and Hamiltonian matrices
            H_ab = H_ab_blocks.get(w)
            if not unique_onsite:
                M_a = M_a_blocks.get(w)
            ### do the actual calculation
            if op == MAT_ELS:
                tmp = 0
                for i in range(b_norbs):
                    for j in range(a_norbs):
                        for k in range(a_norbs):
                            tmp += (bra[b_s + i].conjugate() *
                                    H_ab[j * b_norbs + i].conjugate() *
                                    M_a[j * a_norbs + k] * ket[a_s + k]
                                  - bra[a_s + j].conjugate() *
                                    M_a[j * a_norbs + k] *
                                    H_ab[k * b_norbs + i] * ket[b_s + i])
                out_data[w] = 1j * tmp
            elif op == ACT:
                for i in range(b_norbs):
                    for j in range(a_norbs):
                        for k in range(a_norbs):
                            out_data[b_s + i] = (
                                out_data[b_s + i] +
                                1j * H_ab[j * b_norbs + i].conjugate() *
                                M_a[j * a_norbs + k] * ket[a_s + k])
                            out_data[a_s + j] = (
                                out_data[a_s + j] -
                                1j * M_a[j * a_norbs + k] * H_ab[k * b_norbs + i] *
                                ket[b_s + i])


cdef class Source(_LocalOperator):
    """An operator for calculating general sources.

    An instance of this class can be called like a function to evaluate the
    expectation value with a wavefunction. See
    `~kwant.operator.Source.__call__` for details.

    Parameters
    ----------
    syst : `~kwant.system.System`
    onsite : scalar or square matrix or dict or callable
        The onsite matrix that defines the density from which this source is
        defined. If a dict is given, it maps from site families to square
        matrices (scalars are allowed if the site family has 1 orbital per
        site). If a function is given it must take the same arguments as the
        onsite Hamiltonian functions of the system.
    where : sequence of `int` or `~kwant.system.Site`, or callable, optional
        Where to evaluate the operator. If ``syst`` is not a finalized Builder,
        then this should be a sequence of integers. If a function is provided,
        it should take a single `int` or `~kwant.system.Site` (if ``syst`` is
        a finalized builder) and return True or False.  If not provided, the
        operator will be calculated over all sites in the system.
    check_hermiticity : bool
        Check whether the provided ``onsite`` is Hermitian. If it is not
        Hermitian, then an error will be raised when the operator is
        evaluated.
    sum : bool, default: False
        If True, then calling this operator will return a single scalar,
        otherwise a vector will be returned (see
        `~kwant.operator.Source.__call__` for details).

    Notes
    -----
    An example of a "source" is a spin torque. In general, if there is a
    certain "density" (e.g. charge or spin) that is represented by  a square
    matrix :math:`M_i` associated with each site :math:`i`, and :math:`H_{i}`
    is the onsite Hamiltonian on site site `i`, then an instance of this class
    represents the tensor :math:`Q_{iαβ}` which is equal to :math:`(H_{i})^†
    M_i - M_i H_{i}` when α and β are orbitals on site :math:`i`, and zero
    otherwise.
    """

    @cython.embedsignature
    def __init__(self, syst, onsite=1, where=None, *,
                 check_hermiticity=True, sum=False):
        where = _normalize_site_where(syst, where)
        super().__init__(syst, onsite, where,
                         check_hermiticity=check_hermiticity, sum=sum)

    @cython.embedsignature
    def bind(self, args=(), *, params=None):
        """Bind the given arguments to this operator.

        Returns a copy of this operator that does not need to be passed extra
        arguments when subsequently called or when using the ``act`` method.

        Providing positional arguments via 'args' is deprecated,
        instead provide named parameters as a dictionary via 'params'.
        """
        q = super().bind(args, params=params)
        q._bound_hamiltonian = self._eval_hamiltonian(args, params)
        return q

    @cython.boundscheck(False)
    @cython.wraparound(False)
    def _operate(self, complex[:] out_data, complex[:] bra, complex[:] ket,
                 args, operation op, *, params=None):
        # prepare onsite matrices and hamiltonians
        cdef int unique_onsite = not callable(self.onsite)
        cdef complex[:, :] _tmp_mat
        cdef complex *M_a = NULL
        cdef complex *H_aa = NULL
        cdef BlockSparseMatrix M_a_blocks, H_aa_blocks

        if unique_onsite:
            _tmp_mat = self.onsite
            M_a = <complex*> &_tmp_mat[0, 0]
        elif self._bound_onsite:
            M_a_blocks = self._bound_onsite
        else:
            M_a_blocks = self._eval_onsites(args, params)

        if self._bound_hamiltonian:
            H_aa_blocks = self._bound_hamiltonian
        else:
            H_aa_blocks = self._eval_hamiltonian(args, params)

        # main loop
        cdef gint a, a_s, a_norbs
        cdef gint i, j, k, w
        cdef complex tmp, tmp2
        for w in range(self.where.shape[0]):
            ### get the next site, start orbital and number of orbitals
            # row offsets and block size are the same as for columns, as
            # we are only dealing with the block-diagonal part of H
            a_s = H_aa_blocks.block_offsets[w, 0]
            a_norbs = H_aa_blocks.block_shapes[w, 0]
            ### get the next onsite and Hamiltonian matrices
            H_aa = H_aa_blocks.get(w)
            if not unique_onsite:
                M_a = M_a_blocks.get(w)
            ### do the actual calculation
            if op == MAT_ELS:
                tmp2 = 0
                for i in range(a_norbs):
                    tmp = 0
                    for j in range(a_norbs):
                        for k in range(a_norbs):
                            tmp += (H_aa[j * a_norbs + i].conjugate() *
                                    M_a[j * a_norbs + k] * ket[a_s + k]
                                  - M_a[i * a_norbs + j] *
                                    H_aa[j * a_norbs + k] * ket[a_s + k])
                    tmp2 += bra[a_s + i].conjugate() * tmp
                out_data[w] = 1j * tmp2
            elif op == ACT:
                for i in range(a_norbs):
                    tmp = 0
                    for j in range(a_norbs):
                        for k in range(a_norbs):
                            tmp += (H_aa[j * a_norbs + i].conjugate() *
                                    M_a[j * a_norbs + k] * ket[a_s + k]
                                  - M_a[i * a_norbs + j] *
                                    H_aa[j * a_norbs + k] * ket[a_s + k])
                    out_data[a_s + i] = out_data[a_s + i] + 1j * tmp
