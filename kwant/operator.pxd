from .graph.defs cimport gint
from .graph.defs import gint_dtype


cdef gint _bisect(gint[:] a, gint x)

cdef int _is_herm_conj(complex[:, :] a, complex[:, :] b,
                       double atol=*, double rtol=*) except -1

cdef _select(gint[:, :] arr, gint[:] indexes)

cdef int _check_onsite(complex[:, :] M, gint norbs,
                       int check_hermiticity) except -1

cdef int _check_ham(complex[:, :] H, ham, args, params,
                    gint a, gint a_norbs, gint b, gint b_norbs,
                    int check_hermiticity) except -1

cdef void _get_orbs(gint[:, :] site_ranges, gint site,
                    gint *start_orb, gint *norbs)


cdef class BlockSparseMatrix:
    cdef public gint[:, :] block_offsets, block_shapes
    cdef public gint[:] data_offsets
    cdef public complex[:] data

    cdef complex* get(self, gint block_idx)


cdef class _LocalOperator:
    cdef public int check_hermiticity, sum
    cdef public object syst, onsite, _onsite_param_names, _terms
    cdef public gint[:, :]  where, _site_ranges
    cdef public BlockSparseMatrix _bound_onsite, _bound_hamiltonian

    cdef BlockSparseMatrix _eval_onsites(self, args, params)
    cdef BlockSparseMatrix _eval_hamiltonian(self, args, params)


cdef class Density(_LocalOperator):
    pass


cdef class Current(_LocalOperator):
    pass


cdef class Source(_LocalOperator):
    pass
