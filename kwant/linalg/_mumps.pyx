# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

cimport numpy as np
import numpy as np
from . cimport cmumps
from . import cmumps
from .fortran_helpers import assert_fortran_matvec, assert_fortran_mat

int_dtype = cmumps.int_dtype

# Proxy classes for Python access to the control and info parameters of MUMPS

cdef class mumps_int_array:
    cdef cmumps.MUMPS_INT *array

    def __init__(self):
        self.array = NULL

    def __getitem__(self, key):
        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value


# workaround for the fact that cython cannot pass pointers to an __init__()
cdef make_mumps_int_array(cmumps.MUMPS_INT *array):
    wrapper = mumps_int_array()
    wrapper.array = array
    return wrapper


cdef class smumps_real_array:
    cdef cmumps.SMUMPS_REAL *array

    def __init__(self):
        self.array = NULL

    def __getitem__(self, key):
        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value


cdef make_smumps_real_array(cmumps.SMUMPS_REAL *array):
    wrapper = smumps_real_array()
    wrapper.array = array
    return wrapper


cdef class dmumps_real_array:
    cdef cmumps.DMUMPS_REAL *array

    def __init__(self):
        self.array = NULL

    def __getitem__(self, key):
        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value


cdef make_dmumps_real_array(cmumps.DMUMPS_REAL *array):
    wrapper = dmumps_real_array()
    wrapper.array = array
    return wrapper


cdef class cmumps_real_array:
    cdef cmumps.CMUMPS_REAL *array

    def __init__(self):
        self.array = NULL

    def __getitem__(self, key):
        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value


cdef make_cmumps_real_array(cmumps.CMUMPS_REAL *array):
    wrapper = cmumps_real_array()
    wrapper.array = array
    return wrapper


cdef class zmumps_real_array:
    cdef cmumps.ZMUMPS_REAL *array

    def __init__(self):
        self.array = NULL

    def __getitem__(self, key):
        return self.array[key - 1]

    def __setitem__(self, key, value):
        self.array[key - 1] = value


cdef make_zmumps_real_array(cmumps.ZMUMPS_REAL *array):
    wrapper = zmumps_real_array()
    wrapper.array = array
    return wrapper

#############################################################

cdef class zmumps:
    cdef cmumps.ZMUMPS_STRUC_C params

    cdef public mumps_int_array icntl
    cdef public zmumps_real_array cntl
    cdef public mumps_int_array info
    cdef public mumps_int_array infog
    cdef public zmumps_real_array rinfo
    cdef public zmumps_real_array rinfog

    def __init__(self, verbose=False, sym=0):
        self.params.job = -1
        self.params.sym = sym
        self.params.par = 1
        self.params.comm_fortran = -987654

        cmumps.zmumps_c(&self.params)

        self.icntl = make_mumps_int_array(self.params.icntl)
        self.cntl = make_zmumps_real_array(self.params.cntl)
        self.info = make_mumps_int_array(self.params.info)
        self.infog = make_mumps_int_array(self.params.infog)
        self.rinfo = make_zmumps_real_array(self.params.rinfo)
        self.rinfog = make_zmumps_real_array(self.params.rinfog)

        # no diagnostic output (MUMPS is very verbose normally)
        if not verbose:
            self.icntl[1] = 0
            self.icntl[3] = 0

    def __dealloc__(self):
        self.params.job = -2
        cmumps.zmumps_c(&self.params)

    def call(self):
        cmumps.zmumps_c(&self.params)

    def _set_job(self, value):
        self.params.job = value

    def _get_job(self):
        return self.params.job

    job = property(_get_job, _set_job)

    @property
    def sym(self):
        return self.params.sym

    def set_assembled_matrix(self,
                             cmumps.MUMPS_INT N,
                             np.ndarray[cmumps.MUMPS_INT, ndim=1] i,
                             np.ndarray[cmumps.MUMPS_INT, ndim=1] j,
                             np.ndarray[np.complex128_t, ndim=1] a):
        self.params.n = N
        self.params.nz = a.shape[0]
        self.params.irn = <cmumps.MUMPS_INT *>i.data
        self.params.jcn = <cmumps.MUMPS_INT *>j.data
        self.params.a = <cmumps.ZMUMPS_COMPLEX *>a.data

    def set_dense_rhs(self, np.ndarray rhs):

        assert_fortran_matvec(rhs)
        if rhs.dtype != np.complex128:
            raise ValueError("numpy array must be of dtype complex128!")

        if rhs.ndim == 1:
            self.params.nrhs = 1
        else:
            self.params.nrhs = rhs.shape[1]
        self.params.lrhs = rhs.shape[0]
        self.params.rhs = <cmumps.ZMUMPS_COMPLEX *>rhs.data

    def set_sparse_rhs(self,
                       np.ndarray[cmumps.MUMPS_INT, ndim=1] col_ptr,
                       np.ndarray[cmumps.MUMPS_INT, ndim=1] row_ind,
                       np.ndarray[np.complex128_t, ndim=1] data):

        if row_ind.shape[0] != data.shape[0]:
            raise ValueError("Number of entries in row index and value "
                             "array differ!")

        self.params.nz_rhs = data.shape[0]
        self.params.nrhs = col_ptr.shape[0] - 1
        self.params.rhs_sparse = <cmumps.ZMUMPS_COMPLEX *>data.data
        self.params.irhs_sparse = <cmumps.MUMPS_INT *>row_ind.data
        self.params.irhs_ptr = <cmumps.MUMPS_INT *>col_ptr.data

    def set_schur(self,
                  np.ndarray[np.complex128_t, ndim=2, mode='c'] schur,
                  np.ndarray[cmumps.MUMPS_INT, ndim=1] schur_vars):

        if schur.shape[0] != schur.shape[1]:
            raise ValueError("Schur matrix must be squared!")
        if schur.shape[0] != schur_vars.shape[0]:
            raise ValueError("Number of Schur variables must agree "
                             "with Schur complement size!")

        self.params.size_schur = schur.shape[0]
        self.params.schur = <cmumps.ZMUMPS_COMPLEX *>schur.data
        self.params.listvar_schur = <cmumps.MUMPS_INT *>schur_vars.data
