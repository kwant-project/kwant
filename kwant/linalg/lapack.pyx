# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Low-level access to LAPACK functions. """

__all__ = ['getrf',
           'getrs',
           'gecon',
           'ggev',
           'gees',
           'trsen',
           'strevc', 'dtrevc', 'ctrevc', 'ztrevc',
           'sgges', 'dgges', 'cgges', 'zgges',
           'stgsen', 'dtgsen', 'ctgsen', 'ztgsen',
           'stgevc', 'dtgevc', 'ctgevc', 'ztgevc',
           'prepare_for_lapack']

import numpy as np
cimport numpy as np

cimport scipy.linalg.cython_lapack as lapack

ctypedef int l_int
ctypedef bint l_logical

int_dtype = np.int32
logical_dtype = np.int32

ctypedef float complex float_complex
ctypedef double complex double_complex

ctypedef fused scalar:
    float
    double
    float complex
    double complex

ctypedef fused single_precision:
    float
    float complex

ctypedef fused double_precision:
    double
    double complex

ctypedef fused cmplx:
    float complex
    double complex

ctypedef fused floating:
    float
    double

# exceptions

class LinAlgError(RuntimeError):
    pass


# some helper functions
def filter_args(select, args):
    return tuple([arg for sel, arg in zip(select, args) if sel])

def assert_fortran_mat(*mats):
    # This is a workaround for a bug in NumPy version < 2.0,
    # where 1x1 matrices do not have the F_Contiguous flag set correctly.
    for mat in mats:
        if (mat is not None and (mat.shape[0] > 1 or mat.shape[1] > 1) and
            not mat.flags["F_CONTIGUOUS"]):
            raise ValueError("Input matrix must be Fortran contiguous")


def getrf(np.ndarray[scalar, ndim=2] A):
    cdef l_int M, N, info
    cdef np.ndarray[l_int] ipiv

    assert_fortran_mat(A)

    M = A.shape[0]
    N = A.shape[1]
    ipiv = np.empty(min(M,N), dtype = int_dtype)

    if scalar is float:
        lapack.sgetrf(&M, &N, <float *>A.data, &M,
                      <l_int *>ipiv.data, &info)
    elif scalar is double:
        lapack.dgetrf(&M, &N, <double *>A.data, &M,
                      <l_int *>ipiv.data, &info)
    elif scalar is float_complex:
        lapack.cgetrf(&M, &N, <float complex *>A.data, &M,
                      <l_int *>ipiv.data, &info)
    elif scalar is double_complex:
        lapack.zgetrf(&M, &N, <double complex *>A.data, &M,
                      <l_int *>ipiv.data, &info)

    assert info >= 0, "Argument error in getrf"

    return (A, ipiv, info > 0 or M != N)


def getrs(np.ndarray[scalar, ndim=2] LU, np.ndarray[l_int] IPIV,
          np.ndarray B):
    cdef l_int N, NRHS, info

    assert_fortran_mat(LU)

    # Consistency checks for LU and B

    if B.descr.type_num != LU.descr.type_num:
        raise TypeError('B must have same dtype as LU')

    # Workaround for 1x1-Fortran bug in NumPy < v2.0
    if ((B.ndim == 2 and (B.shape[0] > 1 or B.shape[1] > 1) and
         not B.flags["F_CONTIGUOUS"])):
        raise ValueError("B must be Fortran ordered")

    if B.ndim > 2:
        raise ValueError("B must be a vector or matrix")

    if LU.shape[0] != B.shape[0]:
        raise ValueError('LU and B have incompatible shapes')

    N = LU.shape[0]

    if B.ndim == 1:
        NRHS = 1
    elif B.ndim == 2:
        NRHS = B.shape[1]

    if scalar is float:
        lapack.sgetrs("N", &N, &NRHS, <float *>LU.data, &N,
                      <l_int *>IPIV.data, <float *>B.data, &N,
                      &info)
    elif scalar is double:
        lapack.dgetrs("N", &N, &NRHS, <double *>LU.data, &N,
                      <l_int *>IPIV.data, <double *>B.data, &N,
                      &info)
    elif scalar is float_complex:
        lapack.cgetrs("N", &N, &NRHS, <float complex *>LU.data, &N,
                      <l_int *>IPIV.data, <float complex *>B.data, &N,
                      &info)
    elif scalar is double_complex:
        lapack.zgetrs("N", &N, &NRHS, <double complex *>LU.data, &N,
                      <l_int *>IPIV.data, <double complex *>B.data, &N,
                      &info)

    assert info == 0, "Argument error in getrs"

    return B


def gecon(np.ndarray[scalar, ndim=2] LU, double normA, char *norm = b"1"):
    cdef l_int N, info
    cdef float srcond, snormA
    cdef double drcond

    # Parameter checks

    assert_fortran_mat(LU)
    if norm[0] != b"1" and norm[0] != b"I":
        raise ValueError("'norm' must be either '1' or 'I'")
    if scalar in single_precision:
        snormA = normA

    # Allocate workspaces

    N = LU.shape[0]

    cdef np.ndarray[l_int] iwork
    if scalar in floating:
        iwork = np.empty(N, dtype=int_dtype)

    cdef np.ndarray[scalar] work
    if scalar in floating:
        work = np.empty(4 * N, dtype=LU.dtype)
    else:
        work = np.empty(2 * N, dtype=LU.dtype)

    cdef np.ndarray rwork
    if scalar is float_complex:
        rwork = np.empty(2 * N, dtype=np.float32)
    elif scalar is double_complex:
        rwork = np.empty(2 * N, dtype=np.float64)

    # The actual calculation

    if scalar is float:
        lapack.sgecon(norm, &N, <float *>LU.data, &N, &snormA,
                      &srcond, <float *>work.data,
                      <l_int *>iwork.data, &info)
    elif scalar is double:
        lapack.dgecon(norm, &N, <double *>LU.data, &N, &normA,
                      &drcond, <double *>work.data,
                      <l_int *>iwork.data, &info)
    elif scalar is float_complex:
        lapack.cgecon(norm, &N, <float complex *>LU.data, &N, &snormA,
                      &srcond, <float complex *>work.data,
                      <float *>rwork.data, &info)
    elif scalar is double_complex:
        lapack.zgecon(norm, &N, <double complex *>LU.data, &N, &normA,
                      &drcond, <double complex *>work.data,
                      <double *>rwork.data, &info)

    assert info == 0, "Argument error in gecon"

    if scalar in single_precision:
        return srcond
    else:
        return drcond


# Helper function for xGGEV
def ggev_postprocess(dtype, alphar, alphai, vl_r=None, vr_r=None):
    # depending on whether the eigenvalues are purely real or complex,
    # some post-processing of the eigenvalues and -vectors is necessary

    indx = (alphai > 0.0).nonzero()[0]

    if indx.size:
        alpha = alphar + 1j * alphai

        if vl_r is not None:
            vl = np.array(vl_r, dtype = dtype)
            for i in indx:
                vl.imag[:, i] = vl_r[:,i+1]
                vl[:, i+1] = np.conj(vl[:, i])
        else:
            vl = None

        if vr_r is not None:
            vr = np.array(vr_r, dtype = dtype)
            for i in indx:
                vr.imag[:, i] = vr_r[:,i+1]
                vr[:, i+1] = np.conj(vr[:, i])
        else:
            vr = None
    else:
        alpha = alphar
        vl = vl_r
        vr = vr_r

    return (alpha, vl, vr)


def ggev(np.ndarray[scalar, ndim=2] A, np.ndarray[scalar, ndim=2] B,
         left=False, right=True):
    cdef l_int N, info, lwork

    # Parameter checks

    assert_fortran_mat(A, B)

    if A.ndim != 2 or A.ndim != 2:
        raise ValueError("gen_eig requires both a and be to be matrices")

    if A.shape[0] != A.shape[1]:
        raise ValueError("gen_eig requires square matrix input")

    if A.shape[0] != B.shape[0] or A.shape[1] != B.shape[1]:
        raise ValueError("A and B do not have the same shape")

    # Allocate workspaces

    N = A.shape[0]

    cdef np.ndarray[scalar] alphar, alphai
    if scalar in cmplx:
        alphar = np.empty(N, dtype=A.dtype)
        alphai = None
    else:
        alphar = np.empty(N, dtype=A.dtype)
        alphai = np.empty(N, dtype=A.dtype)

    cdef np.ndarray[scalar] beta = np.empty(N, dtype=A.dtype)

    cdef np.ndarray rwork = None
    if scalar is float_complex:
        rwork = np.empty(8 * N, dtype=np.float32)
    elif scalar is double_complex:
        rwork = np.empty(8 * N, dtype=np.float64)

    cdef np.ndarray vl
    cdef scalar *vl_ptr
    cdef char *jobvl
    if left:
        vl = np.empty((N,N), dtype=A.dtype, order='F')
        vl_ptr = <scalar *>vl.data
        jobvl = "V"
    else:
        vl = None
        vl_ptr = NULL
        jobvl = "N"

    cdef np.ndarray vr
    cdef scalar *vr_ptr
    cdef char *jobvr
    if right:
        vr = np.empty((N,N), dtype=A.dtype, order='F')
        vr_ptr = <scalar *>vr.data
        jobvr = "V"
    else:
        vr = None
        vr_ptr = NULL
        jobvr = "N"

    # Workspace query
    # Xggev expects &qwork as a <scalar *> (even though it's an integer)
    lwork = -1
    cdef scalar qwork

    if scalar is float:
        lapack.sggev(jobvl, jobvr, &N, <float *>A.data, &N,
                     <float *>B.data, &N,
                     <float *>alphar.data, <float *> alphai.data,
                     <float *>beta.data,
                     vl_ptr, &N, vr_ptr, &N,
                     &qwork, &lwork, &info)
    elif scalar is double:
        lapack.dggev(jobvl, jobvr, &N, <double *>A.data, &N,
                     <double *>B.data, &N,
                     <double *>alphar.data, <double *> alphai.data,
                     <double *>beta.data,
                     vl_ptr, &N, vr_ptr, &N,
                     &qwork, &lwork, &info)
    elif scalar is float_complex:
        lapack.cggev(jobvl, jobvr, &N, <float complex *>A.data, &N,
                     <float complex *>B.data, &N,
                     <float complex *>alphar.data, <float complex *>beta.data,
                     vl_ptr, &N, vr_ptr, &N,
                     &qwork, &lwork,
                     <float *>rwork.data, &info)
    elif scalar is double_complex:
        lapack.zggev(jobvl, jobvr, &N, <double complex *>A.data, &N,
                     <double complex *>B.data, &N,
                     <double complex *>alphar.data, <double complex *>beta.data,
                     vl_ptr, &N, vr_ptr, &N,
                     &qwork, &lwork,
                     <double *>rwork.data, &info)

    assert info == 0, "Argument error in ggev"

    if scalar in floating:
        lwork = <l_int>qwork
    else:
        lwork = <l_int>qwork.real
    cdef np.ndarray[scalar] work = np.empty(lwork, dtype=A.dtype)

    # The actual calculation

    if scalar is float:
        lapack.sggev(jobvl, jobvr, &N, <float *>A.data, &N,
                     <float *>B.data, &N,
                     <float *>alphar.data, <float *> alphai.data,
                     <float *>beta.data,
                     vl_ptr, &N, vr_ptr, &N,
                     <float *>work.data, &lwork, &info)
    elif scalar is double:
        lapack.dggev(jobvl, jobvr, &N, <double *>A.data, &N,
                     <double *>B.data, &N,
                     <double *>alphar.data, <double *> alphai.data,
                     <double *>beta.data,
                     vl_ptr, &N, vr_ptr, &N,
                     <double *>work.data, &lwork, &info)
    elif scalar is float_complex:
        lapack.cggev(jobvl, jobvr, &N, <float complex *>A.data, &N,
                     <float complex *>B.data, &N,
                     <float complex *>alphar.data, <float complex *>beta.data,
                     vl_ptr, &N, vr_ptr, &N,
                     <float complex *>work.data, &lwork,
                     <float *>rwork.data, &info)
    elif scalar is double_complex:
        lapack.zggev(jobvl, jobvr, &N, <double complex *>A.data, &N,
                     <double complex *>B.data, &N,
                     <double complex *>alphar.data, <double complex *>beta.data,
                     vl_ptr, &N, vr_ptr, &N,
                     <double complex *>work.data, &lwork,
                     <double *>rwork.data, &info)

    if info > 0:
        raise LinAlgError("QZ iteration failed to converge in sggev")

    assert info == 0, "Argument error in ggev"

    if scalar is float:
        post_dtype = np.complex64
    elif scalar is double:
        post_dtype = np.complex128

    cdef np.ndarray alpha
    alpha = alphar
    if scalar in floating:
        alpha, vl, vr = ggev_postprocess(post_dtype, alphar, alphai, vl, vr)

    return filter_args((True, True, left, right), (alpha, beta, vl, vr))


def gees(np.ndarray[scalar, ndim=2] A, calc_q=True, calc_ev=True):
    cdef l_int N, lwork, sdim, info

    assert_fortran_mat(A)

    if A.ndim != 2:
        raise ValueError("Expect matrix as input")

    if A.shape[0] != A.shape[1]:
        raise ValueError("Expect square matrix")

    # Allocate workspaces

    N = A.shape[0]

    cdef np.ndarray[scalar] wr, wi
    if scalar in cmplx:
        wr = np.empty(N, dtype=A.dtype)
        wi = None
    else:
        wr = np.empty(N, dtype=A.dtype)
        wi = np.empty(N, dtype=A.dtype)

    cdef np.ndarray rwork
    if scalar is float_complex:
        rwork = np.empty(N, dtype=np.float32)
    elif scalar is double_complex:
        rwork = np.empty(N, dtype=np.float64)

    cdef char *jobvs
    cdef scalar *vs_ptr
    cdef np.ndarray[scalar, ndim=2] vs
    if calc_q:
        vs = np.empty((N,N), dtype=A.dtype, order='F')
        vs_ptr = <scalar *>vs.data
        jobvs = "V"
    else:
        vs = None
        vs_ptr = NULL
        jobvs = "N"

    # Workspace query
    # Xgees expects &qwork as a <scalar *> (even though it's an integer)
    lwork = -1
    cdef scalar qwork

    if scalar is float:
        lapack.sgees(jobvs, "N", NULL, &N, <float *>A.data, &N,
                     &sdim, <float *>wr.data, <float *>wi.data, vs_ptr, &N,
                     &qwork, &lwork, NULL, &info)
    elif scalar is double:
        lapack.dgees(jobvs, "N", NULL, &N, <double *>A.data, &N,
                     &sdim, <double *>wr.data, <double *>wi.data, vs_ptr, &N,
                     &qwork, &lwork, NULL, &info)
    elif scalar is float_complex:
        lapack.cgees(jobvs, "N", NULL, &N, <float complex *>A.data, &N,
                     &sdim, <float complex *>wr.data, vs_ptr, &N,
                     &qwork, &lwork, <float *>rwork.data, NULL, &info)
    elif scalar is double_complex:
        lapack.zgees(jobvs, "N", NULL, &N, <double complex *>A.data, &N,
                     &sdim, <double complex *>wr.data, vs_ptr, &N,
                     &qwork, &lwork, <double *>rwork.data, NULL, &info)

    assert info == 0, "Argument error in sgees"

    if scalar in floating:
        lwork = <l_int>qwork
    else:
        lwork = <l_int>qwork.real
    cdef np.ndarray[scalar] work = np.empty(lwork, dtype=A.dtype)

    # The actual calculation

    if scalar is float:
        lapack.sgees(jobvs, "N", NULL, &N, <float *>A.data, &N,
                     &sdim, <float *>wr.data, <float *>wi.data, vs_ptr, &N,
                     <float *>work.data, &lwork, NULL, &info)
    elif scalar is double:
        lapack.dgees(jobvs, "N", NULL, &N, <double *>A.data, &N,
                     &sdim, <double *>wr.data, <double *>wi.data, vs_ptr, &N,
                     <double *>work.data, &lwork, NULL, &info)
    elif scalar is float_complex:
        lapack.cgees(jobvs, "N", NULL, &N, <float complex *>A.data, &N,
                     &sdim, <float complex *>wr.data, vs_ptr, &N,
                     <float complex *>work.data, &lwork,
                     <float *>rwork.data, NULL, &info)
    elif scalar is double_complex:
        lapack.zgees(jobvs, "N", NULL, &N, <double complex *>A.data, &N,
                     &sdim, <double complex *>wr.data, vs_ptr, &N,
                     <double complex *>work.data, &lwork,
                     <double *>rwork.data, NULL, &info)

    if info > 0:
        raise LinAlgError("QR iteration failed to converge in gees")

    assert info == 0, "Argument error in gees"

    # Real inputs possibly produce complex output
    cdef np.ndarray w
    w = wr
    if scalar in floating:
        if wi.nonzero()[0].size:
            w = wr + 1j * wi

    return filter_args((True, calc_q, calc_ev), (A, vs, w))


def trsen(np.ndarray[l_logical] select,
          np.ndarray[scalar, ndim=2] T,
          np.ndarray[scalar, ndim=2] Q,
          calc_ev=True):
    cdef l_int N, M, lwork, liwork, qiwork, info

    assert_fortran_mat(T, Q)

    # Allocate workspaces

    N = T.shape[0]

    cdef np.ndarray[scalar] wr, wi
    if scalar in cmplx:
        wr = np.empty(N, dtype=T.dtype)
        wi = None
    else:
        wr = np.empty(N, dtype=T.dtype)
        wi = np.empty(N, dtype=T.dtype)

    cdef char *compq
    cdef scalar *q_ptr
    if Q is not None:
        compq = "V"
        q_ptr = <scalar *>Q.data
    else:
        compq = "N"
        q_ptr = NULL

    # Workspace query
    # Xtrsen expects &qwork as a <scalar *> (even though it's an integer)
    cdef scalar qwork
    lwork = liwork = -1

    if scalar is float:
        lapack.strsen("N", compq, <l_logical *>select.data,
                      &N, <float *>T.data, &N, q_ptr, &N,
                      <float *>wr.data, <float *>wi.data, &M, NULL, NULL,
                      &qwork, &lwork, &qiwork, &liwork, &info)
    elif scalar is double:
        lapack.dtrsen("N", compq, <l_logical *>select.data,
                      &N, <double *>T.data, &N, q_ptr, &N,
                      <double *>wr.data, <double *>wi.data, &M, NULL, NULL,
                      &qwork, &lwork, &qiwork, &liwork, &info)
    elif scalar is float_complex:
        lapack.ctrsen("N", compq, <l_logical *>select.data,
                      &N, <float complex *>T.data, &N, q_ptr, &N,
                      <float complex *>wr.data, &M, NULL, NULL,
                      &qwork, &lwork, &info)
    elif scalar is double_complex:
        lapack.ztrsen("N", compq, <l_logical *>select.data,
                      &N, <double complex *>T.data, &N, q_ptr, &N,
                      <double complex *>wr.data, &M, NULL, NULL,
                      &qwork, &lwork, &info)

    assert info == 0, "Argument error in trsen"

    cdef np.ndarray[l_int] iwork = None
    if scalar in floating:
        lwork = <l_int>qwork
        liwork = qiwork
        iwork = np.empty(liwork, dtype=int_dtype)
    else:
        lwork = <l_int>qwork.real
    cdef np.ndarray[scalar, ndim=1] work = np.empty(lwork, dtype=T.dtype)

    # Tha actual calculation

    if scalar is float:
        lapack.strsen("N", compq, <l_logical *>select.data,
                      &N, <float *>T.data, &N, q_ptr, &N,
                      <float *>wr.data, <float *>wi.data, &M, NULL, NULL,
                      <float *>work.data, &lwork,
                      <l_int *>iwork.data, &liwork, &info)
    elif scalar is double:
        lapack.dtrsen("N", compq, <l_logical *>select.data,
                      &N, <double *>T.data, &N, q_ptr, &N,
                      <double *>wr.data, <double *>wi.data, &M, NULL, NULL,
                      <double *>work.data, &lwork,
                      <l_int *>iwork.data, &liwork, &info)
    elif scalar is float_complex:
        lapack.ctrsen("N", compq, <l_logical *>select.data,
                      &N, <float complex *>T.data, &N, q_ptr, &N,
                      <float complex *>wr.data, &M, NULL, NULL,
                      <float complex *>work.data, &lwork, &info)
    elif scalar is double_complex:
        lapack.ztrsen("N", compq, <l_logical *>select.data,
                      &N, <double complex *>T.data, &N, q_ptr, &N,
                      <double complex *>wr.data, &M, NULL, NULL,
                      <double complex *>work.data, &lwork, &info)

    if info > 0:
        raise LinAlgError("Reordering failed; problem is very ill-conditioned")

    assert info == 0, "Argument error in trsen"

    # Real inputs possibly produce complex output
    cdef np.ndarray w
    w = wr
    if scalar in floating:
        if wi.nonzero()[0].size:
            w = wr + 1j * wi

    return filter_args((True, Q is not None, calc_ev), (T, Q, w))


# Helper function for xTREVC and xTGEVC
def txevc_postprocess(dtype, T, vreal, np.ndarray[l_logical] select):
    cdef int N, M, i, m, indx

    N = T.shape[0]
    if select is None:
        select = np.ones(N, dtype = logical_dtype)
    selindx = select.nonzero()[0]
    M = selindx.size

    v = np.empty((N, M), dtype = dtype, order='F')

    indx = 0
    for m in range(M):
        k = selindx[m]

        if k < N-1 and T[k+1,k]:
            # we have the situation of a 2x2 block, and
            # the eigenvalue with the positive imaginary part desired
            v[:, m] = vreal[:, indx] + 1j * vreal[:, indx + 1]

            # Check if the eigenvalue with negative real part is also
            # selected, if it is, we need the same entries in vr
            if not select[k+1]:
                indx += 2
        elif k > 0 and T[k,k-1]:
            # we have the situation of a 2x2 block, and
            # the eigenvalue with the negative imaginary part desired
            v[:, m] = vreal[:, indx] - 1j * vreal[:, indx + 1]

            indx += 2
        else:
            # real eigenvalue
            v[:, m] = vreal[:, indx]

            indx += 1
    return v


# Wrappers for xTREVC
def strevc(np.ndarray[np.float32_t, ndim=2] T,
           np.ndarray[np.float32_t, ndim=2] Q=None,
           np.ndarray[l_logical] select=None,
           left=False, right=True):
    cdef l_int N, info, M, MM
    cdef char *side
    cdef char *howmny
    cdef np.ndarray[np.float32_t, ndim=2] vl_r, vr_r
    cdef float *vl_r_ptr
    cdef float *vr_r_ptr
    cdef np.ndarray[l_logical] select_cpy
    cdef l_logical *select_ptr
    cdef np.ndarray[np.float32_t] work

    assert_fortran_mat(T, Q)

    N = T.shape[0]
    work = np.empty(4*N, dtype = np.float32)

    if left and right:
        side = "B"
    elif left:
        side = "L"
    elif right:
        side = "R"
    else:
        return

    if select is not None:
        howmny = "S"
        MM = select.nonzero()[0].size
        # Correct for possible additional storage if a single complex
        # eigenvalue is selected.
        # For that: Figure out the positions of the 2x2 blocks.
        cmplxindx = np.diagonal(T, -1).nonzero()[0]
        for i in cmplxindx:
            if bool(select[i]) != bool(select[i+1]):
                MM += 1

        # Select is overwritten in strevc.
        select_cpy = np.array(select, dtype = logical_dtype,
                              order = 'F')
        select_ptr = <l_logical *>select_cpy.data
    else:
        MM = N
        select_ptr = NULL
        if Q is not None:
            howmny = "B"
        else:
            howmny = "A"

    if left:
        if Q is not None and select is None:
            vl_r = np.asfortranarray(Q.copy())
        else:
            vl_r = np.empty((N, MM), dtype = np.float32, order='F')
        vl_r_ptr = <float *>vl_r.data
    else:
        vl_r_ptr = NULL

    if right:
        if Q is not None and select is None:
            vr_r = np.asfortranarray(Q.copy())
        else:
            vr_r = np.empty((N, MM), dtype = np.float32, order='F')
        vr_r_ptr = <float *>vr_r.data
    else:
        vr_r_ptr = NULL

    lapack.strevc(side, howmny, select_ptr,
                     &N, <float *>T.data, &N,
                     vl_r_ptr, &N, vr_r_ptr, &N, &MM, &M,
                     <float *>work.data, &info)

    assert info == 0, "Argument error in strevc"
    assert MM == M, "Unexpected number of eigenvectors returned in strevc"

    if select is not None and Q is not None:
        if left:
            vl_r = np.asfortranarray(np.dot(Q, vl_r))
        if right:
            vr_r = np.asfortranarray(np.dot(Q, vr_r))

    # If there are complex eigenvalues, we need to postprocess the
    # eigenvectors.
    if np.diagonal(T, -1).nonzero()[0].size:
        if left:
            vl = txevc_postprocess(np.complex64, T, vl_r, select)
        if right:
            vr = txevc_postprocess(np.complex64, T, vr_r, select)
    else:
        if left:
            vl = vl_r
        if right:
            vr = vr_r

    if left and right:
        return (vl, vr)
    elif left:
        return vl
    else:
        return vr


def dtrevc(np.ndarray[np.float64_t, ndim=2] T,
           np.ndarray[np.float64_t, ndim=2] Q=None,
           np.ndarray[l_logical] select=None,
           left=False, right=True):
    cdef l_int N, info, M, MM
    cdef char *side
    cdef char *howmny
    cdef np.ndarray[np.float64_t, ndim=2] vl_r, vr_r
    cdef double *vl_r_ptr
    cdef double *vr_r_ptr
    cdef np.ndarray[l_logical] select_cpy
    cdef l_logical *select_ptr
    cdef np.ndarray[np.float64_t] work

    assert_fortran_mat(T, Q)

    N = T.shape[0]
    work = np.empty(4*N, dtype = np.float64)

    if left and right:
        side = "B"
    elif left:
        side = "L"
    elif right:
        side = "R"
    else:
        return

    if select is not None:
        howmny = "S"
        MM = select.nonzero()[0].size
        # Correct for possible additional storage if a single complex
        # eigenvalue is selected.
        # For that: Figure out the positions of the 2x2 blocks.
        cmplxindx = np.diagonal(T, -1).nonzero()[0]
        for i in cmplxindx:
            if bool(select[i]) != bool(select[i+1]):
                MM += 1

        # Select is overwritten in dtrevc.
        select_cpy = np.array(select, dtype = logical_dtype,
                              order = 'F')
        select_ptr = <l_logical *>select_cpy.data
    else:
        MM = N
        select_ptr = NULL
        if Q is not None:
            howmny = "B"
        else:
            howmny = "A"

    if left:
        if Q is not None and select is None:
            vl_r = np.asfortranarray(Q.copy())
        else:
            vl_r = np.empty((N, MM), dtype = np.float64, order='F')
        vl_r_ptr = <double *>vl_r.data
    else:
        vl_r_ptr = NULL

    if right:
        if Q is not None and select is None:
            vr_r = np.asfortranarray(Q.copy())
        else:
            vr_r = np.empty((N, MM), dtype = np.float64, order='F')
        vr_r_ptr = <double *>vr_r.data
    else:
        vr_r_ptr = NULL

    lapack.dtrevc(side, howmny, select_ptr,
                     &N, <double *>T.data, &N,
                     vl_r_ptr, &N, vr_r_ptr, &N, &MM, &M,
                     <double *>work.data, &info)

    assert info == 0, "Argument error in dtrevc"
    assert MM == M, "Unexpected number of eigenvectors returned in dtrevc"

    if select is not None and Q is not None:
        if left:
            vl_r = np.asfortranarray(np.dot(Q, vl_r))
        if right:
            vr_r = np.asfortranarray(np.dot(Q, vr_r))

    # If there are complex eigenvalues, we need to postprocess the eigenvectors
    if np.diagonal(T, -1).nonzero()[0].size:
        if left:
            vl = txevc_postprocess(np.complex128, T, vl_r, select)
        if right:
            vr = txevc_postprocess(np.complex128, T, vr_r, select)
    else:
        if left:
            vl = vl_r
        if right:
            vr = vr_r

    if left and right:
        return (vl, vr)
    elif left:
        return vl
    else:
        return vr


def ctrevc(np.ndarray[np.complex64_t, ndim=2] T,
           np.ndarray[np.complex64_t, ndim=2] Q=None,
           np.ndarray[l_logical] select=None,
           left=False, right=True):
    cdef l_int N, info, M, MM
    cdef char *side
    cdef char *howmny
    cdef np.ndarray[np.complex64_t, ndim=2] vl, vr
    cdef float complex *vl_ptr
    cdef float complex *vr_ptr
    cdef l_logical *select_ptr
    cdef np.ndarray[np.complex64_t] work
    cdef np.ndarray[np.float32_t] rwork

    assert_fortran_mat(T, Q)

    N = T.shape[0]
    work = np.empty(2*N, dtype = np.complex64)
    rwork = np.empty(N, dtype = np.float32)

    if left and right:
        side = "B"
    elif left:
        side = "L"
    elif right:
        side = "R"
    else:
        return

    if select is not None:
        howmny = "S"
        MM = select.nonzero()[0].size
        select_ptr = <l_logical *>select.data
    else:
        MM = N
        select_ptr = NULL
        if Q is not None:
            howmny = "B"
        else:
            howmny = "A"

    if left:
        if Q is not None and select is None:
            vl = np.asfortranarray(Q.copy())
        else:
            vl = np.empty((N, MM), dtype = np.complex64, order='F')
        vl_ptr = <float complex *>vl.data
    else:
        vl_ptr = NULL

    if right:
        if Q is not None and select is None:
            vr = np.asfortranarray(Q.copy())
        else:
            vr = np.empty((N, MM), dtype = np.complex64, order='F')
        vr_ptr = <float complex *>vr.data
    else:
        vr_ptr = NULL

    lapack.ctrevc(side, howmny, select_ptr,
                     &N, <float complex *>T.data, &N,
                     vl_ptr, &N, vr_ptr, &N, &MM, &M,
                     <float complex *>work.data, <float *>rwork.data, &info)

    assert info == 0, "Argument error in ctrevc"
    assert MM == M, "Unexpected number of eigenvectors returned in ctrevc"

    if select is not None and Q is not None:
        if left:
            vl = np.asfortranarray(np.dot(Q, vl))
        if right:
            vr = np.asfortranarray(np.dot(Q, vr))

    if left and right:
        return (vl, vr)
    elif left:
        return vl
    else:
        return vr


def ztrevc(np.ndarray[np.complex128_t, ndim=2] T,
           np.ndarray[np.complex128_t, ndim=2] Q=None,
           np.ndarray[l_logical] select=None,
           left=False, right=True):
    cdef l_int N, info, M, MM
    cdef char *side
    cdef char *howmny
    cdef np.ndarray[np.complex128_t, ndim=2] vl, vr
    cdef double complex *vl_ptr
    cdef double complex *vr_ptr
    cdef l_logical *select_ptr
    cdef np.ndarray[np.complex128_t] work
    cdef np.ndarray[np.float64_t] rwork

    assert_fortran_mat(T, Q)

    N = T.shape[0]
    work = np.empty(2*N, dtype = np.complex128)
    rwork = np.empty(N, dtype = np.float64)

    if left and right:
        side = "B"
    elif left:
        side = "L"
    elif right:
        side = "R"
    else:
        return

    if select is not None:
        howmny = "S"
        MM = select.nonzero()[0].size
        select_ptr = <l_logical *>select.data
    else:
        MM = N
        select_ptr = NULL
        if Q is not None:
            howmny = "B"
        else:
            howmny = "A"

    if left:
        if Q is not None and select is None:
            vl = np.asfortranarray(Q.copy())
        else:
            vl = np.empty((N, MM), dtype = np.complex128, order='F')
        vl_ptr = <double complex *>vl.data
    else:
        vl_ptr = NULL

    if right:
        if Q is not None and select is None:
            vr = np.asfortranarray(Q.copy())
        else:
            vr = np.empty((N, MM), dtype = np.complex128, order='F')
        vr_ptr = <double complex *>vr.data
    else:
        vr_ptr = NULL

    lapack.ztrevc(side, howmny, select_ptr,
                     &N, <double complex *>T.data, &N,
                     vl_ptr, &N, vr_ptr, &N, &MM, &M,
                     <double complex *>work.data, <double *>rwork.data, &info)

    assert info == 0, "Argument error in ztrevc"
    assert MM == M, "Unexpected number of eigenvectors returned in ztrevc"

    if select is not None and Q is not None:
        if left:
            vl = np.asfortranarray(np.dot(Q, vl))
        if right:
            vr = np.asfortranarray(np.dot(Q, vr))

    if left and right:
        return (vl, vr)
    elif left:
        return vl
    else:
        return vr


# wrappers for xGGES
def sgges(np.ndarray[np.float32_t, ndim=2] A,
          np.ndarray[np.float32_t, ndim=2] B,
          calc_q=True, calc_z=True, calc_ev=True):
    cdef l_int N, lwork, sdim, info
    cdef char *jobvsl
    cdef char *jobvsr
    cdef float *vsl_ptr
    cdef float *vsr_ptr
    cdef float qwork
    cdef np.ndarray[np.float32_t, ndim=2] vsl, vsr
    cdef np.ndarray[np.float32_t] alphar, alphai, beta, work

    assert_fortran_mat(A, B)

    N = A.shape[0]
    alphar = np.empty(N, dtype = np.float32)
    alphai = np.empty(N, dtype = np.float32)
    beta = np.empty(N, dtype = np.float32)

    if calc_q:
        vsl = np.empty((N,N), dtype = np.float32, order='F')
        vsl_ptr = <float *>vsl.data
        jobvsl = "V"
    else:
        vsl = None
        vsl_ptr = NULL
        jobvsl = "N"

    if calc_z:
        vsr = np.empty((N,N), dtype = np.float32, order='F')
        vsr_ptr = <float *>vsr.data
        jobvsr = "V"
    else:
        vsr = None
        vsr_ptr = NULL
        jobvsr = "N"

    # workspace query
    lwork = -1
    lapack.sgges(jobvsl, jobvsr, "N", NULL,
                    &N, <float *>A.data, &N,
                    <float *>B.data, &N, &sdim,
                    <float *>alphar.data, <float *>alphai.data,
                    <float *>beta.data,
                    vsl_ptr, &N, vsr_ptr, &N,
                    &qwork, &lwork, NULL, &info)

    assert info == 0, "Argument error in zgees"

    lwork = <int>qwork
    work = np.empty(lwork, dtype = np.float32)

    # Now the real calculation
    lapack.sgges(jobvsl, jobvsr, "N", NULL,
                    &N, <float *>A.data, &N,
                    <float *>B.data, &N, &sdim,
                    <float *>alphar.data, <float *>alphai.data,
                    <float *>beta.data,
                    vsl_ptr, &N, vsr_ptr, &N,
                    <float *>work.data, &lwork, NULL, &info)

    if info > 0:
        raise LinAlgError("QZ iteration failed to converge in sgges")

    assert info == 0, "Argument error in zgees"

    if alphai.nonzero()[0].size:
        alpha = alphar + 1j * alphai
    else:
        alpha = alphar

    return filter_args((True, True, calc_q, calc_z, calc_ev, calc_ev),
                       (A, B, vsl, vsr, alpha, beta))


def dgges(np.ndarray[np.float64_t, ndim=2] A,
          np.ndarray[np.float64_t, ndim=2] B,
          calc_q=True, calc_z=True, calc_ev=True):
    cdef l_int N, lwork, sdim, info
    cdef char *jobvsl
    cdef char *jobvsr
    cdef double *vsl_ptr
    cdef double *vsr_ptr
    cdef double qwork
    cdef np.ndarray[np.float64_t, ndim=2] vsl, vsr
    cdef np.ndarray[np.float64_t] alphar, alphai, beta, work

    assert_fortran_mat(A, B)

    N = A.shape[0]
    alphar = np.empty(N, dtype = np.float64)
    alphai = np.empty(N, dtype = np.float64)
    beta = np.empty(N, dtype = np.float64)

    if calc_q:
        vsl = np.empty((N,N), dtype = np.float64, order='F')
        vsl_ptr = <double *>vsl.data
        jobvsl = "V"
    else:
        vsl = None
        vsl_ptr = NULL
        jobvsl = "N"

    if calc_z:
        vsr = np.empty((N,N), dtype = np.float64, order='F')
        vsr_ptr = <double *>vsr.data
        jobvsr = "V"
    else:
        vsr = None
        vsr_ptr = NULL
        jobvsr = "N"

    # workspace query
    lwork = -1
    lapack.dgges(jobvsl, jobvsr, "N", NULL,
                    &N, <double *>A.data, &N,
                    <double *>B.data, &N, &sdim,
                    <double *>alphar.data, <double *>alphai.data,
                    <double *>beta.data,
                    vsl_ptr, &N, vsr_ptr, &N,
                    &qwork, &lwork, NULL, &info)

    assert info == 0, "Argument error in zgees"

    lwork = <int>qwork
    work = np.empty(lwork, dtype = np.float64)

    # Now the real calculation
    lapack.dgges(jobvsl, jobvsr, "N", NULL,
                    &N, <double *>A.data, &N,
                    <double *>B.data, &N, &sdim,
                    <double *>alphar.data, <double *>alphai.data,
                    <double *>beta.data,
                    vsl_ptr, &N, vsr_ptr, &N,
                    <double *>work.data, &lwork, NULL, &info)

    if info > 0:
        raise LinAlgError("QZ iteration failed to converge in dgges")

    assert info == 0, "Argument error in zgees"

    if alphai.nonzero()[0].size:
        alpha = alphar + 1j * alphai
    else:
        alpha = alphar

    return filter_args((True, True, calc_q, calc_z, calc_ev, calc_ev),
                       (A, B, vsl, vsr, alpha, beta))


def cgges(np.ndarray[np.complex64_t, ndim=2] A,
          np.ndarray[np.complex64_t, ndim=2] B,
          calc_q=True, calc_z=True, calc_ev=True):
    cdef l_int N, lwork, sdim, info
    cdef char *jobvsl
    cdef char *jobvsr
    cdef float complex *vsl_ptr
    cdef float complex *vsr_ptr
    cdef float complex qwork
    cdef np.ndarray[np.complex64_t, ndim=2] vsl, vsr
    cdef np.ndarray[np.complex64_t] alpha, beta, work
    cdef np.ndarray[np.float32_t] rwork

    assert_fortran_mat(A, B)

    N = A.shape[0]
    alpha = np.empty(N, dtype = np.complex64)
    beta = np.empty(N, dtype = np.complex64)
    rwork = np.empty(8*N, dtype = np.float32)

    if calc_q:
        vsl = np.empty((N,N), dtype = np.complex64, order='F')
        vsl_ptr = <float complex *>vsl.data
        jobvsl = "V"
    else:
        vsl = None
        vsl_ptr = NULL
        jobvsl = "N"

    if calc_z:
        vsr = np.empty((N,N), dtype = np.complex64, order='F')
        vsr_ptr = <float complex *>vsr.data
        jobvsr = "V"
    else:
        vsr = None
        vsr_ptr = NULL
        jobvsr = "N"

    # workspace query
    lwork = -1
    lapack.cgges(jobvsl, jobvsr, "N", NULL,
                    &N, <float complex *>A.data, &N,
                    <float complex *>B.data, &N, &sdim,
                    <float complex *>alpha.data, <float complex *>beta.data,
                    vsl_ptr, &N, vsr_ptr, &N,
                    &qwork, &lwork, <float *>rwork.data, NULL, &info)

    assert info == 0, "Argument error in zgees"

    lwork = <int>qwork.real
    work = np.empty(lwork, dtype = np.complex64)

    # Now the real calculation
    lapack.cgges(jobvsl, jobvsr, "N", NULL,
                    &N, <float complex *>A.data, &N,
                    <float complex *>B.data, &N, &sdim,
                    <float complex *>alpha.data, <float complex *>beta.data,
                    vsl_ptr, &N, vsr_ptr, &N,
                    <float complex *>work.data, &lwork,
                    <float *>rwork.data, NULL, &info)

    if info > 0:
        raise LinAlgError("QZ iteration failed to converge in cgges")

    assert info == 0, "Argument error in zgees"

    return filter_args((True, True, calc_q, calc_z, calc_ev, calc_ev),
                       (A, B, vsl, vsr, alpha, beta))


def zgges(np.ndarray[np.complex128_t, ndim=2] A,
          np.ndarray[np.complex128_t, ndim=2] B,
          calc_q=True, calc_z=True, calc_ev=True):
    cdef l_int N, lwork, sdim, info
    cdef char *jobvsl
    cdef char *jobvsr
    cdef double complex *vsl_ptr
    cdef double complex *vsr_ptr
    cdef double complex qwork
    cdef np.ndarray[np.complex128_t, ndim=2] vsl, vsr
    cdef np.ndarray[np.complex128_t] alpha, beta, work
    cdef np.ndarray[np.float64_t] rwork

    assert_fortran_mat(A, B)

    N = A.shape[0]
    alpha = np.empty(N, dtype = np.complex128)
    beta = np.empty(N, dtype = np.complex128)
    rwork = np.empty(8*N, dtype = np.float64)

    if calc_q:
        vsl = np.empty((N,N), dtype = np.complex128, order='F')
        vsl_ptr = <double complex *>vsl.data
        jobvsl = "V"
    else:
        vsl = None
        vsl_ptr = NULL
        jobvsl = "N"

    if calc_z:
        vsr = np.empty((N,N), dtype = np.complex128, order='F')
        vsr_ptr = <double complex *>vsr.data
        jobvsr = "V"
    else:
        vsr = None
        vsr_ptr = NULL
        jobvsr = "N"

    # workspace query
    lwork = -1
    lapack.zgges(jobvsl, jobvsr, "N", NULL,
                    &N, <double complex *>A.data, &N,
                    <double complex *>B.data, &N, &sdim,
                    <double complex *>alpha.data, <double complex *>beta.data,
                    vsl_ptr, &N, vsr_ptr, &N,
                    &qwork, &lwork, <double *>rwork.data, NULL, &info)

    assert info == 0, "Argument error in zgees"

    lwork = <int>qwork.real
    work = np.empty(lwork, dtype = np.complex128)

    # Now the real calculation
    lapack.zgges(jobvsl, jobvsr, "N", NULL,
                    &N, <double complex *>A.data, &N,
                    <double complex *>B.data, &N, &sdim,
                    <double complex *>alpha.data, <double complex *>beta.data,
                    vsl_ptr, &N, vsr_ptr, &N,
                    <double complex *>work.data, &lwork,
                    <double *>rwork.data, NULL, &info)

    if info > 0:
        raise LinAlgError("QZ iteration failed to converge in zgges")

    assert info == 0, "Argument error in zgees"

    return filter_args((True, True, calc_q, calc_z, calc_ev, calc_ev),
                       (A, B, vsl, vsr, alpha, beta))


# wrappers for xTGSEN
def stgsen(np.ndarray[l_logical] select,
           np.ndarray[np.float32_t, ndim=2] S,
           np.ndarray[np.float32_t, ndim=2] T,
           np.ndarray[np.float32_t, ndim=2] Q=None,
           np.ndarray[np.float32_t, ndim=2] Z=None,
           calc_ev=True):
    cdef l_int N, M, lwork, liwork, qiwork, info, ijob
    cdef l_logical wantq, wantz
    cdef float qwork
    cdef float *q_ptr
    cdef float *z_ptr
    cdef np.ndarray[np.float32_t] alphar, alphai, beta, work
    cdef np.ndarray[l_int] iwork

    assert_fortran_mat(S, T, Q, Z)

    N = S.shape[0]
    alphar = np.empty(N, dtype = np.float32)
    alphai = np.empty(N, dtype = np.float32)
    beta = np.empty(N, dtype = np.float32)
    ijob = 0

    if Q is not None:
        wantq = 1
        q_ptr = <float *>Q.data
    else:
        wantq = 0
        q_ptr = NULL

    if Z is not None:
        wantz = 1
        z_ptr = <float *>Z.data
    else:
        wantz = 0
        z_ptr = NULL

    # workspace query
    lwork = -1
    liwork = -1
    lapack.stgsen(&ijob, &wantq, &wantz, <l_logical *>select.data,
                     &N, <float *>S.data, &N,
                     <float *>T.data, &N,
                     <float *>alphar.data, <float *>alphai.data,
                     <float *>beta.data,
                     q_ptr, &N, z_ptr, &N, &M, NULL, NULL, NULL,
                     &qwork, &lwork, &qiwork, &liwork, &info)

    assert info == 0, "Argument error in stgsen"

    lwork = <int>qwork
    work = np.empty(lwork, dtype = np.float32)
    liwork = qiwork
    iwork = np.empty(liwork, dtype = int_dtype)

    # Now the real calculation
    lapack.stgsen(&ijob, &wantq, &wantz, <l_logical *>select.data,
                     &N, <float *>S.data, &N,
                     <float *>T.data, &N,
                     <float *>alphar.data, <float *>alphai.data,
                     <float *>beta.data,
                     q_ptr, &N, z_ptr, &N, &M, NULL, NULL, NULL,
                     <float *>work.data, &lwork,
                     <l_int *>iwork.data, &liwork, &info)

    if info > 0:
        raise LinAlgError("Reordering failed; problem is very ill-conditioned")

    assert info == 0, "Argument error in stgsen"

    if alphai.nonzero()[0].size:
        alpha = alphar + 1j * alphai
    else:
        alpha = alphar

    return filter_args((True, True, Q is not None, Z is not None,
                        calc_ev, calc_ev),
                       (S, T, Q, Z, alpha, beta))


def dtgsen(np.ndarray[l_logical] select,
           np.ndarray[np.float64_t, ndim=2] S,
           np.ndarray[np.float64_t, ndim=2] T,
           np.ndarray[np.float64_t, ndim=2] Q=None,
           np.ndarray[np.float64_t, ndim=2] Z=None,
           calc_ev=True):
    cdef l_int N, M, lwork, liwork, qiwork, info, ijob
    cdef l_logical wantq, wantz
    cdef double qwork
    cdef double *q_ptr
    cdef double *z_ptr
    cdef np.ndarray[np.float64_t] alphar, alphai, beta, work
    cdef np.ndarray[l_int] iwork

    assert_fortran_mat(S, T, Q, Z)

    N = S.shape[0]
    alphar = np.empty(N, dtype = np.float64)
    alphai = np.empty(N, dtype = np.float64)
    beta = np.empty(N, dtype = np.float64)
    ijob = 0

    if Q is not None:
        wantq = 1
        q_ptr = <double *>Q.data
    else:
        wantq = 0
        q_ptr = NULL

    if Z is not None:
        wantz = 1
        z_ptr = <double *>Z.data
    else:
        wantz = 0
        z_ptr = NULL

    # workspace query
    lwork = -1
    liwork = -1
    lapack.dtgsen(&ijob, &wantq, &wantz, <l_logical *>select.data,
                     &N, <double *>S.data, &N,
                     <double *>T.data, &N,
                     <double *>alphar.data, <double *>alphai.data,
                     <double *>beta.data,
                     q_ptr, &N, z_ptr, &N, &M, NULL, NULL, NULL,
                     &qwork, &lwork, &qiwork, &liwork, &info)

    assert info == 0, "Argument error in dtgsen"

    lwork = <int>qwork
    work = np.empty(lwork, dtype = np.float64)
    liwork = qiwork
    iwork = np.empty(liwork, dtype = int_dtype)

    # Now the real calculation
    lapack.dtgsen(&ijob, &wantq, &wantz, <l_logical *>select.data,
                     &N, <double *>S.data, &N,
                     <double *>T.data, &N,
                     <double *>alphar.data, <double *>alphai.data,
                     <double *>beta.data,
                     q_ptr, &N, z_ptr, &N, &M, NULL, NULL, NULL,
                     <double *>work.data, &lwork,
                     <l_int *>iwork.data, &liwork, &info)

    if info > 0:
        raise LinAlgError("Reordering failed; problem is very ill-conditioned")

    assert info == 0, "Argument error in dtgsen"

    if alphai.nonzero()[0].size:
        alpha = alphar + 1j * alphai
    else:
        alpha = alphar

    return filter_args((True, True, Q is not None, Z is not None,
                        calc_ev, calc_ev),
                       (S, T, Q, Z, alpha, beta))


def ctgsen(np.ndarray[l_logical] select,
           np.ndarray[np.complex64_t, ndim=2] S,
           np.ndarray[np.complex64_t, ndim=2] T,
           np.ndarray[np.complex64_t, ndim=2] Q=None,
           np.ndarray[np.complex64_t, ndim=2] Z=None,
           calc_ev=True):
    cdef l_int N, M, lwork, liwork, qiwork, info, ijob
    cdef l_logical wantq, wantz
    cdef float complex qwork
    cdef float complex *q_ptr
    cdef float complex *z_ptr
    cdef np.ndarray[np.complex64_t] alpha, beta, work
    cdef np.ndarray[l_int] iwork

    assert_fortran_mat(S, T, Q, Z)

    N = S.shape[0]
    alpha = np.empty(N, dtype = np.complex64)
    beta = np.empty(N, dtype = np.complex64)
    ijob = 0

    if Q is not None:
        wantq = 1
        q_ptr = <float complex *>Q.data
    else:
        wantq = 0
        q_ptr = NULL

    if Z is not None:
        wantz = 1
        z_ptr = <float complex *>Z.data
    else:
        wantz = 0
        z_ptr = NULL

    # workspace query
    lwork = -1
    liwork = -1
    lapack.ctgsen(&ijob, &wantq, &wantz, <l_logical *>select.data,
                     &N, <float complex *>S.data, &N,
                     <float complex *>T.data, &N,
                     <float complex *>alpha.data, <float complex *>beta.data,
                     q_ptr, &N, z_ptr, &N, &M, NULL, NULL, NULL,
                     &qwork, &lwork, &qiwork, &liwork, &info)

    assert info == 0, "Argument error in ctgsen"

    lwork = <int>qwork.real
    work = np.empty(lwork, dtype = np.complex64)
    liwork = qiwork
    iwork = np.empty(liwork, dtype = int_dtype)

    # Now the real calculation
    lapack.ctgsen(&ijob, &wantq, &wantz, <l_logical *>select.data,
                     &N, <float complex *>S.data, &N,
                     <float complex *>T.data, &N,
                     <float complex *>alpha.data, <float complex *>beta.data,
                     q_ptr, &N, z_ptr, &N, &M, NULL, NULL, NULL,
                     <float complex *>work.data, &lwork,
                     <l_int *>iwork.data, &liwork, &info)

    if info > 0:
        raise LinAlgError("Reordering failed; problem is very ill-conditioned")

    assert info == 0, "Argument error in ctgsen"

    return filter_args((True, True, Q is not None, Z is not None,
                        calc_ev, calc_ev),
                       (S, T, Q, Z, alpha, beta))


def ztgsen(np.ndarray[l_logical] select,
           np.ndarray[np.complex128_t, ndim=2] S,
           np.ndarray[np.complex128_t, ndim=2] T,
           np.ndarray[np.complex128_t, ndim=2] Q=None,
           np.ndarray[np.complex128_t, ndim=2] Z=None,
           calc_ev=True):
    cdef l_int N, M, lwork, liwork, qiwork, info, ijob
    cdef l_logical wantq, wantz
    cdef double complex qwork
    cdef double complex *q_ptr
    cdef double complex *z_ptr
    cdef np.ndarray[np.complex128_t] alpha, beta, work
    cdef np.ndarray[l_int] iwork

    assert_fortran_mat(S, T, Q, Z)

    N = S.shape[0]
    alpha = np.empty(N, dtype = np.complex128)
    beta = np.empty(N, dtype = np.complex128)
    ijob = 0

    if Q is not None:
        wantq = 1
        q_ptr = <double complex *>Q.data
    else:
        wantq = 0
        q_ptr = NULL

    if Z is not None:
        wantz = 1
        z_ptr = <double complex *>Z.data
    else:
        wantz = 0
        z_ptr = NULL

    # workspace query
    lwork = -1
    liwork = -1
    lapack.ztgsen(&ijob, &wantq, &wantz, <l_logical *>select.data,
                     &N, <double complex *>S.data, &N,
                     <double complex *>T.data, &N,
                     <double complex *>alpha.data, <double complex *>beta.data,
                     q_ptr, &N, z_ptr, &N, &M, NULL, NULL, NULL,
                     &qwork, &lwork, &qiwork, &liwork, &info)

    assert info == 0, "Argument error in ztgsen"

    lwork = <int>qwork.real
    work = np.empty(lwork, dtype = np.complex128)
    liwork = qiwork
    iwork = np.empty(liwork, dtype = int_dtype)

    # Now the real calculation
    lapack.ztgsen(&ijob, &wantq, &wantz, <l_logical *>select.data,
                     &N, <double complex *>S.data, &N,
                     <double complex *>T.data, &N,
                     <double complex *>alpha.data, <double complex *>beta.data,
                     q_ptr, &N, z_ptr, &N, &M, NULL, NULL, NULL,
                     <double complex *>work.data, &lwork,
                     <l_int *>iwork.data, &liwork, &info)

    if info > 0:
        raise LinAlgError("Reordering failed; problem is very ill-conditioned")

    assert info == 0, "Argument error in ztgsen"

    return filter_args((True, True, Q is not None, Z is not None,
                        calc_ev, calc_ev),
                       (S, T, Q, Z, alpha, beta))


# xTGEVC
def stgevc(np.ndarray[np.float32_t, ndim=2] S,
           np.ndarray[np.float32_t, ndim=2] T,
           np.ndarray[np.float32_t, ndim=2] Q=None,
           np.ndarray[np.float32_t, ndim=2] Z=None,
           np.ndarray[l_logical] select=None,
           left=False, right=True):
    cdef l_int N, info, M, MM
    cdef char *side
    cdef char *howmny
    cdef np.ndarray[np.float32_t, ndim=2] vl_r, vr_r
    cdef float *vl_r_ptr
    cdef float *vr_r_ptr
    cdef np.ndarray[l_logical] select_cpy
    cdef l_logical *select_ptr
    cdef np.ndarray[np.float32_t] work

    assert_fortran_mat(S, T, Q, Z)

    N = S.shape[0]
    work = np.empty(6*N, dtype = np.float32)

    if left and right:
        side = "B"
    elif left:
        side = "L"
    elif right:
        side = "R"
    else:
        return

    backtr = False

    if select is not None:
        howmny = "S"
        MM = select.nonzero()[0].size
        # Correct for possible additional storage if a single complex
        # eigenvalue is selected.
        # For that: Figure out the positions of the 2x2 blocks.
        cmplxindx = np.diagonal(S, -1).nonzero()[0]
        for i in cmplxindx:
            if bool(select[i]) != bool(select[i+1]):
                MM += 1

        # select is overwritten in stgevc
        select_cpy = np.array(select, dtype = logical_dtype,
                              order = 'F')
        select_ptr = <l_logical *>select_cpy.data
    else:
        MM = N
        select_ptr = NULL
        if ((left and right and Q is not None and Z is not None) or
            (left and not right and Q is not None) or
            (right and not left and Z is not None)):
            howmny = "B"
            backtr = True
        else:
            howmny = "A"

    if left:
        if backtr:
            vl_r = Q
        else:
            vl_r = np.empty((N, MM), dtype = np.float32, order='F')
        vl_r_ptr = <float *>vl_r.data
    else:
        vl_r_ptr = NULL

    if right:
        if backtr:
            vr_r = Z
        else:
            vr_r = np.empty((N, MM), dtype = np.float32, order='F')
        vr_r_ptr = <float *>vr_r.data
    else:
        vr_r_ptr = NULL

    lapack.stgevc(side, howmny, select_ptr,
                     &N, <float *>S.data, &N,
                     <float *>T.data, &N,
                     vl_r_ptr, &N, vr_r_ptr, &N, &MM, &M,
                     <float *>work.data, &info)

    assert info == 0, "Argument error in stgevc"
    assert MM == M, "Unexpected number of eigenvectors returned in stgevc"

    if not backtr:
        if left:
            vl_r = np.asfortranarray(np.dot(Q, vl_r))
        if right:
            vr_r = np.asfortranarray(np.dot(Z, vr_r))

    # If there are complex eigenvalues, we need to postprocess the eigenvectors
    if np.diagonal(S, -1).nonzero()[0].size:
        if left:
            vl = txevc_postprocess(np.complex64, S, vl_r, select)
        if right:
            vr = txevc_postprocess(np.complex64, S, vr_r, select)
    else:
        if left:
            vl = vl_r
        if right:
            vr = vr_r

    if left and right:
        return (vl, vr)
    elif left:
        return vl
    else:
        return vr


def dtgevc(np.ndarray[np.float64_t, ndim=2] S,
           np.ndarray[np.float64_t, ndim=2] T,
           np.ndarray[np.float64_t, ndim=2] Q=None,
           np.ndarray[np.float64_t, ndim=2] Z=None,
           np.ndarray[l_logical] select=None,
           left=False, right=True):
    cdef l_int N, info, M, MM
    cdef char *side
    cdef char *howmny
    cdef np.ndarray[np.float64_t, ndim=2] vl_r, vr_r
    cdef double *vl_r_ptr
    cdef double *vr_r_ptr
    cdef np.ndarray[l_logical] select_cpy
    cdef l_logical *select_ptr
    cdef np.ndarray[np.float64_t] work

    assert_fortran_mat(S, T, Q, Z)

    N = S.shape[0]
    work = np.empty(6*N, dtype = np.float64)

    if left and right:
        side = "B"
    elif left:
        side = "L"
    elif right:
        side = "R"
    else:
        return

    backtr = False

    if select is not None:
        howmny = "S"
        MM = select.nonzero()[0].size
        # Correct for possible additional storage if a single complex
        # eigenvalue is selected.
        # For that: Figure out the positions of the 2x2 blocks.
        cmplxindx = np.diagonal(S, -1).nonzero()[0]
        for i in cmplxindx:
            if bool(select[i]) != bool(select[i+1]):
                MM += 1

        # select is overwritten in dtgevc
        select_cpy = np.array(select, dtype = logical_dtype,
                              order = 'F')
        select_ptr = <l_logical *>select_cpy.data
    else:
        MM = N
        select_ptr = NULL
        if ((left and right and Q is not None and Z is not None) or
            (left and not right and Q is not None) or
            (right and not left and Z is not None)):
            howmny = "B"
            backtr = True
        else:
            howmny = "A"

    if left:
        if backtr:
            vl_r = Q
        else:
            vl_r = np.empty((N, MM), dtype = np.float64, order='F')
        vl_r_ptr = <double *>vl_r.data
    else:
        vl_r_ptr = NULL

    if right:
        if backtr:
            vr_r = Z
        else:
            vr_r = np.empty((N, MM), dtype = np.float64, order='F')
        vr_r_ptr = <double *>vr_r.data
    else:
        vr_r_ptr = NULL

    lapack.dtgevc(side, howmny, select_ptr,
                     &N, <double *>S.data, &N,
                     <double *>T.data, &N,
                     vl_r_ptr, &N, vr_r_ptr, &N, &MM, &M,
                     <double *>work.data, &info)

    assert info == 0, "Argument error in dtgevc"
    assert MM == M, "Unexpected number of eigenvectors returned in dtgevc"

    if not backtr:
        if left:
            vl_r = np.asfortranarray(np.dot(Q, vl_r))
        if right:
            vr_r = np.asfortranarray(np.dot(Z, vr_r))

    # If there are complex eigenvalues, we need to postprocess the
    # eigenvectors.
    if np.diagonal(S, -1).nonzero()[0].size:
        if left:
            vl = txevc_postprocess(np.complex128, S, vl_r, select)
        if right:
            vr = txevc_postprocess(np.complex128, S, vr_r, select)
    else:
        if left:
            vl = vl_r
        if right:
            vr = vr_r

    if left and right:
        return (vl, vr)
    elif left:
        return vl
    else:
        return vr


def ctgevc(np.ndarray[np.complex64_t, ndim=2] S,
           np.ndarray[np.complex64_t, ndim=2] T,
           np.ndarray[np.complex64_t, ndim=2] Q=None,
           np.ndarray[np.complex64_t, ndim=2] Z=None,
           np.ndarray[l_logical] select=None,
           left=False, right=True):
    cdef l_int N, info, M, MM
    cdef char *side
    cdef char *howmny
    cdef np.ndarray[np.complex64_t, ndim=2] vl, vr
    cdef float complex *vl_ptr
    cdef float complex *vr_ptr
    cdef l_logical *select_ptr
    cdef np.ndarray[np.complex64_t] work
    cdef np.ndarray[np.float32_t] rwork

    assert_fortran_mat(S, T, Q, Z)

    N = S.shape[0]
    work = np.empty(2*N, dtype = np.complex64)
    rwork = np.empty(2*N, dtype = np.float32)

    if left and right:
        side = "B"
    elif left:
        side = "L"
    elif right:
        side = "R"
    else:
        return

    backtr = False

    if select is not None:
        howmny = "S"
        MM = select.nonzero()[0].size
        select_ptr = <l_logical *>select.data
    else:
        MM = N
        select_ptr = NULL
        if ((left and right and Q is not None and Z is not None) or
            (left and not right and Q is not None) or
            (right and not left and Z is not None)):
            howmny = "B"
            backtr = True
        else:
            howmny = "A"

    if left:
        if backtr:
            vl = Q
        else:
            vl = np.empty((N, MM), dtype = np.complex64, order='F')
        vl_ptr = <float complex *>vl.data
    else:
        vl_ptr = NULL

    if right:
        if backtr:
            vr = Z
        else:
            vr = np.empty((N, MM), dtype = np.complex64, order='F')
        vr_ptr = <float complex *>vr.data
    else:
        vr_ptr = NULL

    lapack.ctgevc(side, howmny, select_ptr,
                     &N, <float complex *>S.data, &N,
                     <float complex *>T.data, &N,
                     vl_ptr, &N, vr_ptr, &N, &MM, &M,
                     <float complex *>work.data, <float *>rwork.data, &info)

    assert info == 0, "Argument error in ctgevc"
    assert MM == M, "Unexpected number of eigenvectors returned in ctgevc"

    if not backtr:
        if left:
            vl = np.asfortranarray(np.dot(Q, vl))
        if right:
            vr = np.asfortranarray(np.dot(Z, vr))

    if left and right:
        return (vl, vr)
    elif left:
        return vl
    else:
        return vr


def ztgevc(np.ndarray[np.complex128_t, ndim=2] S,
           np.ndarray[np.complex128_t, ndim=2] T,
           np.ndarray[np.complex128_t, ndim=2] Q=None,
           np.ndarray[np.complex128_t, ndim=2] Z=None,
           np.ndarray[l_logical] select=None,
           left=False, right=True):
    cdef l_int N, info, M, MM
    cdef char *side
    cdef char *howmny
    cdef np.ndarray[np.complex128_t, ndim=2] vl, vr
    cdef double complex *vl_ptr
    cdef double complex *vr_ptr
    cdef l_logical *select_ptr
    cdef np.ndarray[np.complex128_t] work
    cdef np.ndarray[np.float64_t] rwork

    assert_fortran_mat(S, T, Q, Z)

    N = S.shape[0]
    work = np.empty(2*N, dtype = np.complex128)
    rwork = np.empty(2*N, dtype = np.float64)

    if left and right:
        side = "B"
    elif left:
        side = "L"
    elif right:
        side = "R"
    else:
        return

    backtr = False

    if select is not None:
        howmny = "S"
        MM = select.nonzero()[0].size
        select_ptr = <l_logical *>select.data
    else:
        MM = N
        select_ptr = NULL
        if ((left and right and Q is not None and Z is not None) or
            (left and not right and Q is not None) or
            (right and not left and Z is not None)):
            howmny = "B"
            backtr = True
        else:
            howmny = "A"

    if left:
        if backtr:
            vl = Q
        else:
            vl = np.empty((N, MM), dtype = np.complex128, order='F')
        vl_ptr = <double complex *>vl.data
    else:
        vl_ptr = NULL

    if right:
        if backtr:
            vr = Z
        else:
            vr = np.empty((N, MM), dtype = np.complex128, order='F')
        vr_ptr = <double complex *>vr.data
    else:
        vr_ptr = NULL

    lapack.ztgevc(side, howmny, select_ptr,
                     &N, <double complex *>S.data, &N,
                     <double complex *>T.data, &N,
                     vl_ptr, &N, vr_ptr, &N, &MM, &M,
                     <double complex *>work.data, <double *>rwork.data, &info)

    assert info == 0, "Argument error in ztgevc"
    assert MM == M, "Unexpected number of eigenvectors returned in ztgevc"

    if not backtr:
        if left:
            vl = np.asfortranarray(np.dot(Q, vl))
        if right:
            vr = np.asfortranarray(np.dot(Z, vr))

    if left and right:
        return (vl, vr)
    elif left:
        return vl
    else:
        return vr


def prepare_for_lapack(overwrite, *args):
    """Convert arrays to Fortran format.

    This function takes a number of array objects in `args` and converts them
    to a format that can be directly passed to a Fortran function (Fortran
    contiguous NumPy array). If the arrays have different data type, they
    converted arrays are cast to a common compatible data type (one of NumPy's
    `float32`, `float64`, `complex64`, `complex128` data types).

    If `overwrite` is ``False``, an NumPy array that would already be in the
    correct format (Fortran contiguous, right data type) is neverthelessed
    copied. (Hence, overwrite = True does not imply that acting on the
    converted array in the return values will overwrite the original array in
    all cases -- it does only so if the original array was already in the
    correct format. The conversions require copying. In fact, that's the same
    behavior as in SciPy, it's just not explicitly stated there)

    If an argument is ``None``, it is just passed through and not used to
    determine the proper LAPACK type.

    `prepare_for_lapack` returns a character indicating the proper LAPACK data
    type ('s', 'd', 'c', 'z') and a list of properly converted arrays.
    """

    # Make sure we have NumPy arrays
    mats = [None]*len(args)
    for i in range(len(args)):
        if args[i] is not None:
            arr = np.asanyarray(args[i])
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError("Argument cannot be interpreted "
                                 "as a numeric array")

            mats[i] = (arr, arr is not args[i] or overwrite)
        else:
            mats[i] = (None, True)

    # First figure out common dtype
    # Note: The return type of common_type is guaranteed to be a floating point
    #       kind.
    dtype = np.common_type(*[arr for arr, ovwrt in mats if arr is not None])

    if dtype == np.float32:
        lapacktype = 's'
    elif dtype == np.float64:
        lapacktype = 'd'
    elif dtype == np.complex64:
        lapacktype = 'c'
    elif dtype == np.complex128:
        lapacktype = 'z'
    else:
        raise AssertionError("Unexpected data type from common_type")

    ret = [ lapacktype ]
    for npmat, ovwrt in mats:
        # Now make sure that the array is contiguous, and copy if necessary.
        if npmat is not None:
            if npmat.ndim == 2:
                if not npmat.flags["F_CONTIGUOUS"]:
                    npmat = np.asfortranarray(npmat, dtype = dtype)
                elif npmat.dtype != dtype:
                    npmat = npmat.astype(dtype)
                elif not ovwrt:
                    # ugly here: copy makes always C-array, no way to tell it
                    # to make a Fortran array.
                    npmat = np.asfortranarray(npmat.copy())
            elif npmat.ndim == 1:
                if not npmat.flags["C_CONTIGUOUS"]:
                    npmat = np.ascontiguousarray(npmat, dtype = dtype)
                elif npmat.dtype != dtype:
                    npmat = npmat.astype(dtype)
                elif not ovwrt:
                    npmat = np.asfortranarray(npmat.copy())
            else:
                raise ValueError("Dimensionality of array is not 1 or 2")

        ret.append(npmat)

    return tuple(ret)
