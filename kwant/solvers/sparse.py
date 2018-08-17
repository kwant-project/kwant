# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['smatrix', 'greens_function', 'ldos', 'wave_function', 'Solver']

import numpy as np
import scipy.sparse as sp
from . import common

# Note: previous code would have failed if UMFPACK was provided by scikit
import scipy.sparse.linalg.dsolve.linsolve as linsolve

# check if we are actually using UMFPACK or rather SuperLU
# TODO: remove the try (only using the except clause) once we depend on
# scipy >= 0.14.0.
try:
    uses_umfpack = linsolve.isUmfpack
except AttributeError:
    uses_umfpack = linsolve.useUmfpack

if uses_umfpack:
    umfpack = linsolve.umfpack

if uses_umfpack:
    def factorized(A, piv_tol=1.0, sym_piv_tol=1.0):
        """
        Return a fuction for solving a sparse linear system, with A
        pre-factorized.

        Parameters
        ----------
        A : csc_matrix
            matrix to be factorized
        piv_tol : float, 0 <= piv_tol <= 1.0
        sym_piv_tol : float, 0 <= piv_tol <= 1.0
            thresholds used by UMFPACK for pivoting. 0 means no pivoting, 1.0
            means full pivoting as in dense matrices (guaranteeing stability,
            but reducing possibly sparsity). Defaults of UMFPACK are 0.1 and
            0.001 respectively. Whether piv_tol or sym_piv_tol are used is
            decided internally by UMFPACK, depending on whether the matrix is
            "symmetric" enough.

        Examples
        --------
        solve = factorized(A) # Makes LU decomposition.
        x1 = solve(rhs1) # Uses the LU factors.
        x2 = solve(rhs2) # Uses again the LU factors.
        """

        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)

        A.sort_indices()
        A = A.asfptype()  # upcast to a floating point format

        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                             " .astype()")

        family = {'d': 'di', 'D': 'zi'}
        umf = umfpack.UmfpackContext(family[A.dtype.char])

        # adjust pivot thresholds
        umf.control[umfpack.UMFPACK_PIVOT_TOLERANCE] = piv_tol
        umf.control[umfpack.UMFPACK_SYM_PIVOT_TOLERANCE] = sym_piv_tol

        # Make LU decomposition.
        umf.numeric(A)

        def solve(b):
            return umf.solve(umfpack.UMFPACK_A, A, b, autoTranspose=True)

        return solve
else:
    # no UMFPACK found. SuperLU is being used, but usually abysmally slow
    # (SuperLu is not bad per se, somehow the SciPy version isn't good).
    # Since scipy doesn't include UMFPACK anymore due to software rot,
    # there is no warning here.
    factorized = linsolve.factorized


class Solver(common.SparseSolver):
    "Sparse Solver class based on the sparse direct solvers provided by SciPy."
    lhsformat = 'csc'
    rhsformat = 'csc'
    nrhs = 1

    def _factorized(self, a):
        a = sp.csc_matrix(a)
        return factorized(a)

    def _solve_linear_sys(self, factorized_a, b, kept_vars):
        if b.shape[1] == 0:
            return b[kept_vars]

        sols = []
        vec = np.empty(b.shape[0], complex)
        for j in range(b.shape[1]):
            vec[:] = b[:, j].toarray().flatten()
            sols.append(factorized_a(vec)[kept_vars])

        return np.asarray(sols).transpose()


default_solver = Solver()

smatrix = default_solver.smatrix
greens_function = default_solver.greens_function
ldos = default_solver.ldos
wave_function = default_solver.wave_function
