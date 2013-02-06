# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['solve', 'ldos', 'wave_func', 'Solver']

import warnings
import numpy as np
import scipy.sparse as sp
from . import common

# Note: previous code would have failed if UMFPACK was provided by scikit
import scipy.sparse.linalg.dsolve.linsolve as linsolve
umfpack = linsolve.umfpack
uses_umfpack = linsolve.isUmfpack

# check if we are actually using UMFPACK or rather SuperLU

if uses_umfpack:
    # This patches a memory leak in SciPy:
    # http://projects.scipy.org/scipy/ticket/1597
    #
    # TODO: Remove this code once it is likely that the official bug fix has
    # reached all of our users.
    def del_for_umfpackcontext(self):
        self.free()
    if not hasattr(umfpack.UmfpackContext, '__del__'):
        umfpack.UmfpackContext.__del__ = del_for_umfpackcontext
    del del_for_umfpackcontext

    def factorized(A, piv_tol=1.0, sym_piv_tol=1.0):
        """
        Return a fuction for solving a sparse linear system, with A
        pre-factorized.

        Example:
        solve = factorized(A) # Makes LU decomposition.
        x1 = solve(rhs1) # Uses the LU factors.
        x2 = solve(rhs2) # Uses again the LU factors.

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
    # (SuperLu is not bad per se, somehow the SciPy version isn't good)
    warnings.warn("The installed SciPy does not use UMFPACK. Instead, "
                  "SciPy will use the version of SuperLu it is shipped with. "
                  "Performance can be very poor in this case.", RuntimeWarning)

    factorized = linsolve.factorized


class Solver(common.SparseSolver):
    "Sparse Solver class based on the sparse direct solvers provided by SciPy."
    lhsformat = 'csc'
    rhsformat = 'csc'
    nrhs = 1

    def _factorized(self, a):
        a = sp.csc_matrix(a)
        return factorized(a), a.shape

    def _solve_linear_sys(self, factorized_a, b, kept_vars=None):
        slv, a_shape = factorized_a

        if kept_vars is None:
            kept_vars = slice(a_shape[1])

        sols = []
        vec = np.empty(a_shape[0], complex)
        for mat in b:
            if mat.shape[1] != 0:
                # See comment about zero-shaped sparse matrices at the top of
                # common.py.
                mat = sp.csr_matrix(mat)
            for j in xrange(mat.shape[1]):
                vec[:] = mat[:, j].todense().flatten()
                sols.append(slv(vec)[kept_vars])

        if len(sols):
            return np.asarray(sols).transpose()
        else:
            return np.asarray(np.zeros(shape=(len(kept_vars), 0)))


default_solver = Solver()

solve = default_solver.solve
ldos = default_solver.ldos
wave_func = default_solver.wave_func
