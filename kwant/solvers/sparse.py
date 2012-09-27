"""Implementation of the sparse solver framework using the direct sparse solver
provided by `scipy.sparse.linalg
<http://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`.

Scipy currently uses internally either the direct sparse solver Umfpack or if
that is not installed, SuperLU. Often, scipy's SuperLU will give quite poor
performance and you will be warned if only SuperLU is found.  The module
variable `uses_umfpack` can be checked to determine if Umfpack is being used.

`sparse` does not introduce any additional options as compared to the generic
sparse solver framework.
"""

__all__ = ['solve', 'ldos', 'Solver']

from functools import reduce
import warnings
import numpy as np
import scipy.sparse as sp
from . import common

# Note: previous code would have failed if umfpack was provided by scikit
import scipy.sparse.linalg.dsolve.linsolve as linsolve
umfpack = linsolve.umfpack
uses_umfpack = linsolve.isUmfpack

from .. import system, physics

# check if we are actually using umfpack or rather SuperLU

if uses_umfpack:
    # This patches a memory leak in scipy:
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
        solve = factorized( A ) # Makes LU decomposition.
        x1 = solve( rhs1 ) # Uses the LU factors.
        x2 = solve( rhs2 ) # Uses again the LU factors.

        Parameters
        ----------
        A : csc_matrix
            matrix to be factorized
        piv_tol : float, 0 <= piv_tol <= 1.0
        sym_piv_tol : float, 0 <= piv_tol <= 1.0
            thresholds used by umfpack for pivoting. 0 means no pivoting, 1.0
            means full pivoting as in dense matrices (guaranteeing stability,
            but reducing possibly sparsity). Defaults of umfpack are 0.1 and
            0.001 respectively. Whether piv_tol or sym_piv_tol are used is
            decided internally by umfpack, depending on whether the matrix is
            "symmetric" enough.
        """

        if not sp.isspmatrix_csc(A):
            A = sp.csc_matrix(A)

        A.sort_indices()
        A = A.asfptype()  #upcast to a floating point format

        if A.dtype.char not in 'dD':
            raise ValueError("convert matrix data to double, please, using"
                             " .astype()")

        family = {'d' : 'di', 'D' : 'zi'}
        umf = umfpack.UmfpackContext( family[A.dtype.char] )

        # adjust pivot thresholds
        umf.control[umfpack.UMFPACK_PIVOT_TOLERANCE] = piv_tol
        umf.control[umfpack.UMFPACK_SYM_PIVOT_TOLERANCE] = sym_piv_tol

        # Make LU decomposition.
        umf.numeric( A )

        def solve( b ):
            return umf.solve( umfpack.UMFPACK_A, A, b, autoTranspose = True )

        return solve
else:
    # no Umfpack found. SuperLU is being used, but usually abysmally slow
    # (SuperLu is not bad per se, somehow the scipy version isn't good)
    warnings.warn("The installed scipy does not use Umfpack. Instead, "
                  "scipy will use the version of SuperLu it is shipped with. "
                  "Performance can be very poor in this case.", RuntimeWarning)

    factorized = linsolve.factorized


class Solver(common.SparseSolver):
    """Sparse Solver class based on the sparse direct solvers provided
    by scipy.
    """

    lhsformat = 'csc'
    rhsformat = 'csc'
    nrhs = 1

    def solve_linear_sys(self, a, b, kept_vars=None, factored=None):
        """
        Solve matrix system of equations a x = b with sparse input,
        using UMFPACK.

        Parameters
        ----------
        a : a scipy.sparse.csc_matrix sparse matrix
        b : a list of matrices.
            Sizes of these matrices may be smaller than needed, the missing
            entries at the end are padded with zeros.
        kept_vars : slice object or list of integers
            a list of numbers of variables to keep in the solution

        Returns
        -------
        output : a numpy matrix
            solution to the system of equations.

        Notes
        -----
        This function is largely a wrapper to `factorized`.
        """
        a = sp.csc_matrix(a)

        if kept_vars == None:
            kept_vars = [range(a.shape[1])]

        if not factored:
            slv = factorized(a)
        else:
            slv = factored

        sols = []
        vec = np.empty(a.shape[0], complex)
        for mat in b:
            if mat.shape[1] != 0:
                # See comment about zero-shaped sparse matrices at the top of
                # _sparse.py.
                mat = sp.csr_matrix(mat)
            for j in xrange(mat.shape[1]):
                vec[:] = mat[:, j].todense().flatten()
                sols.append(slv(vec)[kept_vars])

        if len(sols):
            return np.asarray(sols).transpose(), factored
        else:
            return np.asarray(np.zeros(shape=(len(kept_vars), 0))), factored


default_solver = Solver()

solve = default_solver.solve
ldos = default_solver.ldos
