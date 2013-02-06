# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['lu_factor', 'lu_solve', 'rcond_from_lu']

import numpy as np
from . import lapack


def lu_factor(a, overwrite_a=False):
    """Compute the LU factorization of a matrix A = P * L * U. The function
    returns a tuple (lu, p, singular), where lu contains the LU factorization
    storing the unit lower triangular matrix L in the strictly lower triangle
    (the unit diagonal is not stored) and the upper triangular matrix U in the
    upper triangle. p is a vector of pivot indices, and singular a Boolean
    value indicating whether the matrix A is singular up to machine precision.

    NOTE: This function mimics the behavior of scipy.linalg.lu_factor (except
    that it has in addition the flag singular). The main reason is that
    lu_factor in SciPy has a bug that depending on the type of NumPy matrix
    passed to it, it would not return what was descirbed in the
    documentation. This bug will be (probably) fixed in 0.10.0 but until this
    is standard, this version is better to use.

    Parameters
    ----------
    a : array, shape (M, M)
        Matrix to factorize
    overwrite_a : boolean
        Whether to overwrite data in a (may increase performance)

    Returns
    -------
    lu : array, shape (N, N)
        Matrix containing U in its upper triangle, and L in its lower triangle.
        The unit diagonal elements of L are not stored.
    piv : array, shape (N,)
        Pivot indices representing the permutation matrix P:
        row i of matrix was interchanged with row piv[i].
    singular : boolean
        Whether the matrix a is singular (up to machine precision)
    """

    ltype, a = lapack.prepare_for_lapack(overwrite_a, a)

    if a.ndim != 2:
        raise ValueError("lu_factor expects a matrix")

    if ltype == 'd':
        return lapack.dgetrf(a)
    elif ltype == 'z':
        return lapack.zgetrf(a)
    elif ltype == 's':
        return lapack.sgetrf(a)
    else:
        return lapack.cgetrf(a)


def lu_solve((lu, ipiv, singular), b):
    """Solve a linear system of equations, a x = b, given the LU
    factorization of a

    Parameters
    ----------
    (lu, piv, singular)
        Factorization of the coefficient matrix a, as given by lu_factor
    b : array (vector or matrix)
        Right-hand side

    Returns
    -------
    x : array (vector or matrix)
        Solution to the system
    """

    if singular:
        raise RuntimeWarning("In lu_solve: the flag singular indicates "
                             "a singular matrix. Result of solve step "
                             "are probably unreliable")

    ltype, lu, b = lapack.prepare_for_lapack(False, lu, b)
    ipiv = np.ascontiguousarray(np.asanyarray(ipiv), dtype=lapack.int_dtype)

    if b.ndim > 2:
        raise ValueError("lu_solve: b must be a vector or matrix")

    if lu.shape[0] != b.shape[0]:
        raise ValueError("lu_solve: incompatible dimensions of b")

    if ltype == 'd':
        return lapack.dgetrs(lu, ipiv, b)
    elif ltype == 'z':
        return lapack.zgetrs(lu, ipiv, b)
    elif ltype == 's':
        return lapack.sgetrs(lu, ipiv, b)
    else:
        return lapack.cgetrs(lu, ipiv, b)


def rcond_from_lu((lu, ipiv, singular), norm_a, norm="1"):
    """Compute the reciprocal condition number from the LU decomposition as
    returned from lu_factor(), given additionally the norm of the matrix a in
    norm_a.

    The reciprocal condition number is given as 1/(||A||*||A^-1||), where
    ||...|| is a matrix norm.

    Parameters
    ----------
    (lu, piv, singular)
        Factorization of the matrix a, as given by lu_factor
    norm_a : float or complex
        norm of the original matrix a (type of norm is specified in norm)
    norm : {'1', 'I'}, optional
        type of matrix norm which should be used to compute the condition
        number ("1": 1-norm, "I": infinity norm). Default: '1'.

    Returns
    -------
    rcond : float or complex
        reciprocal condition number of a with respect to the type of matrix
        norm specified in norm
    """

    if not norm in ("1", "I"):
        raise ValueError("norm in rcond_from_lu must be either '1' or 'I'")

    ltype, lu = lapack.prepare_for_lapack(False, lu)

    if ltype == 'd':
        return lapack.dgecon(lu, norm_a, norm)
    elif ltype == 'z':
        return lapack.zgecon(lu, norm_a, norm)
    elif ltype == 's':
        return lapack.sgecon(lu, norm_a, norm)
    else:
        return lapack.cgecon(lu, norm_a, norm)
