# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
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
    a = lapack.prepare_for_lapack(overwrite_a, a)
    return lapack.getrf(a)


def lu_solve(matrix_factorization, b):
    """Solve a linear system of equations, a x = b, given the LU
    factorization of a

    Parameters
    ----------
    matrix_factorization
        Factorization of the coefficient matrix a, as given by lu_factor
    b : array (vector or matrix)
        Right-hand side

    Returns
    -------
    x : array (vector or matrix)
        Solution to the system
    """
    (lu, ipiv, singular) = matrix_factorization
    if singular:
        raise RuntimeWarning("In lu_solve: the flag singular indicates "
                             "a singular matrix. Result of solve step "
                             "are probably unreliable")

    lu, b = lapack.prepare_for_lapack(False, lu, b)
    ipiv = np.ascontiguousarray(np.asanyarray(ipiv), dtype=lapack.int_dtype)
    return lapack.getrs(lu, ipiv, b)


def rcond_from_lu(matrix_factorization, norm_a, norm="1"):
    """Compute the reciprocal condition number from the LU decomposition as
    returned from lu_factor(), given additionally the norm of the matrix a in
    norm_a.

    The reciprocal condition number is given as 1/(||A||*||A^-1||), where
    ||...|| is a matrix norm.

    Parameters
    ----------
    matrix_factorization
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
    (lu, ipiv, singular) = matrix_factorization
    norm = norm.encode('utf8')  # lapack expects bytes
    lu = lapack.prepare_for_lapack(False, lu)
    return lapack.gecon(lu, norm_a, norm)
