# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['gen_eig']

from . import lapack


def gen_eig(a, b, left=False, right=True, overwrite_ab=False):
    """Compute the eigenvalues and -vectors of the matrix pencil (a,b), i.e. of
    the generalized (unsymmetric) eigenproblem a v = lambda b v where a and b
    are square (unsymmetric) matrices, v the eigenvector and lambda the
    eigenvalues.

    The eigenvalues are returned as numerator alpha and denominator beta,
    i.e. lambda = alpha/beta. This is advantageous, as lambda can be infinity
    which is well-defined in this case as beta = 0.

    Parameters
    ----------
    a : array, shape (M, M)
    b : array, shape (M, M)
        `a` and `b` are the two matrices defining the generalized eigenproblem
    left : boolean
        Whether to calculate and return left eigenvectors
    right : boolean
        Whether to calculate and return right eigenvectors

    overwrite_ab : boolean
        Whether to overwrite data in `a` and `b` (may improve performance)

    Returns
    -------
    alpha : complex array, shape (M,)
    beta : real or complex array, shape (M,)
        The eigenvalues in the form ``alpha/beta``

    (if left == True)
    vl : double or complex array, shape (M, M)
        The left eigenvector corresponding to the eigenvalue
        ``alpha[i]/beta[i]`` is the column ``vl[:,i]``.

    (if right == True)
    vr : double or complex array, shape (M, M)
        The right eigenvector corresponding to the eigenvalue
        ``alpha[i]/beta[i]`` is the column ``vr[:,i]``.
    """

    ltype, a, b = lapack.prepare_for_lapack(overwrite_ab, a, b)

    if a.ndim != 2 or b.ndim != 2:
        raise ValueError("gen_eig requires both a and be to be matrices")

    if a.shape[0] != a.shape[1]:
        raise ValueError("gen_eig requires square matrix input")

    if b.shape[0] != a.shape[0] or b.shape[1] != a.shape[1]:
        raise ValueError("gen_eig requires a and be to have the same shape")

    ggev = getattr(lapack, ltype + "ggev")

    return ggev(a, b, left, right)
