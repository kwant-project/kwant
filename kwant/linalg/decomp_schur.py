# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['schur', 'convert_r2c_schur', 'order_schur', 'evecs_from_schur',
           'gen_schur', 'order_gen_schur', 'convert_r2c_gen_schur',
           'evecs_from_gen_schur']

from math import sqrt
import numpy as np
from . import lapack


def schur(a, calc_q=True, calc_ev=True, overwrite_a=False):
    """Compute the Schur form of a square matrix a.

    The Schur form is a decomposition of the form a = q * t * q^dagger, where q
    is a unitary matrix and t a upper triagonal matrix when computing the Schur
    form of a complex matrix, and a quasi-upper triagonal matrix with only 1x1
    and 2x2 blocks on the diagonal when computing the Schur form of a real
    matrix (In the latter case, the 1x1 blocks correspond to real eigenvalues,
    the 2x2 blocks to conjugate pairs of complex eigenvalues).

    The Schur form is closely related to the eigenvalue problem (the entries of
    the diagonal of the complex Schur form are the eigenvalues of the matrix),
    and the routine can optionally also return the eigenvalues.

    Parameters
    ----------
    a : array, shape (M, M)
        Matrix for which to compute the Schur form.
    calc_q : boolean
        Whether to compute the unitary/orthogonal matrix `q`.
    calc_ev : boolean
        Whether to return the eigenvalues as a separate array.
    overwrite_a : boolean
        Whether to overwrite data in `a` (may increase performance).

    Returns
    -------
    t : array, shape (M, M)
        Schur form of the original matrix (complex or real, depending on the
        input matrix).

    (if calc_q == True)
    q : array, shape (M, M)
        Unitary transformation matrix.

    (if calc_ev == True)
    ev: array, shape (M,)
        Array of eigenvalues of the matrix `a`. Can be complex even if a is
        real. In the latter case, the complex eigenvalues come in conjugated
        pairs with the eigenvalue with positive imaginary part coming
        first.

    Raises
    ------
    LinAlgError
        If the underlying QR iteration fails to converge.
    """
    a = lapack.prepare_for_lapack(overwrite_a, a)
    return lapack.gees(a, calc_q, calc_ev)


def convert_r2c_schur(t, q):
    """Convert a real Schur form (with possibly 2x2 blocks on the diagonal)
    into a complex Schur form that is completely triangular.

    This function is equivalent to the scipy.linalg.rsf2csf pendant (though the
    implementation is different), but there is additionally the guarantee that
    in the case of a 2x2 block at rows and columns i and i+1, t[i, i] will
    contain the eigenvalue with the positive part, and t[i+1, i+1] the one with
    the negative part.  This ensures that the list of eigenvalues (more
    precisely, their order) returned originally from schur() is still valid for
    the newly formed complex Schur form.

    Parameters
    ----------
    t : array, shape (M, M)
        Real Schur form of the original matrix
    q : array, shape (M, M)
        Schur transformation matrix

    Returns
    -------
    t : array, shape (M, M)
        Complex Schur form of the original matrix
    q : array, shape (M, M)
        Schur transformation matrix corresponding to the complex form
    """

    # First find the positions of 2x2-blocks
    blockpos = np.diagonal(t, -1).nonzero()[0]

    # Check if there are actually any 2x2-blocks
    if not blockpos.size:
        return (t, q)
    else:
        t2 = t.astype(np.common_type(t, np.array([], np.complex64)))
        q2 = q.astype(np.common_type(q, np.array([], np.complex64)))

    for i in blockpos:
        # Bringing a 2x2 block to complex triangular form is relatively simple:
        # the 2x2 blocks are guaranteed to be of the form [[a, b], [c, a]],
        # where b*c < 0. The eigenvalues of this matrix are a +/- i sqrt(-b*c),
        # the corresponding eigenvectors are [ +/- sqrt(-b*c), c].  The Schur
        # form can be achieved by a unitary 2x2 matrix with one of the
        # eigenvectors in the first column, and the second column an orthogonal
        # vector.

        a = t[i, i]
        b = t[i, i+1]
        c = t[i+1, i]

        x = 1j * sqrt(-b * c)
        y = c
        norm = sqrt(-b * c + c * c)

        U = np.array([[x / norm, -y / norm], [y / norm, -x / norm]])

        t2[i, i] = a + x
        t2[i+1, i] = 0
        t2[i, i+1] = -b - c
        t2[i+1, i+1] = a - x

        t2[:i, i:i+2] = np.dot(t2[:i, i:i+2], U)
        t2[i:i+2, i+2:] = np.dot(np.conj(U.T), t2[i:i+2, i+2:])

        q2[:, i:i+2] = np.dot(q2[:, i:i+2], U)

    return t2, q2


def order_schur(select, t, q, calc_ev=True, overwrite_tq=False):
    """Reorder the Schur form, selecting a cluster of eigenvalues.

    This function reorders the generalized Schur form such that the cluster of
    eigenvalues determined by select appears in the leading diagonal block of
    the Schur form (this is useful, as the Schur vectors corresponding to the
    leading diagonal block form an orthogonal basis for the subspace of
    eigenvectors).

    If a real Schur form is reordered, it is converted to complex form
    (eliminating the 2x2 blocks on the diagonal) if in a complex conjugated
    pair of eigenvalues only one eigenvalue is chosen.  In this case, the real
    Schur from cannot be reordered in real form without splitting a 2x2 block
    on the diagonal, hence switching to complex form is mandatory.

    Parameters
    ----------
    t : array, shape (M, M)
        Schur form
    q : array, shape (M, M)
        Unitary/orthogonal transformation matrices.
    calc_ev : boolean, optional
        Whether to return the reordered generalized eigenvalues of as two
        separate arrays. Default: True
    overwrite_tq : boolean, optional
        Whether to overwrite data in `t` and `q` (may increase performance)
        Default: False

    Returns
    -------
    t : array, shape (M, M)
        Reordered Schur form. If the original Schur form is real, and the
        desired reordering separates complex conjugated pairs of generalized
        eigenvalues, the resulting Schur form will be complex.
    q : array, shape (M, M)
        Unitary/orthogonal transformation matrix. Only computed if q is
        provided (not None) as input. If the Schur form is converted from real
        to complex, the transformation matrix is also converted from real
        orthogonal to complex unitary
    alpha : array, shape (M)
    beta : array, shape (M)
        Reordered eigenvalues. If the reordered Schur form is real, complex
        conjugated pairs of eigenvalues are ordered such that the eigenvalue
        with the positive imaginary part comes first.  Only computed if
        ``calc_ev == True``
    """

    t, q = lapack.prepare_for_lapack(overwrite_tq, t, q)

    # Figure out if select is a function or array.
    isfun = isarray = True
    try:
        select(0)
    except:
        isfun = False
    try:
        select[0]
    except:
        isarray = False

    if not (isarray or isfun):
        raise ValueError("select must be either a function or an array")
    elif isarray:
        select = np.array(select, dtype=lapack.logical_dtype, order='F')
    else:
        select = np.array(np.vectorize(select)(np.arange(t.shape[0])),
                          dtype=lapack.logical_dtype, order='F')

    # Now check if the reordering can actually be done as desired,
    # if we have a real Schur form (i.e. if the 2x2 blocks would be
    # separated). If this is the case, convert to complex Schur form first.
    for i in np.diagonal(t, -1).nonzero()[0]:
        if bool(select[i]) != bool(select[i+1]):
            t, q = convert_r2c_schur(t, q)
            return order_schur(select, t, q, calc_ev, True)

    return lapack.trsen(select, t, q, calc_ev)


def evecs_from_schur(t, q, select=None, left=False, right=True,
                     overwrite_tq=False):
    """Compute eigenvectors from Schur form.

    This function computes either all or selected eigenvectors for the matrix
    that is represented by the Schur form t and the unitary matrix q, (not the
    eigenvectors of t, but of q*t*q^dagger).

    Parameters
    ----------
    t : array, shape (M, M)
        Schur form
    q : array, shape (M, M)
        Unitary/orthogonal transformation matrix.
    select : boolean function or array, optional
        The value of ``select(i)`` or ``select[i]`` is used to decide whether
        the eigenvector corresponding to the i-th eigenvalue should be
        computed or not. If select is not provided (None), all eigenvectors
        are computed. Default: None
    left : boolean, optional
        Whether to compute left eigenvectors. Default: False
    right : boolean, optional
        Whether to compute right eigenvectors. Default: True
    overwrite_tq : boolean, optional
        Whether to overwrite data in `t` and `q` (may increase performance)
        Default: False

    Returns
    -------
    vl : array, shape(M, N)
        Left eigenvectors. N is the number of eigenvectors selected b
        `select`, or equal to M if select is not provided. The eigenvectors
        may be complex, even if `t` and `q` are real. Only computed if
        ``left == True``.
    vr : array, shape(M, N)
        Right eigenvectors. N is the number of eigenvectors selected by
        `select`, or equal to M if select is not provided. The eigenvectors
        may be complex, even if `t` and `q` are real. Only computed if
        ``right == True``.
    """

    t, q = lapack.prepare_for_lapack(overwrite_tq, t, q)

    # check if select is a function or an array
    if select is not None:
        isfun = isarray = True
        try:
            select(0)
        except:
            isfun = False

        try:
            select[0]
        except:
            isarray = False

        if not (isarray or isfun):
            raise ValueError("select must be either a function, "
                             "an array or None")
        elif isarray:
            selectarr = np.array(select, dtype=lapack.logical_dtype,
                                 order='F')
        else:
            selectarr = np.array(np.vectorize(select)(np.arange(t.shape[0])),
                                 dtype=lapack.logical_dtype, order='F')
    else:
        selectarr = None

    return lapack.trevc(t, q, selectarr, left, right)


def gen_schur(a, b, calc_q=True, calc_z=True, calc_ev=True,
              overwrite_ab=False):
    """Compute the generalized Schur form of a matrix pencil (a, b).

    The generalized Schur form is a decomposition of the form a = q * s *
    z^dagger and b = q * t * z^dagger, where q and z are unitary matrices
    (orthogonal for real input), t is an upper triagonal matrix with
    non-negative real diagonal, and s is a upper triangular matrix for complex
    matrices, and a quasi-upper triangular matrix with only 1x1 and 2x2 blocks
    on the diagonal for real matrices.  (In the latter case, the 1x1 blocks
    correspond to real generalized eigenvalues, the 2x2 blocks to conjugate
    pairs of complex generalized eigenvalues).

    The generalized Schur form is closely related to the generalized eigenvalue
    problem (the entries of the diagonal of the complex Schur form are the
    eigenvalues of the matrix, for example), and the routine can optionally
    also return the generalized eigenvalues in the form (alpha, beta), such
    that alpha/beta is a generalized eigenvalue of the pencil (a, b) (see also
    gen_eig()).

    Parameters
    ----------
    a : array, shape (M, M)
    b : array, shape (M, M)
        Matrix pencil for which to compute the generalized Schur form
    calc_q : boolean, optional
    calc_z : boolean, optional
        Whether to compute the unitary/orthogonal matrices `q` and `z`.
        Default: True
    calc_ev : boolean, optional
        Whether to return the generalized eigenvalues as two separate
        arrays. Default: True
    overwrite_ab : boolean, optional
        Whether to overwrite data in `a` and `b` (may increase performance)
        Default: False

    Returns
    -------
    s : array, shape (M, M)
    t : array, shape (M, M)
        Generalized Schur form of the original matrix pencil (`a`,`b`)
        (complex or real, depending on the input matrices)
    q : array, shape (M, M)
    z : array, shape (M, M)
        Unitary/orthogonal transformation matrices. Only computed if
        ``calc_q == True`` or ``calc_z == True``, respectively.
    alpha : array, shape (M)
    beta : array, shape (M)
        Generalized eigenvalues of the matrix pencil (`a`, `b`) given
        as numerator (`alpha`) and denominator (`beta`), such that the
        generalized eigenvalues are given as ``alpha/beta``. alpha can
        be complex even if a is real. In the latter case, complex
        eigenvalues come in conjugated pairs with the eigenvalue with
        positive imaginary part coming first. Only computed if
        ``calc_ev == True``.

    Raises
    ------
    LinAlError
        If the underlying QZ iteration fails to converge.
    """
    a, b = lapack.prepare_for_lapack(overwrite_ab, a, b)
    return lapack.gges(a, b, calc_q, calc_z, calc_ev)


def order_gen_schur(select, s, t, q=None, z=None, calc_ev=True,
                    overwrite_stqz=False):
    """Reorder the generalized Schur form.

    This function reorders the generalized Schur form such that the cluster of
    eigenvalues determined by select appears in the leading diagonal blocks of
    the Schur form (this is useful, as the Schur vectors corresponding to the
    leading diagonal blocks form an orthogonal basis for the subspace of
    eigenvectors).

    If a real generalized Schur form is reordered, it is converted to complex
    form (eliminating the 2x2 blocks on the diagonal) if in a complex
    conjugated pair of eigenvalues only one eigenvalue is chosen.  In this
    case, the real Schur from cannot be reordered in real form without
    splitting a 2x2 block on the diagonal, hence switching to complex form is
    mandatory.

    Parameters
    ----------
    s : array, shape (M, M)
    t : array, shape (M, M)
        Matrices describing the generalized Schur form.
    q : array, shape (M, M), optional
    z : array, shape (M, M), optional
        Unitary/orthogonal transformation matrices. Default: None.
    calc_ev : boolean, optional
        Whether to return the reordered generalized eigenvalues of as two
        separate arrays. Default: True.
    overwrite_stqz : boolean, optional
        Whether to overwrite data in `s`, `t`, `q`, and `z` (may
        increase performance) Default: False.

    Returns
    -------
    s : array, shape (M, M)
    t : array, shape (M, M)
        Reordered general Schur form. If the original Schur form is real, and
        the desired reordering separates complex conjugated pairs of
        generalized eigenvalues, the resulting Schur form will be complex.
    q : array, shape (M, M)
    z : array, shape (M, M)
        Unitary/orthogonal transformation matrices. Only computed if
        `q` and `z` are provided (not None) on entry, respectively. If
        the generalized Schur form is converted from real to complex,
        the transformation matrices are also converted from real
        orthogonal to complex unitary
    alpha : array, shape (M)
    beta : array, shape (M)
        Reordered generalized eigenvalues. If the reordered Schur form is real,
        complex conjugated pairs of eigenvalues are ordered such that the
        eigenvalue with the positive imaginary part comes first.  Only computed
        if ``calc_ev == True``.

    Raises
    ------
    LinAlError
        If the problem is too ill-conditioned.
    """
    s, t, q, z = lapack.prepare_for_lapack(overwrite_stqz, s, t, q, z)


    # Figure out if select is a function or array.
    isfun = isarray = True
    try:
        select(0)
    except:
        isfun = False
    try:
        select[0]
    except:
        isarray = False

    if not (isarray or isfun):
        raise ValueError("select must be either a function or an array")
    elif isarray:
        select = np.array(select, dtype=lapack.logical_dtype, order='F')
    else:
        select = np.array(np.vectorize(select)(np.arange(t.shape[0])),
                          dtype=lapack.logical_dtype, order='F')

    # Now check if the reordering can actually be done as desired, if we have a
    # real Schur form (i.e. if the 2x2 blocks would be separated). If this is
    # the case, convert to complex Schur form first.
    for i in np.diagonal(s, -1).nonzero()[0]:
        if bool(select[i]) != bool(select[i+1]):
            # Convert to complex Schur form
            if q is not None and z is not None:
                s, t, q, z = convert_r2c_gen_schur(s, t, q, z)
            elif q is not None:
                s, t, q = convert_r2c_gen_schur(s, t, q=q, z=None)
            elif z is not None:
                s, t, z = convert_r2c_gen_schur(s, t, q=None, z=z)
            else:
                s, t = convert_r2c_gen_schur(s, t)

            return order_gen_schur(select, s, t, q, z, calc_ev, True)

    return lapack.tgsen(select, s, t, q, z, calc_ev)


def convert_r2c_gen_schur(s, t, q=None, z=None):
    """Convert a real generallzed Schur form (with possibly 2x2 blocks on the
    diagonal) into a complex Schur form that is completely triangular.  If the
    input is already completely triagonal (real or complex), the input is
    returned unchanged.

    This function guarantees that in the case of a 2x2 block at rows and
    columns i and i+1, the converted, complex Schur form will contain the
    generalized eigenvalue with the positive imaginary part in s[i,i] and
    t[i,i], and the one with the negative imaginary part in s[i+1,i+1] and
    t[i+1,i+1].  This ensures that the list of eigenvalues (more precisely,
    their order) returned originally from gen_schur() is still valid for the
    newly formed complex Schur form.

    Parameters
    ----------
    s : array, shape (M, M)
    t : array, shape (M, M)
        Real generalized Schur form of the original matrix
    q : array, shape (M, M), optional
    z : array, shape (M, M), optional
        Schur transformation matrix. Default: None

    Returns
    -------
    s : array, shape (M, M)
    t : array, shape (M, M)
        Complex generalized Schur form of the original matrix,
        completely triagonal
    q : array, shape (M, M)
    z : array, shape (M, M)
        Schur transformation matrices corresponding to the complex
        form. `q` or `z` are only computed if they are provided (not
        None) on input.

    Raises
    ------
    LinAlgError
        If it fails to convert a 2x2 block into complex form (unlikely).
    """

    s, t, q, z = lapack.prepare_for_lapack(True, s, t, q, z)
    # Note: overwrite=True does not mean much here, the arrays are all copied

    if (s.ndim != 2 or t.ndim != 2 or
        (q is not None and q.ndim != 2) or
        (z is not None and z.ndim != 2)):
        raise ValueError("Expect matrices as input")

    if ((s.shape[0] != s.shape[1] or t.shape[0] != t.shape[1] or
         s.shape[0] != t.shape[0]) or
        (q is not None and (q.shape[0] != q.shape[1] or
                            s.shape[0] != q.shape[0])) or
        (z is not None and (z.shape[0] != z.shape[1] or
                            s.shape[0] != z.shape[0]))):
        raise ValueError("Invalid Schur decomposition as input")

    # First, find the positions of 2x2-blocks.
    blockpos = np.diagonal(s, -1).nonzero()[0]

    # Check if there are actually any 2x2-blocks.
    if not blockpos.size:
        s2 = s
        t2 = t
        q2 = q
        z2 = z
    else:
        s2 = s.astype(np.common_type(s, np.array([], np.complex64)))
        t2 = t.astype(np.common_type(t, np.array([], np.complex64)))
        if q is not None:
            q2 = q.astype(np.common_type(q, np.array([], np.complex64)))
        if z is not None:
            z2 = z.astype(np.common_type(z, np.array([], np.complex64)))

    for i in blockpos:
        # In the following, we use gen_schur on individual 2x2 blocks (that are
        # promoted to complex form) to compute the complex generalized Schur
        # form. If necessary, order_gen_schur is used to ensure the desired
        # order of eigenvalues.

        sb, tb, qb, zb, alphab, betab = gen_schur(s2[i:i+2, i:i+2],
                                                  t2[i:i+2, i:i+2])

        # Ensure order of eigenvalues. (betab is positive)
        if alphab[0].imag < alphab[1].imag:
            sb, tb, qb, zb, alphab, betab = order_gen_schur([False, True],
                                                            sb, tb, qb, zb)

        s2[i:i+2, i:i+2] = sb
        t2[i:i+2, i:i+2] = tb

        s2[:i, i:i+2] = np.dot(s2[:i, i:i+2], zb)
        s2[i:i+2, i+2:] = np.dot(qb.T.conj(), s2[i:i+2, i+2:])
        t2[:i, i:i+2] = np.dot(t2[:i, i:i+2], zb)
        t2[i:i+2, i+2:] = np.dot(qb.T.conj(), t2[i:i+2, i+2:])

        if q is not None:
            q2[:, i:i+2] = np.dot(q[:, i:i+2], qb)
        if z is not None:
            z2[:, i:i+2] = np.dot(z[:, i:i+2], zb)

    if q is not None and z is not None:
        return s2, t2, q2, z2
    elif q is not None:
        return s2, t2, q2
    elif z is not None:
        return s2, t2, z2
    else:
        return s2, t2


def evecs_from_gen_schur(s, t, q=None, z=None, select=None,
                         left=False, right=True, overwrite_qz=False):
    """Compute eigenvectors from Schur form.

    This function computes either all or selected eigenvectors for the matrix
    that is represented by the generalized Schur form (s, t) and the unitary
    matrices q and z, (not the generalized eigenvectors of (s,t), but of
    (q*s*z^dagger, q*t*z^dagger)).

    Parameters
    ----------
    s : array, shape (M, M)
    t : array, shape (M, M)
        Generalized Schur form.
    q : array, shape (M, M), optional
    z : array, shape (M, M), optional
        Unitary/orthogonal transformation matrices. If the left eigenvectors
        are to be computed, `q` must be provided, if the right eigenvectors are
        to be computed, `z` must be provided.
    select : boolean function or array, optional
        The value of ``select(i)`` or ``select[i]`` is used to decide
        whether the eigenvector corresponding to the i-th eigenvalue
        should be computed or not. If select is not provided, all
        eigenvectors are computed. Default: None.
    left : boolean, optional
        Whether to compute left eigenvectors. Default: False.
    right : boolean, optional
        Whether to compute right eigenvectors. Default: True.
    overwrite_qz : boolean, optional
        Whether to overwrite data in `q` and `z` (may increase performance).
        Note that s and t remain always unchanged Default: False.

    Returns
    -------
    (if left == True)
    vl : array, shape(M, N)
        Left generalized eigenvectors. N is the number of eigenvectors
        selected by select, or equal to M if select is not
        provided. The eigenvectors may be complex, even if `s`, `t`,
        `q` and `z` are real.

    (if right == True)
    vr : array, shape(M, N)
        Right generalized eigenvectors. N is the number of
        eigenvectors selected by select, or equal to M if select is
        not provided. The eigenvectors may be complex, even if `s`,
        `t`, `q` and `z` are real.

    """

    s, t, q, z = lapack.prepare_for_lapack(overwrite_qz, s, t, q, z)

    if left and q is None:
        raise ValueError("Matrix q must be provided for left eigenvectors")

    if right and z is None:
        raise ValueError("Matrix z must be provided for right eigenvectors")

    # Check if select is a function or an array.
    if select is not None:
        isfun = isarray = True
        try:
            select(0)
        except:
            isfun = False

        try:
            select[0]
        except:
            isarray = False

        if not (isarray or isfun):
            raise ValueError("select must be either a function, "
                             "an array or None")
        elif isarray:
            selectarr = np.array(select, dtype=lapack.logical_dtype,
                                 order='F')
        else:
            selectarr = np.array(np.vectorize(select)(np.arange(t.shape[0])),
                                 dtype=lapack.logical_dtype, order='F')
    else:
        selectarr = None

    return lapack.tgevc(s, t, q, z, selectarr, left, right)
