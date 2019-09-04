# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.


from math import sin, cos, sqrt, pi, copysign
from collections import namedtuple

from itertools import combinations_with_replacement
import numpy as np
import numpy.linalg as npl
import scipy.linalg as la
from .. import linalg as kla
from scipy.linalg import block_diag
from scipy.sparse import (identity as sp_identity, hstack as sp_hstack,
                          csr_matrix)

dot = np.dot

__all__ = ['selfenergy', 'modes', 'PropagatingModes', 'StabilizedModes']


# TODO: Use scipy block_diag once we depend on scipy>=0.19
try:
    # Throws ValueError, but if fixed ensure that works as intended
    bdiag_broken = block_diag(np.zeros((1,1)), np.zeros((2,0))).shape != (3, 1)
except ValueError:  # skip coverage
    bdiag_broken = True
if bdiag_broken:  # skip coverage
    def block_diag(*matrices):
        """Construct a block diagonal matrix out of the input matrices.

        Like scipy.linalg.block_diag, but works for zero size matrices."""
        rows, cols = np.sum([mat.shape for mat in matrices], axis=0)
        b_mat = np.zeros((rows,cols), dtype='complex')
        rows, cols = 0, 0
        for mat in matrices:
            new_rows = rows + mat.shape[0]
            new_cols = cols + mat.shape[1]
            b_mat[rows:new_rows, cols:new_cols] = mat
            rows, cols = new_rows, new_cols
        return b_mat


# TODO: Remove the workaround once we depend on scipy >= 1.0
def lstsq(a, b):
    """Least squares version that works also with 0-shaped matrices."""
    if a.shape[1] == 0:
        return np.empty((0, 0), dtype=np.common_type(a, b))
    return la.lstsq(a, b)[0]


def nonzero_symm_projection(matrix):
    """Check whether a discrete symmetry relation between two blocks of the
    Hamiltonian vanishes or not.

    For a discrete symmetry S, the projection that maps from the block i to
    block j of the Hamiltonian with projectors P_i and P_j is
    S_ji = P_j^+ S P_i.
    This function determines whether S_ji vanishes or not, i. e. whether a
    symmetry relation exists between blocks i and j.

    Here, discrete symmetries and projectors are in canonical form, such that
    S maps each block at most either to itself or to a single other block.
    In other words, for each j, the block S_ji is nonzero at most for one i,
    while all other blocks vanish. If a nonzero block exists, it is unitary
    and hence has norm 1. To identify nonzero blocks without worrying about
    numerical errors, we thus check that its norm is larger than 0.5.
    """
    if not isinstance(matrix, np.ndarray):
        matrix = matrix.data
    return np.linalg.norm(matrix) > 0.5


def group_halves(arr_list):
    """Split and rearrange a list of arrays.

    Combine a list of arrays into a single array where the first halves
    of each array appear first:
    `[a b], [c d], [e f] -> [a c e b d f]`
    """
    list_ = [np.split(arr, 2) for arr in arr_list]
    lefts, rights = zip(*list_)
    return np.r_[tuple(lefts + rights)]


# Container classes
Linsys = namedtuple('Linsys', ['eigenproblem', 'v', 'extract'])


class PropagatingModes:
    """The calculated propagating modes of a lead.

    Attributes
    ----------
    wave_functions : numpy array
        The wave functions of the propagating modes.
    momenta : numpy array
        Momenta of the modes.
    velocities : numpy array
        Velocities of the modes.
    block_nmodes: list of integers
        Number of left or right moving propagating modes
        per conservation law block of the Hamiltonian.

    Notes
    =====
    The sort order of all the three arrays is identical. The first half of the
    modes have negative velocity, the second half have positive velocity.
    Within these halves the modes are ordered by the eigenvalues of any
    declared conservation laws. Within blocks with the same conservation law
    eigenvalue the modes with negative velocity are ordered by increasing
    momentum, and the modes with positive velocity are ordered by decreasing
    momentum. Finally, modes are ordered by the magnitude of their velocity.
    To summarize, the modes are ordered according to the key
    `(sign(v), conserved_quantity, sign(v) * k , abs(v))` where `v` is
    velocity, `k` is momentum and `conserved_quantity` is the conservation
    law eigenvalue.

    In the above, the positive velocity and momentum directions are defined
    with respect to the translational symmetry direction of the system.

    The first dimension of `wave_functions` corresponds to the orbitals of all
    the sites in a unit cell, the second one to the number of the mode.  Each
    mode is normalized to carry unit current. If several modes have the same
    momentum and velocity, an arbitrary orthonormal basis in the subspace of
    these modes is chosen.

    If a conservation law is specified to block diagonalize the Hamiltonian,
    then `block_nmodes[i]` is the number of left or right moving propagating
    modes in conservation law block `i`. The ordering of blocks is the same as
    the ordering of the projectors used to block diagonalize the Hamiltonian.
    """
    def __init__(self, wave_functions, velocities, momenta):
        kwargs = locals()
        kwargs.pop('self')
        self.__dict__.update(kwargs)


class StabilizedModes:
    """Stabilized eigendecomposition of the translation operator.

    Due to the lack of Hermiticity of the translation operator, its
    eigendecomposition is frequently poorly conditioned. Solvers in Kwant use
    this stabilized decomposition of the propagating and evanescent modes in
    the leads. If the hopping between the unit cells of an infinite system is
    invertible, the translation eigenproblem is written in the basis `psi_n,
    h_hop^+ * psi_(n+1)`, with ``h_hop`` the hopping between unit cells.  If
    `h_hop` is not invertible, and has the singular value decomposition `u s
    v^+`, then the eigenproblem is written in the basis `sqrt(s) v^+ psi_n,
    sqrt(s) u^+ psi_(n+1)`. In this basis we calculate the eigenvectors of the
    propagating modes, and the Schur vectors (an orthogonal basis) in the space
    of evanescent modes.

    `vecs` and `vecslmbdainv` are the first and the second halves of the wave
    functions.  The first `nmodes` are eigenmodes moving in the negative
    direction (hence they are incoming into the system in Kwant convention),
    the second `nmodes` are eigenmodes moving in the positive direction. The
    remaining modes are the Schur vectors of the modes evanescent in the
    positive direction. Propagating modes with the same eigenvalue are
    orthogonalized, and all the propagating modes are normalized to carry unit
    current. Finally the `sqrt_hop` attribute is `v sqrt(s)`.

    Attributes
    ----------
    vecs : numpy array
        Translation eigenvectors.
    vecslmbdainv : numpy array
        Translation eigenvectors divided by the corresponding eigenvalues.
    nmodes : int
        Number of left-moving (or right-moving) modes.
    sqrt_hop : numpy array or None
        Part of the SVD of `h_hop`, or None if the latter is invertible.
    """

    def __init__(self, vecs, vecslmbdainv, nmodes, sqrt_hop=None):
        kwargs = locals()
        kwargs.pop('self')
        self.__dict__.update(kwargs)

    def selfenergy(self):
        """
        Compute the self-energy generated by lead modes.

        Returns
        -------
        Sigma : numpy array, real or complex, shape (M,M)
            The computed self-energy. Note that even if `h_cell` and `h_hop` are
            both real, `Sigma` will typically be complex. (More precisely, if
            there is a propagating mode, `Sigma` will definitely be complex.)
        """
        v = self.sqrt_hop
        vecs = self.vecs[:, self.nmodes:]
        vecslmbdainv = self.vecslmbdainv[:, self.nmodes:]
        return dot(v, dot(vecs, la.solve(vecslmbdainv, v.T.conj())))


# Auxiliary functions that perform different parts of the calculation.
def setup_linsys(h_cell, h_hop, tol=1e6, stabilization=None):
    """Make an eigenvalue problem for eigenvectors of translation operator.

    Parameters
    ----------
    h_cell : numpy array with shape (n, n)
        Hamiltonian of a single lead unit cell.
    h_hop : numpy array with shape (n, m), m <= n
        Hopping Hamiltonian from a cell to the next one.
    tol : float
        Numbers are considered zero when they are smaller than `tol` times
        the machine precision.
    stabilization : sequence of 2 booleans or None
        Which steps of the eigenvalue problem stabilization to perform. If the
        value is `None`, then Kwant chooses the fastest (and least stable)
        algorithm that is expected to be sufficient.  For any other value,
        Kwant forms the eigenvalue problem in the basis of the hopping singular
        values.  The first element set to `True` forces Kwant to add an
        anti-Hermitian term to the cell Hamiltonian before inverting. If it is
        set to `False`, the extra term will only be added if the cell
        Hamiltonian isn't invertible. The second element set to `True` forces
        Kwant to solve a generalized eigenvalue problem, and not to reduce it
        to the regular one.  If it is `False`, reduction to a regular problem
        is performed if possible.

    Returns
    -------
    linsys : namedtuple
        A named tuple containing `matrices` a matrix pencil defining
        the eigenproblem, `v` a hermitian conjugate of the last matrix in
        the hopping singular value decomposition, and functions for
        extracting the wave function in the unit cell from the wave function
        in the basis of the nonzero singular exponents of the hopping.

    Notes
    -----
    The lead problem with degenerate hopping is rather complicated, and the
    details of the algorithm will be published elsewhere.

    """
    n = h_cell.shape[0]
    m = h_hop.shape[1]
    if stabilization is not None:
        stabilization = list(stabilization)

    if not np.any(h_hop):  # skip coverage
        # Inter-cell hopping is zero.  The current algorithm is not suited to
        # treat this extremely singular case.
        raise ValueError("Inter-cell hopping is exactly zero.")

    # If both h and t are real, it may be possible to use the real eigenproblem.
    if (not np.any(h_hop.imag)) and (not np.any(h_cell.imag)):
        h_hop = h_hop.real
        h_cell = h_cell.real

    eps = np.finfo(np.common_type(h_cell, h_hop)).eps * tol

    # First check if the hopping matrix has singular values close to 0.
    # (Close to zero is defined here as |x| < eps * tol * s[0] , where
    # s[0] is the largest singular value.)

    u, s, vh = la.svd(h_hop)
    assert m == vh.shape[1], "Corrupt output of svd."
    n_nonsing = np.sum(s > eps * s[0])

    if (n_nonsing == n and stabilization is None):
        # The hopping matrix is well-conditioned and can be safely inverted.
        # Hence the regular transfer matrix may be used.
        h_hop_sqrt = sqrt(np.linalg.norm(h_hop))
        A = h_hop / h_hop_sqrt
        B = h_hop_sqrt
        B_H_inv = 1.0 / B     # just a real scalar here
        A_inv = la.inv(A)

        lhs = np.zeros((2*n, 2*n), dtype=np.common_type(h_cell, h_hop))
        lhs[:n, :n] = -dot(A_inv, h_cell) * B_H_inv
        lhs[:n, n:] = -A_inv * B
        lhs[n:, :n] = A.T.conj() * B_H_inv

        def extract_wf(psi, lmbdainv):
            return B_H_inv * np.copy(psi[:n])

        matrices = (lhs, None)
        v_out = h_hop_sqrt * np.eye(n)
    else:
        if stabilization is None:
            stabilization = [None, False]

        # The hopping matrix has eigenvalues close to 0 - those
        # need to be eliminated.

        # Recast the svd of h_hop = u s v^dagger such that
        # u, v are matrices with shape n x n_nonsing.
        u = u[:, :n_nonsing]
        s = s[:n_nonsing]
        u = u * np.sqrt(s)
        # pad v with zeros if necessary
        v = np.zeros((n, n_nonsing), dtype=vh.dtype)
        v[:vh.shape[1]] = vh[:n_nonsing].T.conj()
        v = v * np.sqrt(s)

        # Eliminating the zero eigenvalues requires inverting the on-site
        # Hamiltonian, possibly including a self-energy-like term.  The
        # self-energy-like term stabilizes the inversion, but the most stable
        # choice is inherently complex. This can be disadvantageous if the
        # Hamiltonian is real, since staying in real arithmetics can be
        # significantly faster.  The strategy here is to add a complex
        # self-energy-like term always if the original Hamiltonian is complex,
        # and check for invertibility first if it is real

        matrices_real = issubclass(np.common_type(h_cell, h_hop), np.floating)
        add_imaginary = stabilization[0] or ((stabilization[0] is None) and
                                             not matrices_real)
        # Check if there is a chance we will not need to add an imaginary term.
        if not add_imaginary:
            h = h_cell
            sol = kla.lu_factor(h)
            rcond = kla.rcond_from_lu(sol, npl.norm(h, 1))

            if rcond < eps:
                need_to_stabilize = True
            else:
                need_to_stabilize = False

        if add_imaginary or need_to_stabilize:
            need_to_stabilize = True
            # Matrices are complex or need self-energy-like term to be
            # stabilized.
            temp = dot(u, u.T.conj()) + dot(v, v.T.conj())
            h = h_cell + 1j * temp

            sol = kla.lu_factor(h)
            rcond = kla.rcond_from_lu(sol, npl.norm(h, 1))

            # If the condition number of the stabilized h is
            # still bad, there is nothing we can do.
            if rcond < eps:
                raise RuntimeError("Flat band encountered at the requested "
                                   "energy, result is badly defined.")

        # The function that can extract the full wave function psi from
        # the projected one (v^dagger psi lambda^-1, s u^dagger psi).

        def extract_wf(psi, lmbdainv):
            wf = -dot(u, psi[: n_nonsing] * lmbdainv) - dot(v, psi[n_nonsing:])
            if need_to_stabilize:
                wf += 1j * (dot(v, psi[: n_nonsing]) +
                            dot(u, psi[n_nonsing:] * lmbdainv))
            return kla.lu_solve(sol, wf)

        # Setup the generalized eigenvalue problem.

        A = np.zeros((2 * n_nonsing, 2 * n_nonsing), np.common_type(h, h_hop))
        B = np.zeros((2 * n_nonsing, 2 * n_nonsing), np.common_type(h, h_hop))

        begin, end = slice(n_nonsing), slice(n_nonsing, None)

        A[end, begin] = np.identity(n_nonsing)
        temp = kla.lu_solve(sol, v)
        temp2 = dot(u.T.conj(), temp)
        if need_to_stabilize:
            A[begin, begin] = -1j * temp2
        A[begin, end] = temp2
        temp2 = dot(v.T.conj(), temp)
        if need_to_stabilize:
            A[end, begin] -= 1j *temp2
        A[end, end] = temp2

        B[begin, end] = -np.identity(n_nonsing)
        temp = kla.lu_solve(sol, u)
        temp2 = dot(u.T.conj(), temp)
        B[begin, begin] = -temp2
        if need_to_stabilize:
            B[begin, end] += 1j * temp2
        temp2 = dot(v.T.conj(), temp)
        B[end, begin] = -temp2
        if need_to_stabilize:
            B[end, end] = 1j * temp2

        v_out = v[:m]

        # Solving a generalized eigenproblem is about twice as expensive
        # as solving a regular eigenvalue problem.
        # Computing the LU factorization is negligible compared to both
        # (approximately 1/30th of a regular eigenvalue problem).
        # Because of this, it makes sense to try to reduce
        # the generalized eigenvalue problem to a regular one, provided
        # the matrix B can be safely inverted.

        lu_b = kla.lu_factor(B)
        if not stabilization[1]:
            rcond = kla.rcond_from_lu(lu_b, npl.norm(B, 1))
            # A more stringent condition is used here since errors can
            # accumulate from here to the eigenvalue calculation later.
            stabilization[1] = rcond > eps * tol

        if stabilization[1]:
            matrices = (kla.lu_solve(lu_b, A), None)
        else:
            matrices = (A, B)
    return Linsys(matrices, v_out, extract_wf)


def unified_eigenproblem(a, b=None, tol=1e6):
    """A helper routine for modes(), that wraps eigenproblems.

    This routine wraps the regular and general eigenproblems that can arise
    in a unfied way.

    Parameters
    ----------
    a : numpy array
        The matrix on the left hand side of a regular or generalized eigenvalue
        problem.
    b : numpy array or None
        The matrix on the right hand side of the generalized eigenvalue problem.
    tol : float
        The tolerance for separating eigenvalues with absolute value 1 from the
        rest.

    Returns
    -------
    ev : numpy array
        An array of eigenvalues (can contain NaNs and Infs, but those
        are not accessed in `modes()`) The number of eigenvalues equals
        twice the number of nonzero singular values of
        `h_hop` (so `2*h_cell.shape[0]` if `h_hop` is invertible).
    evanselect : numpy integer array
        Index array of right-decaying modes.
    propselect : numpy integer array
        Index array of propagating modes (both left and right).
    vec_gen(select) : function
        A function that computes the eigenvectors chosen by the array select.
    ord_schur(select) : function
        A function that computes the unitary matrix (corresponding to the right
        eigenvector space) of the (general) Schur decomposition reordered such
        that the eigenvalues chosen by the array select are in the top left
        block.
    """
    if b is None:
        eps = np.finfo(a.dtype).eps * tol
        t, z, ev = kla.schur(a)

        # Right-decaying modes.
        select = abs(ev) > 1 + eps
        # Propagating modes.
        propselect = abs(abs(ev) - 1) < eps

        vec_gen = lambda x: kla.evecs_from_schur(t, z, select=x)
        ord_schur = lambda x: kla.order_schur(x, t, z, calc_ev=False)[1]

    else:
        eps = np.finfo(np.common_type(a, b)).eps * tol
        s, t, z, alpha, beta = kla.gen_schur(a, b, calc_q=False)

        # Right-decaying modes.
        select = abs(alpha) > (1 + eps) * abs(beta)
        # Propagating modes.
        propselect = (abs(abs(alpha) - abs(beta)) < eps * abs(beta))

        with np.errstate(divide='ignore', invalid='ignore'):
            ev = alpha / beta
        # Note: the division is OK here, since we later only access
        #       eigenvalues close to the unit circle

        vec_gen = lambda x: kla.evecs_from_gen_schur(s, t, z=z, select=x)
        ord_schur = lambda x: kla.order_gen_schur(x, s, t, z=z,
                                                  calc_ev=False)[2]

    return ev, select, propselect, vec_gen, ord_schur


def phs_symmetrization(wfs, particle_hole):
    """Makes the wave functions that have the same velocity at a time-reversal
    invariant momentum (TRIM) particle-hole symmetric.

    If P is the particle-hole operator and P^2 = 1, then a particle-hole
    symmetric wave function at a TRIM is an eigenstate of P with eigenvalue 1.
    If P^2 = -1, wave functions with the same velocity at a TRIM come in pairs.
    Such a pair is particle-hole symmetric if the wave functions are related by
    P, i. e. the pair can be expressed as [psi_n, P psi_n] where psi_n is a wave
    function.

    To ensure proper ordering of modes, this function also returns an array
    of indices which ensures that particle-hole partners are properly ordered
    in this subspace of modes. These are later used with np.lexsort to ensure
    proper ordering.

    Parameters
    ----------
    wfs : numpy array
        A matrix of propagating wave functions at a TRIM that all have the same
        velocity. The orthonormal wave functions form the columns of this matrix.
    particle_hole : numpy array
        The matrix representation of the unitary part of the particle-hole
        operator, expressed in the tight binding basis.

    Returns
    -------
    new_wfs : numpy array
        The matrix of particle-hole symmetric wave functions.
    TRIM_sort: numpy integer array
        Index array that stores the proper sort order of particle-hole symmetric
        wave functions in this subspace.
    """

    def Pdot(mat):
        """Apply the particle-hole operator to an array. """
        return particle_hole.dot(mat.conj())

    # Take P in the subspace of W = wfs: U = W^+ @ P @ W^*.
    U = wfs.T.conj().dot(Pdot(wfs))
    # Check that wfs are orthonormal and the space spanned
    # by them is closed under ph, meaning U is unitary.
    if not np.allclose(U.dot(U.T.conj()), np.eye(U.shape[0])):
        raise ValueError('wfs are not orthonormal or not closed under particle_hole.')
    P_squared = U.dot(U.conj())
    if np.allclose(P_squared, np.eye(U.shape[0])):
        P_squared = 1
    elif np.allclose(P_squared, -np.eye(U.shape[0])):
        P_squared = -1
    else:
        raise ValueError('particle_hole must square to +1 or -1. P_squared = {}'.format(P_squared))

    if P_squared == 1:
        # Use the matrix square root method from
        # Applied Mathematics and Computation 234 (2014) 380-384.
        assert np.allclose(U, U.T)
        # Schur decomposition guarantees that vecs are orthonormal.
        vals, vecs = la.schur(U)
        # U should be normal, so vals is diagonal.
        assert np.allclose(np.diag(np.diag(vals)), vals)
        vals = np.diag(vals)
        # Need to take safe square root of U, the branch cut should not go
        # through any eigenvalues. Otherwise the square root may not be symmetric.
        # Find largest gap between eigenvalues
        phases = np.sort(np.angle(vals))
        dph = np.append(np.diff(phases), phases[0] + 2*np.pi - phases[-1])
        i = np.argmax(dph)
        shift = -np.pi - (phases[i] + dph[i]/2)
        # Take matrix square root with branch cut in largest gap
        vals = np.sqrt(vals * np.exp(1j * shift)) * np.exp(-0.5j * shift)
        sqrtU = vecs.dot(np.diag(vals)).dot(vecs.T.conj())
        # For symmetric U sqrt(U) is also symmetric.
        assert np.allclose(sqrtU, sqrtU.T)
        # We want a new basis W_new such that W_new^+ @ P @ W_new^* = 1.
        # This is achieved by W_new = W @ sqrt(U).
        new_wfs = wfs.dot(sqrtU)
        # If P^2 = 1, there is no need to sort the modes further.
        TRIM_sort = np.zeros((wfs.shape[1],), dtype=int)
    else:
        # P^2 = -1.
        # Iterate over wave functions to construct
        # particle-hole partners.
        new_wfs = []
        # The number of modes. This is always an even number >=2.
        N_modes = wfs.shape[1]
        # If there are only two modes in this subspace, they are orthogonal
        # so we replace the second one with the P applied to the first one.
        if N_modes == 2:
            wf = wfs[:, 0]
            # Store psi_n and P psi_n.
            new_wfs.append(wf)
            new_wfs.append(Pdot(wf))
        # If there are more than two modes, iterate over wave functions
        # and construct their particle-hole partners one by one.
        else:
            # We construct pairs of modes that are particle-hole partners.
            # Need to iterate over all pairs except the final one.
            iterations = range((N_modes-2)//2)
            for i in iterations:
                # Take a mode psi_n from the basis - the first column
                # of the matrix of remaining modes.
                wf = wfs[:, 0]
                # Store psi_n and P psi_n.
                new_wfs.append(wf)
                P_wf = Pdot(wf)
                new_wfs.append(P_wf)
                # Remove psi_n and P psi_n from the basis matrix of modes.
                # First remove psi_n.
                wfs = wfs[:, 1:]
                # Now we project the remaining modes onto the orthogonal
                # complement of P psi_n. projector:
                projector = wfs.dot(wfs.T.conj()) - \
                            np.outer(P_wf, P_wf.T.conj())
                # After the projection, the mode matrix is rank deficient -
                # the span of the column space has dimension one less than
                # the number of columns.
                wfs = projector.dot(wfs)
                wfs = la.qr(wfs, mode='economic', pivoting=True)[0]
                # Remove the redundant column.
                wfs = wfs[:, :-1]
                # If this is the final iteration, we only have two modes
                # left and can construct particle-hole partners without
                # the projection.
                if i == iterations[-1]:
                    assert wfs.shape[1] == 2
                    wf = wfs[:, 0]
                    # Store psi_n and P psi_n.
                    new_wfs.append(wf)
                    new_wfs.append(Pdot(wf))
                assert np.allclose(wfs.T.conj().dot(wfs),
                                   np.eye(wfs.shape[1]))
        new_wfs = np.hstack([col.reshape(len(col), 1)/npl.norm(col) for
                             col in new_wfs])
        assert np.allclose(new_wfs[:, 1::2], Pdot(new_wfs[:, ::2]))
        # Store sort ordering in this subspace of modes
        TRIM_sort = np.arange(new_wfs.shape[1])
    assert np.allclose(new_wfs.T.conj().dot(new_wfs), np.eye(new_wfs.shape[1]))
    return new_wfs, TRIM_sort


def make_proper_modes(lmbdainv, psi, extract, tol, particle_hole,
                      time_reversal, chiral):
    """
    Find, normalize and sort the propagating eigenmodes.

    Special care is taken of the case of degenerate k-values, where the
    numerically computed modes are typically a superposition of the real
    modes. In this case, also the proper (orthogonal) modes are computed.
    """
    vel_eps = np.finfo(psi.dtype).eps * tol

    nmodes = psi.shape[1]
    n = len(psi) // 2

    # Array for the velocities.
    velocities = np.empty(nmodes, dtype=float)

    # Array of indices to sort modes at a TRIM by PHS.
    TRIM_PHS_sort = np.zeros(nmodes, dtype=int)

    # Calculate the full wave function in real space.
    full_psi = extract(psi, lmbdainv)

    # Cast the types if any of the symmetry operators is complex
    for symm in time_reversal, particle_hole, chiral:
        if symm is None:
            continue
        full_psi = full_psi.astype(np.common_type(symm, full_psi))
        psi = psi.astype(np.common_type(symm, psi))

    # Find clusters of nearby eigenvalues. Since the eigenvalues occupy the
    # unit circle, special care has to be taken to not introduce a cut at
    # lambda = -1.
    eps = np.finfo(lmbdainv.dtype).eps * tol
    angles = np.angle(lmbdainv)
    sort_order = np.resize(np.argsort(angles), (2 * len(angles,)))
    boundaries = np.argwhere(np.abs(np.diff(lmbdainv[sort_order]))
                             > eps).flatten() + 1

    # Detect the singular case of all eigenvalues equal.
    if boundaries.shape == (0,) and len(angles):
        boundaries = np.array([0, len(angles)])

    for interval in zip(boundaries[:-1], boundaries[1:]):
        if interval[1] > boundaries[0] + len(angles):
            break

        indx = sort_order[interval[0] : interval[1]]

        # If there is a degenerate eigenvalue with several different
        # eigenvectors, the numerical routines return some arbitrary
        # overlap of the real, physical solutions. In order
        # to figure out the correct wave function, we need to
        # have the full, not the projected wave functions
        # (at least to our current knowledge).

        # Finding the true modes is done in two steps:

        # 1. The true transversal modes should be orthogonal to each other, as
        # they share the same Bloch momentum (note that transversal modes with
        # different Bloch momenta k1 and k2 need not be orthogonal, the full
        # modes are orthogonal because of the longitudinal dependence e^{i k1
        # x} and e^{i k2 x}).  The modes with the same k are therefore
        # orthogonalized. Moreover for the velocity to have a proper value the
        # modes should also be normalized.

        q, r = la.qr(full_psi[:, indx], mode='economic')

        # If the eigenvectors were purely real up to this stage,
        # they will typically become complex after the rotation.
        if psi.dtype != np.common_type(psi, r):
            psi = psi.astype(np.common_type(psi, r))
        if full_psi.dtype != np.common_type(full_psi, q):
            full_psi = full_psi.astype(np.common_type(psi, q))

        full_psi[:, indx] = q
        psi[:, indx] = la.solve(r.T, psi[:, indx].T).T

        # 2. Moving infinitesimally away from the degeneracy
        # point, the modes should diagonalize the velocity
        # operator (i.e. when they are non-degenerate any more)
        # The modes are therefore rotated properly such that they
        # diagonalize the velocity operator.
        # Note that step 2. does not give a unique result if there are
        # two modes with the same velocity, or if the modes stay
        # degenerate even for a range of Bloch momenta (and hence
        # must have the same velocity). However, this does not matter,
        # since we are happy with any superposition in this case.

        vel_op = -1j * dot(psi[n:, indx].T.conj(), psi[:n, indx])
        vel_op = vel_op + vel_op.T.conj()
        vel_vals, rot = la.eigh(vel_op)

        # If the eigenvectors were purely real up to this stage,
        # they will typically become complex after the rotation.

        if psi.dtype != np.common_type(psi, rot):
            psi = psi.astype(np.common_type(psi, rot))
        if full_psi.dtype != np.common_type(full_psi, rot):
            full_psi = full_psi.astype(np.common_type(psi, rot))

        psi[:, indx] = dot(psi[:, indx], rot)
        full_psi[:, indx] = dot(full_psi[:, indx], rot)
        velocities[indx] = vel_vals

        # With particle-hole symmetry, treat TRIMs individually.
        # Particle-hole conserves velocity.
        # If P^2 = 1, we can pick modes at a TRIM as particle-hole eigenstates.
        # If P^2 = -1, a mode at a TRIM and its particle-hole partner are
        # orthogonal, and we pick modes such that they are related by
        # particle-hole symmetry.

        # At a TRIM, propagating translation eigenvalues are +1 or -1.
        if (particle_hole is not None and
            (np.abs(np.abs(lmbdainv[indx].real) - 1) < eps).all()):
            assert not len(indx) % 2
            # Set the eigenvalues to the exact TRIM values.
            if (np.abs(lmbdainv[indx].real - 1) < eps).all():
                lmbdainv[indx] = 1
            else:
                # Momenta are the negative arguments of the translation
                # eigenvalues, as computed below using np.angle. np.angle of -1
                # is pi, so this assigns k = -pi to modes with translation
                # eigenvalue -1.
                lmbdainv[indx] = -1

            # Original wave functions
            orig_wf = full_psi[:, indx]

            # Modes are already sorted by velocity in ascending order, as
            # returned by la.eigh above. The first half is thus incident,
            # and the second half outgoing.
            # Here we work within a subspace of modes with a fixed velocity.
            # Mostly, this is done to ensure that modes of different velocities
            # are not mixed when particle-hole partners are constructed for
            # P^2 = -1. First, we identify which modes have the same velocity.
            # In each such subspace of modes, we construct wave functions that
            # are particle-hole partners.
            vels = velocities[indx]
            # Velocities are sorted in ascending order. Find the indices of the
            # last instance of each unique velocity.
            inds = [ind+1 for ind, vel in enumerate(vels[:-1])
                    if np.abs(vel-vels[ind+1])>vel_eps]
            inds = [0] + inds + [len(vels)]
            inds = zip(inds[:-1], inds[1:])
            # Now possess an iterator over tuples, where each tuple (i,j)
            # contains the starting and final indices i and j of a submatrix
            # of the modes matrix, such that all modes in the submatrix
            # have the same velocity.

            # Iterate over all submatrices of modes with the same velocity.
            new_wf = []
            TRIM_sorts = []
            for ind_tuple in inds:
                # Pick out wave functions that have a given velocity
                wfs = orig_wf[:, slice(*ind_tuple)]
                # Make particle-hole symmetric modes
                new_modes, TRIM_sort = phs_symmetrization(wfs, particle_hole)
                new_wf.append(new_modes)
                # Store sorting indices of the TRIM modes with the given
                # velocity.
                TRIM_sorts.append(TRIM_sort)
            # Gather into a matrix of modes
            new_wf = np.hstack(new_wf)
            # Store the sort order of all modes at the TRIM.
            # Used later with np.lexsort when the ordering
            # of modes is done.
            TRIM_PHS_sort[indx] = np.hstack(TRIM_sorts)
            # Replace the old modes.
            full_psi[:, indx] = new_wf
            # For both cases P^2 = +-1, must rotate wave functions in the
            # singular value basis. Find the rotation from new basis to old.
            rot = new_wf.T.conj().dot(orig_wf)
            # Rotate the wave functions in the singular value basis
            psi[:, indx] = psi[:, indx].dot(rot.T.conj())

        # Ensure proper usage of chiral symmetry.
        if chiral is not None and time_reversal is None:
            out_orig = full_psi[:, indx[len(indx)//2:]]
            out = chiral.dot(full_psi[:, indx[:len(indx)//2]])
            # No least squares below because the modes should be orthogonal.
            rot = out_orig.T.conj().dot(out)
            full_psi[:, indx[len(indx)//2:]] = out
            psi[:, indx[len(indx)//2:]] = psi[:, indx[len(indx)//2:]].dot(rot)

    if np.any(abs(velocities) < vel_eps):
        raise RuntimeError("Found a mode with zero or close to zero velocity.")
    if 2 * np.sum(velocities < 0) != len(velocities):
        raise RuntimeError("Numbers of left- and right-propagating "
                           "modes differ, possibly due to a numerical "
                           "instability.")

    momenta = -np.angle(lmbdainv)
    # Sort the modes. The modes are sorted first by velocity and momentum,
    # and finally TRIM modes are properly ordered.
    order = np.lexsort([TRIM_PHS_sort, velocities,
                        -np.sign(velocities) * momenta, np.sign(velocities)])

    velocities = velocities[order]
    momenta = momenta[order]
    full_psi = full_psi[:, order]
    psi = psi[:, order]

    # Use particle-hole symmetry to relate modes that propagate in the
    # same direction but at opposite momenta.
    # Modes are sorted by velocity (first incident, then outgoing).
    # Incident modes are then sorted by momentum in ascending order,
    # and outgoing modes in descending order.
    # Adopted convention is to get modes with negative k (both in and out)
    # by applying particle-hole operator to modes with positive k.
    if particle_hole is not None:
        N = nmodes//2  # Number of incident or outgoing modes.
        # With particle-hole symmetry, N must be an even number.
        # Incident modes
        positive_k = (np.pi - eps > momenta[:N]) * (momenta[:N] > eps)
        # Original wave functions with negative values of k
        orig_neg_k = full_psi[:, :N][:, positive_k[::-1]]
        # For incident modes, ordering of wfs by momentum as returned by kwant
        # is [-k2, -k1, k1, k2], if k2, k1 > 0 and k2 > k1.
        # To maintain this ordering with ki and -ki as particle-hole partners,
        # reverse the order of the product at the end.
        wf_neg_k = particle_hole.dot(
                (full_psi[:, :N][:, positive_k]).conj())[:, ::-1]
        rot = lstsq(orig_neg_k, wf_neg_k)
        full_psi[:, :N][:, positive_k[::-1]] = wf_neg_k
        psi[:, :N][:, positive_k[::-1]] = \
                psi[:, :N][:, positive_k[::-1]].dot(rot)

        # Outgoing modes
        positive_k = (np.pi - eps > momenta[N:]) * (momenta[N:] > eps)
        # Original wave functions with negative values of k
        orig_neg_k = full_psi[:, N:][:, positive_k[::-1]]
        # For outgoing modes, ordering of wfs by momentum as returned by kwant
        # is like [k2, k1, -k1, -k2], if k2, k1 > 0 and k2 > k1.

        # Reverse order at the end to match momenta of opposite sign.
        wf_neg_k = particle_hole.dot(
                full_psi[:, N:][:, positive_k].conj())[:, ::-1]
        rot = lstsq(orig_neg_k, wf_neg_k)
        full_psi[:, N:][:, positive_k[::-1]] = wf_neg_k
        psi[:, N:][:, positive_k[::-1]] = \
                psi[:, N:][:, positive_k[::-1]].dot(rot)

    # Modes are ordered by velocity.
    # Use time-reversal symmetry to relate modes of opposite velocity.
    if time_reversal is not None:
        # Note: within this function, nmodes refers to the total number
        # of propagating modes, not either left or right movers.
        out_orig = full_psi[:, nmodes//2:]
        out = time_reversal.dot(full_psi[:, :nmodes//2].conj())
        rot = lstsq(out_orig, out)
        full_psi[:, nmodes//2:] = out
        psi[:, nmodes//2:] = psi[:, nmodes//2:].dot(rot)

    norm = np.sqrt(abs(velocities))
    full_psi = full_psi / norm
    psi = psi / norm

    return psi, PropagatingModes(full_psi, velocities, momenta)


def compute_block_modes(h_cell, h_hop, tol, stabilization,
                        time_reversal, particle_hole, chiral):
    """Calculate modes corresponding to a single projector. """
    n, m = h_hop.shape

    # Defer most of the calculation to helper routines.
    matrices, v, extract = setup_linsys(h_cell, h_hop, tol, stabilization)
    ev, evanselect, propselect, vec_gen, ord_schur = unified_eigenproblem(
        *(matrices + (tol,)))

    # v is never None.
    # h_hop.shape[0] and v.shape[1] not always the same.
    # Adding this makes the difference for some tests
    n = v.shape[1]

    nrightmovers = np.sum(propselect) // 2
    nevan = n - nrightmovers
    evan_vecs = ord_schur(evanselect)[:, :nevan]

    # Compute the propagating eigenvectors.
    prop_vecs = vec_gen(propselect)
    # Compute their velocity, and, if necessary, rotate them.

    # prop_vecs here is 'psi' in make_proper_modes, i.e. the wf in the SVD
    # basis.  It is in turn used to construct vecs and vecslmbdainv (the
    # propagating parts).  The evanescent parts of vecs and vecslmbdainv come
    # from evan_vecs above.
    prop_vecs, real_space_data = make_proper_modes(ev[propselect], prop_vecs,
                                                   extract, tol, particle_hole,
                                                   time_reversal, chiral)

    vecs = np.c_[prop_vecs[n:], evan_vecs[n:]]
    vecslmbdainv = np.c_[prop_vecs[:n], evan_vecs[:n]]

    # Prepare output for a single block
    wave_functions = real_space_data.wave_functions
    momenta = real_space_data.momenta
    velocities = real_space_data.velocities

    return (wave_functions, momenta, velocities, vecs, vecslmbdainv, v)


def transform_modes(modes_data, unitary=None, time_reversal=None,
                    particle_hole=None, chiral=None):
    """Transform the modes data for a given block of the Hamiltonian using a
    discrete symmetry (see arguments). The symmetry operator can also be
    specified as can also be identity, for the case when blocks are identical.

    Assume that modes_data has the form returned by compute_block_modes, i.e. a
    tuple (wave_functions, momenta, velocities, nmodes, vecs, vecslmbdainv, v)
    containing the block modes data.

    Assume that the symmetry operator is written in the proper basis (block
    basis, not full TB).

    """
    wave_functions, momenta, velocities, vecs, vecslmbdainv, v = modes_data

    # Copy to not overwrite modes from previous blocks
    wave_functions = wave_functions.copy()
    momenta = momenta.copy()
    velocities = velocities.copy()
    vecs = vecs.copy()
    vecslmbdainv = vecslmbdainv.copy()
    v = v.copy()

    nmodes = wave_functions.shape[1] // 2

    if unitary is not None:
        perm = np.arange(2*nmodes)
        conj = False
        flip_energy = False
    elif time_reversal is not None:
        unitary = time_reversal
        perm = np.arange(2*nmodes)[::-1]
        conj = True
        flip_energy = False
    elif particle_hole is not None:
        unitary = particle_hole
        perm = ((-1-np.arange(2*nmodes)) % nmodes +
                nmodes * (np.arange(2*nmodes) // nmodes))
        conj = True
        flip_energy = True
    elif chiral is not None:
        unitary = chiral
        perm = (np.arange(2*nmodes) % nmodes +
                nmodes * (np.arange(2*nmodes) < nmodes))
        conj = False
        flip_energy = True
    else:  # skip coverage
        raise ValueError("No relation between blocks was provided.")

    if conj:
        momenta *= -1
        vecs = vecs.conj()
        vecslmbdainv = vecslmbdainv.conj()
        wave_functions = wave_functions.conj()
        v = v.conj()

    if flip_energy:
        vecslmbdainv *= -1

    if flip_energy != conj:
        velocities *= -1

    wave_functions = unitary.dot(wave_functions)[:, perm]
    v = unitary.dot(v)
    vecs[:, :2*nmodes] = vecs[:, perm]
    vecslmbdainv[:, :2*nmodes] = vecslmbdainv[:, perm]
    velocities = velocities[perm]
    momenta = momenta[perm]
    return (wave_functions, momenta, velocities, vecs, vecslmbdainv, v)


def modes(h_cell, h_hop, tol=1e6, stabilization=None, *,
          discrete_symmetry=None, projectors=None, time_reversal=None,
          particle_hole=None, chiral=None):
    """Compute the eigendecomposition of a translation operator of a lead.

    Parameters
    ----------
    h_cell : numpy array, real or complex, shape (N,N) The unit cell
        Hamiltonian of the lead unit cell.
    h_hop : numpy array, real or complex, shape (N,M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers and differences are considered zero when they are smaller
        than `tol` times the machine precision.
    stabilization : sequence of 2 booleans or None
        Which steps of the eigenvalue prolem stabilization to perform. If the
        value is `None`, then Kwant chooses the fastest (and least stable)
        algorithm that is expected to be sufficient.  For any other value,
        Kwant forms the eigenvalue problem in the basis of the hopping singular
        values.  The first element set to `True` forces Kwant to add an
        anti-Hermitian term to the cell Hamiltonian before inverting. If it is
        set to `False`, the extra term will only be added if the cell
        Hamiltonian isn't invertible. The second element set to `True` forces
        Kwant to solve a generalized eigenvalue problem, and not to reduce it
        to the regular one.  If it is `False`, reduction to a regular problem
        is performed if possible.  Selecting the stabilization manually is
        mostly necessary for testing purposes.
    particle_hole : sparse or dense square matrix
        The unitary part of the particle-hole symmetry operator.
    time_reversal : sparse or dense square matrix
        The unitary part of the time-reversal symmetry operator.
    chiral : sparse or dense square matrix
        The chiral symmetry operator.
    projectors : an iterable of sparse or dense matrices
        Projectors that block diagonalize the Hamiltonian in accordance
        with a conservation law.

    Returns
    -------
    propagating : `~kwant.physics.PropagatingModes`
        Contains the array of the wave functions of propagating modes, their
        momenta, and their velocities. It can be used to identify the gauge in
        which the scattering problem is solved.
    stabilized : `~kwant.physics.StabilizedModes`
        A basis of propagating and evanescent modes used by the solvers.

    Notes
    -----
    The sorting of the propagating modes is fully described in the
    documentation for `~kwant.physics.PropagatingModes`.  In simple cases where
    bands do not cross, this ordering corresponds to "lowest modes first". In
    general, however, it is necessary to examine the band structure --
    something this function is not doing by design.

    Propagating modes with the same momentum are orthogonalized. All the
    propagating modes are normalized by current.

    `projectors`, `time_reversal`, `particle_hole`, and `chiral` affect the
    basis in which the scattering modes are expressed - see
    `~kwant.physics.DiscreteSymmetry` for details.

    This function uses the most stable and efficient algorithm for calculating
    the mode decomposition that the Kwant authors are aware about. Its details
    are to be published.
    """
    if discrete_symmetry is not None:
        projectors, time_reversal, particle_hole, chiral = discrete_symmetry
    n, m = h_hop.shape

    if h_cell.shape != (n, n):
        raise ValueError("Incompatible matrix sizes for h_cell and h_hop.")

    if not np.any(h_hop):
        wf = np.zeros((n, 0))
        v = np.zeros((m, 0))
        m = np.zeros((0, 0))
        vec = np.zeros((0,))
        return (PropagatingModes(wf, vec, vec), StabilizedModes(m, m, 0, v))

    ham = h_cell
    # Avoid the trouble of dealing with non-square hopping matrices.
    # TODO: How to avoid this while not doing a lot of book-keeping?
    hop = np.empty_like(ham, dtype=h_hop.dtype)
    hop[:, :m] = h_hop
    hop[:, m:] = 0

    # Provide default values to not deal with special cases.
    if projectors is None:
        projectors = [sp_identity(ham.shape[0], format='csr')]

    def to_dense_or_csr(matrix):
        if matrix is None:
            return csr_matrix(ham.shape)
        try:
            return matrix.tocsr()
        except AttributeError:
            return matrix

    symmetries = map(to_dense_or_csr,
                     (time_reversal, particle_hole, chiral))
    time_reversal, particle_hole, chiral = symmetries

    offsets = np.cumsum([0] + [projector.shape[1] for projector in projectors])
    indices = [slice(*i) for i in np.vstack([offsets[:-1], offsets[1:]]).T]
    projection_op = sp_hstack(projectors)

    def basis_change(a, antiunitary=False):
        b = projection_op
        # We need the extra transposes to ensure that sparse dot is used.
        if antiunitary:
            # b.T.conj() @ a @ b.conj()
            return (b.T.conj().dot((b.T.conj().dot(a)).T)).T
        else:
            # b.T.conj() @ a @ b
            return (b.T.dot((b.T.conj().dot(a)).T)).T

    # Conservation law basis
    ham_cons = basis_change(ham)
    hop_cons = basis_change(hop)
    trs = basis_change(time_reversal, True)
    phs = basis_change(particle_hole, True)
    sls = basis_change(chiral)

    # Check that the Hamiltonian has the conservation law
    block_modes = len(projectors) * [None]
    numbers_coords = combinations_with_replacement(enumerate(indices), 2)
    for (i, x), (j, y) in numbers_coords:
        h = ham_cons[x, y]
        t = hop_cons[x, y]
        # Symmetries that project from block x to block y
        symmetries = [symm[y, x] for symm in (trs, phs, sls)]
        symmetries = [(symm if nonzero_symm_projection(symm) else None) for
                      symm in symmetries]
        if i == j:
            if block_modes[i] is not None:
                continue
            # We did not compute this block yet.
            block_modes[i] = compute_block_modes(h, t, tol, stabilization,
                                                 *symmetries)
        else:
            if block_modes[j] is not None:
                # Modes in the block already computed.
                continue
            x, y = tuple(np.mgrid[x, x]), tuple(np.mgrid[y, y])

            if ham_cons[x].shape != ham_cons[y].shape:
                continue
            if (np.allclose(ham_cons[x], ham_cons[y]) and
                np.allclose(hop_cons[x], hop_cons[y])):
                unitary = sp_identity(h.shape[0])
            else:
                unitary = None
            if any(op is not None for op in symmetries + [unitary]):
                block_modes[j] = transform_modes(block_modes[i], unitary,
                                                 *symmetries)
    (wave_functions, momenta, velocities,
     vecs, vecslmbdainv, sqrt_hops) = zip(*block_modes)

    # Reorder by direction of propagation
    wave_functions = group_halves([(projector.dot(wf)).T for wf, projector in
                                   zip(wave_functions, projectors)]).T
    # Propagating modes object to return
    prop_modes = PropagatingModes(wave_functions, group_halves(velocities),
                                  group_halves(momenta))
    nmodes = [len(v) // 2 for v in velocities]
    # Store the number of modes per block as an attribute.
    # nmodes[i] is the number of left or right moving modes in block i.
    # In the module that makes leads with conservation laws, this is necessary
    # to keep track of the block structure of the scattering matrix.
    prop_modes.block_nmodes = nmodes

    parts = zip(*([v[:, :n], v[:, n:2*n], v[:, 2*n:]]
                  for n, v in zip(nmodes, vecs)))
    vecs = np.hstack([block_diag(*part) for part in parts])

    parts = zip(*([v[:, :n], v[:, n:2*n], v[:, 2*n:]]
                  for n, v in zip(nmodes, vecslmbdainv)))
    vecslmbdainv = np.hstack([block_diag(*part) for part in parts])

    sqrt_hops = np.hstack([projector.dot(hop) for projector, hop in
                             zip(projectors, sqrt_hops)])[:h_hop.shape[1]]

    stab_modes = StabilizedModes(vecs, vecslmbdainv, sum(nmodes), sqrt_hops)

    return prop_modes, stab_modes


def selfenergy(h_cell, h_hop, tol=1e6):
    """
    Compute the self-energy generated by the lead.

    Parameters
    ----------
    h_cell : numpy array, real or complex, shape (N,N) The unit cell Hamiltonian
        of the lead unit cell.
    h_hop : numpy array, real or complex, shape (N,M)
        The hopping matrix from a lead cell to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).
    tol : float
        Numbers are considered zero when they are smaller than `tol` times
        the machine precision.

    Returns
    -------
    Sigma : numpy array, real or complex, shape (M,M)
        The computed self-energy. Note that even if `h_cell` and `h_hop` are
        both real, `Sigma` will typically be complex. (More precisely, if there
        is a propagating mode, `Sigma` will definitely be complex.)

    Notes
    -----
    For simplicity this function internally calculates the modes first.
    This may cause a small slowdown, and can be improved if necessary.
    """
    stabilized = modes(h_cell, h_hop, tol)[1]
    return stabilized.selfenergy()


def square_selfenergy(width, hopping, fermi_energy):
    """
    Calculate analytically the self energy for a square lattice.

    The lattice is assumed to have a single orbital per site and
    nearest-neighbor hopping.

    Parameters
    ----------
    width : integer
        width of the lattice
    """

    # Following appendix C of M. Wimmer's diploma thesis:
    # http://www.physik.uni-regensburg.de/forschung/\
    # richter/richter/media/research/publications2004/wimmer-Diplomarbeit.pdf

    # p labels transversal modes.  i and j label the sites of a cell.

    # Precalculate the transverse wave function.
    psi_p_i = np.empty((width, width))
    factor = pi / (width + 1)
    prefactor = sqrt(2 / (width + 1))
    for p in range(width):
        for i in range(width):
            psi_p_i[p, i] = prefactor * sin(factor * (p + 1) * (i + 1))

    # Precalculate the integrals of the longitudinal wave functions.
    def f(q):
        if abs(q) <= 2:
            return q/2 - 1j * sqrt(1 - (q / 2) ** 2)
        else:
            return q/2 - copysign(sqrt((q / 2) ** 2 - 1), q)
    f_p = np.empty((width,), dtype=complex)
    for p in range(width):
        e = 2 * hopping * (1 - cos(factor * (p + 1)))
        q = (fermi_energy - e) / hopping - 2
        f_p[p] = f(q)

    # Put everything together into the self energy and return it.
    result = np.empty((width, width), dtype=complex)
    for i in range(width):
        for j in range(width):
            result[i, j] = hopping * (
                psi_p_i[:, i] * psi_p_i[:, j].conj() * f_p).sum()
    return result
