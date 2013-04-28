# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from __future__ import division
from math import sin, cos, sqrt, pi, copysign
from collections import namedtuple
from itertools import izip
import numpy as np
import numpy.linalg as npl
import scipy.linalg as la
from .. import linalg as kla

dot = np.dot

__all__ = ['selfenergy', 'modes', 'Modes']

Linsys = namedtuple('Linsys', ['eigenproblem', 'v', 'extract', 'project'])

def setup_linsys(h_onslice, h_hop, tol=1e6):
    """
    Make an eigenvalue problem for eigenvectors of translation operator.

    Parameters
    ----------
    h_onslice : numpy array with shape (n, n)
        Hamiltonian of a single lead slice.
    h_hop : numpy array with shape (n, m), m <= n
        Hopping Hamiltonian from the slice to the next one.

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
    n = h_onslice.shape[0]
    m = h_hop.shape[1]

    if not (np.any(h_hop.real) or np.any(h_hop.imag)):
        # Inter-slice hopping is zero.  The current algorithm is not suited to
        # treat this extremely singular case.
        # Note: np.any(h_hop) returns (at least from numpy 1.6.*)
        #       False if h_hop is purely imaginary
        raise ValueError("Inter-slice hopping is exactly zero.")

    eps = np.finfo(np.common_type(h_onslice, h_hop)).eps * tol

    # First check if the hopping matrix has singular values close to 0.
    # (Close to zero is defined here as |x| < eps * tol * s[0] , where
    # s[0] is the largest singular value.)

    u, s, vh = la.svd(h_hop)
    assert m == vh.shape[1], "Corrupt output of svd."
    n_nonsing = np.sum(s > eps * s[0])

    if n_nonsing == n:
        # The hopping matrix is well-conditioned and can be safely inverted.
        # Hence the regular transfer matrix may be used.
        hop_inv = la.inv(h_hop)

        A = np.zeros((2*n, 2*n), dtype=np.common_type(h_onslice, h_hop))
        A[:n, :n] = dot(hop_inv, -h_onslice)
        A[:n, n:] = -hop_inv
        A[n:, :n] = h_hop.T.conj()

        # The function that can extract the full wave function psi from the
        # projected one. Here it is almost trivial, but used for simplifying
        # the logic.

        def extract_wf(psi, lmbdainv):
            return psi[:n]

        # Project the full wave function back (also trivial).

        def project_wf(psi, lmbdainv):
            return np.r_[psi * lmbdainv, dot(h_hop.T.conj(), psi)]

        matrices = (A, None)
        v_out = None
    else:
        # The hopping matrix has eigenvalues close to 0 - those
        # need to be eliminated.

        # Recast the svd of h_hop = u s v^dagger such that
        # u, v are matrices with shape n x n_nonsing.
        u = u[:, :n_nonsing]
        s = s[:n_nonsing]
        # pad v with zeros if necessary
        v = np.zeros((n, n_nonsing), dtype=vh.dtype)
        v[:vh.shape[1]] = vh[:n_nonsing].T.conj()

        # Eliminating the zero eigenvalues requires inverting the
        # on-site Hamiltonian, possibly including a self-energy-like term.
        # The self-energy-like term stabilizes the inversion, but the most
        # stable choice is inherently complex. This can be disadvantageous
        # if the Hamiltonian is real - as staying in real arithmetics can be
        # significantly faster.
        # The strategy here is to add a complex self-energy-like term
        # always if the original Hamiltonian is complex, and check for
        # invertibility first if it is real

        need_to_stabilize = True

        if issubclass(np.common_type(h_onslice, h_hop), np.floating):
            # Check if stabilization is needed.
            h = h_onslice
            sol = kla.lu_factor(h)
            rcond = kla.rcond_from_lu(sol, npl.norm(h, 1))

            if rcond > eps:
                need_to_stabilize = False

        if need_to_stabilize:
            # Matrices are complex or need self-energy-like term to be
            # stabilized.
            temp = dot(u, s.reshape(-1, 1) * u.T.conj()) + dot(v, v.T.conj())
            h = h_onslice + 1j * temp

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
            wf = - dot(u * s, psi[: n_nonsing] * lmbdainv) - \
                 dot(v, psi[n_nonsing:])
            if need_to_stabilize:
                wf += 1j * (dot(v, psi[: n_nonsing]) +
                            dot(u, psi[n_nonsing:] * lmbdainv))
            return kla.lu_solve(sol, wf)

        # Project the full wave function back.

        def project_wf(psi, lmbdainv):
            return np.r_[dot(v.T.conj(), psi * lmbdainv),
                         dot(s.reshape(-1, 1) * u.T.conj(), psi)]

        # Setup the generalized eigenvalue problem.

        A = np.zeros((2 * n_nonsing, 2 * n_nonsing), np.common_type(h, h_hop))
        B = np.zeros((2 * n_nonsing, 2 * n_nonsing), np.common_type(h, h_hop))

        A[:n_nonsing, :n_nonsing] = -np.identity(n_nonsing)

        B[n_nonsing:, n_nonsing:] = np.identity(n_nonsing)
        u_s = u * s

        temp = kla.lu_solve(sol, v)
        temp2 = dot(u_s.T.conj(), temp)
        if need_to_stabilize:
            A[n_nonsing:, :n_nonsing] = 1j * temp2
        A[n_nonsing:, n_nonsing:] = -temp2
        temp2 = dot(v.T.conj(), temp)
        if need_to_stabilize:
            A[:n_nonsing, :n_nonsing] += temp2
        A[:n_nonsing, n_nonsing:] = -temp2

        temp = kla.lu_solve(sol, u_s)
        temp2 = dot(u_s.T.conj(), temp)
        B[n_nonsing:, :n_nonsing] = temp2
        if need_to_stabilize:
            B[n_nonsing:, n_nonsing:] -= 1j * temp2
        temp2 = dot(v.T.conj(), temp)
        B[:n_nonsing, :n_nonsing] = temp2
        if need_to_stabilize:
            B[:n_nonsing, n_nonsing:] = -1j * temp2

        # Solving a generalized eigenproblem is about twice as expensive
        # as solving a regular eigenvalue problem.
        # Computing the LU factorization is negligible compared to both
        # (approximately 1/30th of a regular eigenvalue problem).
        # Because of this, it makes sense to try to reduce
        # the generalized eigenvalue problem to a regular one, provided
        # the matrix B can be safely inverted.

        lu_b = kla.lu_factor(B)
        rcond = kla.rcond_from_lu(lu_b, npl.norm(B, 1))

        # A more stringent condition is used here since errors can accumulate
        # from here to the eigenvalue calculation later.
        v_out = v[:m]
        if rcond > eps * tol:
            matrices = (kla.lu_solve(lu_b, A), None)
        else:
            matrices = (A, B)

    return Linsys(matrices, v_out, extract_wf, project_wf)


def make_proper_modes(lmbdainv, psi, extract, project, tol=1e6):
    """
    Determine the velocities and direction of the propagating eigenmodes.

    Special care is taken of the case of degenerate k-values, where the
    numerically computed modes are typically a superposition of the real
    modes. In this case, also the proper (orthogonal) modes are computed.
    """
    vel_eps = np.finfo(psi.dtype).eps * tol

    # h_hop is either the full hopping matrix, or the singular
    # values vector of the svd.
    nmodes = psi.shape[1]
    n = len(psi) // 2

    # Array for the velocities.
    velocities = np.empty(nmodes, dtype=float)

    # Mark the right-going modes.
    rightselect = np.zeros(nmodes, dtype=bool)

    # Find clusters of nearby points.
    eps = np.finfo(lmbdainv.dtype).eps * tol
    angles = np.angle(lmbdainv)
    sort_order = np.resize(np.argsort(angles), (2 * len(angles,)))
    boundaries = np.argwhere(np.abs(np.diff(lmbdainv[sort_order]))
                             > eps).flatten() + 1

    for interval in izip(boundaries[:-1], boundaries[1:]):
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

        # 1. The true transversal modes should be orthogonal to
        # each other, as they share the same Bloch momentum (note
        # that transversal modes with different Bloch momenta k1
        # and k2 need not be orthogonal, the full modes are
        # orthogonal because of the longitudinal dependence
        # e^{i k1 x} and e^{i k2 x}).
        # The modes with the same k are therefore orthogonalized:

        if len(indx) > 1:
            full_psi = extract(psi[:, indx], lmbdainv[indx])

            # Note: Here's a workaround for the fact that the interface
            # to qr changed from SciPy 0.8.0 to 0.9.0
            try:
                full_psi = la.qr(full_psi, econ=True, mode='qr')[0]
            except TypeError:
                full_psi = la.qr(full_psi, mode='economic')[0]

            psi[:, indx] = project(full_psi, lmbdainv[indx])

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
        psi[:, indx] = dot(psi[:, indx], rot)
        velocities[indx] = vel_vals

    rightselect = velocities > vel_eps
    if np.any(abs(velocities) < vel_eps):
        raise RuntimeError("Found a mode with zero or close to zero velocity.")
    if 2 * np.sum(rightselect) != len(velocities):
        raise RuntimeError("Numbers of left- and right-propagating "
                           "modes differ, possibly due to a numerical "
                           "instability.")

    return psi, velocities, rightselect


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
        `h_hop` (so `2*h_onslice.shape[0]` if `h_hop` is invertible).
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

        warning_settings = np.seterr(divide='ignore', invalid='ignore')
        ev = alpha / beta
        np.seterr(**warning_settings)
        # Note: the division is OK here, since we later only access
        #       eigenvalues close to the unit circle

        vec_gen = lambda x: kla.evecs_from_gen_schur(s, t, z=z, select=x)
        ord_schur = lambda x: kla.order_gen_schur(x, s, t, z=z,
                                                  calc_ev=False)[2]

    return ev, select, propselect, vec_gen, ord_schur


Modes = namedtuple('Modes', ['vecs', 'vecslmbdainv', 'nmodes', 'svd'])


def modes(h_onslice, h_hop, tol=1e6):
    """
    Compute the eigendecomposition of a translation operator of a lead.

    Parameters
    ----------
    h_onslice : numpy array, real or complex, shape (N,N) The unit cell
        Hamiltonian of the lead slice.
    h_hop : numpy array, real or complex, shape (N,M)
        The hopping matrix from a lead slice to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).

    Returns
    -------
    Modes(vecs, vecslmbdainv, nmodes, v) : a named tuple
        `vecs` is the matrix of eigenvectors of the translation operator.
        `vecslmbdainv` is the matrix of eigenvectors multiplied with their
        corresponding inverse eigenvalue and with the hermitian conjugate
        of the hopping matrix. `nmodes` is the number of propagating modes
        in either direction. `v` is the last matrix of the hopping matrix
        singular value decomposition (u, s, v), or None if `h_hop` is
        invertible.

    Notes
    -----
    Only the propagating modes and the modes decaying away from the system are
    returned: The first `nmodes` columns in `vecs` correspond to incoming modes
    (coming from the lead into the system), the following `nmodes` columns
    correspond to outgoing modes (going into the lead, away from the system),
    the remaining columns are evanescent modes, decaying away from the system.

    The propagating modes are sorted according to the longitudinal component of
    their k-vector, with incoming modes having k sorted in descending order,
    and outgoing modes having k sorted in ascending order.  In simple cases
    where bands do not cross, this ordering corresponds to "lowest modes
    first". In general, however, it is necessary to examine the band structure
    -- something this function is not doing by design.

    If `h_hop` is invertible, the full transverse wave functions are returned.
    If it is singular, the projections (s u^dagger psi, v^dagger psi lambda^-1)
    are returned.

    In order for the linear system to be well-defined, instead of the
    evanescent modes, an orthogonal basis in the space of evanescent modes is
    returned.

    Propagating modes with the same lambda are orthogonalized. All the
    propagating modes are normalized by current.

    This function uses the most stable and efficient algorithm for calculating
    the mode decomposition. Its details will be published elsewhere.
    """
    m = h_hop.shape[1]

    if (h_onslice.shape[0] != h_onslice.shape[1] or
        h_onslice.shape[0] != h_hop.shape[0]):
        raise ValueError("Incompatible matrix sizes for h_onslice and h_hop.")

    # Note: np.any(h_hop) returns (at least from numpy 1.6.1 - 1.8-devel)
    #       False if h_hop is purely imaginary
    if not (np.any(h_hop.real) or np.any(h_hop.imag)):
        n = h_hop.shape[0]
        v = np.empty((0, m))
        return Modes(np.empty((0, 0)), np.empty((0, 0)), 0, v)

    # Defer most of the calculation to helper routines.
    matrices, v, extract, project = setup_linsys(h_onslice, h_hop, tol)
    ev, evanselect, propselect, vec_gen, ord_schur =\
         unified_eigenproblem(*(matrices + (tol,)))

    if v is not None:
        n = v.shape[1]
    else:
        n = h_onslice.shape[0]

    nprop = np.sum(propselect)
    nevan = n - nprop // 2
    evanselect_bool = np.zeros((2*n), dtype='bool')
    evanselect_bool[evanselect] = True
    evan_vecs = ord_schur(evanselect)[:, :nevan]

    # Compute the propagating eigenvectors.
    prop_vecs = vec_gen(propselect)
    # Compute their velocity, and, if necessary, rotate them
    prop_vecs, vel, rprop = \
            make_proper_modes(ev[propselect], prop_vecs, extract, project, tol)

    # Normalize propagating eigenvectors by velocities.
    prop_vecs /= np.sqrt(abs(vel))

    # Fix the phase factor - make maximum of the transverse wave function real
    # TODO (Anton): Take care of multiple maxima when normalizing.
    maxnode = prop_vecs[n + np.argmax(abs(prop_vecs[n:, :]), axis=0),
                        np.arange(prop_vecs.shape[1])]
    maxnode /= abs(maxnode)
    prop_vecs /= maxnode

    lprop = np.logical_not(rprop)
    nmodes = np.sum(rprop)

    # Sort modes according to their k-vector (1/lambda = e^{-ik}):
    # - right-going modes: sort that k is in ascending order
    # - left-going modes: sort that k is in descending order
    # (note that k can be positive or negative). With this convention,
    # the modes of a simple square lattice strip are ordered as
    # expected (lowest subband first, etc.)

    prop_ev = ev[propselect]
    rsort = np.argsort((-1j * np.log(prop_ev[rprop])).real)
    lsort = np.argsort((1j * np.log(prop_ev[lprop])).real)

    # The following is necessary due to how numpy deals with indexing of empty.
    # arrays.
    if nmodes == 0:
        lprop = rprop = rsort = lsort = slice(None)

    vecs = np.c_[prop_vecs[n:, lprop][:, lsort],
                 prop_vecs[n:, rprop][:, rsort],
                 evan_vecs[n:]]
    vecslmbdainv = np.c_[prop_vecs[:n, lprop][:, lsort],
                         prop_vecs[:n, rprop][:, rsort],
                         evan_vecs[:n]]

    return Modes(vecs, vecslmbdainv, nmodes, v)


def selfenergy(lead_modes, tol=1e6):
    """
    Compute the self-energy generated by a lead.

    The lead is described either by the unit-cell Hamiltonian h_onslice and the
    hopping matrix h_hop, or by its modes.

    Parameters
    ----------
    lead_modes : Modes(vecs, vecslmbdainv, nmodes, v) a named tuple
        The modes in the lead, as calculated by
        `kwant.physics.modes`.
    tol : double
        Numerical tolerance used in several places.

    Returns
    -------
    Sigma : numpy array, real or complex, shape (M,M)
        The computed self-energy. Note that even if `h_onslice` and `h_hop`
        are both real, `Sigma` will typically be complex. (More precisely, if
        there is a propagating mode, `Sigma` will definitely be complex.)

    Notes
    -----
    For simplicity this function relies on the calculation of modes as input.
    This may cause a small slowdown, and can be improved if necessary.
    """
    vecs, vecslmbdainv, nmodes, v = lead_modes
    vecs = vecs[:, nmodes:]
    vecslmbdainv = vecslmbdainv[:, nmodes:]
    if v is not None:
        return dot(v, dot(vecs, la.solve(vecslmbdainv, v.T.conj())))
    else:
        return la.solve(vecslmbdainv.T, vecs.T).T


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

    # p labels transversal modes.  i and j label the sites of a slice.

    # Precalculate the transverse wave function.
    psi_p_i = np.empty((width, width))
    factor = pi / (width + 1)
    prefactor = sqrt(2 / (width + 1))
    for p in xrange(width):
        for i in xrange(width):
            psi_p_i[p, i] = prefactor * sin(factor * (p + 1) * (i + 1))

    # Precalculate the integrals of the longitudinal wave functions.
    def f(q):
        if abs(q) <= 2:
            return q/2 - 1j * sqrt(1 - (q / 2) ** 2)
        else:
            return q/2 - copysign(sqrt((q / 2) ** 2 - 1), q)
    f_p = np.empty((width,), dtype=complex)
    for p in xrange(width):
        e = 2 * hopping * (1 - cos(factor * (p + 1)))
        q = (fermi_energy - e) / hopping - 2
        f_p[p] = f(q)

    # Put everything together into the self energy and return it.
    result = np.empty((width, width), dtype=complex)
    for i in xrange(width):
        for j in xrange(width):
            result[i, j] = hopping * \
                (psi_p_i[:, i] * psi_p_i[:, j].conj() * f_p).sum()
    return result
