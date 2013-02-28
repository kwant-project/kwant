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
import numpy as np
import numpy.linalg as npl
import scipy.linalg as la
from .. import linalg as kla

dot = np.dot

__all__ = ['self_energy', 'modes', 'Modes']


def setup_linsys(h_onslice, h_hop, tol=1e6):
    """
    Make an eigenvalue problem for eigenvectors of translation operator.

    Parameters
    ----------
    h_onslice : NumPy array with shape (n, n)
        Hamiltonian of a single lead slice.
    h_hop : NumPy array with shape (n, m), m <= n
        Hopping Hamiltonian from the slice to the next one.

    Returns
    -------
    linsys : matrix or tuple
        if the hopping is nonsingular, a single matrix defining an eigenvalue
        problem is returned, othewise a tuple of two matrices defining a
        generalized eigenvalue problem together with additional information is
        returned.

    Notes
    -----
    The lead problem with degenerate hopping is rather complicated, and it is
    described in kwant/doc/other/lead_modes.pdf.
    """
    n = h_onslice.shape[0]
    m = h_hop.shape[1]

    # Inter-slice hopping is zero.  The current algorithm is not suited to
    # treat this extremely singular case.
    # Note: np.any(h_hop) returns (at least from NumPy 1.6.1 - 1.8-devel)
    #       False if h_hop is purely imaginary
    assert np.any(h_hop.real) or np.any(h_hop.imag)

    eps = np.finfo(np.common_type(h_onslice, h_hop)).eps

    # First check if the hopping matrix has eigenvalues close to 0.
    u, s, vh = la.svd(h_hop)

    assert m == vh.shape[1], "Corrupt output of svd."

    # Count the number of singular values close to zero.
    # (Close to zero is defined here as |x| < eps * tol * s[0] , where
    #  s[0] is the largest singular value.)
    n_nonsing = np.sum(s > eps * tol * s[0])

    if n_nonsing == n:
        # The hopping matrix is well-conditioned and can be safely inverted.
        sol = kla.lu_factor(h_hop)

        A = np.empty((2*n, 2*n), dtype=np.common_type(h_onslice, h_hop))

        A[0:n, 0:n] = kla.lu_solve(sol, -h_onslice)
        A[0:n, n:2*n] = kla.lu_solve(sol, -h_hop.T.conj())
        A[n:2*n, 0:n] = np.identity(n)
        A[n:2*n, n:2*n] = 0

        return A
    else:
        # The hopping matrix has eigenvalues close to 0 - those
        # need to be eliminated.

        # Recast the svd of h_hop = u s v^dagger such that
        # u, v are matrices with shape n x n_nonsing.
        u = u[:, :n_nonsing]
        s = s[:n_nonsing]
        # pad v with zeros if necessary
        v = np.zeros((n, n_nonsing), dtype=vh.dtype)
        v[:vh.shape[1], :] = vh[:n_nonsing, :].T.conj()

        # Eliminating the zero eigenvalues requires inverting the
        # on-site Hamiltonian, possibly including a self-energy-like term.
        # The self-energy-like term stabilizes the inversion, but the most
        # stable choice is inherently complex. This can be disadvantageous
        # if the Hamiltonian is real - as staying in real arithmetics can be
        # significantly faster.
        # The strategy here is to add a complex self-energy-like term
        # always if the original Hamiltonian is complex, and check for
        # invertibility first if it is real

        gamma = None

        if issubclass(np.common_type(h_onslice, h_hop), np.floating):

            # Check if stabilization is needed.
            h = h_onslice

            sol = kla.lu_factor(h)
            rcond = kla.rcond_from_lu(sol, npl.norm(h, 1))

            if rcond > eps * tol:
                gamma = 0

        if gamma is None:
            # Matrices are complex or need self-energy-like term  to be
            # stabilized.

            # Normalize such that the maximum entry in the
            # self-energy-like term has a value comparable to the
            # maximum entry in h_onslice.

            temp = dot(u, u.T.conj()) + dot(v, v.T.conj())

            max_h = np.amax(abs(h_onslice))
            max_temp = np.amax(abs(temp))

            gamma = max_h / max_temp * 1j

            h = h_onslice + gamma * temp

            sol = kla.lu_factor(h)
            rcond = kla.rcond_from_lu(sol, npl.norm(h, 1))

            # If the condition number of the stabilized h is
            # still bad, there is nothing we can do.
            if rcond < eps * tol:
                raise RuntimeError("Flat band encountered at the requested "
                                   "energy, result is badly defined.")

        # Function that can extract the full wave function psi from
        # the projected one (v^dagger psi lambda^-1, u^dagger psi).

        def extract_wf(psi, lmbdainv):
            return kla.lu_solve(sol,
                                gamma * dot(v, psi[: n_nonsing]) +
                                gamma * dot(u, psi[n_nonsing:] * lmbdainv) -
                                dot(u * s, psi[: n_nonsing] * lmbdainv) -
                                dot(v * s, psi[n_nonsing:]))

        # Project a full wave function back.

        def project_wf(psi, lmbdainv):
            return np.asarray(np.bmat([[dot(v.T.conj(), psi * lmbdainv)],
                                       [dot(u.T.conj(), psi)]]))

        # Setup the generalized eigenvalue problem.

        A = np.empty((2 * n_nonsing, 2 * n_nonsing), np.common_type(h, h_hop))
        B = np.empty((2 * n_nonsing, 2 * n_nonsing), np.common_type(h, h_hop))

        A[:n_nonsing, :n_nonsing] = -np.eye(n_nonsing)

        B[n_nonsing: 2 * n_nonsing,
          n_nonsing: 2 * n_nonsing] = np.eye(n_nonsing)

        temp = kla.lu_solve(sol, v)
        temp2 = dot(u.T.conj(), temp)
        A[n_nonsing : 2 * n_nonsing, :n_nonsing] = gamma * temp2
        A[n_nonsing : 2 * n_nonsing, n_nonsing: 2 * n_nonsing] = - temp2 * s
        temp2 = dot(v.T.conj(), temp)
        A[:n_nonsing, :n_nonsing] += gamma * temp2
        A[:n_nonsing, n_nonsing : 2 * n_nonsing] = - temp2 * s

        temp = kla.lu_solve(sol, u)
        temp2 = dot(u.T.conj(), temp)
        B[n_nonsing : 2 * n_nonsing, :n_nonsing] = temp2 * s
        B[n_nonsing : 2 * n_nonsing, n_nonsing : 2 * n_nonsing] -= \
            gamma * temp2
        temp2 = dot(v.T.conj(), temp)
        B[:n_nonsing, :n_nonsing] = temp2 * s
        B[:n_nonsing, n_nonsing : 2 * n_nonsing] = - gamma * temp2

        # Solving a generalized eigenproblem is about twice as expensive
        # as solving a regular eigenvalue problem.
        # Computing the LU factorization is negligible compared to both
        # (approximately 1/30th of a regular eigenvalue problem).
        # Because of this, it makes sense to try to reduce
        # the generalized eigenvalue problem to a regular one, provided
        # the matrix B can be safely inverted.

        lu_b = kla.lu_factor(B)
        rcond = kla.rcond_from_lu(lu_b, npl.norm(B, 1))

        # I put a more stringent condition here - errors can accumulate
        # from here to the eigenvalue calculation later.
        if rcond > eps * tol**2:
            return (kla.lu_solve(lu_b, A), (u, s, v[:m, :]),
                    (extract_wf, project_wf))
        else:
            return (A, B, (u, s, v[:m]), (extract_wf, project_wf))


def split_degenerate(evs, tol=1e6):
    """
    Find sets of approximately degenerate list elements on a unit circle.

    Given a list of eigenvalues on the unit circle, return a list containing
    tuples of indices of eigenvalues that are numerically degenerate. Two
    eigenvalues ev[i] and ev[j] are considered to be numerically degenerate if
    abs(ev[i] - ev[j]) < eps * tol, where eps is the machine precision.

    Example
    -------
    >>> split_degenerate(np.array([1,-1,1,1], dtype=complex))
    [(1,), (0, 2, 3)].
    """
    eps = np.finfo(evs.dtype).eps

    n = evs.size
    evlist = []

    # Figure out if there are degenerate eigenvalues.
    # For this, sort according to k, which is i*log(ev) (ev is exp(-ik)).
    k = np.log(evs).imag
    sortindx = np.argsort(k)
    evs_sorted = evs[sortindx]

    # Note that we sorted eigenvalues on the unit circle, cutting
    # the unit circle at -1. We thus must search for degeneracies also
    # across this cut

    start = 0
    while (start - 1 > -n and
           abs(evs_sorted[start - 1] - evs_sorted[start]) < eps * tol):
        start = start - 1

    stop = n + start

    while start < stop:
        deglist = [sortindx[start]]
        while (start + 1 < stop and
               abs(evs_sorted[start] - evs_sorted[start + 1]) < eps * tol):
            start += 1
            deglist.append(sortindx[start])

        evlist.append(tuple(deglist))
        start += 1

    return evlist


def make_proper_modes(lmbdainv, psi, h_hop, extract=None,
                      project=None, tol=1e6):
    """
    Determine the velocities and direction of the propagating eigenmodes.

    Special care is taken of the case of degenerate k-values, where the
    numerically computed modes are typically a superposition of the real
    modes. In this case, also the proper (orthogonal) modes are computed.
    """

    vel_eps = np.finfo(np.common_type(psi, h_hop)).eps * tol

    # h_hop is either the full hopping matrix, or the singular
    # values vector of the svd.
    if h_hop.ndim == 2:
        n = h_hop.shape[0]
        m = h_hop.shape[1]
    else:
        n = h_hop.size

    nmodes = psi.shape[1]

    if nmodes == 0:
        raise ValueError('Empty mode array.')

    # Array for the velocities.
    v = np.empty(nmodes, dtype=float)

    # Mark the right-going modes.
    rightselect = np.zeros(nmodes, dtype=bool)

    n_left = n_right = 0
    crossing = False

    indxclust = split_degenerate(lmbdainv)

    for indx in indxclust:
        if len(indx) > 1:
            # Several degenerate propagating modes. In this case, the computed
            # eigenvectors do not orthogonalize the velocity
            # operator, i.e. they do not have a proper velocity.

            indx = np.array(indx)

            # If there is a degenerate eigenvalue with several different
            # eigenvectors, the numerical routines return some arbitrary
            # overlap of the real, physical solutions. In order
            # to figure out the correct wave function, we need to
            # have the full, not the projected wave functions
            # (at least to our current knowledge).

            if extract is not None:
                full_psi = extract(psi[:, indx], lmbdainv[indx])
            else:
                full_psi = psi[:n, indx]

            # Finding the true modes is done in two steps:

            # 1. The true transversal modes should be orthogonal to
            # each other, as they share the same Bloch momentum (note
            # that transversal modes with different Bloch momenta k1
            # and k2 need not be orthogonal, the full modes are
            # orthogonal because of the longitudinal dependence
            # e^{i k1 x} and e^{i k2 x}).
            # The modes are therefore orthogonalized:

            # Note: Here's a workaround for the fact that the interface
            # to qr changed from SciPy 0.8.0 to 0.9.0
            try:
                full_psi = la.qr(full_psi, econ=True, mode='qr')[0]
            except TypeError:
                full_psi = la.qr(full_psi, mode='economic')[0]

            if project:
                psi[:, indx] = project(full_psi, lmbdainv[indx])
            else:
                psi[:n, indx] = full_psi * lmbdainv[indx]
                psi[n:2*n, indx] = full_psi

            # 2. Moving infinitesimally away from the degeneracy
            # point, the modes should diagonalize the velocity
            # operator (i.e. when they are non-degenerate any more)
            # The modes are therefore rotated properly such that they
            # diagonalize the velocity operator.
            # Note that step 2. does not give a unique result if there are
            # two modes with the same velocity, or if the modes stay
            # degenerate even for a range of Bloch momenta (and hence
            # must have the same velocity). However, this does not matter,
            # as we are happy with any superposition in this case.

            if h_hop.ndim == 2:
                vel_op = -1j * dot(psi[n:, indx].T.conj(),
                                     dot(h_hop, psi[:m, indx]))
            else:
                vel_op = -1j * dot(psi[n:, indx].T.conj() * h_hop,
                                     psi[:n, indx])

            vel_op = vel_op + vel_op.T.conj()

            vel_vals, rot = la.eigh(vel_op)

            # If the eigenvectors were purely real up to this stage,
            # will typically become complex after the rotation.
            if psi.dtype != np.common_type(psi, rot):
                psi = psi.astype(np.common_type(psi, rot))

            psi[:, indx] = dot(psi[:, indx], rot)

            v[indx] = vel_vals

            # For some of the self-energy methods it matters
            # whether the degeneracy is a crossing with velocities
            # of different sign
            if not ((vel_vals > 0).all() or (vel_vals < 0).all()):
                crossing = True

            for (vel, k) in zip(vel_vals, indx):
                if vel > vel_eps:
                    n_right += 1
                    rightselect[k] = True
                elif vel < -vel_eps:
                    n_left += 1
                else:
                    raise RuntimeError("Found a mode with zero or close to "
                                       "zero velocity.")
        else:
            # A single, unique propagating mode
            k = indx[0]

            if h_hop.ndim == 2:
                v[k] = 2 * dot(dot(psi[n:2*n, k:k+1].T.conj(), h_hop),
                               psi[:m, k:k+1]).imag
            else:
                v[k] = 2 * dot(psi[n:2*n, k:k+1].T.conj() * h_hop,
                               psi[0:n, k:k+1]).imag

            if v[k] > vel_eps:
                rightselect[k] = True
                n_right += 1
            elif v[k] < -vel_eps:
                n_left += 1
            else:
                raise RuntimeError("Found a mode with zero or close to "
                                   "zero velocity.")

    if n_left != n_right:
        raise RuntimeError("Numbers of left- and right-propagating "
                           "modes differ.")

    return psi, v, rightselect, crossing


def unified_eigenproblem(h_onslice, h_hop, tol):
    """A helper routine for general() and modes(), that wraps eigenproblems.

    This routine wraps the different types of eigenproblems that can arise
    in a unfied way.

    Returns
    -------
    ev : NumPy array
        an array of eigenvalues (can contain NaNs and Infs, but those
        are not accessed in `general()` and `modes()`) The number of
        eigenvalues is given by twice the number of nonzero singular values of
        `h_hop` (i.e. `2*h_onslice.shape[0]` if `h_hop` is invertible).
    evanselect : NumPy array
        index array of right-decaying modes.
    propselect : NumPy array
        index array of propagating modes (both left and right).
    vec_gen(select) : function
        a function that computes the eigenvectors chosen by the array select.
    ord_schur(select) : function
        a function that computes the unitary matrix (corresponding to the right
        eigenvector space) of the (general) Schur decomposition reordered such
        that the eigenvalues chosen by the array select are in the top left
        block.
    u, w, v :
        if the hopping is singular, the svd of the hopping matrix, otherwise
        they all the three values are None.
    extract, project : functions
        functions to extract the full wave function from the projected wave
        functions, and project it back. Both are equal to None if the hopping
        is invertible.
    """

    eps = np.finfo(np.common_type(h_onslice, h_hop)).eps

    linsys = setup_linsys(h_onslice, h_hop)

    if isinstance(linsys, tuple):
        # In the singular case, it depends on the details of the system
        # whether one needs to solve a regular or a generalized
        # eigenproblem.

        assert len(linsys) == 3 or len(linsys) == 4, \
            "Corrupt lead eigenproblem data."

        if len(linsys) == 3:
            t, z, ev = kla.schur(linsys[0])

            # Right-decaying modes.
            select = abs(ev) > 1 + eps * tol
            # Propagating modes.
            propselect = abs(abs(ev) - 1) < eps * tol

            u, w, v = linsys[1]
            extract, project = linsys[2]

            vec_gen = lambda x: kla.evecs_from_schur(t, z, select=x)
            ord_schur = lambda x: kla.order_schur(x, t, z, calc_ev=False)[1]

        else:
            s, t, z, alpha, beta = kla.gen_schur(linsys[0], linsys[1],
                                                 calc_q=False)

            # Right-decaying modes.
            select = abs(alpha) > (1 + eps * tol) * abs(beta)
            # Propagating modes.
            propselect = (abs(abs(alpha) - abs(beta)) <
                          eps * tol * abs(beta))

            warning_settings = np.seterr(divide='ignore', invalid='ignore')
            ev = alpha / beta
            np.seterr(**warning_settings)
            # Note: the division is OK here, as we later only access
            #       eigenvalues close to the unit circle

            u, w, v = linsys[2]
            extract, project = linsys[3]

            vec_gen = lambda x: kla.evecs_from_gen_schur(s, t, z=z, select=x)
            ord_schur = lambda x: kla.order_gen_schur(x, s, t,
                                                      z=z, calc_ev=False)[2]
    else:
        # Hopping matrix can be safely inverted -> regular eigenproblem can be
        # used. This also means, that the hopping matrix is n x n square.

        t, z, ev = kla.schur(linsys)

        # Right-decaying modes.
        select = abs(ev) > 1 + eps * tol
        # Propagating modes.
        propselect = abs(abs(ev) - 1) < eps * tol

        # Signal that we are in the regular case.
        u = v = w = None
        extract = project = None

        vec_gen = lambda x: kla.evecs_from_schur(t, z, select=x)
        ord_schur = lambda x: kla.order_schur(x, t, z, calc_ev=False)[1]

    return ev, select, propselect, vec_gen, ord_schur,\
        u, w, v, extract, project


def self_energy(h_onslice, h_hop, tol=1e6):
    """
    Compute the self-energy generated by a lead.

    The lead is described by the unit-cell
    Hamiltonian h_onslice and the hopping matrix h_hop.

    Parameters
    ----------
    h_onslice : NumPy array, real or complex, shape (N,N) The unit cell
        Hamiltonian of the lead slice.
    h_hop : NumPy array, real or complex, shape (N,M)
        the hopping matrix from a lead slice to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).

    Returns
    -------
    Sigma : NumPy array, real or complex, shape (M,M)
        The computed self-energy. Note that even if `h_onslice` and `h_hop`
        are both real, `Sigma` will typically be complex. (More precisely, if
        there is a propagating mode, `Sigma` will definitely be complex.)

    Notes
    -----
    This function uses the most stable and efficient algorithm for calculating
    self-energy, described in kwant/doc/other/lead_modes.pdf
    """

    m = h_hop.shape[1]

    # Note: np.any(h_hop) returns (at least from NumPy 1.6.1 - 1.8-devel)
    #       False if h_hop is purely imaginary
    if not (np.any(h_hop.real) or np.any(h_hop.imag)):
        return np.zeros((m, m))

    if (h_onslice.shape[0] != h_onslice.shape[1] or
        h_onslice.shape[0] != h_hop.shape[0]):
        raise ValueError("Incompatible matrix sizes for h_onslice and h_hop.")

    #defer most of the calculation to a helper routine (also used by modes)
    ev, select, propselect, vec_gen, ord_schur,\
        u, w, v, extract, project = unified_eigenproblem(h_onslice, h_hop, tol)

    if w is not None:
        n = w.size
        h_hop = w
    else:
        n = h_onslice.shape[0]

    # Compute the propagating eigenvectors, if they are present.
    nprop = np.sum(propselect)

    if nprop > 0:
        prop_vecs = vec_gen(propselect)

        prop_vecs, vel, rselect, crossing = \
            make_proper_modes(ev[propselect], prop_vecs, h_hop,
                              extract, project)
    else:
        # Without propagating modes, the Schur methods certainly work.
        crossing = False

    if crossing:
        # Schur decomposition method does not work in this case, we need to
        # compute all the eigenvectors.

        # We already have the propagating ones, now we just need the
        # evanescent ones in addition.

        if nprop > 0:
            vecs = np.empty((2*n, n),
                            dtype=np.common_type(ev, prop_vecs))
        else:
            vecs = np.empty((2*n, n), dtype=ev.dtype)
        # Note: rationale for the dtype: only if all the eigenvalues are real,
        #       (which can only happen if the original eigenproblem was
        #       real) and all propagating modes are real, the matrix of
        #       eigenvectors will be real, too.

        if nprop > 0:
            nmodes = np.sum(rselect)
            vecs[:, :nmodes] = prop_vecs[:, rselect]
        else:
            nmodes = 0

        vecs[:, nmodes:] = vec_gen(select)

        if v is not None:
            return dot(v * w, dot(vecs[n:], dot(npl.inv(vecs[:n]),
                                                v.T.conj())))
        else:
            return dot(h_hop.T.conj(), dot(vecs[n:], npl.inv(vecs[:n])))
    else:
        # Reorder all the right-going eigenmodes to the top left part of
        # the Schur decomposition.

        if nprop > 0:
            select[propselect] = rselect

        z = ord_schur(select)

        if v is not None:
            return dot(v * w, dot(z[n:, :n], dot(npl.inv(z[:n, :n]),
                                                   v.T.conj())))
        else:
            return dot(h_hop.T.conj(), dot(z[n:, :n], npl.inv(z[:n, :n])))

Modes = namedtuple('Modes', ['vecs', 'vecslmbdainv', 'nmodes', 'svd'])


def modes(h_onslice, h_hop, tol=1e6):
    """
    Compute the eigendecomposition of a translation operator of a lead.

    Parameters
    ----------
    h_onslice : NumPy array, real or complex, shape (N,N) The unit cell
        Hamiltonian of the lead slice.
    h_hop : NumPy array, real or complex, shape (N,M)
        the hopping matrix from a lead slice to the one on which self-energy
        has to be calculated (and any other hopping in the same direction).

    Returns
    -------
    (vecs, vecslmbdainv, nmodes, svd) : Modes
        `vecs` is the matrix of eigenvectors of the translation operator.
        `vecslmbdainv` is the matrix of eigenvectors multiplied with their
        corresponding inverse eigenvalue.  `nmodes` is the number of
        propagating modes in either direction.  `svd` is a tuple (u, s, v)
        holding the singular value decomposition of the hopping matrix, or a
        single None if `h_hop` is invertible.

    Notes
    -----
    Only propagating modes and modes decaying away from the system are
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
    If it is singular, the projections (u^dagger psi, v^dagger psi lambda^-1)
    are returned.

    In order for the linear system to be well-defined, instead of the
    evanescent modes, an orthogonal basis in the space of evanescent modes is
    returned.

    Propagating modes with the same lambda are orthogonalized. All the
    propagating modes are normalized by current.

    This function uses the most stable and efficient algorithm for calculating
    self-energy, described in kwant/doc/other/lead_modes.pdf
    """

    m = h_hop.shape[1]

    if (h_onslice.shape[0] != h_onslice.shape[1] or
        h_onslice.shape[0] != h_hop.shape[0]):
        raise ValueError("Incompatible matrix sizes for h_onslice and h_hop.")

    # Note: np.any(h_hop) returns (at least from NumPy 1.6.1 - 1.8-devel)
    #       False if h_hop is purely imaginary
    if not (np.any(h_hop.real) or np.any(h_hop.imag)):
        n = h_hop.shape[0]
        svd = (np.empty((n, 0)), np.empty((0, 0)), np.empty((0, m)))
        return Modes(np.empty((0, 0)), np.empty((0, 0)), 0, svd)

    # Defer most of the calculation to a helper routine.
    ev, evanselect, propselect, vec_gen, ord_schur, \
        u, s, v, extract, project = unified_eigenproblem(h_onslice, h_hop, tol)

    if s is not None:
        n = s.size
    else:
        n = h_onslice.shape[0]

    nprop = np.sum(propselect)
    nevan = n - nprop // 2
    evanselect_bool = np.zeros((2*n), dtype='bool')
    evanselect_bool[evanselect] = True
    evan_vecs = ord_schur(evanselect)[:, :nevan]

    if nprop > 0:
        # Compute the propagating eigenvectors.
        prop_vecs = vec_gen(propselect)

        # Compute their velocity, and, if necessary, rotate them
        if s is not None:
            prop_vecs, vel, rprop, crossing = \
                make_proper_modes(ev[propselect], prop_vecs, s,
                                  extract, project)
        else:
            prop_vecs, vel, rprop, crossing = \
                make_proper_modes(ev[propselect], prop_vecs, h_hop)

        # Normalize propagating eigenvectors by velocities.
        prop_vecs /= np.sqrt(abs(vel))

        # Fix phase factor - make maximum of transverse wave function real
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

        vecs = np.c_[prop_vecs[n:, lprop][:, lsort],
                     prop_vecs[n:, rprop][:, rsort],
                     evan_vecs[n:]]
        vecslmbdainv = np.c_[prop_vecs[:n, lprop][:, lsort],
                             prop_vecs[:n, rprop][:, rsort],
                             evan_vecs[:n]]

    else:
        vecs = evan_vecs[n:]
        vecslmbdainv = evan_vecs[:n]
        nmodes = 0

    svd = None if s is None else (u, s, v)
    return Modes(vecs, vecslmbdainv, nmodes, svd)


def square_self_energy(width, hopping, potential, fermi_energy):
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
        q = (fermi_energy - potential - e) / hopping - 2
        f_p[p] = f(q)

    # Put everything together into the self energy and return it.
    result = np.empty((width, width), dtype=complex)
    for i in xrange(width):
        for j in xrange(width):
            result[i, j] = hopping * \
                (psi_p_i[:, i] * psi_p_i[:, j].conj() * f_p).sum()
    return result
