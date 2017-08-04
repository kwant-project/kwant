# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['gaussian', 'circular']

import numpy as np
from ._common import ensure_rng

qr = np.linalg.qr

sym_list = 'A', 'AI', 'AII', 'AIII', 'BDI', 'CII', 'D', 'DIII', 'C', 'CI'


def t(sym):
    """Return the value of time-reversal symmetry squared (1, 0, or -1)"""
    if sym not in sym_list:
        raise ValueError('Non-existent symmetry class.')
    if sym in ('CI', 'AI', 'BDI'):
        return 1
    elif sym in ('CII', 'AII', 'DIII'):
        return -1
    else:
        return 0


def p(sym):
    """Return the value of particle-hole symmetry squared (1, 0, or -1)"""
    if sym not in sym_list:
        raise ValueError('Non-existent symmetry class.')
    if sym in ('D', 'DIII', 'BDI'):
        return 1
    elif sym in ('C', 'CI', 'CII'):
        return -1
    else:
        return 0


def c(sym):
    """Return 1 if the system has chiral symmetry, and 0 otherwise."""
    if (t(sym) and p(sym)) or sym == 'AIII':
        return 1
    else:
        return 0


h_t_matrix = {'AI': [[1]], 'CI': [[0, 1], [1, 0]], 'BDI': [[1, 0], [0, -1]],
              'AII': [[0, 1j], [-1j, 0]],
              'CII': [[0, 0, 1j, 0], [0, 0, 0, 1j],
                      [-1j, 0, 0, 0], [0, -1j, 0, 0]],
              'DIII': [[0, 1j], [-1j, 0]]}
h_p_matrix = {'C': [[0, 1j], [-1j, 0]], 'CI': [[0, 1j], [-1j, 0]],
              'CII': [[0, 0, 1j, 0], [0, 0, 0, -1j],
                      [-1j, 0, 0, 0], [0, 1j, 0, 0]],
              'D': [[1]], 'DIII': [[0, 1], [1, 0]], 'BDI': [[1]]}


def gaussian(n, sym='A', v=1., rng=None):
    """Make a n * n random Gaussian Hamiltonian.

    Parameters
    ----------
    n : int
        Size of the Hamiltonian. It should be even for all the classes
        except A, D, and AI, and in class CII it should be a multiple of 4.
    sym : one of 'A', 'AI', 'AII', 'AIII', 'BDI', 'CII', 'D', 'DIII', 'C', 'CI'
         Altland-Zirnbauer symmetry class of the Hamiltonian.
    v : float
        Variance every degree of freedom of the Hamiltonian. The probaility
        distribution of the Hamiltonian is `P(H) = exp(-Tr(H^2) / 2 v^2)`.
    rng: int or rng (optional)
        Seed or random number generator. If no 'rng' is provided the random
        number generator from numpy will be used.

    Returns
    -------
    h : numpy.ndarray
        A numpy array drawn from a corresponding Gaussian ensemble.

    Notes
    -----
    The representations of symmetry operators are chosen according to
    Phys. Rev. B 85, 165409.

    Matrix indices are grouped first according to orbital number,
    then sigma-index, then tau-index.

    Chiral (sublattice) symmetry C always reads:
        H = -tau_z H tau_z.
    Time reversal symmetry T reads:
        AI: H = H^*.
        BDI: H = tau_z H^* tau_z.
        CI: H = tau_x H^* tau_x.
        AII, CII: H = sigma_y H^* sigma_y.
        DIII: H = tau_y H^* tau_y.
    Particle-hole symmetry reads:
        C, CI: H = -tau_y H^* tau_y.
        CII: H = -tau_z sigma_y H^* tau_z sigma_y.
        D, BDI: H = -H^*.
        DIII: H = -tau_x H^* tau_x.

    This implementation should be sufficiently efficient for large matrices,
    since it avoids any matrix multiplication.
    """
    if sym not in sym_list:
        raise ValueError('Unknown symmetry type.')
    if (c(sym) or t(sym) == -1 or p(sym) == -1):
        if n % 2:
            raise ValueError('Matrix dimension should be even in'
                             ' chosen symmetry class.')
        else:
            tau_z = np.array((n // 2) * [1, -1])
            idx_x = np.arange(n) + tau_z

    if sym == 'CII' and n % 4:
        raise ValueError('Matrix dimension should be a multiple of 4 in'
                         'symmetry class CII.')
    factor = v / np.sqrt(2)

    # define random number generator
    rng = ensure_rng(rng)
    randn = rng.randn

    # Generate a Gaussian matrix of appropriate dtype.
    if sym == 'AI':
        h = randn(n, n)
    elif sym in ('D', 'BDI'):
        h = 1j * randn(n, n)
    else:
        h = randn(n, n) + 1j * randn(n, n)

    # Ensure Hermiticity.
    # TODO: write this as h += h.T.conj() once we rely on numpy >= 1.13.0
    h = h + h.T.conj()

    # Ensure Chiral symmetry.
    if c(sym):
        h -= tau_z.reshape(-1, 1) * h * tau_z
        factor *= 0.5

    # Ensure the necessary anti-unitary symmetry.
    if sym in ('AII', 'DIII'):
        h += tau_z.reshape(-1, 1) * h[idx_x][:, idx_x].conj() * tau_z
        factor /= np.sqrt(2)
    elif sym in ('C', 'CI'):
        h -= tau_z.reshape(-1, 1) * h[idx_x][:, idx_x].conj() * tau_z
        factor /= np.sqrt(2)
    elif sym == 'CII':
        sigma_z = np.array((n // 4) * [1, 1, -1, -1])
        idx_sigma_x = np.arange(n) + 2 * sigma_z
        h += (sigma_z.reshape(-1, 1) * h[idx_sigma_x][:, idx_sigma_x].conj() *
              sigma_z)
        factor /= np.sqrt(2)

    h *= factor

    return h


def circular(n, sym='A', charge=None, rng=None):
    """Make a n * n matrix belonging to a symmetric circular ensemble.

    Parameters
    ----------
    n : int
        Size of the matrix. It should be even for the classes C, CI, CII,
        AII, DIII (either T^2 = -1 or P^2 = -1).
    sym : one of 'A', 'AI', 'AII', 'AIII', 'BDI', 'CII', 'D', 'DIII', 'C', 'CI'
         Altland-Zirnbauer symmetry class of the matrix.
    charge : int or None
        Topological invariant of the matrix. Should be one of 1, -1 in symmetry
        classes D and DIII, should be from 0 to n in classes AIII and BDI,
        and should be from 0 to n / 2 in class CII. If charge is None,
        it is drawn from a binomial distribution with p = 1 / 2.
    rng: int or rng (optional)
        Seed or random number generator. If no 'rng' is passed, the random
        number generator provided by numpy will be used.

    Returns
    -------
    s : numpy.ndarray
        A numpy array drawn from a corresponding circular ensemble.

    Notes
    -----
    The representations of symmetry operators are chosen according to
    Phys. Rev. B 85, 165409, except class D.

    Matrix indices are grouped first according to channel number,
    then sigma-index.

    Chiral (sublattice) symmetry C always reads:
        s = s^+.
    Time reversal symmetry T reads:
        AI, BDI: r = r^T.
        CI: r = -sigma_y r^T sigma_y.
        AII, DIII: r = -r^T.
        CII: r = sigma_y r^T sigma_y.
    Particle-hole symmetry reads:
        CI: r = -sigma_y r^* sigma_y
        C, CII: r = sigma_y r^* sigma_y
        D, BDI: r = r^*.
        DIII: -r = r^*.

    This function uses QR decomposition to probe symmetric compact groups,
    as detailed in arXiv:math-ph/0609050. For a reason as yet unknown, scipy
    implementation of QR decomposition also works for symplectic matrices.
    """
    rng = ensure_rng(rng)
    randn = rng.randn

    if sym not in sym_list:
        raise ValueError('Unknown symmetry type.')
    if (t(sym) == -1 or p(sym) == -1) and n % 2:
        raise ValueError('n must be even in chosen symmetry class.')

    # Prepare a real, complex or symplectic Gaussian matrix
    # Real case.
    if p(sym) == 1:
        h = randn(n, n)
    # Complex case.
    elif p(sym) == 0:
        h = randn(n, n) + 1j * randn(n, n)
    # Symplectic case.
    else:  # p(sym) == -1
        h = randn(n, n) + 1j * randn(n, n)
        tau_z = np.array((n // 2) * [1, -1])
        idx_x = np.arange(n) + tau_z
        h -= tau_z.reshape(-1, 1) * h[idx_x][:, idx_x].conj() * tau_z
        h *= 1j

    # Generate a random matrix with proper electron-hole symmetry.
    # This matrix is a random element of groups O(N), U(N), or USp(N).
    s, h = qr(h)
    if p(sym) == -1 and not np.allclose(np.diag(h, 1)[::2], 0, atol=1e-8):
        raise RuntimeError('QR decomposition symmetry failure.')
    h = np.diag(h).copy()
    h /= np.abs(h)
    s *= h

    # Ensure proper topological invariant in classes D and DIII.
    if sym in ('D', 'DIII') and charge is not None:
        if charge not in (-1, 1):
            raise ValueError('Impossible value of topological invariant.')
        det = np.linalg.det(s)
        if sym == 'DIII':
            det *= (-1) ** (n // 2)
        if (charge > 0) != (det > 0):
            idx = np.arange(n)
            idx[-1] -= 1
            idx[-2] += 1
            s = s[idx]

    # Add the proper time-reversal symmetry:
    if sym in ('AI', 'CI'):
        s = np.dot(s.T, s)
        if sym == 'CI':
            tau_z = np.array((n // 2) * [1, -1])
            idx_x = np.arange(n) + tau_z
            s = 1j * tau_z * s[:, idx_x]
    elif sym == 'AII' or sym == 'DIII':
        tau_z = np.array((n // 2) * [1, -1])
        idx_x = np.arange(n) + tau_z
        s = 1j * np.dot(s.T * tau_z, s[idx_x])

    # Add the chiral symmetry:
    elif sym in ('AIII', 'BDI', 'CII'):
        if sym != 'CII':
            if charge is None:
                diag = 2 * rng.randint(2, size=(n,)) - 1
            elif (0 <= charge <= n) and int(charge) == charge:
                diag = np.array(charge * [-1] + (n - charge) * [1])
            else:
                raise ValueError('Impossible value of topological invariant.')
        else:
            if charge is None:
                diag = 2 * rng.randint(2, size=(n // 2,)) - 1
                diag = np.resize(diag, (2, n // 2)).T.flatten()
            elif (0 <= charge <= n // 2) and int(charge) == charge:
                charge *= 2
                diag = np.array(charge * [-1] + (n - charge) * [1])
            else:
                raise ValueError('Impossible value of topological invariant.')

        s = np.dot(diag * s.T.conj(), s)

    return s
