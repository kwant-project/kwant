from __future__ import division
import numpy as np
from numpy.testing import assert_almost_equal
import kwant.physics.selfenergy as se

def test_analytic_numeric():
    w = 5                       # width
    t = 0.5                     # hopping element
    v = 2                       # potential
    e = 3.3                     # Fermi energy

    h_hop = -t * np.identity(w)
    h_onslice = ((v + 4 * t - e)
                 * np.identity(w))
    h_onslice.flat[1 :: w + 1] = -t
    h_onslice.flat[w :: w + 1] = -t

    assert_almost_equal(se.square_self_energy(w, t, v, e),
                        se.self_energy(h_onslice, h_hop))

def test_regular_fully_degenerate():
    """This testcase features an invertible hopping matrix,
    and bands that are fully degenerate.

    This case can still be treated with the Schur technique."""

    w = 5                       # width
    t = 0.5                     # hopping element
    v = 2                       # potential
    e = 3.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_onslice_s = ((v + 4 * t - e)
                 * np.identity(w))
    h_onslice_s.flat[1 :: w + 1] = -t
    h_onslice_s.flat[w :: w + 1] = -t

    h_hop = np.zeros((2*w, 2*w))
    h_hop[0:w, 0:w] = h_hop_s
    h_hop[w:2*w, w:2*w] = h_hop_s

    h_onslice = np.zeros((2*w, 2*w))
    h_onslice[0:w, 0:w] = h_onslice_s
    h_onslice[w:2*w, w:2*w] = h_onslice_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[0:w, 0:w] = se.square_self_energy(w, t, v, e)
    g[w:2*w, w:2*w] = se.square_self_energy(w, t, v, e)

    assert_almost_equal(g,
                        se.self_energy(h_onslice, h_hop))

def test_regular_degenerate_with_crossing():
    """This is a testcase with invertible hopping matrices,
    and degenerate k-values with a crossing such that one
    mode has a positive velocity, and one a negative velocity

    For this case the fall-back technique must be used.
    """

    w = 5                       # width
    t = 0.5                     # hopping element
    v = 2                       # potential
    e = 3.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_onslice_s = ((v + 4 * t - e)
                 * np.identity(w))
    h_onslice_s.flat[1 :: w + 1] = -t
    h_onslice_s.flat[w :: w + 1] = -t

    h_hop = np.zeros((2*w, 2*w))
    h_hop[0:w, 0:w] = h_hop_s
    h_hop[w:2*w, w:2*w] = -h_hop_s

    h_onslice = np.zeros((2*w, 2*w))
    h_onslice[0:w, 0:w] = h_onslice_s
    h_onslice[w:2*w, w:2*w] = -h_onslice_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[0:w, 0:w] = se.square_self_energy(w, t, v, e)
    g[w:2*w, w:2*w] = -np.conj(se.square_self_energy(w, t, v, e))

    assert_almost_equal(g,
                        se.self_energy(h_onslice, h_hop))

def test_singular():
    """This testcase features a rectangular (and hence singular)
     hopping matrix without degeneracies.

    This case can be treated with the Schur technique."""

    w = 5                       # width
    t = 0.5                     # hopping element
    v = 2                       # potential
    e = 3.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_onslice_s = ((v + 4 * t - e)
                 * np.identity(w))
    h_onslice_s.flat[1 :: w + 1] = -t
    h_onslice_s.flat[w :: w + 1] = -t

    h_hop = np.zeros((2*w, w))
    h_hop[w:2*w, 0:w] = h_hop_s

    h_onslice = np.zeros((2*w, 2*w))
    h_onslice[0:w, 0:w] = h_onslice_s
    h_onslice[0:w, w:2*w] = h_hop_s
    h_onslice[w:2*w, 0:w] = h_hop_s
    h_onslice[w:2*w, w:2*w] = h_onslice_s

    assert_almost_equal(se.square_self_energy(w, t, v, e),
                        se.self_energy(h_onslice, h_hop))

def test_singular_but_square():
    """This testcase features a singular, square hopping matrices
    without degeneracies.

    This case can be treated with the Schur technique."""

    w = 5                       # width
    t = 0.5                     # hopping element
    v = 2                       # potential
    e = 3.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_onslice_s = ((v + 4 * t - e)
                 * np.identity(w))
    h_onslice_s.flat[1 :: w + 1] = -t
    h_onslice_s.flat[w :: w + 1] = -t

    h_hop = np.zeros((2*w, 2*w))
    h_hop[w:2*w, 0:w] = h_hop_s

    h_onslice = np.zeros((2*w, 2*w))
    h_onslice[0:w, 0:w] = h_onslice_s
    h_onslice[0:w, w:2*w] = h_hop_s
    h_onslice[w:2*w, 0:w] = h_hop_s
    h_onslice[w:2*w, w:2*w] = h_onslice_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[0:w, 0:w] = se.square_self_energy(w, t, v, e)

    assert_almost_equal(g,
                        se.self_energy(h_onslice, h_hop))

def test_singular_fully_degenerate():
    """This testcase features a rectangular (and hence singular)
     hopping matrix with complete degeneracy.

    This case can still be treated with the Schur technique."""

    w = 5                       # width
    t = 0.5                     # hopping element
    v = 2                       # potential
    e = 3.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_onslice_s = ((v + 4 * t - e)
                 * np.identity(w))
    h_onslice_s.flat[1 :: w + 1] = -t
    h_onslice_s.flat[w :: w + 1] = -t

    h_hop = np.zeros((4*w, 2*w))
    h_hop[2*w:3*w, 0:w] = h_hop_s
    h_hop[3*w:4*w, w:2*w] = h_hop_s

    h_onslice = np.zeros((4*w, 4*w))
    h_onslice[0:w, 0:w] = h_onslice_s
    h_onslice[0:w, 2*w:3*w] = h_hop_s
    h_onslice[w:2*w, w:2*w] = h_onslice_s
    h_onslice[w:2*w, 3*w:4*w] = h_hop_s
    h_onslice[2*w:3*w, 0:w] = h_hop_s
    h_onslice[2*w:3*w, 2*w:3*w] = h_onslice_s
    h_onslice[3*w:4*w, w:2*w] = h_hop_s
    h_onslice[3*w:4*w, 3*w:4*w] = h_onslice_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[0:w, 0:w] = se.square_self_energy(w, t, v, e)
    g[w:2*w, w:2*w] = se.square_self_energy(w, t, v, e)

    assert_almost_equal(g,
                        se.self_energy(h_onslice, h_hop))

def test_singular_degenerate_with_crossing():
    """This testcase features a rectangular (and hence singular)
     hopping matrix with degeneracy k-values including a crossing
     with velocities of opposite sign.

    This case must be treated with the fall-back technique."""

    w = 5                       # width
    t = 0.5                     # hopping element
    v = 2                       # potential
    e = 3.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_onslice_s = ((v + 4 * t - e)
                 * np.identity(w))
    h_onslice_s.flat[1 :: w + 1] = -t
    h_onslice_s.flat[w :: w + 1] = -t

    h_hop = np.zeros((4*w, 2*w))
    h_hop[2*w:3*w, 0:w] = h_hop_s
    h_hop[3*w:4*w, w:2*w] = -h_hop_s

    h_onslice = np.zeros((4*w, 4*w))
    h_onslice[0:w, 0:w] = h_onslice_s
    h_onslice[0:w, 2*w:3*w] = h_hop_s
    h_onslice[w:2*w, w:2*w] = -h_onslice_s
    h_onslice[w:2*w, 3*w:4*w] = -h_hop_s
    h_onslice[2*w:3*w, 0:w] = h_hop_s
    h_onslice[2*w:3*w, 2*w:3*w] = h_onslice_s
    h_onslice[3*w:4*w, w:2*w] = -h_hop_s
    h_onslice[3*w:4*w, 3*w:4*w] = -h_onslice_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[0:w, 0:w] = se.square_self_energy(w, t, v, e)
    g[w:2*w, w:2*w] = -np.conj(se.square_self_energy(w, t, v, e))

    assert_almost_equal(g,
                        se.self_energy(h_onslice, h_hop))

def test_modes():
    h, t = .3, .7
    vecs, vecslinv, nrpop, svd = se.modes(np.mat(h), np.mat(t))
    l = (np.sqrt(h**2 - 4 * t**2 + 0j) - h) / (2 * t)
    current = np.sqrt(4 * t**2 - h**2)
    assert nrpop == 1
    assert svd is None
    np.testing.assert_almost_equal(vecs, [2 * [1/np.sqrt(current)]])
    np.testing.assert_almost_equal(vecslinv,
                                   vecs * np.array([1/l, 1/l.conj()]))
