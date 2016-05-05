# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.


import numpy as np
from itertools import product
from numpy.testing import assert_almost_equal
from kwant.physics import leads
import kwant

modes_se = leads.selfenergy

def h_cell_s_func(t, w, e):
    h = (4 * t - e) * np.identity(w)
    h.flat[1 :: w + 1] = -t
    h.flat[w :: w + 1] = -t
    return h


def test_analytic_numeric():
    w = 5                       # width
    t = 0.78                    # hopping element
    e = 1.3                     # Fermi energy

    assert_almost_equal(leads.square_selfenergy(w, t, e),
                        modes_se(h_cell_s_func(t, w, e), -t * np.identity(w)))


def test_regular_fully_degenerate():
    """Selfenergy with an invertible hopping matrix, and degenerate bands."""

    w = 6                       # width
    t = 0.5                     # hopping element
    e = 1.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_cell_s = h_cell_s_func(t, w, e)

    h_hop = np.zeros((2*w, 2*w))
    h_hop[:w, :w] = h_hop_s
    h_hop[w:, w:] = h_hop_s

    h_cell = np.zeros((2*w, 2*w))
    h_cell[:w, :w] = h_cell_s
    h_cell[w:, w:] = h_cell_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[:w, :w] = leads.square_selfenergy(w, t, e)
    g[w:, w:] = leads.square_selfenergy(w, t, e)

    assert_almost_equal(g, modes_se(h_cell, h_hop))


def test_regular_degenerate_with_crossing():
    """This is a testcase with invertible hopping matrices,
    and degenerate k-values with a crossing such that one
    mode has a positive velocity, and one a negative velocity

    For this case the fall-back technique must be used.
    """

    w = 4                       # width
    t = 0.5                     # hopping element
    e = 1.8                     # Fermi energy

    global h_hop
    h_hop_s = -t * np.identity(w)
    h_cell_s = h_cell_s_func(t, w, e)

    hop = np.zeros((2*w, 2*w))
    hop[:w, :w] = h_hop_s
    hop[w:, w:] = -h_hop_s

    h_cell = np.zeros((2*w, 2*w))
    h_cell[:w, :w] = h_cell_s
    h_cell[w:, w:] = -h_cell_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[:w, :w] = leads.square_selfenergy(w, t, e)
    g[w:, w:] = -np.conj(leads.square_selfenergy(w, t, e))

    assert_almost_equal(g, modes_se(h_cell, hop))


def test_singular():
    """This testcase features a rectangular (and hence singular)
     hopping matrix without degeneracies.

    This case can be treated with the Schur technique."""

    w = 5                       # width
    t = .5                     # hopping element
    e = 0.4                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_cell_s = h_cell_s_func(t, w, e)

    h_hop = np.zeros((2*w, w))
    h_hop[w:, :w] = h_hop_s

    h_cell = np.zeros((2*w, 2*w))
    h_cell[:w, :w] = h_cell_s
    h_cell[:w, w:] = h_hop_s
    h_cell[w:, :w] = h_hop_s
    h_cell[w:, w:] = h_cell_s
    g = leads.square_selfenergy(w, t, e)

    assert_almost_equal(g, modes_se(h_cell, h_hop))


def test_singular_but_square():
    """This testcase features a singular, square hopping matrices
    without degeneracies.

    This case can be treated with the Schur technique."""

    w = 5                       # width
    t = 0.9                     # hopping element
    e = 2.38                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_cell_s = h_cell_s_func(t, w, e)

    h_hop = np.zeros((2*w, 2*w))
    h_hop[w:, :w] = h_hop_s

    h_cell = np.zeros((2*w, 2*w))
    h_cell[:w, :w] = h_cell_s
    h_cell[:w, w:] = h_hop_s
    h_cell[w:, :w] = h_hop_s
    h_cell[w:, w:] = h_cell_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[:w, :w] = leads.square_selfenergy(w, t, e)
    assert_almost_equal(g, modes_se(h_cell, h_hop))


def test_singular_fully_degenerate():
    """This testcase features a rectangular (and hence singular)
     hopping matrix with complete degeneracy.

    This case can still be treated with the Schur technique."""

    w = 5                       # width
    t = 1.5                     # hopping element
    e = 3.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_cell_s = h_cell_s_func(t, w, e)

    h_hop = np.zeros((4*w, 2*w))
    h_hop[2*w:3*w, :w] = h_hop_s
    h_hop[3*w:4*w, w:2*w] = h_hop_s

    h_cell = np.zeros((4*w, 4*w))
    h_cell[:w, :w] = h_cell_s
    h_cell[:w, 2*w:3*w] = h_hop_s
    h_cell[w:2*w, w:2*w] = h_cell_s
    h_cell[w:2*w, 3*w:4*w] = h_hop_s
    h_cell[2*w:3*w, :w] = h_hop_s
    h_cell[2*w:3*w, 2*w:3*w] = h_cell_s
    h_cell[3*w:4*w, w:2*w] = h_hop_s
    h_cell[3*w:4*w, 3*w:4*w] = h_cell_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[:w, :w] = leads.square_selfenergy(w, t, e)
    g[w:, w:] = leads.square_selfenergy(w, t, e)

    assert_almost_equal(g, modes_se(h_cell, h_hop))


def test_singular_degenerate_with_crossing():
    """This testcase features a rectangular (and hence singular)
     hopping matrix with degeneracy k-values including a crossing
     with velocities of opposite sign.

    This case must be treated with the fall-back technique."""

    w = 5                       # width
    t = 20.5                     # hopping element
    e = 3.3                     # Fermi energy

    h_hop_s = -t * np.identity(w)
    h_cell_s = h_cell_s_func(t, w, e)

    h_hop = np.zeros((4*w, 2*w))
    h_hop[2*w:3*w, :w] = h_hop_s
    h_hop[3*w:4*w, w:2*w] = -h_hop_s

    h_cell = np.zeros((4*w, 4*w))
    h_cell[:w, :w] = h_cell_s
    h_cell[:w, 2*w:3*w] = h_hop_s
    h_cell[w:2*w, w:2*w] = -h_cell_s
    h_cell[w:2*w, 3*w:4*w] = -h_hop_s
    h_cell[2*w:3*w, :w] = h_hop_s
    h_cell[2*w:3*w, 2*w:3*w] = h_cell_s
    h_cell[3*w:4*w, w:2*w] = -h_hop_s
    h_cell[3*w:4*w, 3*w:4*w] = -h_cell_s

    g = np.zeros((2*w, 2*w), dtype=complex)
    g[:w, :w] = leads.square_selfenergy(w, t, e)
    g[w:, w:] = -np.conj(leads.square_selfenergy(w, t, e))

    assert_almost_equal(g, modes_se(h_cell, h_hop))


def test_singular_h_and_t():
    h = 0.1 * np.identity(6)
    t = np.eye(6, 6, 4)
    sigma = modes_se(h, t)
    sigma_should_be = np.zeros((6,6))
    sigma_should_be[4, 4] = sigma_should_be[5, 5] = -10
    assert_almost_equal(sigma, sigma_should_be)


def test_modes():
    h, t = .3, .7
    k = np.arccos(-h / (2 * t))
    v = 2 * t * np.sin(k)
    prop, stab = leads.modes(np.array([[h]]), np.array([[t]]))
    assert stab.nmodes == 1
    assert stab.sqrt_hop is None
    np.testing.assert_almost_equal(prop.velocities, [-v, v])
    np.testing.assert_almost_equal(prop.momenta, [k, -k])
    # Test for normalization by current.
    np.testing.assert_almost_equal(
        2 * (stab.vecs[0] * stab.vecslmbdainv[0].conj()).imag, [1, -1])


def test_modes_bearded_ribbon():
    # Check if bearded graphene ribbons work.
    lat = kwant.lattice.honeycomb()
    syst = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    syst[lat.shape((lambda pos: -20 < pos[1] < 20),
                  (0, 0))] = 0.3
    syst[lat.neighbors()] = -1
    syst = syst.finalized()
    h, t = syst.cell_hamiltonian(), syst.inter_cell_hopping()
    # The number of expected modes is calculated by plotting the dispersion.
    assert leads.modes(h, t)[1].nmodes == 8


def test_algorithm_equivalence():
    np.random.seed(400)
    n = 12
    h = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    h += h.T.conj()
    t = np.random.randn(n, n) + 1j * np.random.randn(n, n)
    u, s, vh = np.linalg.svd(t)
    u, v = u * np.sqrt(s), vh.T.conj() * np.sqrt(s)
    prop_vecs = []
    evan_vecs = []
    algos = [None] + list(product(*(2 * [(True, False)])))
    for algo in algos:
        result = leads.modes(h, t, stabilization=algo)[1]

        vecs, vecslmbdainv = result.vecs, result.vecslmbdainv

        # Bring the calculated vectors to real space
        if algo is not None:
            vecs = np.dot(v, vecs)
            np.testing.assert_almost_equal(result.sqrt_hop, v)
        else:
            vecslmbdainv = np.dot(v.T.conj(), vecslmbdainv)
        full_vecs = np.r_[vecslmbdainv, vecs]

        prop_vecs.append(full_vecs[:, : 2 * result.nmodes])
        evan_vecs.append(full_vecs[:, 2 * result.nmodes :])

    msg = 'Stabilization {0} failed.'
    for vecs, algo in zip(prop_vecs, algos):
        # Propagating modes should have identical ordering, and only vary
        # By a phase
        np.testing.assert_allclose(np.abs(np.sum(vecs/prop_vecs[0],
                                                 axis=0)), vecs.shape[0],
                                   err_msg=msg.format(algo))

    for vecs, algo in zip(evan_vecs, algos):
        # Evanescent modes must span the same linear space.
        assert (np.linalg.matrix_rank(np.c_[vecs, evan_vecs[0]], tol=1e-12) ==
                vecs.shape[1]), msg.format(algo)


def test_for_all_evs_equal():
    """Test an 'ideal lead' which has all eigenvalues e^ik equal."""

    onsite = np.array([[0., 1.], [1., 0.]], dtype=complex)
    hopping = np.array([[0.0], [-1.0]], dtype=complex)

    modes = leads.modes(onsite, hopping)[1]

    assert modes.vecs.shape == (1, 2)
    assert modes.vecslmbdainv.shape == (1, 2)
    assert modes.nmodes == 1


def test_dtype_linsys():
    """Test that setup_linsys stays in real arithmetics when possible."""
    h_cell = np.array([[2.0, -1.0], [-1.0, 2.0]], dtype=np.float64)
    h_hop = np.array([[0.0],[-1.0]], dtype=np.float64)

    lsyst = kwant.physics.leads.setup_linsys(h_cell - 0.3*np.eye(2),
                                            h_hop)
    assert lsyst.eigenproblem[0].dtype == np.float64

    lsyst = kwant.physics.leads.setup_linsys(h_cell.astype(np.complex128)
                                            - 0.3*np.eye(2),
                                            h_hop.astype(np.complex128))
    assert lsyst.eigenproblem[0].dtype == np.float64

    # energy=1 is an eigenstate of the isolated cell Hamiltonian,
    # i.e. a complex self-energy stabilization is necessary
    lsyst = kwant.physics.leads.setup_linsys(h_cell - 1*np.eye(2),
                                            h_hop)
    assert lsyst.eigenproblem[0].dtype == np.complex128

    # with complex input, output must be complex, too
    h_hop = np.array([[0.0],[-1.0 + 0.1j]], dtype=np.complex128)
    lsyst = kwant.physics.leads.setup_linsys(h_cell - 0.3*np.eye(2),
                                            h_hop)
    assert lsyst.eigenproblem[0].dtype == np.complex128


def test_zero_hopping():
    h_cell = np.identity(2)
    h_hop = np.zeros((2, 1))
    expected = (leads.PropagatingModes(np.zeros((2, 0)), np.zeros((0,)),
                                       np.zeros((0,))),
                leads.StabilizedModes(np.zeros((0, 0)), np.zeros((0, 0)), 0,
                                      np.zeros((1, 0))))
    actual = leads.modes(h_cell, h_hop)
    assert all(np.alltrue(getattr(actual[1], attr) ==
                          getattr(expected[1], attr)) for attr
                   in ('vecs', 'vecslmbdainv', 'nmodes', 'sqrt_hop'))
    assert all(np.alltrue(getattr(actual[0], attr) ==
                          getattr(expected[0], attr)) for attr
                   in ('wave_functions', 'velocities', 'momenta'))
