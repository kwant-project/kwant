# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.


import numpy as np
from numpy.testing import assert_almost_equal
import scipy.linalg as la
from scipy import sparse
from kwant.physics import leads
from kwant._common import ensure_rng
import kwant

modes_se = leads.selfenergy


def current_conserving(stabilized, info=None):
    vecs, vecsl = stabilized.vecs, stabilized.vecslmbdainv
    n = stabilized.nmodes
    current = 1j * vecs.T.conj().dot(vecsl)
    current = current + current.T.conj()
    should_be = np.zeros_like(current)
    should_be[:2*n, :2*n] = np.diag(n * [1] + n * [-1])
    if not np.allclose(current, should_be):
        raise AssertionError(np.round(current, 4), n, info)


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
    # Now with conservation laws and symmetries.
    conserved = np.identity(2*w)
    projectors = [sparse.csr_matrix(i) for i in [conserved[:, :w],
                                                 conserved[:, w:]]]
    modes2 = leads.modes(h_cell, h_hop, projectors=projectors)
    current_conserving(modes2[1])
    assert_almost_equal(g, modes2[1].selfenergy())

    trs = sparse.identity(2*w)
    modes3 = leads.modes(h_cell, h_hop, projectors=projectors,
                         time_reversal=trs)
    current_conserving(modes3[1])
    assert_almost_equal(g, modes3[1].selfenergy())

    phs = np.eye(2*w, 2*w, w) + np.eye(2*w, 2*w, -w)

    modes4 = leads.modes(h_cell, h_hop, projectors=projectors,
                         time_reversal=trs, particle_hole=phs)
    current_conserving(modes4[1])
    assert_almost_equal(g, modes4[1].selfenergy())

def test_regular_degenerate_with_crossing():
    """This is a testcase with invertible hopping matrices,
    and degenerate k-values with a crossing such that one
    mode has a positive velocity, and one a negative velocity

    For this case the fall-back technique must be used.
    """

    w = 4                       # width
    t = 0.5                     # hopping element
    e = 1.8                     # Fermi energy

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
    current_conserving(stab)
    assert stab.nmodes == 1
    assert stab.sqrt_hop[0] == np.sqrt(np.linalg.norm(t))
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


def check_equivalence(h, t, n, sym='', particle_hole=None, chiral=None,
                      time_reversal=None):
    """Compare modes stabilization algorithms for a given Hamiltonian."""
    u, s, vh = la.svd(t)
    u, v = u * np.sqrt(s), vh.T.conj() * np.sqrt(s)
    prop_vecs = []
    evan_vecs = []
    algos = [None, (True, True), (True, False), (False, True), (False, False)]
    for algo in algos:
        result = leads.modes(h, t, stabilization=algo, chiral=chiral,
                             particle_hole=particle_hole,
                             time_reversal=time_reversal)[1]
        current_conserving(result, (sym, algo, n))
        vecs, vecslmbdainv = result.vecs, result.vecslmbdainv

        # Bring the calculated vectors to real space
        if algo is not None:
            vecs = np.dot(v, vecs)
            np.testing.assert_almost_equal(result.sqrt_hop, v)
        else:
            vecslmbdainv = (np.dot(v.T.conj(), vecslmbdainv) /
                            np.sqrt(np.linalg.norm(t)))
            vecs = vecs * np.sqrt(np.linalg.norm(t))
        full_vecs = np.r_[vecslmbdainv, vecs]

        prop_vecs.append(full_vecs[:, : 2 * result.nmodes])
        evan_vecs.append(full_vecs[:, 2 * result.nmodes :])

    msg = 'Stabilization {0} failed.in symmetry class {1}'
    for vecs, algo in zip(prop_vecs, algos):
        # Propagating modes should have identical ordering, and only vary
        # By a phase
        np.testing.assert_allclose(np.abs(np.sum(vecs/prop_vecs[0], axis=0)),
                                   vecs.shape[0],
                                   err_msg=msg.format(algo, sym))

    for vecs, algo in zip(evan_vecs, algos):
        # Evanescent modes must span the same linear space.
        mat = np.c_[vecs, evan_vecs[0]]
        # Scale largest singular value to 1 if the array is not empty
        mat = mat/np.linalg.norm(mat, ord=2)
        # As a tolerance, take the square root of machine precision times the
        # largest matrix dimension.
        tol = np.abs(np.sqrt(max(mat.shape)*np.finfo(mat.dtype).eps))
        assert (np.linalg.matrix_rank(mat, tol=tol) ==
                vecs.shape[1]), msg.format(algo)+' in symmetry class '+sym


def test_symm_algorithm_equivalence():
    """Test different stabilization methods in the computation of modes,
    in the presence and/or absence of the discrete symmetries."""
    rng = ensure_rng(400)
    n = 8
    for sym in kwant.rmt.sym_list:
        # Random onsite and hopping matrices in symmetry class
        h_cell = kwant.rmt.gaussian(n, sym, rng=rng)
        # Hopping is an offdiagonal block of a Hamiltonian. We rescale it
        # to ensure that there are modes at the Fermi level.
        h_hop = 10 * kwant.rmt.gaussian(2*n, sym, rng=rng)[:n, n:]

        if kwant.rmt.p(sym):
            p_mat = np.array(kwant.rmt.h_p_matrix[sym])
            p_mat = np.kron(np.identity(n // len(p_mat)), p_mat)
        else:
            p_mat = None

        if kwant.rmt.t(sym):
            t_mat = np.array(kwant.rmt.h_t_matrix[sym])
            t_mat = np.kron(np.identity(n // len(t_mat)), t_mat)
        else:
            t_mat = None

        if kwant.rmt.c(sym):
            c_mat = np.kron(np.identity(n // 2), np.diag([1, -1]))
        else:
            c_mat = None

        check_equivalence(h_cell, h_hop, n, sym=sym, particle_hole=p_mat,
                          chiral=c_mat, time_reversal=t_mat)


def test_for_all_evs_equal():
    """Test an 'ideal lead' which has all eigenvalues e^ik equal."""

    onsite = np.array([[0., 1.], [1., 0.]], dtype=complex)
    hopping = np.array([[0.0], [-1.0]], dtype=complex)

    modes = leads.modes(onsite, hopping)[1]
    current_conserving(modes)
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


def make_clean_lead(W, E, t):
    syst = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    lat = kwant.lattice.square()
    syst[(lat(0, j) for j in range(W))] = E
    syst[lat.neighbors()] = -t
    return syst.finalized()


def test_momenta():
    """Test whether the two systems have the same momenta,
    these should not change when the Hamiltonian is scaled."""
    momenta = [make_clean_lead(10, s, s).modes()[0].momenta for s in [1, 1e20]]
    assert_almost_equal(*momenta)


def check_PHS(TRIM, moms, velocities, wfs, pmat):
    """Check PHS of incident or outgoing modes at a TRIM momentum.

    Input are momenta, velocities and wave functions of incident or outgoing
    modes.
    """
    # Pick out TRIM momenta - in this test, all momenta are either 0 or -pi,
    # so the boundaries of the interval here are not important.
    TRIM_moms = (TRIM-0.1 < moms) * (moms < TRIM+0.1)
    assert_almost_equal(moms[TRIM_moms], TRIM)
    # At a given momentum, incident modes are sorted in ascending order by
    # velocity. Pick out modes with the same velocity.
    vels = velocities[TRIM_moms]
    inds = [ind+1 for ind, vel in enumerate(vels[:-1])
            if np.abs(vel-vels[ind+1])>1e-8]
    inds = [0] + inds + [len(vels)]
    inds = zip(inds[:-1], inds[1:])
    for ind_tuple in inds:
        vel_wfs = wfs[:, slice(*ind_tuple)]
        assert_almost_equal(vels[slice(*ind_tuple)], vels[slice(*ind_tuple)][0])
        assert_almost_equal(vel_wfs[:, 1::2], pmat.dot(vel_wfs[:, ::2].conj()),
                            err_msg='Particle-hole symmetry broken at a TRIM')

def test_PHS_TRIM_degenerate_ordering():
    """ Test PHS at a TRIM, both when it squares to 1 and -1.

    Take a Hamiltonian with 3 degenerate bands, the degeneracy of each is given
    in the tuple dims. The bands have different velocities. All bands intersect
    zero energy only at k = 0 and at the edge of the BZ, so all momenta are 0
    or -pi. We thus have multiple TRIM modes, both with the same and different
    velocities.

    If P^2 = 1, all TRIM modes are eigenmodes of P.
    If P^2 = -1, TRIM modes come in pairs of particle-hole partners, ordered by
    a predefined convention."""
    sy = np.array([[0,-1j],[1j,0]])
    sz = np.array([[1,0],[0,-1]])

    # P squares to 1.
    rng = ensure_rng(42)
    dims = (4, 8, 12)
    ts = (1.0, 1.7, 13.8)
    rand_hop = 1j*(0.1+rng.random_sample())
    hop = la.block_diag(*[t*rand_hop*np.eye(dim) for t, dim in zip(ts, dims)])

    pmat = np.eye(sum(dims))
    onsite = np.zeros(hop.shape, dtype=complex)
    prop, stab = leads.modes(onsite, hop, particle_hole=pmat)
    current_conserving(stab)
    assert np.all([np.any(momentum - np.array([0, -np.pi])) for momentum in
                   prop.momenta])
    assert np.all([np.allclose(wf, pmat.dot(wf.conj())) for wf in
                   prop.wave_functions.T])

    # P squares to -1
    dims = (1, 4, 10)
    ts = (1.0, 17.2, 13.4)

    hop_mat = np.kron(sz, 1j * (0.1 + rng.random_sample()) * np.eye(2))
    blocks = []
    for t, dim in zip(ts, dims):
        blocks += dim*[t*hop_mat]
    hop = la.block_diag(*blocks)
    pmat = np.kron(np.eye(sum(dims)), 1j*np.kron(sz, sy))

    assert_almost_equal(pmat.dot(pmat.conj()), -np.eye(pmat.shape[0]))
    # The Hamiltonian anticommutes with P
    assert_almost_equal(pmat.dot(hop.conj()).dot(np.linalg.inv(pmat)), -hop)

    onsite = np.zeros(hop.shape, dtype=complex)
    prop, stab = leads.modes(onsite, hop, particle_hole=pmat)
    current_conserving(stab)
    # By design, all momenta are either 0 or -pi.
    assert np.all([np.any(momentum - np.array([0, -np.pi])) for momentum in
                   prop.momenta])

    wfs = prop.wave_functions
    momenta = prop.momenta
    velocities = prop.velocities
    nmodes = stab.nmodes

    # By design, all modes are at a TRIM here. Each must thus have a
    # particle-hole partner at the same TRIM and with the same velocity.
    # Incident modes
    check_PHS(0, momenta[:nmodes], velocities[:nmodes], wfs[:, :nmodes], pmat)
    check_PHS(-np.pi, momenta[:nmodes], velocities[:nmodes], wfs[:, :nmodes],
              pmat)
    # Outgoing modes
    check_PHS(0, momenta[nmodes:], velocities[nmodes:], wfs[:, nmodes:], pmat)
    check_PHS(-np.pi, momenta[nmodes:], velocities[nmodes:], wfs[:, nmodes:],
              pmat)


def test_modes_symmetries():
    rng = ensure_rng(10)
    for n in (4, 8, 40, 60):
        for sym in kwant.rmt.sym_list:
            # Random onsite and hopping matrices in symmetry class
            h_cell = kwant.rmt.gaussian(n, sym, rng=rng)
            # Hopping is an offdiagonal block of a Hamiltonian. We rescale it
            # to ensure that there are modes at the Fermi level.
            h_hop = 10 * kwant.rmt.gaussian(2*n, sym, rng=rng)[:n, n:]

            if kwant.rmt.p(sym):
                p_mat = np.array(kwant.rmt.h_p_matrix[sym])
                p_mat = np.kron(np.identity(n // len(p_mat)), p_mat)
            else:
                p_mat = None

            if kwant.rmt.t(sym):
                t_mat = np.array(kwant.rmt.h_t_matrix[sym])
                t_mat = np.kron(np.identity(n // len(t_mat)), t_mat)
            else:
                t_mat = None

            if kwant.rmt.c(sym):
                c_mat = np.kron(np.identity(n // 2), np.diag([1, -1]))
            else:
                c_mat = None

            prop_modes, stab_modes = leads.modes(h_cell, h_hop,
                                                 particle_hole=p_mat,
                                                 time_reversal=t_mat,
                                                 chiral=c_mat)
            current_conserving(stab_modes)
            wave_functions = prop_modes.wave_functions
            momenta = prop_modes.momenta
            nmodes = stab_modes.nmodes

            if t_mat is not None:
                assert_almost_equal(wave_functions[:, nmodes:],
                        t_mat.dot(wave_functions[:, :nmodes].conj()),
                        err_msg='TRS broken in ' + sym)

            if c_mat is not None:
                assert_almost_equal(wave_functions[:, nmodes:],
                        c_mat.dot(wave_functions[:, :nmodes][:, ::-1]),
                        err_msg='SLS broken in ' + sym)

            if p_mat is not None:
                # If P^2 = -1, then P psi(-k) = -psi(k) for k>0, so one must
                # look at positive and negative momenta separately.  Test
                # positive momenta.
                first, last = momenta[:nmodes], momenta[nmodes:]
                in_positive_k = (np.pi > first) * (first > 0)
                out_positive_k = (np.pi > last) * (last > 0)

                wf_first = wave_functions[:, :nmodes]
                wf_last = wave_functions[:, nmodes:]
                assert_almost_equal(wf_first[:, in_positive_k[::-1]],
                        p_mat.dot((wf_first[:, in_positive_k][:, ::-1]).conj()),
                        err_msg='PHS broken in ' + sym)
                assert_almost_equal(wf_last[:, out_positive_k[::-1]],
                        p_mat.dot((wf_last[:, out_positive_k][:, ::-1]).conj()),
                        err_msg='PHS broken in ' + sym)

                # Test negative momenta. Need the sign of P^2 here.
                p_squared_sign = np.sign(p_mat.dot(p_mat.conj())[0, 0].real)
                in_neg_k = (-np.pi < first) * (first < 0)
                out_neg_k = (-np.pi < last) * (last < 0)

                assert_almost_equal(p_squared_sign*wf_first[:, in_neg_k[::-1]],
                        p_mat.dot((wf_first[:, in_neg_k][:, ::-1]).conj()),
                        err_msg='PHS broken in ' + sym)
                assert_almost_equal(p_squared_sign*wf_last[:, out_neg_k[::-1]],
                        p_mat.dot((wf_last[:, out_neg_k][:, ::-1]).conj()),
                        err_msg='PHS broken in ' + sym)


def test_chiral_symm():
    """Test modes for a single block Hamiltonian with chiral symmetry.

    To ensure that there are modes at the Fermi level, a conservation law is
    included, no projectors are used.
    """
    for n in (2, 8, 20):
        h_cell, h_hop = random_onsite_hop(n)
        c_mat = np.kron(np.identity(n // 2), np.diag([1, -1]))
        sx = np.array([[0, 1], [1, 0]])
        C = np.kron(sx, c_mat)
        H_cell = la.block_diag(h_cell, -c_mat.dot(h_cell).dot(c_mat.T.conj()))
        H_hop = 10*la.block_diag(h_hop, -c_mat.dot(h_hop).dot(c_mat.T.conj()))
        assert_almost_equal(C.dot(C.T.conj()), np.eye(2*n))
        assert_almost_equal(H_cell.dot(C) + C.dot(H_cell), 0)
        assert_almost_equal(H_hop.dot(C) + C.dot(H_hop), 0)
        prop_modes, stab_modes = kwant.physics.leads.modes(H_cell, H_hop,
                                                           chiral=C)
        current_conserving(stab_modes)
        wave_functions = prop_modes.wave_functions
        nmodes = stab_modes.nmodes
        assert_almost_equal(wave_functions[:, nmodes:],
                            C.dot(wave_functions[:, :nmodes][:, ::-1]))


def test_PHS_TRIM():
    """Test the function that makes particle-hole symmetric modes at a TRIM. """
    rng = ensure_rng(10)
    for n in (4, 8, 16, 60):
        for sym in kwant.rmt.sym_list:
            if kwant.rmt.p(sym):
                p_mat = np.array(kwant.rmt.h_p_matrix[sym])
                p_mat = np.kron(np.identity(n // len(p_mat)), p_mat)
                P_squared = 1 if np.allclose(p_mat.conj().dot(p_mat),
                                             np.eye(*p_mat.shape)) else -1
                if P_squared == 1:
                    for nmodes in (1, 3, n//4, n//2, n):
                        # Random matrix of 'modes.' Take part of a unitary
                        # matrix to ensure that the modes form a basis.
                        modes = kwant.rmt.circular(n, 'A', rng=rng)[:, :nmodes]
                        # Ensure modes are particle-hole symmetric and
                        # orthonormal
                        modes = modes + p_mat.dot(modes.conj())
                        modes = la.qr(modes, mode='economic')[0]
                        # Mix the modes with a random unitary transformation
                        U = kwant.rmt.circular(nmodes, 'A', rng=rng)
                        modes = modes.dot(U)
                        # Make the modes PHS symmetric using the method for a
                        # TRIM.
                        phs_modes = leads.phs_symmetrization(modes, p_mat)[0]
                        assert_almost_equal(phs_modes,
                                            p_mat.dot(phs_modes.conj()),
                                            err_msg='PHS broken at a TRIM in '
                                                    + sym)
                        assert_almost_equal(phs_modes.T.conj().dot(phs_modes),
                                            np.eye(phs_modes.shape[1]),
                                            err_msg='Modes are not orthonormal,'
                                                    'TRIM PHS in ' + sym)
                elif P_squared == -1:
                    # Need even number of modes =< n
                    for nmodes in (2, 4, n//2, n):
                        # Random matrix of 'modes.' Take part of a unitary
                        # matrix to ensure that the modes form a basis.
                        modes = kwant.rmt.circular(n, 'A', rng=rng)[:, :nmodes]
                        # Ensure modes are particle-hole symmetric and
                        # orthonormal.
                        modes[:, nmodes//2:] = \
                                p_mat.dot(modes[:, :nmodes//2].conj())
                        modes = la.qr(modes, mode='economic')[0]
                        # Mix the modes with a random unitary transformation
                        U = kwant.rmt.circular(nmodes, 'A', rng=rng)
                        modes = modes.dot(U)
                        # Make the modes PHS symmetric using the method for a
                        # TRIM.
                        phs_modes = leads.phs_symmetrization(modes, p_mat)[0]
                        assert_almost_equal(phs_modes[:, 1::2],
                                            p_mat.dot(phs_modes[:, ::2].conj()),
                                            err_msg='PHS broken at a TRIM in '
                                                    + sym)
                        assert_almost_equal(phs_modes.T.conj().dot(phs_modes),
                                            np.eye(phs_modes.shape[1]),
                                            err_msg='Modes are not orthonormal,'
                                                    ' TRIM PHS in ' + sym)
        # Test the off-diagonal case when p_mat = sigma_x
        p_mat = np.array([[0, 1], [1, 0]])
        p_mat = np.kron(p_mat, np.identity(n // len(p_mat)))
        for nmodes in (1, 3, n//4, n//2):
            if nmodes > n//2:
                continue
            # Random matrix of 'modes.' Take part of a unitary
            # matrix to ensure that the modes form a basis, all modes
            # are only in half of the space
            modes = kwant.rmt.circular(n//2, 'A', rng=rng)[:, :nmodes]
            modes = np.vstack((modes, np.zeros((n//2, nmodes))))
            # Add an orthogonal set of vectors that are ph images
            modes = np.hstack((modes, p_mat.dot(modes.conj())))
            # Make the modes PHS symmetric using the method for a
            # TRIM.
            phs_modes = leads.phs_symmetrization(modes, p_mat)[0]
            assert_almost_equal(phs_modes,
                                p_mat.dot(phs_modes.conj()),
                                err_msg='PHS broken at a TRIM in '
                                        'off-diagonal test')
            assert_almost_equal(phs_modes.T.conj().dot(phs_modes),
                                np.eye(phs_modes.shape[1]),
                                err_msg='Modes are not orthonormal,'
                                        'off-diagonal test')


def random_onsite_hop(n, rng=0):
    rng = ensure_rng(rng)
    onsite = rng.randn(n, n) + 1j * rng.randn(n, n)
    onsite = onsite + onsite.T.conj()
    hop = rng.rand(n, n) + 1j * rng.rand(n, n)
    return onsite, hop

def test_cons_singular_hopping():
    # Conservation law with square but singular hopping
    n = 20
    hc1, hh1 = random_onsite_hop(n)
    hc2, hh2 = random_onsite_hop(n)
    # Square but singular hopping matrices
    hh1[:, n//2:] = 0
    hh2[:, ::2] = 0
    H_cell = la.block_diag(hc1, hc2)
    H_hop = la.block_diag(hh1, hh2)
    # Eigenvalues of conservation law - two blocks,
    # eigenvalues -1 and 1.
    cs = np.diag(n*[-1] + n*[1])
    assert_almost_equal(H_cell.dot(cs) - cs.dot(H_cell), 0)
    assert_almost_equal(H_hop.dot(cs) - cs.dot(H_hop), 0)
    # Mix the blocks with a random unitary
    U = kwant.rmt.circular(2*n, 'A', rng=78)
    cs_t = U.T.conj().dot(cs).dot(U)
    H_cell_t = U.T.conj().dot(H_cell).dot(U)
    H_hop_t = U.T.conj().dot(H_hop).dot(U)
    assert_almost_equal(cs_t.dot(H_cell_t) - H_cell_t.dot(cs_t), 0)
    assert_almost_equal(cs_t.dot(H_hop_t) - H_hop_t.dot(cs_t), 0)
    # Get the projectors.
    evals, evecs = np.linalg.eigh(cs_t)
    # Make sure the ordering is correct.
    assert_almost_equal(evals-np.array(n*[-1] + n*[1]), 0)
    # First projector projects onto block with
    # eigenvalue -1, second to eigenvalue 1.
    # Both blocks are of size n.
    projectors = [np.reshape(evecs[:,:n], (2*n, n)),
                  np.reshape(evecs[:,n:2*n], (2*n, n))]
    # Check that projectors are correctly defined.
    assert_almost_equal(evecs, np.hstack(projectors))
    # Make the projectors sparse.
    projectors = [sparse.csr_matrix(p) for p in projectors]
    # Without projectors
    modes1 = kwant.physics.leads.modes(H_cell_t, H_hop_t)
    current_conserving(modes1[1])
    # With projectors
    modes2 = kwant.physics.leads.modes(H_cell_t, H_hop_t,
                                       projectors=projectors)
    current_conserving(modes2[1])
    assert_almost_equal(modes1[1].selfenergy(), modes2[1].selfenergy())

def test_cons_rectangular_hopping():
    # Conservation law with rectangular (singular) hopping
    n = 20
    hc1, hh1 = random_onsite_hop(n)
    hc2, hh2 = random_onsite_hop(n)

    H_cell = la.block_diag(hc1, hc2)
    H_hop = la.block_diag(hh1, hh2)
    # Remove half of block 1 from the hopping.
    H_hop = H_hop[:, :-n//2]

    # Get the projectors.
    projectors = np.split(np.eye(2*n), 2, 1)
    projectors = [np.reshape(p, (2*n, n)) for p in projectors]
    # Make the projectors sparse.
    projectors = [sparse.csr_matrix(p) for p in projectors]
    # Without projectors
    modes1 = kwant.physics.leads.modes(H_cell, H_hop)
    current_conserving(modes1[1])
    # With projectors
    modes2 = kwant.physics.leads.modes(H_cell, H_hop, projectors=projectors)
    current_conserving(modes2[1])
    assert_almost_equal(modes1[1].selfenergy(), modes2[1].selfenergy())

def check_bdiag_modes(modes, block_rows, block_cols):
    for vs in modes:
        # Now check that the blocks do not vanish, and that the arrays are 0
        # outside of the block diagonal.
        for rows, cols in zip(block_rows, block_cols):
            # Check that the block is not the empty,
            # i.e. that there are modes in the block
            if vs[rows, cols].size:
                # If there are modes, this block should not vanish.
                assert not np.allclose(vs[rows, cols], 0)
                # Now set this block to zero - to check that with all
                # blocks set to zero, the array is filled with 0.
                vs[rows, cols] = np.zeros(vs[rows, cols].shape)
        assert_almost_equal(vs, 0)

def test_cons_blocks_sizes():
    # Hamiltonian with a conservation law consisting of three blocks of
    # different size.
    n = 8
    # Three blocks of different sizes.
    cs = np.diag(n*[-1] + 2*n*[1] + 3*n*[2])
    # Onsite and hopping
    hc, hh = random_onsite_hop(6*n)
    hc[:n, n:] = 0
    hc[n:, :n] = 0
    hc[n:3*n, 3*n:] = 0
    hc[3*n:, n:3*n] = 0
    assert_almost_equal(cs.dot(hc) - hc.dot(cs), 0)
    hh[:n, n:] = 0
    hh[n:, :n] = 0
    hh[n:3*n, 3*n:] = 0
    hh[3*n:, n:3*n] = 0
    assert_almost_equal(cs.dot(hh) - hh.dot(cs), 0)
    # Mix them with a random unitary
    U = kwant.rmt.circular(6*n, 'A', rng=23)
    cs_t = U.T.conj().dot(cs).dot(U)
    hc_t = U.T.conj().dot(hc).dot(U)
    hh_t = U.T.conj().dot(hh).dot(U)
    assert_almost_equal(cs_t.dot(hc_t) - hc_t.dot(cs_t), 0)
    assert_almost_equal(cs_t.dot(hh_t) - hh_t.dot(cs_t), 0)
    # Get the projectors. There are three of them, composed of
    # the eigenvectors of the conservation law.
    evals, evecs = np.linalg.eigh(cs_t)
    # Make sure the ordering is correct.
    assert_almost_equal(evals-np.array(n*[-1] + 2*n*[1] + 3*n*[2]), 0)
    # First projector projects onto block with
    # eigenvalue -1 of size n, second to eigenvalue 1 of
    # size 2n, third onto block with eigenvalue 2 of size 3n.
    projectors = [np.reshape(evecs[:,:n], (6*n, n)),
                  np.reshape(evecs[:,n:3*n], (6*n, 2*n)),
                  np.reshape(evecs[:,3*n:6*n], (6*n, 3*n))]
    # Check that projectors are correctly defined.
    assert_almost_equal(evecs, np.hstack(projectors))

    ########
    # Compute self energy with and without specifying projectors.
    # Make the projectors sparse.
    projectors = [sparse.csr_matrix(p) for p in projectors]
    # Modes without projectors
    modes1 = leads.modes(hc_t, hh_t)
    current_conserving(modes1[1])
    # With projectors
    modes2 = leads.modes(hc_t, hh_t, projectors=projectors)
    current_conserving(modes2[1])
    assert_almost_equal(modes1[1].selfenergy(), modes2[1].selfenergy())

    ########
    # Check that the number of modes per block with projectors matches the
    # total number of modes, with and without projectors.
    prop1, stab1 = modes1  # No projectors
    prop2, stab2 = modes2  # Projectors
    assert_almost_equal(stab1.nmodes, sum(prop1.block_nmodes))
    assert_almost_equal(stab2.nmodes, sum(prop2.block_nmodes))
    assert_almost_equal(stab1.nmodes, sum(prop2.block_nmodes))

    ########
    # Check that with projectors specified, in/out propagating modes and
    # evanescent modes in vecs and vecslmbdainv are block diagonal.
    # Do this by checking explicitly that the blocks are nonzero, and that
    # if all blocks are set to zero manually, the stabilized modes only contain
    # zeros (meaning that anything off the block diagonal is zero).
    nmodes = stab2.nmodes
    block_nmodes = prop2.block_nmodes
    vecs, vecslmbdainv = stab2.vecs, stab2.vecslmbdainv
    # Row indices for the blocks.
    # The first block is of size n, second of size 2n
    # and the third of size 3n.
    block_rows = [slice(0, n), slice(n, 3*n), slice(3*n, 6*n)]
    ### Check propagating modes ###
    # Column indices for the blocks.
    offsets = np.cumsum([0]+block_nmodes)
    block_cols = [slice(*i) for i in np.vstack([offsets[:-1], offsets[1:]]).T]
    prop_modes = (vecs[:, :nmodes], vecs[:, nmodes:2*nmodes],
                  vecslmbdainv[:, :nmodes], vecslmbdainv[:, nmodes:2*nmodes])
    check_bdiag_modes(prop_modes, block_rows, block_cols)
    ### Check evanescent modes ###
    # First figure out the number of evanescent modes per block.
    # The number of relevant evanescent modes (outward decaying) in a block is
    # N - nmodes, where N is the dimension of the block and nmodes the number
    # of incident or outgoing propagating modes.
    block_cols = [N - nmodes for nmodes, N in zip(block_nmodes, [n, 2*n, 3*n])]
    offsets = np.cumsum([0]+block_cols)
    block_cols = [slice(*i) for i in np.vstack([offsets[:-1], offsets[1:]]).T]
    ev_modes = (vecs[:, 2*nmodes:], vecslmbdainv[:, 2*nmodes:])
    check_bdiag_modes(ev_modes, block_rows, block_cols)


def check_identical_modes(modes, block_rows, block_cols):
    for vs in modes:
        rows0, cols0 = block_rows[0], block_cols[0]
        rows1, cols1 = block_rows[1], block_cols[1]
        if vs[rows0, cols0].size:
            assert_almost_equal(vs[rows0, cols0], vs[rows1, cols1])
        else:
            # If one is empty, so is the other one.
            assert not vs[rows1, cols1].size


def test_block_relations_cons_PHS():
    # Four blocks. Two identical blocks, each with particle-hole symmetry. Then
    # two blocks, related by particle-hole symmetry, but neither possessing it
    # on its own. These two blocks are not identical.  There is a conservation
    # law relating the first two, and a discrete symmetry relating the latter
    # two.  Also check the case when the latter two blocks have singular
    # hopping.
    n = 8
    rng = ensure_rng(99)
    sym = 'C'  # Particle-hole squares to -1
    # Onsite and hopping blocks with particle-hole symm
    hP_cell = kwant.rmt.gaussian(n, sym, rng=rng)
    hP_hop = 10 * kwant.rmt.gaussian(2*n, sym, rng=rng)[:n, n:]
    p_mat = np.array(kwant.rmt.h_p_matrix[sym])
    p_mat = np.kron(np.identity(n // len(p_mat)), p_mat)
    # Random onsite and hopping blocks
    h_cell, h_hop = random_onsite_hop(n)
    # Full onsite and hoppings
    H_cell = la.block_diag(hP_cell, hP_cell, h_cell,
                           -p_mat.dot(h_cell.conj()).dot(p_mat.T.conj()))
    H_hop = la.block_diag(hP_hop, hP_hop, h_hop,
                          -p_mat.dot(h_hop.conj()).dot(p_mat.T.conj()))
    # Also check the case when the hopping is singular (but square) in the
    # second two blocks.
    h_hop[:, ::2] = 0
    H_hop_s = la.block_diag(hP_hop, hP_hop, h_hop,
                            -p_mat.dot(h_hop.conj()).dot(p_mat.T.conj()))
    sx = np.array([[0,1],[1,0]])
    # Particle-hole symmetry operator
    P_mat = la.block_diag(p_mat, p_mat, np.kron(sx, p_mat))
    assert_almost_equal(P_mat.dot(H_cell.conj()) + H_cell.dot(P_mat), 0)
    assert_almost_equal(P_mat.dot(H_hop.conj()) + H_hop.dot(P_mat), 0)
    assert_almost_equal(P_mat.dot(H_hop_s.conj()) + H_hop_s.dot(P_mat), 0)
    assert_almost_equal(P_mat.dot(P_mat.conj()), -np.eye(P_mat.shape[0]))

    # Projectors
    projectors = np.split(np.eye(4*n), 4, 1)
    # Make the projectors sparse.
    projectors = [sparse.csr_matrix(p) for p in projectors]

    # Cover both singular and nonsingular hopping
    ham = [(H_cell, H_hop), (H_cell, H_hop_s)]
    for (H_cell, H_hop) in ham:
        prop, stab = kwant.physics.leads.modes(H_cell, H_hop,
                                               particle_hole=P_mat,
                                               projectors=projectors)
        current_conserving(stab)
        nmodes = stab.nmodes
        block_nmodes = prop.block_nmodes
        vecs, vecslmbdainv = stab.vecs, stab.vecslmbdainv
        # Row indices for the blocks.
        # All 4 blocks are of size n.
        block_rows = [slice(0, n), slice(n, 2*n),
                      slice(2*n, 3*n), slice(3*n, 4*n)]

        ######## Check that the first two blocks are identical.
        ### Propagating modes ###
        # Column indices for the blocks.
        offsets = np.cumsum([0]+block_nmodes)
        block_cols = [slice(*i) for i in np.vstack([offsets[:-1],
                                                    offsets[1:]]).T]
        prop_modes = (vecs[:, :nmodes], vecs[:, nmodes:2*nmodes],
                      vecslmbdainv[:, :nmodes],
                      vecslmbdainv[:, nmodes:2*nmodes])
        check_identical_modes(prop_modes, block_rows, block_cols)
        ### Evanescent modes ###
        # First figure out the number of evanescent modes per block.  The
        # number of relevant evanescent modes (outward decaying) in a block is
        # N - nmodes, where N is the dimension of the block and nmodes the
        # number of incident or outgoing propagating modes.
        block_cols = [N - nmodes for nmodes, N in zip(block_nmodes, 4*[n])]
        offsets = np.cumsum([0]+block_cols)
        block_cols = [slice(*i) for i in np.vstack([offsets[:-1],
                                                    offsets[1:]]).T]
        ev_modes = (vecs[:, 2*nmodes:], vecslmbdainv[:, 2*nmodes:])
        check_identical_modes(ev_modes, block_rows, block_cols)

        # Check that the second two blocks are related by PHS.

        # Here we only look at propagating modes. Compare the stabilized modes
        # of incident and outgoing modes for the two blocks that are related by
        # particle-hole symmetry, i.e. blocks 2 and 3.  Column indices for the
        # blocks.
        offsets = np.cumsum([0]+block_nmodes)
        block_cols = [slice(*i) for i in np.vstack([offsets[:-1],
                                                    offsets[1:]]).T]
        # Need both incident and outgoing stabilized modes to make the
        # comparison between blocks
        prop_modes = [(vecs[:, :nmodes], vecs[:, nmodes:2*nmodes], 1),
                      (vecslmbdainv[:, :nmodes],
                       vecslmbdainv[:, nmodes:2*nmodes], -1)]
        # P is antiunitary, such that vecslmbdainv changes sign when
        # used between blocks to construct modes.
        for (in_modes, out_modes, vecs_sign) in prop_modes:
            # Coordinates of blocks 2 and 3
            rows2, cols2 = block_rows[2], block_cols[2]
            cols3 = block_cols[3]
            bnmodes = block_nmodes[2] # Number of modes in block 2
            # Mode rearrangement by particle-hole symmetry
            perm = ((-1-np.arange(2*bnmodes)) % bnmodes +
                    bnmodes * (np.arange(2*bnmodes) // bnmodes))
            # Make sure the blocks are not empty
            if in_modes[rows2, cols2].size:
                # Check that the real space representations of the stabilized
                # modes of blocks 2 and 3 are related by particle-hole
                # symmetry.
                sqrt_hop = stab.sqrt_hop
                modes2 = sqrt_hop.dot(np.hstack([in_modes[:, cols2],
                                                 out_modes[:, cols2]]))
                modes3 = sqrt_hop.dot(np.hstack([in_modes[:, cols3],
                                                 out_modes[:, cols3]]))
                # In the algorithm, the blocks are compared in the same order
                # they are specified.  Block 2 is computed before block 3, so
                # block 3 is obtained by particle-hole transforming the modes
                # of block 2. Check that it is so.
                assert_almost_equal(P_mat.dot(modes2.conj())[:, perm], vecs_sign*modes3)


def check_symm_ham(h_cell, h_hop, sym_op, trans_sign, sym):
    """Check that the symmetry operator and Hamiltonian are properly defined"""
    if sym in ['AI', 'AII', 'C', 'D']:
        # Antiunitary
        assert_almost_equal(sym_op.dot(h_cell.conj()) -
                           trans_sign*h_cell.dot(sym_op), 0)
        assert_almost_equal(sym_op.dot(h_hop.conj()) -
                           trans_sign*h_hop.dot(sym_op), 0)
        assert np.allclose(sym_op.dot(sym_op.conj()),
                           np.eye(sym_op.shape[0])) or \
               np.allclose(sym_op.dot(sym_op.conj()), -np.eye(sym_op.shape[0]))
    elif sym in ['AIII']:
        # Unitary
        assert_almost_equal(sym_op.dot(h_cell) - trans_sign*h_cell.dot(sym_op),
                            0)
        assert_almost_equal(sym_op.dot(h_hop) - trans_sign*h_hop.dot(sym_op),
                            0)
        assert_almost_equal(sym_op.dot(sym_op), np.eye(sym_op.shape[0])) or \
               np.allclose(sym_op.dot(sym_op), -np.eye(sym_op.shape[0]))


def test_blocks_symm_complex_projectors():
    # Two blocks of equal size, related by any one of the discrete
    # symmetries. Each block by itself has no symmetry. The system is
    # transformed with a random unitary, such that the projectors onto the
    # blocks are complex.
    n = 8
    rng = ensure_rng(27)
    # Symmetry class, sign of H under symmetry transformation.
    sym_info = [('AI', 1), ('AII', 1), ('D', -1),
                ('C', -1), ('AIII', -1)]
    for (sym, trans_sign) in sym_info:
        # Conservation law values
        cs = n*[-1] + n*[1]
        # Random onsite and hopping blocks
        h_cell, h_hop = random_onsite_hop(n, rng)
        # Symmetry operator
        if sym in ['AI', 'AII']:
            sym_op = np.array(kwant.rmt.h_t_matrix[sym])
            sym_op = np.kron(np.identity(n // len(sym_op)), sym_op)
        elif sym in ['D', 'C']:
            sym_op = np.array(kwant.rmt.h_p_matrix[sym])
            sym_op = np.kron(np.identity(n // len(sym_op)), sym_op)
        elif sym in ['AIII']:
            sym_op = np.kron(np.identity(n // 2), np.diag([1, -1]))
        else:
            raise ValueError('Symmetry class not covered.')
        # Full onsite and hoppings
        if sym in ['AI', 'AII', 'C', 'D']:
            # Antiunitary symmetries
            H_cell = la.block_diag(h_cell, trans_sign*sym_op.dot(
                                   h_cell.conj()).dot(sym_op.T.conj()))
            H_hop = la.block_diag(h_hop, trans_sign*sym_op.dot(
                                  h_hop.conj()).dot(sym_op.T.conj()))
        elif sym in ['AIII']:
            # Unitary symmetries
            H_cell = la.block_diag(h_cell, trans_sign*sym_op.dot(
                                   h_cell).dot(sym_op.T.conj()))
            H_hop = la.block_diag(h_hop, trans_sign*sym_op.dot(
                                  h_hop).dot(sym_op.T.conj()))
        sx = np.array([[0,1],[1,0]])
        # Full symmetry operator relating the blocks
        S = np.kron(sx, sym_op)
        check_symm_ham(H_cell, H_hop, S, trans_sign, sym=sym)
        # Mix with a random unitary
        U = kwant.rmt.circular(2*n, 'A', rng=3)
        H_cell_t = U.T.conj().dot(H_cell).dot(U)
        H_hop_t = U.T.conj().dot(H_hop).dot(U)
        if sym in ['AI', 'AII', 'C', 'D']:
            S_t = U.T.conj().dot(S).dot(U.conj())
        elif sym in ['AIII']:
            S_t = U.T.conj().dot(S).dot(U)
        # Conservation law matrix in the new basis
        cs_t = U.T.conj().dot(np.diag(cs)).dot(U)
        check_symm_ham(H_cell_t, H_hop_t, S_t, trans_sign, sym=sym)

        # Get the projectors.
        evals, evecs = np.linalg.eigh(cs_t)
        # Make sure the ordering is correct.
        assert_almost_equal(evals-np.array(cs), 0)
        projectors = [np.reshape(evecs[:, :n], (2*n, n)),
                      np.reshape(evecs[:, n:2*n], (2*n, n))]
        # Ensure that the projectors sum to a unitary.
        assert_almost_equal(sum(projector.dot(projector.conj().T) for projector
                                in projectors), np.eye(2*n))
        projectors = [sparse.csr_matrix(p) for p in projectors]

        if sym in ['AI', 'AII']:
            prop, stab = kwant.physics.leads.modes(H_cell_t, H_hop_t,
                                                   time_reversal=S_t,
                                                   projectors=projectors)
        elif sym in ['C', 'D']:
            prop, stab = kwant.physics.leads.modes(H_cell_t, H_hop_t,
                                                   particle_hole=S_t,
                                                   projectors=projectors)
        elif sym in ['AIII']:
            prop, stab = kwant.physics.leads.modes(H_cell_t, H_hop_t,
                                                   chiral=S_t,
                                                   projectors=projectors)
        current_conserving(stab)

        nmodes = stab.nmodes
        block_nmodes = prop.block_nmodes
        vecs, vecslmbdainv = stab.vecs, stab.vecslmbdainv
        # Row indices for the blocks. Both are of size n.
        block_rows = [slice(0, n), slice(n, 2*n)]
        ######## Check that the two blocks are related by symmetry.

        # Compare the stabilized propagating modes of incident and outgoing
        # modes for the two blocks that are related by symmetry.  Column
        # indices for the blocks.
        offsets = np.cumsum([0]+block_nmodes)
        block_cols = [slice(*i) for i in np.vstack([offsets[:-1],
                                                    offsets[1:]]).T]
        # Mode rearrangement for each symmetry
        bnmodes = block_nmodes[0] # Number of modes in the first block
        if sym in ['AI', 'AII']:
            perm = np.arange(2*bnmodes)[::-1]
        elif sym in ['C', 'D']:
            perm = ((-1-np.arange(2*bnmodes)) % bnmodes +
                    bnmodes * (np.arange(2*bnmodes) // bnmodes))
        elif sym in ['AIII']:
            perm = (np.arange(2*bnmodes) % bnmodes +
                    bnmodes * (np.arange(2*bnmodes) < bnmodes))

        # Need both incident and outgoing stabilized modes to make the
        # comparison between blocks
        prop_modes = [(vecs[:, :nmodes], vecs[:, nmodes:2*nmodes], 1),
                      (vecslmbdainv[:, :nmodes],
                       vecslmbdainv[:, nmodes:2*nmodes], trans_sign)]
        # Symmetries that flip the sign of energy change the sign of
        # vecslmbdainv when used to construct modes between blocks.
        for (in_modes, out_modes, vecs_sign) in prop_modes:
            rows0, cols0 = block_rows[0], block_cols[0]
            rows1, cols1 = block_rows[1], block_cols[1]
            # Make sure the blocks are not empty
            if in_modes[rows0, cols0].size:
                # Check the real space representations of the stabilized modes
                sqrt_hop = stab.sqrt_hop
                modes0 = sqrt_hop.dot(np.hstack([in_modes[:, cols0],
                                                 out_modes[:, cols0]]))
                modes1 = sqrt_hop.dot(np.hstack([in_modes[:, cols1],
                                                 out_modes[:, cols1]]))
                # In the algorithm, the blocks are compared in the same order
                # they are specified.  Block 0 is computed before block 1, so
                # block 1 is obtained by symmetry transforming the modes of
                # block 0. Check that it is so.
                if sym in ['AI', 'AII', 'C', 'D']:
                    assert_almost_equal(S_t.dot(modes0.conj())[:, perm], vecs_sign*modes1)
                elif sym in ['AIII']:
                    assert_almost_equal(S_t.dot(modes0)[:, perm], vecs_sign*modes1)
            # If first block is empty, so is the second one.
            else:
                assert not in_modes[rows1, cols1].size
