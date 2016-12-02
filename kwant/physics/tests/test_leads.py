# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.


import numpy as np
from numpy.testing import assert_almost_equal
import scipy.linalg as la
import numpy.linalg as npl
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


def check_equivalence(h, t, n, sym='', particle_hole=None, chiral=None, time_reversal=None):
    """Compare modes stabilization algorithms for a given Hamiltonian."""
    u, s, vh = np.linalg.svd(t)
    u, v = u * np.sqrt(s), vh.T.conj() * np.sqrt(s)
    prop_vecs = []
    evan_vecs = []
    algos = [None, (True, True), (True, False), (False, True), (False, False)]
    for algo in algos:
        result = leads.modes(h, t, stabilization=algo, chiral=chiral,
                             particle_hole=particle_hole, time_reversal=time_reversal)[1]

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

    msg = 'Stabilization {0} failed.'
    for vecs, algo in zip(prop_vecs, algos):
        # Propagating modes should have identical ordering, and only vary
        # By a phase
        np.testing.assert_allclose(np.abs(np.sum(vecs/prop_vecs[0],
                                                 axis=0)), vecs.shape[0],
                                   err_msg=msg.format(algo)+' in symmetry class '+sym)

    for vecs, algo in zip(evan_vecs, algos):
        # Evanescent modes must span the same linear space.
        mat = np.c_[vecs, evan_vecs[0]]
        # Scale largest singular value to 1 if the array is not empty
        mat = mat/np.linalg.norm(mat, ord=2)
        # As a tolerance, take the square root of machine precision times the largest
        # matrix dimension.
        tol = np.abs(np.sqrt(max(mat.shape)*np.finfo(mat.dtype).eps))
        assert (np.linalg.matrix_rank(mat, tol=tol) ==
                vecs.shape[1]), msg.format(algo)+' in symmetry class '+sym


def test_symm_algorithm_equivalence():
    """Test different stabilization methods in the computation of modes,
    in the presence and/or absence of the discrete symmetries."""
    np.random.seed(400)
    for n in (12, 20, 40, 60):
        for sym in kwant.rmt.sym_list:
            # Random onsite and hopping matrices in symmetry class
            h_cell = kwant.rmt.gaussian(n, sym)
            # Hopping is an offdiagonal block of a Hamiltonian. We rescale it
            # to ensure that there are modes at the Fermi level.
            h_hop = 10 * kwant.rmt.gaussian(2*n, sym)[:n, n:]

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
    """Check PHS of incident or outgoing modes at a TRIM momentum. Input are momenta,
    velocities and wave functions of incident or outgoing modes. """
    # Pick out TRIM momenta - in this test, all momenta are either 0 or -pi,
    # so the boundaries of the interval here are not important.
    TRIM_moms = (TRIM-0.1 < moms) * (moms < TRIM+0.1)
    assert np.allclose(moms[TRIM_moms], TRIM)
    # At a given momentum, incident modes are sorted in ascending order by velocity.
    # Pick out modes with the same velocity.
    vels = velocities[TRIM_moms]
    inds = [ind+1 for ind, vel in enumerate(vels[:-1])
            if np.abs(vel-vels[ind+1])>1e-8]
    inds = [0] + inds + [len(vels)]
    inds = zip(inds[:-1], inds[1:])
    for ind_tuple in inds:
        vel_wfs = wfs[:, slice(*ind_tuple)]
        assert np.allclose(vels[slice(*ind_tuple)], vels[slice(*ind_tuple)][0])
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
    ### P squares to 1 ###
    np.random.seed(42)
    dims = (4, 10, 20)
    ts = (1.0, 1.7, 13.8)
    rand_hop = 1j*(0.1+np.random.rand())
    hop = la.block_diag(*[t*rand_hop*np.eye(dim) for t, dim in zip(ts, dims)])

    # Particle-hole operator
    pmat = np.eye(sum(dims))
    onsite = np.zeros(hop.shape, dtype=complex)
    prop, stab = leads.modes(onsite, hop, particle_hole=pmat)
    # All momenta are either 0 or -pi.
    assert np.all([np.any(ele - np.array([0, -np.pi])) for ele in prop.momenta])
    # All modes are eigenmodes of P.
    assert np.all([np.allclose(wf, pmat.dot(wf.conj())) for wf in prop.wave_functions.T])
    ###########

    ### P squares to -1 ###
    np.random.seed(1337)
    dims = (1, 4, 40)
    ts = (1.0, 1.7, 13.4)

    hop_mat = np.kron(sz, 1j*(0.1+np.random.rand())*np.eye(2))
    blocks = []
    for t, dim in zip(ts, dims):
        blocks += dim*[t*hop_mat]
    hop = la.block_diag(*blocks)
    # Particle-hole operator
    pmat = np.kron(np.eye(sum(dims)), 1j*np.kron(sz, sy))

    # P squares to -1
    assert np.allclose(pmat.dot(pmat.conj()), -np.eye(pmat.shape[0]))
    # The Hamiltonian anticommutes with P
    assert np.allclose(pmat.dot(hop.conj()).dot(npl.inv(pmat)), -hop)

    onsite = np.zeros(hop.shape, dtype=complex)
    prop, stab = leads.modes(onsite, hop, particle_hole=pmat)
    # By design, all momenta are either 0 or -pi.
    assert np.all([np.any(ele - np.array([0, -np.pi])) for ele in prop.momenta])

    wfs = prop.wave_functions
    momenta = prop.momenta
    velocities = prop.velocities
    nmodes = stab.nmodes

    # By design, all modes are at a TRIM here. Each must thus have a particle-hole
    # partner at the same TRIM and with the same velocity.
    # Incident modes
    check_PHS(0, momenta[:nmodes], velocities[:nmodes], wfs[:, :nmodes], pmat)
    check_PHS(-np.pi, momenta[:nmodes], velocities[:nmodes], wfs[:, :nmodes], pmat)
    # Outgoing modes
    check_PHS(0, momenta[nmodes:], velocities[nmodes:], wfs[:, nmodes:], pmat)
    check_PHS(-np.pi, momenta[nmodes:], velocities[nmodes:], wfs[:, nmodes:], pmat)
    ###########


def test_modes_symmetries():
    np.random.seed(10)
    for n in (4, 8, 40, 100):
        for sym in kwant.rmt.sym_list:
            # Random onsite and hopping matrices in symmetry class
            h_cell = kwant.rmt.gaussian(n, sym)
            # Hopping is an offdiagonal block of a Hamiltonian. We rescale it
            # to ensure that there are modes at the Fermi level.
            h_hop = 10 * kwant.rmt.gaussian(2*n, sym)[:n, n:]

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

            prop_modes, stab_modes = leads.modes(h_cell, h_hop, particle_hole=p_mat,
                                                 time_reversal=t_mat, chiral=c_mat)
            wave_functions = prop_modes.wave_functions
            momenta = prop_modes.momenta
            nmodes = stab_modes.nmodes

            if t_mat is not None:
                assert_almost_equal(wave_functions[:, nmodes:],
                        t_mat.dot(wave_functions[:, :nmodes].conj()),
                        err_msg='TRS broken in ' + sym)

            if c_mat is not None:
                assert_almost_equal(wave_functions[:, nmodes:],
                        c_mat.dot(wave_functions[:, :nmodes:-1]),
                        err_msg='SLS broken in ' + sym)

            if p_mat is not None:
                # If P^2 = -1, then P psi(-k) = -psi(k) for k>0, so one must look at
                # positive and negative momenta separately.
                # Test positive momenta.
                in_positive_k = (np.pi > momenta[:nmodes]) * (momenta[:nmodes] > 0)
                out_positive_k = (np.pi > momenta[nmodes:]) * (momenta[nmodes:] > 0)

                assert_almost_equal(wave_functions[:, :nmodes][:, in_positive_k[::-1]],
                        p_mat.dot((wave_functions[:, :nmodes][:, in_positive_k][:, ::-1]).conj()),
                        err_msg='PHS broken in ' + sym)
                assert_almost_equal(wave_functions[:, nmodes:][:, out_positive_k[::-1]],
                        p_mat.dot((wave_functions[:, nmodes:][:, out_positive_k][:, ::-1]).conj()),
                        err_msg='PHS broken in ' + sym)

                # Test negative momenta. Need the sign of P^2 here.
                p_squared_sign = np.sign(p_mat.dot(p_mat.conj())[0, 0].real)
                in_neg_k = (-np.pi < momenta[:nmodes]) * (momenta[:nmodes] < 0)
                out_neg_k = (-np.pi < momenta[nmodes:]) * (momenta[nmodes:] < 0)

                assert_almost_equal(p_squared_sign*wave_functions[:, :nmodes][:, in_neg_k[::-1]],
                        p_mat.dot((wave_functions[:, :nmodes][:, in_neg_k][:, ::-1]).conj()),
                        err_msg='PHS broken in ' + sym)
                assert_almost_equal(p_squared_sign*wave_functions[:, nmodes:][:, out_neg_k[::-1]],
                        p_mat.dot((wave_functions[:, nmodes:][:, out_neg_k][:, ::-1]).conj()),
                        err_msg='PHS broken in ' + sym)


def test_PHS_TRIM():
    """Test the function that makes particle-hole symmetric modes at a TRIM. """
    np.random.seed(10)
    for n in (4, 8, 16, 40, 100):
        for sym in kwant.rmt.sym_list:
            if kwant.rmt.p(sym):
                p_mat = np.array(kwant.rmt.h_p_matrix[sym])
                p_mat = np.kron(np.identity(n // len(p_mat)), p_mat)
                P_squared = 1 if np.all(np.abs(p_mat.conj().dot(p_mat) -
                                               np.eye(*p_mat.shape)) < 1e-10) else -1
                if P_squared == 1:
                    for nmodes in (1, 3, n//4, n//2, n):
                        # Random matrix of 'modes.' Take part of a unitary matrix to
                        # ensure that the modes form a basis.
                        modes = np.random.rand(n, n) + 1j*np.random.rand(n, n)
                        modes = la.expm(1j*(modes + modes.T.conj()))[:n, :nmodes]
                        # Ensure modes are particle-hole symmetric and normalized
                        modes = modes + p_mat.dot(modes.conj())
                        modes = np.array([col/np.linalg.norm(col) for col in modes.T]).T
                        # Mix the modes with a random unitary transformation
                        U = np.random.rand(nmodes, nmodes) + 1j*np.random.rand(nmodes, nmodes)
                        U = la.expm(1j*(U + U.T.conj()))
                        modes = modes.dot(U)
                        # Make the modes PHS symmetric using the method for a TRIM.
                        phs_modes = leads.phs_symmetrization(modes, p_mat)[0]
                        assert_almost_equal(phs_modes, p_mat.dot(phs_modes.conj()),
                                            err_msg='PHS broken at a TRIM in ' + sym)
                        assert_almost_equal(phs_modes.T.conj().dot(phs_modes), np.eye(phs_modes.shape[1]),
                                           err_msg='Modes are not orthonormal, TRIM PHS in ' + sym)
                elif P_squared == -1:
                    # Need even number of modes =< n
                    for nmodes in (2, 4, n//2, n):
                        # Random matrix of 'modes.' Take part of a unitary matrix to
                        # ensure that the modes form a basis.
                        modes = np.random.rand(n, n) + 1j*np.random.rand(n, n)
                        modes = la.expm(1j*(modes + modes.T.conj()))[:n, :nmodes]
                        # Ensure modes are particle-hole symmetric and orthonormal.
                        modes[:, nmodes//2:] = p_mat.dot(modes[:, :nmodes//2].conj())
                        modes = la.qr(modes, mode='economic')[0]
                        # Mix the modes with a random unitary transformation
                        U = np.random.rand(nmodes, nmodes) + 1j*np.random.rand(nmodes, nmodes)
                        U = la.expm(1j*(U + U.T.conj()))
                        modes = modes.dot(U)
                        # Make the modes PHS symmetric using the method for a TRIM.
                        phs_modes = leads.phs_symmetrization(modes, p_mat)[0]
                        assert_almost_equal(phs_modes[:, 1::2], p_mat.dot(phs_modes[:, ::2].conj()),
                                            err_msg='PHS broken at a TRIM in ' + sym)
                        assert_almost_equal(phs_modes.T.conj().dot(phs_modes), np.eye(phs_modes.shape[1]),
                                           err_msg='Modes are not orthonormal, TRIM PHS in ' + sym)