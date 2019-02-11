# Copyright 2011-2014 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.


from math import cos, sin
import numpy as np
from pytest import raises
from numpy.testing import assert_almost_equal
import kwant
from kwant._common import ensure_rng

n = 5
chain = kwant.lattice.chain()
sq = square = kwant.lattice.square()


class LeadWithOnlySelfEnergy:
    def __init__(self, lead):
        self.lead = lead
        self.parameters = frozenset()

    def selfenergy(self, energy, args=(), *, params=None):
        assert args == ()
        assert params == None
        return self.lead.selfenergy(energy)


def assert_modes_equal(modes1, modes2):
    assert_almost_equal(modes1.velocities, modes2.velocities)
    assert_almost_equal(modes1.momenta, modes2.momenta)
    vecs1, vecs2 = modes1.wave_functions, modes2.wave_functions
    assert_almost_equal(np.abs(np.sum(vecs1/vecs2, axis=0)),
                        vecs1.shape[0])


# Test output sanity: that an error is raised if no output is requested,
# and that solving for a subblock of a scattering matrix is the same as taking
# a subblock of the full scattering matrix.
def test_output(smatrix):
    rng = ensure_rng(3)
    system = kwant.Builder()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    right_lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                    (left_lead, chain(0)), (right_lead, chain(0))]:
        h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
        h += h.conjugate().transpose()
        b[site] = h
    for b, hopp in [(system, (chain(0), chain(1))),
                    (left_lead, (chain(0), chain(1))),
                    (right_lead, (chain(0), chain(1)))]:
        b[hopp] = (10 * rng.random_sample((n, n)) +
                   1j * rng.random_sample((n, n)))
    system.attach_lead(left_lead)
    system.attach_lead(right_lead)
    fsyst = system.finalized()

    result1 = smatrix(fsyst)
    s, modes1 = result1.data, result1.lead_info
    assert s.shape == 2 * (sum(len(i.momenta) for i in modes1) // 2,)
    s1 = result1.submatrix(1, 0)
    result2 = smatrix(fsyst, 0, (), [1], [0])
    s2, modes2 = result2.data, result2.lead_info
    assert s2.shape == (len(modes2[1].momenta) // 2,
                        len(modes2[0].momenta) // 2)
    assert_almost_equal(abs(s1), abs(s2))
    assert_almost_equal(np.dot(s.T.conj(), s),
                        np.identity(s.shape[0]))
    raises(ValueError, smatrix, fsyst, out_leads=[])
    modes = smatrix(fsyst).lead_info
    h = fsyst.leads[0].cell_hamiltonian()
    t = fsyst.leads[0].inter_cell_hopping()
    modes1 = kwant.physics.modes(h, t)[0]
    h = fsyst.leads[1].cell_hamiltonian()
    t = fsyst.leads[1].inter_cell_hopping()
    modes2 = kwant.physics.modes(h, t)[0]
    assert_modes_equal(modes1, modes[0])
    assert_modes_equal(modes2, modes[1])


# Test that a system with one lead has unitary scattering matrix.
def test_one_lead(smatrix):
    rng = ensure_rng(3)
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                    (system, chain(2)), (lead, chain(0))]:
        h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
        h += h.conjugate().transpose()
        b[site] = h
    for b, hopp in [(system, (chain(0), chain(1))),
                    (system, (chain(1), chain(2))),
                    (lead, (chain(0), chain(1)))]:
        b[hopp] = (10 * rng.random_sample((n, n)) +
                   1j * rng.random_sample((n, n)))
    system.attach_lead(lead)
    fsyst = system.finalized()

    for syst in (fsyst, fsyst.precalculate(), fsyst.precalculate(what='all')):
        s = smatrix(syst).data
        assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                            np.identity(s.shape[0]))

    raises(ValueError, smatrix, fsyst.precalculate(what='selfenergy'))

# Test that a system with one lead with no propagating modes has a
# 0x0 S-matrix.
def test_smatrix_shape(smatrix):
    system = kwant.Builder()
    lead0 = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead1 = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                    (system, chain(2))]:
        b[site] = 2
    lead0[chain(0)] = lambda site: lead0_val
    lead1[chain(0)] = lambda site: lead1_val

    for b, hopp in [(system, (chain(0), chain(1))),
                    (system, (chain(1), chain(2))),
                    (lead0, (chain(0), chain(1))),
                    (lead1, (chain(0), chain(1)))]:
        b[hopp] = -1
    system.attach_lead(lead0)
    system.attach_lead(lead1)
    fsyst = system.finalized()

    lead0_val = 4
    lead1_val = 4
    s = smatrix(fsyst, 1.0, (), [1], [0]).data
    assert s.shape == (0, 0)

    lead0_val = 2
    lead1_val = 2
    s = smatrix(fsyst, 1.0, (), [1], [0]).data
    assert s.shape == (1, 1)

    lead0_val = 4
    lead1_val = 2
    s = smatrix(fsyst, 1.0, (), [1], [0]).data
    assert s.shape == (1, 0)

    lead0_val = 2
    lead1_val = 4
    s = smatrix(fsyst, 1.0, (), [1], [0]).data
    assert s.shape == (0, 1)


# Test that a translationally invariant system with two leads has only
# transmission and that transmission does not mix modes.
def test_two_equal_leads(smatrix):
    def check_fsyst(fsyst):
        sol = smatrix(fsyst)
        s, leads = sol.data, sol.lead_info
        assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                            np.identity(s.shape[0]))
        n_modes = len(leads[0].momenta) // 2
        assert len(leads[1].momenta) // 2 == n_modes
        assert_almost_equal(s[: n_modes, : n_modes], 0)
        t_elements = np.sort(abs(np.asarray(s[n_modes :, : n_modes])),
                             axis=None)
        t_el_should_be = n_modes * (n_modes - 1) * [0] + n_modes * [1]
        assert_almost_equal(t_elements, t_el_should_be)
        assert_almost_equal(sol.transmission(1,0), n_modes)
    rng = ensure_rng(11)
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
    h += h.conjugate().transpose()
    h *= 0.8
    t = 4 * rng.random_sample((n, n)) + 4j * rng.random_sample((n, n))
    lead[chain(0)] = system[chain(0)] = h
    lead[chain(0), chain(1)] = t
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    fsyst = system.finalized()
    for syst in (fsyst, fsyst.precalculate(), fsyst.precalculate(what='all')):
        check_fsyst(syst)
    raises(ValueError, check_fsyst, fsyst.precalculate(what='selfenergy'))

    # Test the same, but with a larger scattering region.
    system = kwant.Builder()
    system[[chain(0), chain(1)]] = h
    system[chain(0), chain(1)] = t
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    fsyst = system.finalized()
    for syst in (fsyst, fsyst.precalculate(), fsyst.precalculate(what='all')):
        check_fsyst(syst)
    raises(ValueError, check_fsyst, fsyst.precalculate(what='selfenergy'))


# Test a more complicated graph with non-singular hopping.
def test_graph_system(smatrix):
    rng = ensure_rng(11)
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
    h += h.conjugate().transpose()
    h *= 0.8
    t = 4 * rng.random_sample((n, n)) + 4j * rng.random_sample((n, n))
    t1 = 4 * rng.random_sample((n, n)) + 4j * rng.random_sample((n, n))
    lead[sq(0, 0)] = system[[sq(0, 0), sq(1, 0)]] = h
    lead[sq(0, 1)] = system[[sq(0, 1), sq(1, 1)]] = 4 * h
    for builder in [system, lead]:
        builder[sq(0, 0), sq(1, 0)] = t
        builder[sq(0, 1), sq(1, 0)] = t1
        builder[sq(0, 1), sq(1, 1)] = 1.1j * t1
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    fsyst = system.finalized()

    result = smatrix(fsyst)
    s, leads = result.data, result.lead_info
    assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                        np.identity(s.shape[0]))
    n_modes = len(leads[0].momenta) // 2
    assert len(leads[1].momenta) // 2 == n_modes
    assert_almost_equal(s[: n_modes, : n_modes], 0)
    t_elements = np.sort(abs(np.asarray(s[n_modes:, :n_modes])),
                         axis=None)
    t_el_should_be = n_modes * (n_modes - 1) * [0] + n_modes * [1]
    assert_almost_equal(t_elements, t_el_should_be)


# Test a system with singular hopping.
def test_singular_graph_system(smatrix):
    rng = ensure_rng(11)

    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
    h += h.conjugate().transpose()
    h *= 0.8
    t = 4 * rng.random_sample((n, n)) + 4j * rng.random_sample((n, n))
    t1 = 4 * rng.random_sample((n, n)) + 4j * rng.random_sample((n, n))
    lead[sq(0, 0)] = system[[sq(0, 0), sq(1, 0)]] = h
    lead[sq(0, 1)] = system[[sq(0, 1), sq(1, 1)]] = 4 * h
    for builder in [system, lead]:
        builder[sq(0, 0), sq(1, 0)] = t
        builder[sq(0, 1), sq(1, 0)] = t1
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    fsyst = system.finalized()

    result = smatrix(fsyst)
    s, leads = result.data, result.lead_info
    assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                        np.identity(s.shape[0]))
    n_modes = len(leads[0].momenta) // 2
    assert len(leads[1].momenta) // 2 == n_modes
    assert_almost_equal(s[: n_modes, : n_modes], 0)
    t_elements = np.sort(abs(np.asarray(s[n_modes :, : n_modes])),
                         axis=None)
    t_el_should_be = n_modes * (n_modes - 1) * [0] + n_modes * [1]
    assert_almost_equal(t_elements, t_el_should_be)


# This test features inside the cell Hamiltonian a hopping matrix with more
# zero eigenvalues than the lead hopping matrix. Older version of the
# sparse solver failed here.
def test_tricky_singular_hopping(smatrix):
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((4, 0)))

    interface = []
    for i in range(n):
        site = sq(-1, i)
        interface.append(site)
        system[site] = 0
        for j in range(4):
            lead[sq(j, i)] = 0
    for i in range(n-1):
        system[sq(-1, i), sq(-1, i+1)] = -1
        for j in range(4):
            lead[sq(j, i), sq(j, i+1)] = -1
    for i in range(n):
        for j in range(4):
            lead[sq(j, i), sq(j+1, i)] = -1
    del lead[sq(1, 0), sq(2, 0)]

    system.leads.append(kwant.builder.BuilderLead(lead, interface))
    fsyst = system.finalized()

    s = smatrix(fsyst, -1.3).data
    assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                        np.identity(s.shape[0]))


# Test the consistency of transmission and conductance_matrix for a four-lead
# system without time-reversal symmetry.
def test_many_leads(*factories):
    E=2.1
    B=0.01

    def phase(a, b):
        ap = a.pos
        bp = b.pos
        phase = -B * (0.5 * (ap[1] + bp[1]) * (bp[0] - ap[0]))
        return -complex(cos(phase), sin(phase))

    # Build a square system with four leads and a hole in the center.
    syst = kwant.Builder()
    syst[(sq(x, y) for x in range(-4, 4) for y in range(-4, 4)
         if x**2 + y**2 >= 2)] = 3
    syst[sq.neighbors()] = phase
    for r in [range(-4, -1), range(4)]:
        lead = kwant.Builder(kwant.TranslationalSymmetry([-1, 0]))
        lead[(sq(0, y) for y in r)] = 4
        lead[sq.neighbors()] = phase
        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())
    syst = syst.finalized()

    r4 = list(range(4))
    br = factories[0](syst, E, out_leads=r4, in_leads=r4)
    trans = np.array([[br._transmission(i, j) for j in r4] for i in r4])
    cmat = br.conductance_matrix()
    assert_almost_equal(cmat.sum(axis=0), [0] * 4)
    assert_almost_equal(cmat.sum(axis=1), [0] * 4)
    for i in r4:
        for j in r4:
            assert_almost_equal(
                (br.num_propagating(i) if i == j else 0) - cmat[i, j],
                trans[i, j])

    for out_leads, in_leads in [(r4, r4), ((1, 2, 3), r4), (r4, (0, 2, 3)),
                                ((1, 3), (1, 2, 3)), ((0, 2), (0, 1, 2)),
                                ((0, 1,), (1, 2)), ((3,), (3,))]:
        for f in factories:
            br = f(syst, E, out_leads=out_leads, in_leads=in_leads)
            if len(out_leads) == 3:
                out_leads = r4
            if len(in_leads) == 3:
                in_leads = r4
            for i in r4:
                for j in r4:
                    if i in out_leads and j in in_leads:
                        assert_almost_equal(br.transmission(i, j), trans[i, j])
                    else:
                        raises(ValueError, br.transmission, i, j)
            if len(out_leads) == len(in_leads) == 4:
                assert_almost_equal(br.conductance_matrix(), cmat)


# Test equivalence between self-energy and scattering matrix representations.
# Also check that transmission works.
def test_selfenergy(greens_function, smatrix):
    rng = ensure_rng(4)
    system = kwant.Builder()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    right_lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                 (left_lead, chain(0)), (right_lead, chain(0))]:
        h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
        h += h.conjugate().transpose()
        b[site] = h
    for b, hopp in [(system, (chain(0), chain(1))),
                    (left_lead, (chain(0), chain(1))),
                    (right_lead, (chain(0), chain(1)))]:
        b[hopp] = (10 * rng.random_sample((n, n)) +
                   1j * rng.random_sample((n, n)))
    system.attach_lead(left_lead)
    system.attach_lead(right_lead)
    fsyst = system.finalized()

    t = smatrix(fsyst, 0, (), [1], [0]).data
    eig_should_be = np.linalg.eigvals(t * t.conjugate().transpose())
    n_eig = len(eig_should_be)

    def check_fsyst(fsyst):
        sol = greens_function(fsyst, 0, (), [1], [0])
        ttdagnew = sol._a_ttdagger_a_inv(1, 0)
        eig_are = np.linalg.eigvals(ttdagnew)
        t_should_be = np.sum(eig_are)
        assert_almost_equal(eig_are.imag, 0)
        assert_almost_equal(np.sort(eig_are.real)[-n_eig:],
                            np.sort(eig_should_be.real))
        assert_almost_equal(t_should_be, sol.transmission(1, 0))

    fsyst.leads[1] = LeadWithOnlySelfEnergy(fsyst.leads[1])
    check_fsyst(fsyst)

    fsyst.leads[0] = LeadWithOnlySelfEnergy(fsyst.leads[0])
    check_fsyst(fsyst)

    fsyst = system.finalized()
    for syst in (fsyst, fsyst.precalculate(what='selfenergy'),
                fsyst.precalculate(what='all')):
        check_fsyst(syst)
    raises(ValueError, check_fsyst, fsyst.precalculate(what='modes'))


def test_selfenergy_reflection(greens_function, smatrix):
    rng = ensure_rng(4)
    system = kwant.Builder()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                 (left_lead, chain(0))]:
        h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
        h += h.conjugate().transpose()
        b[site] = h
    for b, hopp in [(system, (chain(0), chain(1))),
                    (left_lead, (chain(0), chain(1)))]:
        b[hopp] = (10 * rng.random_sample((n, n)) +
                   1j * rng.random_sample((n, n)))
    system.attach_lead(left_lead)
    fsyst = system.finalized()

    t = smatrix(fsyst, 0, (), [0], [0])

    fsyst.leads[0] = LeadWithOnlySelfEnergy(fsyst.leads[0])
    sol = greens_function(fsyst, 0, (), [0], [0])
    assert_almost_equal(sol.transmission(0,0), t.transmission(0,0))

    fsyst = system.finalized()
    for syst in (fsyst.precalculate(what='selfenergy'),
                fsyst.precalculate(what='all')):
        sol = greens_function(syst, 0, (), [0], [0])
        assert_almost_equal(sol.transmission(0,0), t.transmission(0,0))
    raises(ValueError, greens_function, fsyst.precalculate(what='modes'),
                  0, (), [0], [0])


def test_very_singular_leads(smatrix):
    syst = kwant.Builder()
    chain = kwant.lattice.chain()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    right_lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    syst[chain(0)] = left_lead[chain(0)] = right_lead[chain(0)] = np.identity(2)
    left_lead[chain(0), chain(1)] = np.zeros((2, 2))
    right_lead[chain(0), chain(1)] = np.identity(2)
    syst.attach_lead(left_lead)
    syst.attach_lead(right_lead)
    fsyst = syst.finalized()
    leads = smatrix(fsyst).lead_info
    assert [len(i.momenta) for i in leads] == [0, 4]


def test_ldos(ldos):
    syst = kwant.Builder()
    chain = kwant.lattice.chain()
    lead = kwant.Builder(kwant.TranslationalSymmetry(chain.vec((1,))))
    syst[chain(0)] = syst[chain(1)] = lead[chain(0)] = 0
    syst[chain(0), chain(1)] = lead[chain(0), chain(1)] = 1
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    fsyst = syst.finalized()

    for finsyst in (fsyst, fsyst.precalculate(what='modes'),
                   fsyst.precalculate(what='all')):
        assert_almost_equal(ldos(finsyst, 0),
                            np.array([1, 1]) / (2 * np.pi))
    raises(ValueError, ldos, fsyst.precalculate(what='selfenergy'), 0)
    fsyst.leads[0] = LeadWithOnlySelfEnergy(fsyst.leads[0])
    raises(NotImplementedError, ldos, fsyst, 0)


def test_wavefunc_ldos_consistency(wave_function, ldos):
    L = 2
    W = 3

    rng = ensure_rng(31)
    syst = kwant.Builder()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    top_lead = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    for b, sites in [(syst, [square(x, y)
                               for x in range(L) for y in range(W)]),
                     (left_lead, [square(0, y) for y in range(W)]),
                     (top_lead, [square(x, 0) for x in range(L)])]:
        for site in sites:
            h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
            h += h.conjugate().transpose()
            b[site] = h
        for hopping_kind in square.neighbors():
            for hop in hopping_kind(b):
                b[hop] = (10 * rng.random_sample((n, n)) +
                          1j * rng.random_sample((n, n)))
    syst.attach_lead(left_lead)
    syst.attach_lead(top_lead)
    syst = syst.finalized()

    def check(syst):
        for energy in [0, 1000]:
            wf = wave_function(syst, energy)
            ldos2 = np.zeros(wf.num_orb, float)
            for lead in range(len(syst.leads)):
                temp = abs(wf(lead))
                temp **= 2
                ldos2 += temp.sum(axis=0)
            ldos2 *= (0.5 / np.pi)

            assert_almost_equal(ldos2, ldos(syst, energy))

    for fsyst in (syst, syst.precalculate(what='modes'),
                 syst.precalculate(what='all')):
        check(fsyst)
    raises(ValueError, check, syst.precalculate(what='selfenergy'))
    syst.leads[0] = LeadWithOnlySelfEnergy(syst.leads[0])
    raises(NotImplementedError, check, syst)


def test_arg_passing(wave_function, ldos, smatrix):

    def onsite(site, a, b):
        return site.pos[0] + site.pos[1] + a + b

    def hopping(site1, site2, a, b):
        return b - a

    W = 3
    L = 4

    syst = kwant.Builder()
    syst[(square(i, j) for i in range(L) for j in range(W))] = onsite
    syst[square.neighbors()] = hopping

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[(square(0, j) for j in range(W))] = onsite
    lead[square.neighbors()] = hopping

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    fsyst = syst.finalized()

    # compare results to those when we pass `args` only
    args = (1, 3)
    params = dict(a=1, b=3)
    np.testing.assert_array_equal(
        wave_function(fsyst, args=args)(0),
        wave_function(fsyst, params=params)(0))
    np.testing.assert_array_equal(
        ldos(fsyst, args=args),
        ldos(fsyst, params=params))
    np.testing.assert_array_equal(
        smatrix(fsyst, args=args).data,
        smatrix(fsyst, params=params).data)
