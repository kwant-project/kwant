# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from __future__ import division
import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_equal, assert_almost_equal
import kwant

n = 5
chain = kwant.lattice.chain()
sq = square = kwant.lattice.square()


# Test output sanity: that an error is raised if no output is requested,
# and that solving for a subblock of a scattering matrix is the same as taking
# a subblock of the full scattering matrix.
def test_output(solve):
    np.random.seed(3)
    system = kwant.Builder()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    right_lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                    (left_lead, chain(0)), (right_lead, chain(0))]:
        h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        h += h.conjugate().transpose()
        b[site] = h
    for b, hopp in [(system, (chain(0), chain(1))),
                    (left_lead, (chain(0), chain(1))),
                    (right_lead, (chain(0), chain(1)))]:
        b[hopp] = 10 * np.random.rand(n, n) + 1j * np.random.rand(n, n)
    system.attach_lead(left_lead)
    system.attach_lead(right_lead)
    fsys = system.finalized()

    result1 = solve(fsys)
    s, modes1 = result1.data, result1.lead_info
    assert s.shape == 2 * (sum(i[2] for i in modes1),)
    s1 = result1.submatrix(1, 0)
    result2 = solve(fsys, 0, [1], [0])
    s2, modes2 = result2.data, result2.lead_info
    assert s2.shape == (modes2[1][2], modes2[0][2])
    assert_almost_equal(s1, s2)
    assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                        np.identity(s.shape[0]))
    assert_raises(ValueError, solve, fsys, 0, [])
    modes = solve(fsys).lead_info
    h = fsys.leads[0].slice_hamiltonian()
    t = fsys.leads[0].inter_slice_hopping()
    modes1 = kwant.physics.modes(h, t)
    h = fsys.leads[1].slice_hamiltonian()
    t = fsys.leads[1].inter_slice_hopping()
    modes2 = kwant.physics.modes(h, t)
    assert_almost_equal(modes1[0], modes[0][0])
    assert_almost_equal(modes2[1], modes[1][1])


# Test that a system with one lead has unitary scattering matrix.
def test_one_lead(solve):
    np.random.seed(3)
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                    (system, chain(2)), (lead, chain(0))]:
        h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        h += h.conjugate().transpose()
        b[site] = h
    for b, hopp in [(system, (chain(0), chain(1))),
                    (system, (chain(1), chain(2))),
                    (lead, (chain(0), chain(1)))]:
        b[hopp] = 10 * np.random.rand(n, n) + 1j * np.random.rand(n, n)
    system.attach_lead(lead)
    fsys = system.finalized()

    s = solve(fsys).data
    assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                        np.identity(s.shape[0]))


# Test that a system with one lead with no propagating modes has a
# 0x0 S-matrix.
def test_smatrix_shape(solve):
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
    fsys = system.finalized()

    lead0_val = 4
    lead1_val = 4
    s = solve(fsys, energy=1.0, out_leads=[1], in_leads=[0]).data
    assert s.shape == (0, 0)

    lead0_val = 2
    lead1_val = 2
    s = solve(fsys, energy=1.0, out_leads=[1], in_leads=[0]).data
    assert s.shape == (1, 1)

    lead0_val = 4
    lead1_val = 2
    s = solve(fsys, energy=1.0, out_leads=[1], in_leads=[0]).data
    assert s.shape == (1, 0)

    lead0_val = 2
    lead1_val = 4
    s = solve(fsys, energy=1.0, out_leads=[1], in_leads=[0]).data
    assert s.shape == (0, 1)


# Test that a translationally invariant system with two leads has only
# transmission and that transmission does not mix modes.
def test_two_equal_leads(solve):
    def check_fsys():
        sol = solve(fsys)
        s, leads = sol.data, sol.lead_info
        assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                            np.identity(s.shape[0]))
        n_modes = leads[0][2]
        assert leads[1][2] == n_modes
        assert_almost_equal(s[: n_modes, : n_modes], 0)
        t_elements = np.sort(abs(np.asarray(s[n_modes :, : n_modes])),
                             axis=None)
        t_el_should_be = n_modes * (n_modes - 1) * [0] + n_modes * [1]
        assert_almost_equal(t_elements, t_el_should_be)
        assert_almost_equal(sol.transmission(1,0), n_modes)
    np.random.seed(11)
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    h += h.conjugate().transpose()
    h *= 0.8
    t = 4 * np.random.rand(n, n) + 4j * np.random.rand(n, n)
    lead[chain(0)] = system[chain(0)] = h
    lead[chain(0), chain(1)] = t
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    fsys = system.finalized()
    check_fsys()

    # Test the same, but with a larger scattering region.
    system = kwant.Builder()
    system[[chain(0), chain(1)]] = h
    system[chain(0), chain(1)] = t
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    fsys = system.finalized()
    check_fsys()


# Test a more complicated graph with non-singular hopping.
def test_graph_system(solve):
    np.random.seed(11)
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    h += h.conjugate().transpose()
    h *= 0.8
    t = 4 * np.random.rand(n, n) + 4j * np.random.rand(n, n)
    t1 = 4 * np.random.rand(n, n) + 4j * np.random.rand(n, n)
    lead[sq(0, 0)] = system[[sq(0, 0), sq(1, 0)]] = h
    lead[sq(0, 1)] = system[[sq(0, 1), sq(1, 1)]] = 4 * h
    for builder in [system, lead]:
        builder[sq(0, 0), sq(1, 0)] = t
        builder[sq(0, 1), sq(1, 0)] = t1
        builder[sq(0, 1), sq(1, 1)] = 1.1j * t1
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    fsys = system.finalized()

    result = solve(fsys)
    s, leads = result.data, result.lead_info
    assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                        np.identity(s.shape[0]))
    n_modes = leads[0][2]
    assert_equal(leads[1][2], n_modes)
    assert_almost_equal(s[: n_modes, : n_modes], 0)
    t_elements = np.sort(abs(np.asarray(s[n_modes:, :n_modes])),
                         axis=None)
    t_el_should_be = n_modes * (n_modes - 1) * [0] + n_modes * [1]
    assert_almost_equal(t_elements, t_el_should_be)


# Test a system with singular hopping.
def test_singular_graph_system(solve):
    np.random.seed(11)

    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    h += h.conjugate().transpose()
    h *= 0.8
    t = 4 * np.random.rand(n, n) + 4j * np.random.rand(n, n)
    t1 = 4 * np.random.rand(n, n) + 4j * np.random.rand(n, n)
    lead[sq(0, 0)] = system[[sq(0, 0), sq(1, 0)]] = h
    lead[sq(0, 1)] = system[[sq(0, 1), sq(1, 1)]] = 4 * h
    for builder in [system, lead]:
        builder[sq(0, 0), sq(1, 0)] = t
        builder[sq(0, 1), sq(1, 0)] = t1
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    fsys = system.finalized()

    result = solve(fsys)
    s, leads = result.data, result.lead_info
    assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                        np.identity(s.shape[0]))
    n_modes = leads[0][2]
    assert leads[1][2] == n_modes
    assert_almost_equal(s[: n_modes, : n_modes], 0)
    t_elements = np.sort(abs(np.asarray(s[n_modes :, : n_modes])),
                         axis=None)
    t_el_should_be = n_modes * (n_modes - 1) * [0] + n_modes * [1]
    assert_almost_equal(t_elements, t_el_should_be)


# This test features inside the onslice Hamiltonian a hopping matrix with more
# zero eigenvalues than the lead hopping matrix. Older version of the
# sparse solver failed here.
def test_tricky_singular_hopping(solve):
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((4, 0)))

    interface = []
    for i in xrange(n):
        site = sq(-1, i)
        interface.append(site)
        system[site] = 0
        for j in xrange(4):
            lead[sq(j, i)] = 0
    for i in xrange(n-1):
        system[sq(-1, i), sq(-1, i+1)] = -1
        for j in xrange(4):
            lead[sq(j, i), sq(j, i+1)] = -1
    for i in xrange(n):
        for j in xrange(4):
            lead[sq(j, i), sq(j+1, i)] = -1
    del lead[sq(1, 0), sq(2, 0)]

    system.leads.append(kwant.builder.BuilderLead(lead, interface))
    fsys = system.finalized()

    s = solve(fsys, -1.3).data
    assert_almost_equal(np.dot(s.conjugate().transpose(), s),
                        np.identity(s.shape[0]))


# Test equivalence between self-energy and scattering matrix representations.
# Also check that transmission works.
def test_self_energy(solve):
    class LeadWithOnlySelfEnergy(object):
        def __init__(self, lead):
            self.lead = lead

        def self_energy(self, energy):
            return self.lead.self_energy(energy)

    np.random.seed(4)
    system = kwant.Builder()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    right_lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                 (left_lead, chain(0)), (right_lead, chain(0))]:
        h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        h += h.conjugate().transpose()
        b[site] = h
    for b, hopp in [(system, (chain(0), chain(1))),
                    (left_lead, (chain(0), chain(1))),
                    (right_lead, (chain(0), chain(1)))]:
        b[hopp] = 10 * np.random.rand(n, n) + 1j * np.random.rand(n, n)
    system.attach_lead(left_lead)
    system.attach_lead(right_lead)
    fsys = system.finalized()

    t = solve(fsys, 0, [1], [0]).data
    eig_should_be = np.linalg.eigvals(t * t.conjugate().transpose())
    n_eig = len(eig_should_be)

    fsys.leads[1] = LeadWithOnlySelfEnergy(fsys.leads[1])
    sol = solve(fsys, 0, [1], [0])
    ttdagnew = sol._a_ttdagger_a_inv(1, 0)
    eig_are = np.linalg.eigvals(ttdagnew)
    t_should_be = np.sum(eig_are)
    assert_almost_equal(eig_are.imag, 0)
    assert_almost_equal(np.sort(eig_are.real)[-n_eig:],
                        np.sort(eig_should_be.real))
    assert_almost_equal(t_should_be, sol.transmission(1, 0))

    fsys.leads[0] = LeadWithOnlySelfEnergy(fsys.leads[0])
    sol = solve(fsys, 0, [1], [0])
    ttdagnew = sol._a_ttdagger_a_inv(1, 0)
    eig_are = np.linalg.eigvals(ttdagnew)
    t_should_be = np.sum(eig_are)
    assert_almost_equal(eig_are.imag, 0)
    assert_almost_equal(np.sort(eig_are.real)[-n_eig :],
                        np.sort(eig_should_be.real))
    assert_almost_equal(t_should_be, sol.transmission(1, 0))


def test_self_energy_reflection(solve):
    class LeadWithOnlySelfEnergy(object):
        def __init__(self, lead):
            self.lead = lead

        def self_energy(self, energy):
            return self.lead.self_energy(energy)

    np.random.seed(4)
    system = kwant.Builder()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    for b, site in [(system, chain(0)), (system, chain(1)),
                 (left_lead, chain(0))]:
        h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
        h += h.conjugate().transpose()
        b[site] = h
    for b, hopp in [(system, (chain(0), chain(1))),
                    (left_lead, (chain(0), chain(1)))]:
        b[hopp] = 10 * np.random.rand(n, n) + 1j * np.random.rand(n, n)
    system.attach_lead(left_lead)
    fsys = system.finalized()

    t = solve(fsys, 0, [0], [0])

    fsys.leads[0] = LeadWithOnlySelfEnergy(fsys.leads[0])
    sol = solve(fsys, 0, [0], [0])

    assert_almost_equal(sol.transmission(0,0), t.transmission(0,0))


def test_very_singular_leads(solve):
    sys = kwant.Builder()
    gr = kwant.lattice.chain()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    right_lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    sys[gr(0)] = left_lead[gr(0)] = right_lead[gr(0)] = np.identity(2)
    left_lead[gr(0), gr(1)] = np.zeros((2, 2))
    right_lead[gr(0), gr(1)] = np.identity(2)
    sys.attach_lead(left_lead)
    sys.attach_lead(right_lead)
    fsys = sys.finalized()
    leads = solve(fsys).lead_info
    assert [i[2] for i in leads] == [0, 2]


def test_ldos(ldos):
    sys = kwant.Builder()
    gr = kwant.lattice.chain()
    lead = kwant.Builder(kwant.TranslationalSymmetry(gr.vec((1,))))
    sys.default_site_group = lead.default_site_group = gr
    sys[gr(0)] = sys[gr(1)] = lead[gr(0)] = 0
    sys[gr(0), gr(1)] = lead[gr(0), gr(1)] = 1
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())
    fsys = sys.finalized()
    assert_almost_equal(ldos(fsys, 0),
                        np.array([1, 1]) / (2 * np.pi))


def test_wavefunc_ldos_consistency(wave_func, ldos):
    L = 2
    W = 3

    np.random.seed(31)
    sys = kwant.Builder()
    left_lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    top_lead = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    for b, sites in [(sys, [square(x, y)
                               for x in range(L) for y in range(W)]),
                     (left_lead, [square(0, y) for y in range(W)]),
                     (top_lead, [square(x, 0) for x in range(L)])]:
        for site in sites:
            h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
            h += h.conjugate().transpose()
            b[site] = h
        for kind in square.nearest:
            for hop in b.possible_hoppings(*kind):
                b[hop] = 10 * np.random.rand(n, n) + 1j * np.random.rand(n, n)
    sys.attach_lead(left_lead)
    sys.attach_lead(top_lead)
    sys = sys.finalized()

    for energy in [0, 1000]:
        wf = wave_func(sys, energy)
        ldos2 = np.zeros(wf.num_orb, float)
        for lead in xrange(len(sys.leads)):
            temp = abs(wf(lead))
            temp **= 2
            ldos2 += temp.sum(axis=0)
        ldos2 *= (0.5 / np.pi)

        assert_almost_equal(ldos2, ldos(sys, energy))
