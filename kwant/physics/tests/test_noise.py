# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np
from nose.tools import assert_raises
from numpy.testing import assert_almost_equal
import kwant
from kwant.physics.noise import two_terminal_shotnoise

n = 5
chain = kwant.lattice.chain()

def _twoterminal_system():
    np.random.seed(11)
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    h = np.random.rand(n, n) + 1j * np.random.rand(n, n)
    h += h.conjugate().transpose()
    h *= 0.8
    t = 4 * np.random.rand(n, n) + 4j * np.random.rand(n, n)
    lead[chain(0)] = h
    system[chain(0)] = h * 1.2
    lead[chain(0), chain(1)] = t
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    return system.finalized()


def test_twoterminal_input():
    """Input checks for two_terminal_shotnoise"""

    fsys = _twoterminal_system()
    sol = kwant.solve(fsys, out_leads=[0], in_leads=[0])
    assert_raises(ValueError, two_terminal_shotnoise, sol)


def test_twoterminal():
    """Shot noise in a two-terminal conductor"""

    fsys = _twoterminal_system()

    sol = kwant.solve(fsys)
    t = sol.submatrix(1, 0)
    Tn = np.linalg.eigvalsh(np.dot(t, t.conj().T))
    noise_should_be = np.sum(Tn * (1 - Tn))

    assert_almost_equal(noise_should_be, two_terminal_shotnoise(sol))

    # replace leads successively with self-energy
    class LeadWithOnlySelfEnergy(object):
        def __init__(self, lead):
            self.lead = lead

        def self_energy(self, energy):
            return self.lead.self_energy(energy)

    fsys.leads[0] = LeadWithOnlySelfEnergy(fsys.leads[0])
    sol = kwant.solve(fsys)
    assert_almost_equal(noise_should_be, two_terminal_shotnoise(sol))

    fsys.leads[1] = LeadWithOnlySelfEnergy(fsys.leads[1])
    sol = kwant.solve(fsys)
    assert_almost_equal(noise_should_be, two_terminal_shotnoise(sol))
