# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np
from pytest import raises
from numpy.testing import assert_almost_equal
import kwant
from kwant.physics import two_terminal_shotnoise
from kwant._common import ensure_rng

n = 5
chain = kwant.lattice.chain()

def twoterminal_system():
    rng = ensure_rng(11)
    system = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((1,)))
    h = rng.random_sample((n, n)) + 1j * rng.random_sample((n, n))
    h += h.conjugate().transpose()
    h *= 0.8
    t = 4 * rng.random_sample((n, n)) + 4j * rng.random_sample((n, n))
    lead[chain(0)] = h
    system[chain(0)] = h * 1.2
    lead[chain(0), chain(1)] = t
    system.attach_lead(lead)
    system.attach_lead(lead.reversed())
    return system


def test_multiterminal_input():
    """Input checks for two_terminal_shotnoise"""

    syst = twoterminal_system()
    syst.attach_lead(syst.leads[0].builder)
    sol = kwant.smatrix(syst.finalized(), out_leads=[0], in_leads=[0])
    raises(ValueError, two_terminal_shotnoise, sol)


def test_twoterminal():
    """Shot noise in a two-terminal conductor"""

    fsyst = twoterminal_system().finalized()

    sol = kwant.smatrix(fsyst)
    t = sol.submatrix(1, 0)
    Tn = np.linalg.eigvalsh(np.dot(t, t.conj().T))
    noise_should_be = np.sum(Tn * (1 - Tn))

    assert_almost_equal(noise_should_be, two_terminal_shotnoise(sol))
