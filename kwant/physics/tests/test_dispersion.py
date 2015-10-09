# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from numpy.testing import assert_array_almost_equal, assert_almost_equal
from nose.tools import assert_raises

import kwant
from math import pi, cos, sin

def test_band_energies(N=5):
    sys = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lat = kwant.lattice.square()
    sys[[lat(0, 0), lat(0, 1)]] = 3
    sys[lat(0, 1), lat(0, 0)] = -1
    sys[((lat(1, y), lat(0, y)) for y in range(2))] = -1

    band_energies = kwant.physics.Bands(sys.finalized())
    for i in range(-N, N):
        k = i * pi / N
        energies = band_energies(k)
        assert_array_almost_equal(sorted(energies),
                                  sorted([2 - 2 * cos(k), 4 - 2 * cos(k)]))

def test_same_as_lead():
    sys = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lat = kwant.lattice.chain()
    sys[lat(0)] = 0
    sys[lat(0), lat(1)] = complex(cos(0.2), sin(0.2))

    sys = sys.finalized()
    momenta = sys.modes()[0].momenta
    bands = kwant.physics.Bands(sys)

    for momentum in momenta:
        assert_almost_equal(bands(momentum)[0], 0)

def test_raise_nonhermitian():
    sys = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lat = kwant.lattice.chain()
    sys[lat(0)] = 1j
    sys[lat(0), lat(1)] = complex(cos(0.2), sin(0.2))
    sys = sys.finalized()
    assert_raises(ValueError, kwant.physics.Bands, sys)
