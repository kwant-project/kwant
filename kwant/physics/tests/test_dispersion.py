# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from numpy.testing import assert_array_almost_equal, assert_almost_equal
from pytest import raises
from numpy import linspace

import kwant
from math import pi, cos, sin


def test_band_energies(N=5):
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lat = kwant.lattice.square()
    syst[[lat(0, 0), lat(0, 1)]] = 3
    syst[lat(0, 1), lat(0, 0)] = -1
    syst[((lat(1, y), lat(0, y)) for y in range(2))] = -1

    band_energies = kwant.physics.Bands(syst.finalized())
    for i in range(-N, N):
        k = i * pi / N
        energies = band_energies(k)
        assert_array_almost_equal(sorted(energies),
                                  sorted([2 - 2 * cos(k), 4 - 2 * cos(k)]))


def test_same_as_lead():
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lat = kwant.lattice.chain()
    syst[lat(0)] = 0
    syst[lat(0), lat(1)] = complex(cos(0.2), sin(0.2))

    syst = syst.finalized()
    momenta = syst.modes()[0].momenta
    bands = kwant.physics.Bands(syst)

    for momentum in momenta:
        assert_almost_equal(bands(momentum)[0], 0)


def test_raise_nonhermitian():
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lat = kwant.lattice.chain()
    syst[lat(0)] = 1j
    syst[lat(0), lat(1)] = complex(cos(0.2), sin(0.2))
    syst = syst.finalized()
    raises(ValueError, kwant.physics.Bands, syst)


def test_band_velocities():
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lat = kwant.lattice.square()
    syst[lat(0, 0)] = 1
    syst[lat(0, 1)] = 3
    syst[lat(1, 0), lat(0, 0)] = -1
    syst[lat(1, 1), lat(0, 1)] = 2
    bands = kwant.physics.Bands(syst.finalized())
    eps = 1E-4
    for k in linspace(-pi, pi, 200):
        vel = bands(k, derivative_order=1)[1]
        # higher order formula for first derivative to get required accuracy
        num_vel = (- bands(k+2*eps) + bands(k-2*eps) +
                   8*(bands(k+eps) - bands(k-eps))) / (12 * eps)
        assert_array_almost_equal(vel, num_vel)


def test_band_velocity_derivative():
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lat = kwant.lattice.square()
    syst[lat(0, 0)] = 1
    syst[lat(0, 1)] = 3
    syst[lat(1, 0), lat(0, 0)] = -1
    syst[lat(1, 1), lat(0, 1)] = 2
    bands = kwant.physics.Bands(syst.finalized())
    eps = 1E-4
    eps2 = eps * eps
    c3 = 1 / 90
    c2 = - 3 / 20
    c1 = 3 / 2
    c0 = - 49 / 18
    for k in linspace(-pi, pi, 200):
        dvel = bands(k, derivative_order=2)[2]
        # higher order formula for second derivative to get required accuracy
        num_dvel = (c3 * (bands(k+3*eps) + bands(k-3*eps)) +
                    c2 * (bands(k+2*eps) + bands(k-2*eps)) +
                    c1 * (bands(k+eps) + bands(k-eps)) +
                    c0 * bands(k)) / eps2
        assert_array_almost_equal(dvel, num_dvel)


def test_raise_implemented():
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lat = kwant.lattice.square()
    syst[[lat(0, 0), lat(0, 1)]] = 3
    syst[lat(0, 1), lat(0, 0)] = -1
    syst[((lat(1, y), lat(0, y)) for y in range(2))] = -1
    bands = kwant.physics.Bands(syst.finalized())
    assert bands(1.).shape == (2,)
    assert bands(1., derivative_order=0).shape == (2,)
    assert bands(1., derivative_order=1).shape == (2, 2)
    assert bands(1., derivative_order=2).shape == (3, 2)
    raises(NotImplementedError, bands, 1., derivative_order=-1)
    raises(NotImplementedError, bands, 1., derivative_order=3)
