# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from __future__ import division
from math import sqrt
import numpy as np
import tinyarray as ta
from nose.tools import assert_raises, assert_not_equal
from numpy.testing import assert_equal
from kwant import lattice


def test_general():
    for lat in (lattice.general(((1, 0), (0.5, 0.5))),
                lattice.general(((1, 0), (0.5, sqrt(3)/2)),
                                     ((0, 0), (0, 1/sqrt(3))))):
        for sl in lat.sublattices:
            tag = (-5, 33)
            site = sl(*tag)
            assert_equal(tag, sl.closest(site.pos))

    # Test 2D lattice with 1 vector.
    lat = lattice.make_lattice([[1, 0]])
    site = lat(0)
    assert_raises(ValueError, lat, 0, 1)


def test_shape():
    def in_circle(pos):
        return pos[0] ** 2 + pos[1] ** 2 < 3

    lat = lattice.general(((1, 0), (0.5, sqrt(3) / 2)),
                               ((0, 0), (0, 1 / sqrt(3))))
    sites = list(lat.shape(in_circle, (0, 0)))
    sites_alt = list()
    sl0, sl1 = lat.sublattices
    for x in xrange(-2, 3):
        for y in xrange(-2, 3):
            tag = (x, y)
            for site in (sl0(*tag), sl1(*tag)):
                if in_circle(site.pos):
                    sites_alt.append(site)
    assert len(sites) == len(sites_alt)
    assert_equal(set(sites), set(sites_alt))
    assert_raises(ValueError, lat.shape(in_circle, (10, 10)).next)


def test_translational_symmetry():
    ts = lattice.TranslationalSymmetry
    g2 = lattice.general(np.identity(2))
    g3 = lattice.general(np.identity(3))
    shifted = lambda site, delta: site.group(*ta.add(site.tag, delta))

    sym = ts((0, 0, 4), (0, 5, 0), (0, 0, 2))
    assert_raises(ValueError, sym.add_site_group, g3)
    sym = ts((3.3, 0))
    assert_raises(ValueError, sym.add_site_group, g2)

    # Test lattices with dimension smaller than dimension of space.
    g2in3 = lattice.general([[4, 4, 0], [4, -4, 0]])
    sym = ts((8, 0, 0))
    sym.add_site_group(g2in3)
    sym = ts((8, 0, 1))
    assert_raises(ValueError, sym.add_site_group, g2in3)

    # Test automatic fill-in of transverse vectors.
    sym = ts((1, 2))
    sym.add_site_group(g2)
    assert_not_equal(sym.site_group_data[g2.canonical_repr][2], 0)
    sym = ts((1, 0, 2), (3, 0, 2))
    sym.add_site_group(g3)
    assert_not_equal(sym.site_group_data[g3.canonical_repr][2], 0)

    transl_vecs = np.array([[10, 0], [7, 7]], dtype=int)
    sym = ts(*transl_vecs)
    assert_equal(sym.num_directions, 2)
    sym2 = ts(*transl_vecs[: 1, :])
    sym2.add_site_group(g2, transl_vecs[1:, :])
    for site in [g2(0, 0), g2(4, 0), g2(2, 1), g2(5, 5), g2(15, 6)]:
        assert sym.in_fd(site)
        assert sym2.in_fd(site)
        assert_equal(sym.which(site), (0, 0))
        assert_equal(sym2.which(site), (0,))
        for v in [(1, 0), (0, 1), (-1, 0), (0, -1), (5, 10), (-111, 573)]:
            site2 = shifted(site, np.dot(v, transl_vecs))
            assert not sym.in_fd(site2)
            assert (v[0] != 0) != sym2.in_fd(site2)
            assert_equal(sym.to_fd(site2), site)
            assert (v[1] == 0) == (sym2.to_fd(site2) == site)
            assert_equal(sym.which(site2), v)
            assert_equal(sym2.which(site2), v[:1])

            for hop in [(0, 0), (100, 0), (0, 5), (-2134, 3213)]:
                assert_equal(sym.to_fd(site2, shifted(site2, hop)),
                             (site, shifted(site, hop)))

    # Test act for hoppings belonging to different lattices.
    g2p = lattice.general(2 * np.identity(2))
    sym = ts(*(2 * np.identity(2)))
    assert sym.act((1, 1), g2(0, 0), g2p(0, 0)) == (g2(2, 2), g2p(1, 1))
    assert sym.act((1, 1), g2p(0, 0), g2(0, 0)) == (g2p(1, 1), g2(2, 2))


def test_translational_symmetry_reversed():
    def assert_equal_symmetry(a, b):
        np.testing.assert_array_almost_equal(a.periods, b.periods)
        for i in a.site_group_data:
            assert i in b.site_group_data
            data = zip(b.site_group_data[i], a.site_group_data[i])
            for j in data:
                assert np.array_equal(j[0], j[1])

    params = [([(3,)], None),
              ([(0, -1)], None),
              ([(1, 1)], None),
              ([(1, 1)], [(-1, 0)]),
              ([(3, 1, 1), (1, 4, 1), (1, 1, 5)], []),
              ([(3, 1, 1), (1, 4, 1)], [(1, 1, 5)]),
              ([(3, 1, 1)], [(1, 4, 1), (1, 1, 5)])]
    for periods, other_vectors in params:
        sym = lattice.TranslationalSymmetry(*periods)
        gr = lattice.general(np.identity(len(periods[0])))
        sym.add_site_group(gr, other_vectors)
        rsym = sym.reversed()
        assert_equal_symmetry(sym, rsym.reversed())
        rperiods = -np.array(periods, dtype=int)
        rsym2 = lattice.TranslationalSymmetry(*rperiods)
        rsym2.add_site_group(gr, other_vectors)
        assert_equal_symmetry(rsym, rsym2)


def test_monatomic_lattice():
    lat = lattice.square()
    lat2 = lattice.general(np.identity(2))
    lat3 = lattice.square(name='no')
    assert len(set([lat, lat2, lat3, lat(0, 0), lat2(0, 0), lat3(0, 0)])) == 4
