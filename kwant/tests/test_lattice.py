from __future__ import division
from math import sqrt
import numpy as np
from nose.tools import assert_raises, assert_not_equal
from numpy.testing import assert_equal
from kwant import lattice, builder


def test_make_lattice():
    for lat in (lattice.make_lattice(((1, 0), (0.5, 0.5))),
                lattice.make_lattice(((1, 0), (0.5, sqrt(3)/2)),
                                     ((0, 0), (0, 1/sqrt(3))))):
        for sl in lat.sublattices:
            tag = (-5, 33)
            site = sl(*tag)
            assert_equal(tag, sl.closest(site.pos))


def test_pack_unpack():
    for dim in [1, 2, 3, 5, 10, 99]:
        group = lattice.make_lattice(np.identity(dim))
        group_by_pgid = {group.packed_group_id : group}
        tag = tuple(xrange(dim))
        site = group(*tag)
        psite = site.packed()
        same_site = builder.unpack(psite, group_by_pgid)
        assert_equal(same_site, site)


def test_shape():
    def in_circle(pos):
        return pos[0]**2 + pos[1]**2 < 3

    lat = lattice.make_lattice(((1, 0), (0.5, sqrt(3)/2)),
                               ((0, 0), (0, 1/sqrt(3))))
    sites = set(lat.shape(in_circle, (0, 0)))
    sites_alt = set()
    sl0, sl1 = lat.sublattices
    for x in xrange(-2, 3):
        for y in xrange(-2, 3):
            tag = (x, y)
            for site in (sl0(*tag), sl1(*tag)):
                if in_circle(site.pos):
                    sites_alt.add(site)
    assert_equal(sites, sites_alt)
    assert_raises(ValueError, lat.shape(in_circle, (10, 10)).next)


def test_translational_symmetry():
    ts = lattice.TranslationalSymmetry
    g2 = lattice.make_lattice(np.identity(2))
    g3 = lattice.make_lattice(np.identity(3))

    sym = ts([(0, 0, 4), (0, 5, 0), (0, 0, 2)])
    assert_raises(ValueError, sym.add_site_group, g3)
    sym = ts([(3.3, 0)])
    assert_raises(ValueError, sym.add_site_group, g2)

    # Test lattices with dimension smaller than dimension of space.

    g2in3 = lattice.make_lattice([[4, 4, 0], [4, -4, 0]])
    sym = ts([(8, 0, 0)])
    sym.add_site_group(g2in3)
    sym = ts([(8, 0, 1)])
    assert_raises(ValueError, sym.add_site_group, g2in3)

    # Test automatic fill-in of transverse vectors.
    sym = ts([(1, 2)])
    sym.add_site_group(g2)
    assert_not_equal(sym.site_group_data[g2][2], 0)
    sym = ts([(1, 0, 2), (3, 0, 2)])
    sym.add_site_group(g3)
    assert_not_equal(sym.site_group_data[g3][2], 0)

    transl_vecs = np.array([[10, 0], [7, 7]], dtype=int)
    sym = ts(transl_vecs)
    assert_equal(sym.num_directions, 2)
    sym2 = ts(transl_vecs[: 1, :])
    sym2.add_site_group(g2, transl_vecs[1:, :])
    for site in [g2(0, 0), g2(4, 0), g2(2, 1), g2(5, 5), g2(15, 6)]:
        assert sym.in_fd(site)
        assert sym2.in_fd(site)
        assert_equal(sym.which(site), (0, 0))
        assert_equal(sym2.which(site), (0,))
        for v in [(1, 0), (0, 1), (-1, 0), (0, -1), (5, 10), (-111, 573)]:
            site2 = site.shifted(np.dot(v, transl_vecs))
            assert not sym.in_fd(site2)
            assert (v[0] != 0) != sym2.in_fd(site2)
            assert_equal(sym.to_fd(site2), site)
            assert (v[1] == 0) == (sym2.to_fd(site2) == site)
            assert_equal(sym.which(site2), v)
            assert_equal(sym2.which(site2), v[:1])

            for hop in [(0, 0), (100, 0), (0, 5), (-2134, 3213)]:
                assert_equal(sym.to_fd(site2, site2.shifted(hop)),
                             (site, site.shifted(hop)))


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
        sym = lattice.TranslationalSymmetry(periods)
        gr = lattice.make_lattice(np.identity(len(periods[0])))
        sym.add_site_group(gr, other_vectors)
        rsym = sym.reversed()
        assert_equal_symmetry(sym, rsym.reversed())
        rperiods = -np.array(periods, dtype=int)
        rsym2 = lattice.TranslationalSymmetry(rperiods)
        rsym2.add_site_group(gr, other_vectors)
        assert_equal_symmetry(rsym, rsym2)
