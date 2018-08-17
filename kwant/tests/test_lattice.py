# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from math import sqrt
import numpy as np
import tinyarray as ta
from pytest import raises
from kwant import lattice, builder
from kwant._common import ensure_rng
import pytest


def test_closest():
    rng = ensure_rng(4)
    lat = lattice.general(((1, 0), (0.5, sqrt(3)/2)))
    for i in range(50):
        point = 20 * rng.random_sample(2)
        closest = lat(*lat.closest(point)).pos
        assert np.linalg.norm(point - closest) <= 1 / sqrt(3)
    lat = lattice.general(rng.randn(3, 3))
    for i in range(50):
        tag = rng.randint(10, size=(3,))
        assert lat.closest(lat(*tag).pos) == tag


def test_general():
    for lat in (lattice.general(((1, 0), (0.5, 0.5))),
                lattice.general(((1, 0), (0.5, sqrt(3)/2)),
                                     ((0, 0), (0, 1/sqrt(3))))):
        for sl in lat.sublattices:
            tag = (-5, 33)
            site = sl(*tag)
            assert tag == sl.closest(site.pos)

    # Test 2D lattice with 1 vector.
    lat = lattice.general([[1, 0]])
    site = lat(0)
    raises(ValueError, lat, 0, 1)


def test_neighbors():
    lat = lattice.honeycomb(1e-10)
    num_nth_nearest = [len(lat.neighbors(n)) for n in range(5)]
    assert num_nth_nearest == [2, 3, 6, 3, 6]
    lat = lattice.general([(0, 1e8, 0, 0), (0, 0, 1e8, 0)])
    num_nth_nearest = [len(lat.neighbors(n)) for n in range(5)]
    assert num_nth_nearest == [1, 2, 2, 2, 4]
    lat = lattice.chain(1e-10)
    num_nth_nearest = [len(lat.neighbors(n)) for n in range(5)]
    assert num_nth_nearest == 5 * [1]


def test_shape():
    def in_circle(pos):
        return pos[0] ** 2 + pos[1] ** 2 < 3

    lat = lattice.honeycomb()
    sites = list(lat.shape(in_circle, (0, 0))())
    sites_alt = list()
    sl0, sl1 = lat.sublattices
    for x in range(-2, 3):
        for y in range(-2, 3):
            tag = (x, y)
            for site in (sl0(*tag), sl1(*tag)):
                if in_circle(site.pos):
                    sites_alt.append(site)
    assert len(sites) == len(sites_alt)
    assert set(sites) == set(sites_alt)
    raises(ValueError, lat.shape(in_circle, (10, 10))().__next__)

    # Check if narrow ribbons work.
    for period in (0, 1), (1, 0), (1, -1):
        vec = lat.vec(period)
        sym = lattice.TranslationalSymmetry(vec)
        def shape(pos):
            return abs(pos[0] * vec[1] - pos[1] * vec[0]) < 10
        sites = list(lat.shape(shape, (0, 0))(sym))
        assert len(sites) > 35


def test_wire():
    rng = ensure_rng(5)
    vecs = rng.randn(3, 3)
    vecs[0] = [1, 0, 0]
    center = rng.randn(3)
    lat = lattice.general(vecs, rng.randn(4, 3))
    syst = builder.Builder(lattice.TranslationalSymmetry((2, 0, 0)))
    def wire_shape(pos):
        pos = np.array(pos)
        return np.linalg.norm(pos[1:] - center[1:])**2 <= 8.6**2
    syst[lat.shape(wire_shape, center)] = 0
    sites2 = set(syst.sites())
    syst = builder.Builder(lattice.TranslationalSymmetry((2, 0, 0)))
    syst[lat.wire(center, 8.6)] = 1
    sites1 = set(syst.sites())
    assert sites1 == sites2


def test_translational_symmetry():
    ts = lattice.TranslationalSymmetry
    f2 = lattice.general(np.identity(2))
    f3 = lattice.general(np.identity(3))
    shifted = lambda site, delta: site.family(*ta.add(site.tag, delta))

    raises(ValueError, ts, (0, 0, 4), (0, 5, 0), (0, 0, 2))
    sym = ts((3.3, 0))
    raises(ValueError, sym.add_site_family, f2)

    # Test lattices with dimension smaller than dimension of space.
    f2in3 = lattice.general([[4, 4, 0], [4, -4, 0]])
    sym = ts((8, 0, 0))
    sym.add_site_family(f2in3)
    sym = ts((8, 0, 1))
    raises(ValueError, sym.add_site_family, f2in3)

    # Test automatic fill-in of transverse vectors.
    sym = ts((1, 2))
    sym.add_site_family(f2)
    assert sym.site_family_data[f2][2] != 0
    sym = ts((1, 0, 2), (3, 0, 2))
    sym.add_site_family(f3)
    assert sym.site_family_data[f3][2] != 0

    transl_vecs = np.array([[10, 0], [7, 7]], dtype=int)
    sym = ts(*transl_vecs)
    assert sym.num_directions == 2
    sym2 = ts(*transl_vecs[: 1, :])
    sym2.add_site_family(f2, transl_vecs[1:, :])
    for site in [f2(0, 0), f2(4, 0), f2(2, 1), f2(5, 5), f2(15, 6)]:
        assert sym.in_fd(site)
        assert sym2.in_fd(site)
        assert sym.which(site) == (0, 0)
        assert sym2.which(site) == (0,)
        for v in [(1, 0), (0, 1), (-1, 0), (0, -1), (5, 10), (-111, 573)]:
            site2 = shifted(site, np.dot(v, transl_vecs))
            assert not sym.in_fd(site2)
            assert (v[0] != 0) != sym2.in_fd(site2)
            assert sym.to_fd(site2) == site
            assert (v[1] == 0) == (sym2.to_fd(site2) == site)
            assert sym.which(site2) == v
            assert sym2.which(site2) == v[:1]

            for hop in [(0, 0), (100, 0), (0, 5), (-2134, 3213)]:
                assert (sym.to_fd(site2, shifted(site2, hop)) ==
                        (site, shifted(site, hop)))

    # Test act for hoppings belonging to different lattices.
    f2p = lattice.general(2 * np.identity(2))
    sym = ts(*(2 * np.identity(2)))
    assert sym.act((1, 1), f2(0, 0), f2p(0, 0)) == (f2(2, 2), f2p(1, 1))
    assert sym.act((1, 1), f2p(0, 0), f2(0, 0)) == (f2p(1, 1), f2(2, 2))

    # Test add_site_family on random lattices and symmetries by ensuring that
    # it's possible to add site groups that are compatible with a randomly
    # generated symmetry with proper vectors.
    rng = ensure_rng(30)
    vec = rng.randn(3, 5)
    lat = lattice.general(vec)
    total = 0
    for k in range(1, 4):
        for i in range(10):
            sym_vec = rng.randint(-10, 10, size=(k, 3))
            if np.linalg.matrix_rank(sym_vec) < k:
                continue
            total += 1
            sym_vec = np.dot(sym_vec, vec)
            sym = ts(*sym_vec)
            sym.add_site_family(lat)
    assert total > 20


def test_translational_symmetry_reversed():
    rng = ensure_rng(30)
    lat = lattice.general(np.identity(3))
    sites = [lat(i, j, k) for i in range(-2, 6) for j in range(-2, 6)
                          for k in range(-2, 6)]
    for i in range(4):
            periods = rng.randint(-5, 5, (3, 3))
            try:
                sym = lattice.TranslationalSymmetry(*periods)
                rsym = sym.reversed()
                for site in sites:
                    assert sym.to_fd(site) == rsym.to_fd(site)
                    assert sym.which(site) == -rsym.which(site)
                    vec = np.array([1, 1, 1])
                    assert sym.act(vec, site), rsym.act(-vec == site)
            except ValueError:
                pass


def test_monatomic_lattice():
    lat = lattice.square()
    lat2 = lattice.general(np.identity(2))
    lat3 = lattice.square(name='no')
    assert len(set([lat, lat2, lat3, lat(0, 0), lat2(0, 0), lat3(0, 0)])) == 4

@pytest.mark.parametrize('prim_vecs, basis', [
    (1, None),
    ([1], None),
    ([1, 0], [[0, 0]]),
    ([[1, 0], [2, 0]], None),
    ([[1, 0], [2, 0]], [[0, 0]]),
    ([[1, 0], [0, 2], [1, 2]], None),
    ([[1, 0], [0, 2], [1, 2]], [[0, 0]]),
])
def test_lattice_constraints(prim_vecs, basis):
    with pytest.raises(ValueError):
        lattice.general(prim_vecs, basis)


def test_norbs():
    id_mat = np.identity(2)
    # Monatomic lattices
    assert lattice.general(id_mat).norbs == None
    assert lattice.general(id_mat, norbs=2).norbs == 2
    # Polyatomic lattices
    lat = lattice.general(id_mat, basis=id_mat, norbs=None)
    for l in lat.sublattices:
        assert l.norbs == None
    lat = lattice.general(id_mat, basis=id_mat, norbs=2)
    for l in lat.sublattices:
        assert l.norbs == 2
    lat = lattice.general(id_mat, basis=id_mat, norbs=[1, 2])
    for l, n in zip(lat.sublattices, [1, 2]):
        assert l.norbs == n
    # should raise ValueError for # of norbs different to length of `basis`
    raises(ValueError, lattice.general, id_mat, id_mat, norbs=[])
    raises(ValueError, lattice.general, id_mat, id_mat, norbs=[1, 2, 3])
    # TypeError if Monatomic lattice
    raises(TypeError, lattice.general, id_mat, norbs=[])
    # should raise ValueError if norbs not an integer
    raises(ValueError, lattice.general, id_mat, norbs=1.5)
    raises(ValueError, lattice.general, id_mat, id_mat, norbs=1.5)
    raises(ValueError, lattice.general, id_mat, id_mat, norbs=[1.5, 1.5])
    # test that lattices with different norbs are compared `not equal`
    lat = lattice.general(id_mat, basis=id_mat, norbs=None)
    lat1 = lattice.general(id_mat, basis=id_mat, norbs=1)
    lat2 = lattice.general(id_mat, basis=id_mat, norbs=2)
    assert lat != lat1
    assert lat != lat2
    assert lat1 != lat2


def test_symmetry_act():
    lat = lattice.square()
    sym = lattice.TranslationalSymmetry((1, 0), (0, 1))
    site = lat(0, 0)
    hopping = (lat(0, 0), lat(1, 0))
    el = (1, 0)

    # Verify that the dtype of tags of sites returned by 'act' is 'int'
    for el in [el, ta.array(el, int)]:
        assert sym.act(el, site).tag.dtype is int
        assert all(s.tag.dtype is int for s in sym.act(el, *hopping))

    for el in [(1.0, 0), (1.5, 0)]:
        with raises(ValueError):
            sym.act(el, site)
        with raises(ValueError):
            sym.act(ta.array(el), site)


def test_symmetry_has_subgroup():
    rng = np.random.RandomState(0)
    ## test whether actual subgroups are detected as such
    vecs = rng.randn(3, 3)
    sym1 = lattice.TranslationalSymmetry(*vecs)
    ns = builder.NoSymmetry()
    assert ns.has_subgroup(ns)
    assert sym1.has_subgroup(sym1)
    assert sym1.has_subgroup(ns)
    assert sym1.has_subgroup(lattice.TranslationalSymmetry(
        2 * vecs[0], 3 * vecs[1] + 4 * vecs[2]))
    assert not lattice.TranslationalSymmetry(*(0.8 * vecs)).has_subgroup(sym1)

    ## test subgroup creation
    for dim in range(1, 4):
        generators = rng.randint(10, size=(dim, 3))
        assert sym1.has_subgroup(sym1.subgroup(*generators))

    # generators are not linearly independent
    with raises(ValueError):
        sym1.subgroup(*rng.randint(10, size=(4, 3)))

    # generators are not integer sequences
    with raises(ValueError):
        sym1.subgroup(*rng.rand(1, 3))
