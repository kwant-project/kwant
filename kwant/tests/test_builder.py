# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import warnings
import pickle
import itertools as it
import functools as ft
from random import Random

import numpy as np
import tinyarray as ta
import pytest
from pytest import raises, warns
from numpy.testing import assert_almost_equal

import kwant
from kwant import builder
from kwant._common import ensure_rng


def test_bad_keys():

    def setitem(key):
        syst[key] = None

    fam = builder.SimpleSiteFamily()
    syst = builder.Builder()

    failures = [
        # Invalid single keys
        ([syst.__contains__, syst.__getitem__, setitem, syst.__delitem__],
         [(TypeError, [123,
                       (0, 1),
                       (fam(0), 123),
                       (123, (fam(0)))]),
          (IndexError, [(fam(0),),
                        (fam(0), fam(1), fam(2))]),
          (ValueError, [(fam(0), fam(0)),
                        (fam(2), fam(2))])]),

        # Hoppings that contain sites that do not belong to the system
        ([syst.__getitem__, setitem, syst.__delitem__],
         [(KeyError, [(fam(0), fam(3)), (fam(2), fam(1)),
                      (fam(2), fam(3))])]),

        # Sequences containing a bad key.
        ([setitem, syst.__delitem__],
         [(TypeError, [[fam(0), fam(1), 123],
                       [fam(0), (fam(1),)],
                       [fam(0), (fam(1), fam(2))],
                       [(fam(0), fam(1)), (0, 1)],
                       [(fam(0), fam(1)), (fam(0), 123)],
                       [(fam(0), fam(1)), (123, fam(0))],
                       [(fam(0), fam(1)), fam(2)]]),
          (IndexError, [[(fam(0), fam(1)), (fam(2),)]]),
          (ValueError, [[(fam(0), fam(1)), (fam(2), fam(2))],
                        [(fam(0), fam(0)), (fam(1), fam(0))]]),
          (KeyError, [[(fam(0), fam(1)), (fam(0), fam(3))],
                      [(fam(0), fam(1)), (fam(2), fam(1))],
                      [(fam(1), fam(2)), (fam(0), fam(1))]])]),

        # Sites that do not belong to the system, also as part of a
        # sequence
        ([syst.__delitem__],
         [(KeyError, [fam(123),
                      [fam(0), fam(123)],
                      [fam(123), fam(1)]])]),

        # Various things that are not sites present in the system.
        ([syst.degree, lambda site: list(syst.neighbors(site))],
         [(TypeError, [123,
                       [0, 1, 2],
                       (0, 1),
                       (fam(0), fam(1)),
                       [fam(0), fam(1)],
                       [fam(1), fam(2)],
                       [fam(3), fam(0)]]),
          (KeyError, [fam(123)])])]

    for funcs, errors in failures:
        for error, keys in errors:
            for key in keys:
                for func in funcs:
                    syst[[fam(0), fam(1)]] = None
                    syst[fam(0), fam(1)] = None
                    try:
                        raises(error, func, key)
                    except AssertionError:
                        print(func, error, key)
                        raise


def test_site_families():
    syst = builder.Builder()
    fam = builder.SimpleSiteFamily()
    ofam = builder.SimpleSiteFamily()
    yafam = builder.SimpleSiteFamily('another_name')

    syst[fam(0)] = 7
    assert syst[fam(0)] == 7

    assert len(set([fam, ofam, fam('a'), ofam('a'), yafam])) == 3
    syst[fam(1)] = 123
    assert syst[fam(1)] == 123
    assert syst[ofam(1)] == 123
    raises(KeyError, syst.__getitem__, yafam(1))

    # test site families compare equal/not-equal
    assert fam == ofam
    assert fam != yafam
    assert fam != None
    assert fam != 'a'

    # test site families sorting
    fam1 = builder.SimpleSiteFamily(norbs=1)
    fam2 = builder.SimpleSiteFamily(norbs=2)
    assert fam1 < fam2  # string '1' is lexicographically less than '2'


class VerySimpleSymmetry(builder.Symmetry):
    def __init__(self, period):
        self.period = period

    @property
    def num_directions(self):
        return 1

    def has_subgroup(self, other):
        if isinstance(other, builder.NoSymmetry):
            return True
        elif isinstance(other, VerySimpleSymmetry):
            return not other.period % self.period
        else:
            return False

    def subgroup(self, *generators):
        generators = ta.array(generators)
        assert generators.shape == (1, 1)
        if generators.dtype != int:
            raise ValueError('Generators must be sequences of integers.')
        g = generators[0, 0]
        return VerySimpleSymmetry(g * self.period)

    def which(self, site):
        return ta.array((site.tag[0] // self.period,), int)

    def act(self, element, a, b=None):
        shifted = lambda site, delta: site.family(*ta.add(site.tag, delta))
        delta = (self.period * element[0],) + (len(a.tag) - 1) * (0,)
        if b is None:
            return shifted(a, delta)
        else:
            return shifted(a, delta), shifted(b, delta)


# The hoppings have to form a ring.  Some other implicit assumptions are also
# made.
def check_construction_and_indexing(sites, sites_fd, hoppings, hoppings_fd,
                                    unknown_hoppings, sym=None):
    fam = builder.SimpleSiteFamily()
    syst = builder.Builder(sym)
    t, V = 1.0j, 0.0
    syst[sites] = V
    for site in sites:
        syst[site] = V
    syst[hoppings] = t
    for hopping in hoppings:
        syst[hopping] = t

    for hopping in unknown_hoppings:
        raises(KeyError, syst.__setitem__, hopping, t)

    assert (fam(5), fam(123)) not in syst
    assert (sites[0], fam(5, 123)) not in syst
    assert (fam(7, 8), sites[0]) not in syst
    for site in sites:
        assert site in syst
        assert syst[site] == V
    for hop in hoppings:
        rev_hop = hop[1], hop[0]
        assert hop in syst
        assert rev_hop in syst
        assert syst[hop] == t
        assert syst[rev_hop] == t.conjugate()

    assert syst.degree(sites[0]) == 2
    assert (sorted(s for s in syst.neighbors(sites[0])) ==
            sorted([sites[1], sites[-1]]))

    del syst[hoppings]
    assert list(syst.hoppings()) == []
    syst[hoppings] = t

    del syst[sites[0]]
    assert sorted(tuple(s) for s in syst.sites()) == sorted(sites_fd[1:])
    assert (sorted((a, b) for a, b in syst.hoppings()) ==
            sorted(hoppings_fd[1:-1]))

    assert (sorted((tuple(site.tag), value)
                        for site, value in syst.site_value_pairs()) ==
            sorted((tuple(site.tag), syst[site]) for site in syst.sites()))
    assert (sorted((tuple(a.tag), tuple(b.tag), value)
                   for (a, b), value in syst.hopping_value_pairs()) ==
            sorted((tuple(a.tag), tuple(b.tag), syst[a, b])
                   for a, b in syst.hoppings()))


def test_construction_and_indexing():
    # Without symmetry
    fam = builder.SimpleSiteFamily()
    sites = [fam(0, 0), fam(0, 1), fam(1, 0)]
    hoppings = [(fam(0, 0), fam(0, 1)),
                (fam(0, 1), fam(1, 0)),
                (fam(1, 0), fam(0, 0))]
    unknown_hoppings = [(fam(0, 1), fam(7, 8)),
                        (fam(12, 14), fam(0, 1))]
    check_construction_and_indexing(sites, sites, hoppings, hoppings,
                                    unknown_hoppings)

    # With symmetry
    sites = [fam(0, 0), fam(1, 1), fam(2, 1), fam(4, 2)]
    sites_fd = [fam(0, 0), fam(1, 1), fam(0, 1), fam(0, 2)]
    hoppings = [(fam(0, 0), fam(1, 1)),
                (fam(1, 1), fam(2, 1)),
                (fam(2, 1), fam(4, 2)),
                (fam(4, 2), fam(0, 0))]
    hoppings_fd = [(fam(0, 0), fam(1, 1)),
                   (fam(1, 1), fam(2, 1)),
                   (fam(0, 1), fam(2, 2)),
                   (fam(0, 2), fam(-4, 0))]
    unknown_hoppings = [(fam(0, 0), fam(0, 3)), (fam(0, 4), fam(0, 0)),
                        (fam(0, 0), fam(2, 3)), (fam(2, 4), fam(0, 0)),
                        (fam(4, 2), fam(6, 3)), (fam(6, 4), fam(4, 2))]
    sym = VerySimpleSymmetry(2)
    check_construction_and_indexing(sites, sites_fd, hoppings, hoppings_fd,
                                    unknown_hoppings, sym)


def test_hermitian_conjugation():
    def f(i, j, arg):
        i, j = i.tag, j.tag
        if j[0] == i[0] + 1:
            return arg * ta.array([[1, 2j], [3 + 1j, 4j]])
        else:
            raise ValueError

    syst = builder.Builder()
    fam = builder.SimpleSiteFamily()
    syst[fam(0)] = syst[fam(1)] = ta.identity(2)

    syst[fam(0), fam(1)] = f
    assert syst[fam(0), fam(1)] is f
    assert isinstance(syst[fam(1), fam(0)], builder.HermConjOfFunc)
    assert (syst[fam(1), fam(0)](fam(1), fam(0), 2) ==
            syst[fam(0), fam(1)](fam(0), fam(1), 2).conjugate().transpose())
    syst[fam(0), fam(1)] = syst[fam(1), fam(0)]
    assert isinstance(syst[fam(0), fam(1)], builder.HermConjOfFunc)
    assert syst[fam(1), fam(0)] is f


def test_value_equality_and_identity():
    m = ta.array([[1, 2], [3j, 4j]])
    syst = builder.Builder()
    fam = builder.SimpleSiteFamily()

    syst[fam(0)] = m
    syst[fam(1)] = m
    assert syst[fam(1)] is m

    syst[fam(0), fam(1)] = m
    assert syst[fam(1), fam(0)] == m.transpose().conjugate()
    assert syst[fam(0), fam(1)] is m

    syst[fam(1), fam(0)] = m
    assert syst[fam(0), fam(1)] == m.transpose().conjugate()
    assert syst[fam(1), fam(0)] is m


def random_onsite_hamiltonian(rng):
    return 2 * rng.random() - 1


def random_hopping_integral(rng):
    return complex(2 * rng.random() - 1, 2 * rng.random() - 1)


def check_onsite(fsyst, sites, subset=False, check_values=True):
    freq = {}
    for node in range(fsyst.graph.num_nodes):
        site = fsyst.sites[node].tag
        freq[site] = freq.get(site, 0) + 1
        if check_values and site in sites:
            assert fsyst.onsites[node][0] is sites[site]
    if not subset:
        # Check that all sites of `fsyst` are in `sites`.
        for site in freq.keys():
            assert site in sites
    # Check that all sites of `sites` are in `fsyst`.
    for site in sites:
        assert freq[site] == 1


def check_hoppings(fsyst, hops):
    assert fsyst.graph.num_edges == 2 * len(hops)
    for edge_id, edge in enumerate(fsyst.graph):
        tail, head = edge
        tail = fsyst.sites[tail].tag
        head = fsyst.sites[head].tag
        value = fsyst.hoppings[edge_id][0]
        if value is builder.Other:
            assert (head, tail) in hops
        else:
            assert (tail, head) in hops
            assert value is hops[tail, head]

def check_id_by_site(fsyst):
    for i, site in enumerate(fsyst.sites):
        assert fsyst.id_by_site[site] == i


def test_finalization():
    """Test the finalization of finite and infinite systems.

    In order to exactly verify the finalization, low-level features of the
    build module are used directly.  This is not the way one would use a
    finalized system in normal code.
    """
    def set_sites(dest):
        while len(dest) < n_sites:
            site = ta.array([rng.randrange(size), rng.randrange(size)])
            if site not in dest:
                dest[site] = random_onsite_hamiltonian(rng)

    def set_hops(dest, sites):
        while len(dest) < n_hops:
            a, b = rng.sample(list(sites), 2)
            if (a, b) not in dest and (b, a) not in dest:
                dest[a, b] = random_hopping_integral(rng)

    rng = Random(123)
    size = 20
    n_sites = 120
    n_hops = 500

    # Make scattering region blueprint.
    sr_sites = {}
    set_sites(sr_sites)
    sr_hops = {}
    set_hops(sr_hops, sr_sites)

    # Make lead blueprint.
    possible_neighbors = rng.sample(list(sr_sites), n_sites // 2)
    lead_sites = {}
    for pn in possible_neighbors:
        lead_sites[pn] = random_hopping_integral(rng)
    set_sites(lead_sites)
    lead_hops = {}        # Hoppings within a single lead unit cell
    set_hops(lead_hops, lead_sites)
    lead_sites_list = list(lead_sites)
    neighbors = set()
    for i in range(n_hops):
        while True:
            a = rng.choice(lead_sites_list)
            b = rng.choice(possible_neighbors)
            neighbors.add(b)
            b = ta.array([b[0] - size, b[1]])
            if rng.randrange(2):
                a, b = b, a
            if (a, b) not in lead_hops and (b, a) not in lead_hops:
                break
        lead_hops[a, b] = random_hopping_integral(rng)
    neighbors = sorted(neighbors)

    # Build scattering region from blueprint and test it.
    syst = builder.Builder()
    fam = kwant.lattice.general(ta.identity(2))
    for site, value in sr_sites.items():
        syst[fam(*site)] = value
    for hop, value in sr_hops.items():
        syst[fam(*hop[0]), fam(*hop[1])] = value
    fsyst = syst.finalized()
    check_id_by_site(fsyst)
    check_onsite(fsyst, sr_sites)
    check_hoppings(fsyst, sr_hops)
    # check that sites are sorted
    assert fsyst.sites == tuple(sorted(fam(*site) for site in sr_sites))

    # Build lead from blueprint and test it.
    lead = builder.Builder(kwant.TranslationalSymmetry((size, 0)))
    for site, value in lead_sites.items():
        shift = rng.randrange(-5, 6) * size
        site = site[0] + shift, site[1]
        lead[fam(*site)] = value
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        lead.finalized()        # Trigger the warning.
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert "disconnected" in str(w[0].message)
    for (a, b), value in lead_hops.items():
        shift = rng.randrange(-5, 6) * size
        a = a[0] + shift, a[1]
        b = b[0] + shift, b[1]
        lead[fam(*a), fam(*b)] = value
    flead = lead.finalized()
    all_sites = list(lead_sites)
    all_sites.extend(ta.array([x - size, y]) for (x, y) in neighbors)
    check_id_by_site(fsyst)
    check_onsite(flead, all_sites, check_values=False)
    check_onsite(flead, lead_sites, subset=True)
    check_hoppings(flead, lead_hops)

    # Attach lead to system with empty interface.
    syst.leads.append(builder.BuilderLead(lead, ()))
    raises(ValueError, syst.finalized)

    # Attach lead with improper interface.
    syst.leads[-1] = builder.BuilderLead(
        lead, 2 * tuple(builder.Site(fam, n) for n in neighbors))
    raises(ValueError, syst.finalized)

    # Attach lead properly.
    syst.leads[-1] = builder.BuilderLead(
        lead, (builder.Site(fam, n) for n in neighbors))
    fsyst = syst.finalized()
    assert len(fsyst.lead_interfaces) == 1
    assert ([fsyst.sites[i].tag for i in fsyst.lead_interfaces[0]] ==
            neighbors)

    # test that we cannot finalize a system with a badly sorted interface order
    raises(ValueError, builder.InfiniteSystem, lead,
           [builder.Site(fam, n) for n in reversed(neighbors)])
    # site ordering independent of whether interface was specified
    flead_order = builder.InfiniteSystem(lead, [builder.Site(fam, n)
                                                for n in neighbors])
    assert flead.sites == flead_order.sites


    syst.leads[-1] = builder.BuilderLead(
        lead, (builder.Site(fam, n) for n in neighbors))
    fsyst = syst.finalized()
    assert len(fsyst.lead_interfaces) == 1
    assert ([fsyst.sites[i].tag for i in fsyst.lead_interfaces[0]] ==
            neighbors)

    # Add a hopping to the lead which couples two next-nearest cells and check
    # whether this leads to an error.
    a = rng.choice(lead_sites_list)
    b = rng.choice(possible_neighbors)
    b = b[0] + 2 * size, b[1]
    lead[fam(*a), fam(*b)] = random_hopping_integral(rng)
    raises(ValueError, lead.finalized)


def test_site_ranges():
    lat1a = kwant.lattice.chain(norbs=1, name='a')
    lat1b = kwant.lattice.chain(norbs=1, name='b')
    lat2 = kwant.lattice.chain(norbs=2)
    site_ranges = builder._site_ranges

    # simple case -- single site family
    for lat in (lat1a, lat2):
        sites = list(map(lat, range(10)))
        ranges = site_ranges(sites)
        expected = [(0, lat.norbs, 0), (10, 0, 10 * lat.norbs)]
        assert ranges == expected

    # pair of site families
    sites = it.chain(map(lat1a, range(4)), map(lat1b, range(6)),
                     map(lat1a, range(4)))
    expected = [(0, 1, 0), (4, 1, 4), (10, 1, 10), (14, 0, 14)]
    assert expected == site_ranges(tuple(sites))
    sites = it.chain(map(lat2, range(4)), map(lat1a, range(6)),
                     map(lat1b, range(4)))
    expected = [(0, 2, 0), (4, 1, 4*2), (10, 1, 4*2+6), (14, 0, 4*2+10)]
    assert expected == site_ranges(tuple(sites))

    # test with an actual builder
    for lat in (lat1a, lat2):
        sites = list(map(lat, range(10)))
        syst = kwant.Builder()
        syst[sites] = np.eye(lat.norbs)
        ranges = syst.finalized().site_ranges
        expected = [(0, lat.norbs, 0), (10, 0, 10 * lat.norbs)]
        assert ranges == expected
        # poison system with a single site with no norbs defined
        syst[kwant.lattice.chain()(0)] = 1
        ranges = syst.finalized().site_ranges
        assert ranges == None


def test_hamiltonian_evaluation():
    def f_onsite(site):
        return site.tag[0]

    def f_hopping(a, b):
        a, b = a.tag, b.tag
        return complex(a[0] + b[0], a[1] - b[1])

    tags = [(0, 0), (1, 1), (2, 2), (3, 3)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2)]

    syst = builder.Builder()
    fam = builder.SimpleSiteFamily()
    sites = [fam(*tag) for tag in tags]
    syst[(fam(*tag) for tag in tags)] = f_onsite
    syst[((fam(*tags[i]), fam(*tags[j])) for (i, j) in edges)] = f_hopping
    fsyst = syst.finalized()

    assert fsyst.graph.num_nodes == len(tags)
    assert fsyst.graph.num_edges == 2 * len(edges)

    for i in range(len(tags)):
        site = fsyst.sites[i]
        assert site in sites
        assert fsyst.hamiltonian(i, i) == syst[site](site)

    for t, h in fsyst.graph:
        tsite = fsyst.sites[t]
        hsite = fsyst.sites[h]
        assert fsyst.hamiltonian(t, h) == syst[tsite, hsite](tsite, hsite)

    # test when user-function raises errors
    def onsite_raises(site):
        raise ValueError()

    def hopping_raises(a, b):
        raise ValueError('error message')

    def test_raising(fsyst, hop):
        a, b = hop
        # exceptions are converted to kwant.UserCodeError and we add our message
        with raises(kwant.UserCodeError) as exc_info:
            fsyst.hamiltonian(a, a)
        msg = 'Error occurred in user-supplied value function "onsite_raises"'
        assert msg in str(exc_info.value)

        for hop in [(a, b), (b, a)]:
            with raises(kwant.UserCodeError) as exc_info:
                fsyst.hamiltonian(*hop)
            msg = ('Error occurred in user-supplied '
                   'value function "hopping_raises"')
            assert msg in str(exc_info.value)

    # test with finite system
    new_hop = (fam(-1, 0), fam(0, 0))
    syst[new_hop[0]] = onsite_raises
    syst[new_hop] = hopping_raises
    fsyst = syst.finalized()
    hop = tuple(map(fsyst.sites.index, new_hop))
    test_raising(fsyst, hop)

    # test with infinite system
    inf_syst = kwant.Builder(VerySimpleSymmetry(2))
    for k, v in it.chain(syst.site_value_pairs(), syst.hopping_value_pairs()):
        inf_syst[k] = v
    inf_fsyst = inf_syst.finalized()
    hop = tuple(map(inf_fsyst.sites.index, new_hop))
    test_raising(inf_fsyst, hop)


def test_dangling():
    def make_system():
        #        1
        #       / \
        #    3-0---2-4-5  6-7  8
        syst = builder.Builder()
        fam = builder.SimpleSiteFamily()
        syst[(fam(i) for i in range(9))] = None
        syst[[(fam(0), fam(1)), (fam(1), fam(2)), (fam(2), fam(0))]] = None
        syst[[(fam(0), fam(3)), (fam(2), fam(4)), (fam(4), fam(5))]] = None
        syst[fam(6), fam(7)] = None
        return syst

    syst0 = make_system()
    assert (sorted(site.tag for site in syst0.dangling()) ==
            sorted([(3,), (5,), (6,), (7,), (8,)]))
    syst0.eradicate_dangling()

    syst1 = make_system()
    while True:
        dangling = list(syst1.dangling())
        if not dangling:
            break
        del syst1[dangling]

    assert (sorted(site.tag for site in syst0.sites()) ==
            sorted([(0,), (1,), (2,)]))
    assert (sorted(site.tag for site in syst0.sites()) ==
            sorted(site.tag for site in syst1.sites()))


def test_builder_with_symmetry():
    g = kwant.lattice.general(ta.identity(3))
    sym = kwant.TranslationalSymmetry((0, 0, 3), (0, 2, 0))
    syst = builder.Builder(sym)

    t, V = 1.0j, 0.0
    hoppings = [(g(5, 0, 0), g(0, 5, 0)),
                (g(0, 5, 0), g(0, 0, 5)),
                (g(0, 0, 5), g(5, 0, 0)),
                (g(0, 3, 0), g(0, 0, 5)),
                (g(0, 7, -6), g(5, 6, -6))]
    hoppings_fd = [(g(5, 0, 0), g(0, 5, 0)),
                   (g(0, 1, 0), g(0, -4, 5)),
                   (g(0, 0, 2), g(5, 0, -3)),
                   (g(0, 1, 0), g(0, -2, 5)),
                   (g(0, 1, 0), g(5, 0, 0))]

    syst[(a for a, b in hoppings)] = V
    syst[hoppings] = t

    # TODO: Once Tinyarray supports "<" the conversion to tuple can be removed.
    assert (sorted(tuple(site.tag) for site in syst.sites()) ==
            sorted(set(tuple(hop[0].tag) for hop in hoppings_fd)))
    for sites in hoppings_fd:
        for site in sites:
            assert site in syst
            assert syst[site] == V

    # TODO: Once Tinyarray supports "<" the conversion to tuple can be removed.
    assert (sorted((tuple(a.tag), tuple(b.tag)) for a, b in syst.hoppings()) ==
            sorted((tuple(a.tag), tuple(b.tag)) for a, b in hoppings_fd))
    for hop in hoppings_fd:
        rhop = hop[1], hop[0]
        assert hop in syst
        assert rhop in syst
        assert syst[hop] == t
        assert syst[rhop] == t.conjugate()

    del syst[g(0, 6, -4), g(0, 11, -9)]
    assert (g(0, 1, 0), g(0, -4, 5)) not in syst

    del syst[g(0, 3, -3)]
    assert (list((a.tag, b.tag) for a, b in syst.hoppings()) == [((0, 0, 2),
                                                                  (5, 0, -3))])


def test_fill():
    g = kwant.lattice.square()
    sym_x = kwant.TranslationalSymmetry((-1, 0))
    sym_xy = kwant.TranslationalSymmetry((-1, 0), (0, 1))

    template_1d = builder.Builder(sym_x)
    template_1d[g(0, 0)] = None
    template_1d[g.neighbors()] = None

    def line_200(site):
        return -100 <= site.pos[0] < 100

    ## Test that copying a builder by "fill" preserves everything.
    for sym, func in [(kwant.TranslationalSymmetry(*np.diag([3, 4, 5])),
                       lambda pos: True),
                      (builder.NoSymmetry(),
                       lambda pos: ta.dot(pos, pos) < 17)]:
        cubic = kwant.lattice.general(ta.identity(3))

        # Make a weird system.
        orig = kwant.Builder(sym)
        sites = cubic.shape(func, (0, 0, 0))
        for i, site in enumerate(orig.expand(sites)):
            if i % 7 == 0:
                continue
            orig[site] = i
        for i, hopp in enumerate(orig.expand(cubic.neighbors(1))):
            if i % 11 == 0:
                continue
            orig[hopp] = i * 1.2345
        for i, hopp in enumerate(orig.expand(cubic.neighbors(2))):
            if i % 13 == 0:
                continue
            orig[hopp] = i * 1j

        # Clone the original using fill.
        clone = kwant.Builder(sym)
        clone.fill(orig, lambda s: True, (0, 0, 0))

        # Verify that both are identical.
        assert set(clone.site_value_pairs()) == set(orig.site_value_pairs())
        assert (set(clone.hopping_value_pairs())
                == set(orig.hopping_value_pairs()))

    ## Test for warning when "start" is outside the filling shape.
    target = builder.Builder()
    for start in [(-101, 0), (101, 0)]:
        with warns(RuntimeWarning):
            target.fill(template_1d, line_200, start)

    ## Test filling of infinite builder.
    for n in [1, 2, 4]:
        sym_n = kwant.TranslationalSymmetry((n, 0))
        for start in [g(0, 0), g(20, 0)]:
            target = builder.Builder(sym_n)
            sites = target.fill(template_1d, lambda s: True, start,
                                max_sites=10)
            assert len(sites) == n
            assert len(list(target.hoppings())) == n
            assert set(sym_n.to_fd(s) for s in sites) == set(target.sites())

    ## test max_sites
    target = builder.Builder()
    for max_sites in (-1, 0):
        with raises(ValueError):
            target.fill(template_1d, lambda site: True, g(0, 0),
                        max_sites=max_sites)
        assert len(list(target.sites())) == 0
    target = builder.Builder()
    with raises(RuntimeError):
        target.fill(template_1d, line_200, g(0, 0) , max_sites=10)
    ## test filling
    target = builder.Builder()
    added_sites = target.fill(template_1d, line_200, g(0, 0))
    assert len(added_sites) == 200
    # raise warning if target already contains all starting sites
    with warns(RuntimeWarning):
        target.fill(template_1d, line_200, g(0, 0))

    ## test multiplying unit cell size in 1D
    n_cells = 10
    sym_nx = kwant.TranslationalSymmetry(*(sym_x.periods * n_cells))
    target = builder.Builder(sym_nx)
    target.fill(template_1d, lambda site: True, g(0, 0))

    should_be_syst = builder.Builder(sym_nx)
    should_be_syst[(g(i, 0) for i in range(n_cells))] = None
    should_be_syst[g.neighbors()] = None

    assert sorted(target.sites()) == sorted(should_be_syst.sites())
    assert sorted(target.hoppings()) == sorted(should_be_syst.hoppings())

    ## test multiplying unit cell size in 2D
    template_2d = builder.Builder(sym_xy)
    template_2d[g(0, 0)] = None
    template_2d[g.neighbors()] = None
    template_2d[builder.HoppingKind((2, 2), g)] = None

    nm_cells = (3, 5)
    sym_nmxy = kwant.TranslationalSymmetry(*(sym_xy.periods * nm_cells))
    target = builder.Builder(sym_nmxy)
    target.fill(template_2d, lambda site: True, g(0, 0))

    should_be_syst = builder.Builder(sym_nmxy)
    should_be_syst[(g(i, j) for i in range(10) for j in range(10))] = None
    should_be_syst[g.neighbors()] = None
    should_be_syst[builder.HoppingKind((2, 2), g)] = None

    assert sorted(target.sites()) == sorted(should_be_syst.sites())
    assert sorted(target.hoppings()) == sorted(should_be_syst.hoppings())

    ## test filling 0D builder with 2D builder
    def square_shape(site):
        x, y = site.tag
        return 0 <= x < 10 and 0 <= y < 10

    target = builder.Builder()
    target.fill(template_2d, square_shape, g(0, 0))

    should_be_syst = builder.Builder()
    should_be_syst[(g(i, j) for i in range(10) for j in range(10))] = None
    should_be_syst[g.neighbors()] = None
    should_be_syst[builder.HoppingKind((2, 2), g)] = None

    assert sorted(target.sites()) == sorted(should_be_syst.sites())
    assert sorted(target.hoppings()) == sorted(should_be_syst.hoppings())

    ## test that 'fill' respects the symmetry of the target builder
    lat = kwant.lattice.chain(a=1)
    template = builder.Builder(kwant.TranslationalSymmetry((-1,)))
    template[lat(0)] = 2
    template[lat.neighbors()] = -1

    target = builder.Builder(kwant.TranslationalSymmetry((-2,)))
    target[lat(0)] = None
    to_target_fd = target.symmetry.to_fd
    # Refuses to fill the target because target already contains the starting
    # site.
    with warns(RuntimeWarning):
        target.fill(template, lambda x: True, lat(0))

    # should only add a single site (and hopping)
    new_sites = target.fill(template, lambda x: True, lat(1))
    assert target[lat(0)] is None  # should not be overwritten by template
    assert target[lat(-1)] == template[lat(0)]
    assert len(new_sites) == 1
    assert to_target_fd(new_sites[0]) == to_target_fd(lat(-1))

    # Test for warning with an empty template
    template = builder.Builder(kwant.TranslationalSymmetry((-1,)))
    target = builder.Builder()
    with warns(RuntimeWarning):
        target.fill(template, lambda x: True, lat(0))

    # Test for warning when one of the starting sites is outside the template
    lat = kwant.lattice.square()
    template = builder.Builder(kwant.TranslationalSymmetry((-1, 0)))
    template[lat(0, 0)] = None
    template[lat.neighbors()] = None
    target = builder.Builder()
    with warns(RuntimeWarning):
        target.fill(template, lambda x: -1 < x.pos[0] < 1,
                    [lat(0, 0), lat(0, 1)])


def test_fill_sticky():
    """Test that adjacent regions are properly interconnected when filled
    separately.
    """
    # Generate model template.
    lat = kwant.lattice.kagome()
    template = kwant.Builder(kwant.TranslationalSymmetry(
        lat.vec((1, 0)), lat.vec((0, 1))))
    for i, sl in enumerate(lat.sublattices):
        template[sl(0, 0)] = i
    for i in range(1, 3):
        for j, hop in enumerate(template.expand(lat.neighbors(i))):
            template[hop] = j * 1j

    def disk(site):
        pos = site.pos
        return ta.dot(pos, pos) < 13

    def halfplane(site):
        return ta.dot(site.pos - (-1, 1), (-0.9, 0.63)) > 0

    # Fill in one go.
    syst0 = kwant.Builder()
    syst0.fill(template, disk, (0, 0))

    # Fill in two stages.
    syst1 = kwant.Builder()
    syst1.fill(template, lambda s: disk(s) and halfplane(s), (-2, 1))
    syst1.fill(template, lambda s: disk(s) and not halfplane(s), (0, 0))

    # Verify that both results are identical.
    assert set(syst0.site_value_pairs()) == set(syst1.site_value_pairs())
    assert (set(syst0.hopping_value_pairs())
            == set(syst1.hopping_value_pairs()))


def test_attach_lead():
    fam = builder.SimpleSiteFamily(norbs=1)
    fam_noncommensurate = builder.SimpleSiteFamily(name='other')

    syst = builder.Builder()
    syst[fam(1)] = 0
    lead = builder.Builder(VerySimpleSymmetry(-2))
    raises(ValueError, syst.attach_lead, lead)

    lead[fam(0)] = 1
    raises(ValueError, syst.attach_lead, lead)
    lead[fam(1)] = 1
    syst.attach_lead(lead)
    raises(ValueError, syst.attach_lead, lead, fam(5))

    syst = builder.Builder()
    # The tag of the site that is added in the following line is an empty tuple.
    # This simulates a site family that is not commensurate with the symmetry of
    # the lead.  Such sites may be present in the system, as long as there are
    # other sites that will interrupt the lead.
    syst[fam_noncommensurate()] = 2
    syst[fam(1)] = 0
    syst[fam(0)] = 1
    lead[fam(0), fam(1)] = lead[fam(0), fam(2)] = 1
    syst.attach_lead(lead)
    assert len(list(syst.sites())) == 4
    assert set(syst.leads[0].interface) == set([fam(-1), fam(0)])
    syst[fam(-10)] = syst[fam(-11)] = 0
    syst.attach_lead(lead)
    assert set(syst.leads[1].interface) == set([fam(-10), fam(-11)])
    assert len(list(syst.sites())) == 6
    syst.attach_lead(lead, fam(-5))
    assert set(syst.leads[0].interface) == set([fam(-1), fam(0)])

    # add some further-than-nearest-neighbor hoppings
    hop_range = 3
    lead = builder.Builder(
        VerySimpleSymmetry(1),
        conservation_law=np.eye(1),
        time_reversal=np.eye(1),
        particle_hole=np.eye(1),
        chiral=np.eye(1))
    lead[fam(0)] = 1
    for i in range(1, hop_range + 1):
        lead[fam(0), fam(i)] = 1
    syst.attach_lead(lead)
    expanded_lead = syst.leads[-1].builder
    assert expanded_lead.symmetry.period == hop_range
    assert len(list(expanded_lead.sites())) == hop_range
    assert expanded_lead.conservation_law is lead.conservation_law
    assert expanded_lead.time_reversal is lead.time_reversal
    assert expanded_lead.particle_hole is lead.particle_hole
    assert expanded_lead.chiral is lead.chiral

    # check that we can actually finalize the system
    syst.finalized()


def test_attach_lead_incomplete_unit_cell():
    lat = kwant.lattice.chain()
    syst = kwant.Builder()
    lead = kwant.Builder(kwant.TranslationalSymmetry((2,)))
    syst[lat(1)] = lead[lat(0)] = lead[lat(1)] = 0
    lead[lat.neighbors()] = 0
    assert(len(syst.attach_lead(lead)) == 0)


def test_neighbors_not_in_single_domain():
    sr = builder.Builder()
    lead = builder.Builder(VerySimpleSymmetry(-1))
    fam = builder.SimpleSiteFamily()
    sr[(fam(x, y) for x in range(3) for y in range(3) if x >= y)] = 0
    sr[builder.HoppingKind((1, 0), fam)] = 1
    sr[builder.HoppingKind((0, 1), fam)] = 1
    lead[(fam(0, y) for y in range(3))] = 0
    lead[((fam(0, y), fam(1, y)) for y in range(3))] = 1
    lead[((fam(0, y), fam(0, y + 1)) for y in range(2))] = 1
    sr.leads.append(builder.BuilderLead(lead, [fam(i, i) for i in range(3)]))
    raises(ValueError, sr.finalized)


def inside_disc(center, rr):
    def shape(site):
        d = site.pos - center
        dd = ta.dot(d, d)
        return dd <= rr
    return shape


def test_closest():
    rng = ensure_rng(10)
    for sym_dim in range(1, 4):
        for space_dim in range(sym_dim, 4):
            lat = kwant.lattice.general(ta.identity(space_dim))

            # Choose random periods.
            while True:
                periods = rng.randint(-10, 11, (sym_dim, space_dim))
                if np.linalg.det(np.dot(periods, periods.T)) > 0.1:
                    # Periods are reasonably linearly independent.
                    break
            syst = builder.Builder(kwant.TranslationalSymmetry(*periods))

            for tag in rng.randint(-30, 31, (4, space_dim)):
                # Add site and connect it to the others.
                old_sites = list(syst.sites())
                new_site = lat(*tag)
                syst[new_site] = None
                syst[((new_site, os) for os in old_sites)] = None

                # Test consistency with fill().
                for point in 200 * rng.random_sample((10, space_dim)) - 100:
                    closest = syst.closest(point)
                    dist = closest.pos - point
                    dist = ta.dot(dist, dist)
                    syst2 = builder.Builder()
                    syst2.fill(syst, inside_disc(point, 2 * dist), closest)
                    assert syst2.closest(point) == closest
                    for site in syst2.sites():
                        dd = site.pos - point
                        dd = ta.dot(dd, dd)
                        assert dd >= 0.999999 * dist


def test_update():
    lat = builder.SimpleSiteFamily()

    syst = builder.Builder()
    syst[[lat(0,), lat(1,)]] = 1
    syst[lat(0,), lat(1,)] = 1

    other_syst = builder.Builder()
    other_syst[[lat(1,), lat(2,)]] = 2
    other_syst[lat(1,), lat(2,)] = 1

    lead0 = builder.Builder(VerySimpleSymmetry(-1))
    lead0[lat(0,)] = 1
    lead0[(lat(0,), lat(1,))] = 1
    lead0 = builder.BuilderLead(lead0, [lat(0,)])
    syst.leads.append(lead0)

    lead1 = builder.Builder(VerySimpleSymmetry(1))
    lead1[lat(2,)] = 1
    lead1[(lat(2,), lat(1,))] = 1
    lead1 = builder.BuilderLead(lead1, [lat(2,)])
    other_syst.leads.append(lead1)

    syst.update(other_syst)
    assert syst.leads == [lead0, lead1]
    expected = sorted([((0,), 1), ((1,), 2), ((2,), 2)])
    assert sorted(((s.tag, v) for s, v in syst.site_value_pairs())) == expected
    expected = sorted([((0,), (1,), 1), ((1,), (2,), 1)])
    assert (sorted(((a.tag, b.tag, v) for (a, b), v in
                    syst.hopping_value_pairs())) == expected)


# y=0:    y=1:
#
# hhhh    hhhh
# gggh    gggh
# hhgh    hhgh
# ghgh    hhgh
#
def test_HoppingKind():
    g = kwant.lattice.general(ta.identity(3), name='some_lattice')
    h = kwant.lattice.general(ta.identity(3), name='another_lattice')
    sym = kwant.TranslationalSymmetry((0, 2, 0))
    syst = builder.Builder(sym)
    syst[((h if max(x, y, z) % 2 else g)(x, y, z)
         for x in range(4) for y in range(2) for z in range(4))] = None
    for delta, ga, gb, n in [((1, 0, 0), g, h, 4),
                             ((1, 0, 0), h, g, 7),
                             ((0, 1, 0), g, h, 1),
                             ((0, 4, 0), h, h, 21),
                             ((0, 0, 1), g, h, 4)]:
        ph = list(builder.HoppingKind(delta, ga, gb)(syst))
        assert len(ph) == n
        ph = set(ph)
        assert len(ph) == n

        ph2 = list((
                sym.to_fd(b, a) for a, b in
                builder.HoppingKind(ta.negative(delta), gb, ga)(syst)))
        assert len(ph2) == n
        ph2 = set(ph2)
        assert ph2 == ph

        for a, b in ph:
            assert a.family == ga
            assert b.family == gb
            assert sym.to_fd(a) == a
            assert a.tag - b.tag == delta

        # test hashability and equality
        hk = builder.HoppingKind((1, 0, 0), g)
        hk2 = builder.HoppingKind((1, 0, 0), g)
        hk3 = builder.HoppingKind((1, 0, 0), g, h)
        assert hk == hk2
        assert hash(hk) == hash(hk2)
        assert hk != hk3
        assert hash(hk) != hash(hk3)
        assert len({hk: 0, hk2:1, hk3: 2}) == 2


def test_invalid_HoppingKind():
    g = kwant.lattice.general(ta.identity(3))
    h = kwant.lattice.general(np.identity(3)[:-1])  # 2D lattice in 3D

    delta = (1, 0, 0)

    # families have incompatible tags
    with raises(ValueError):
        builder.HoppingKind(delta, g, h)

    # delta is incompatible with tags
    with raises(ValueError):
        builder.HoppingKind(delta, h)


def test_ModesLead_and_SelfEnergyLead():
    lat = builder.SimpleSiteFamily()
    hoppings = [builder.HoppingKind((1, 0), lat),
                builder.HoppingKind((0, 1), lat)]
    rng = Random(123)
    L = 5
    t = 1
    energies = [0.9, 1.7]

    syst = builder.Builder()
    for x in range(L):
        for y in range(L):
            syst[lat(x, y)] = 4 * t + rng.random() - 0.5
    syst[hoppings] = -t

    # Attach a lead from the left.
    lead = builder.Builder(VerySimpleSymmetry(-1))
    for y in range(L):
        lead[lat(0, y)] = 4 * t
    lead[hoppings] = -t
    syst.attach_lead(lead)

    # Make the right lead and attach it.
    lead = builder.Builder(VerySimpleSymmetry(1))
    for y in range(L):
        lead[lat(0, y)] = 4 * t
    lead[hoppings] = -t
    syst.attach_lead(lead)

    fsyst = syst.finalized()
    ts = [kwant.smatrix(fsyst, e).transmission(1, 0) for e in energies]

    # Replace lead with it's finalized copy.
    lead = fsyst.leads[1]
    interface = [lat(L-1, lead.sites[i].tag[1]) for i in range(L)]

    # Re-attach right lead as ModesLead.
    syst.leads[1] = builder.ModesLead(lead.modes, interface, lead.parameters)
    fsyst = syst.finalized()
    ts2 = [kwant.smatrix(fsyst, e).transmission(1, 0) for e in energies]
    assert_almost_equal(ts2, ts)

    # Re-attach right lead as ModesLead with old-style modes API
    # that does not take a 'params' keyword parameter.
    syst.leads[1] = builder.ModesLead(
        lambda energy, args: lead.modes(energy, args),
        interface, lead.parameters)
    fsyst = syst.finalized()
    ts2 = [kwant.smatrix(fsyst, e).transmission(1, 0) for e in energies]
    assert_almost_equal(ts2, ts)

    # Re-attach right lead as SelfEnergyLead.
    syst.leads[1] = builder.SelfEnergyLead(lead.selfenergy, interface,
                                           lead.parameters)
    fsyst = syst.finalized()
    ts2 = [kwant.greens_function(fsyst, e).transmission(1, 0) for e in energies]
    assert_almost_equal(ts2, ts)

    # Re-attach right lead as SelfEnergyLead with old-style selfenergy API
    # that does not take a 'params' keyword parameter.
    syst.leads[1] = builder.SelfEnergyLead(
        lambda energy, args: lead.selfenergy(energy, args),
        interface, lead.parameters)
    fsyst = syst.finalized()
    ts2 = [kwant.greens_function(fsyst, e).transmission(1, 0) for e in energies]
    assert_almost_equal(ts2, ts)

    # Append a virtual (=zero self energy) lead.  This should have no effect.
    # Also verifies that the selfenergy callback function can return exotic
    # arraylikes.
    syst.leads.append(builder.SelfEnergyLead(
        lambda *args: list(ta.zeros((L, L))), interface, lead.parameters))
    fsyst = syst.finalized()
    ts2 = [kwant.greens_function(fsyst, e).transmission(1, 0) for e in energies]
    assert_almost_equal(ts2, ts)


def test_site_pickle():
    site = kwant.lattice.square()(0, 0)
    assert pickle.loads(pickle.dumps(site)) == site


def test_discrete_symmetries():
    lat = builder.SimpleSiteFamily(name='ccc', norbs=2)
    lat2 = builder.SimpleSiteFamily(name='bla', norbs=1)
    lat3 = builder.SimpleSiteFamily(name='dd', norbs=4)

    cons_law = {lat: np.diag([1, 2]), lat2: 0}
    syst = builder.Builder(conservation_law=cons_law,
                           time_reversal=(lambda site, p: np.exp(1j*p) *
                                          np.identity(site.family.norbs)))
    syst[lat(1)] = np.identity(2)
    syst[lat2(1)] = 1

    params=dict(p=0)

    sym = syst.finalized().discrete_symmetry(params=params)
    for proj, should_be in zip(sym.projectors, np.identity(3)):
        assert np.allclose(proj.toarray(), should_be.reshape((3, 1)))
    assert np.allclose(sym.time_reversal.toarray(), np.identity(3))
    syst.conservation_law = lambda site, p: cons_law[site.family]
    sym = syst.finalized().discrete_symmetry(params=params)
    for proj, should_be in zip(sym.projectors, np.identity(3)):
        assert np.allclose(proj.toarray(), should_be.reshape((-1, 1)))

    syst = builder.Builder(conservation_law=np.diag([-1, 1]))
    syst[lat(1)] = np.identity(2)
    sym = syst.finalized().discrete_symmetry()
    for proj, should_be in zip(sym.projectors, np.identity(2)):
        assert np.allclose(proj.toarray(), should_be.reshape((-1, 1)))

    syst = builder.Builder(conservation_law=1)
    syst[lat2(1)] = 0
    sym = syst.finalized().discrete_symmetry()
    [proj] = sym.projectors
    assert np.allclose(proj.toarray(), [[1]])

    syst = kwant.Builder(conservation_law=np.diag([-1, 1, -1, 1]))

    syst[lat3(0)] = np.eye(4)

    sym = syst.finalized().discrete_symmetry()
    p1 = np.zeros((4, 2))
    p1[0, 0] = p1[2, 1] = 1
    assert np.allclose(sym.projectors[0].toarray(), p1)
    p2 = np.zeros((4, 2))
    p2[1, 0] = p2[3, 1] = 1
    assert np.allclose(sym.projectors[1].toarray(), p2)

    # test parameter passing to conservation_law
    syst = builder.Builder(conservation_law=lambda site, b: b)
    syst[lat2(1)] = 0
    sym = syst.finalized().discrete_symmetry(params=dict(a=None, b=1))
    [proj] = sym.projectors
    assert np.allclose(proj.toarray(), [[1]])


# We need to keep testing 'args', but we don't want to see
# all the deprecation warnings in the test logs
@pytest.mark.filterwarnings("ignore:.*'args' parameter")
def test_argument_passing():
    chain = kwant.lattice.chain()

    # Test for passing parameters to hamiltonian matrix elements
    def onsite(site, p1, p2):
        return p1 + p2

    def hopping(site1, site2, p1, p2):
        return p1 - p2

    def gen_fill_syst(onsite, hopping, syst):
        syst[(chain(i) for i in range(3))] = onsite
        syst[chain.neighbors()] = hopping
        return syst.finalized()

    fill_syst = ft.partial(gen_fill_syst, onsite, hopping)

    syst = fill_syst(kwant.Builder())
    inf_syst = fill_syst(kwant.Builder(kwant.TranslationalSymmetry((-3,))))

    tests = (
        syst.hamiltonian_submatrix,
        inf_syst.cell_hamiltonian,
        inf_syst.inter_cell_hopping,
        inf_syst.selfenergy,
        lambda *args, **kw: inf_syst.modes(*args, **kw)[0].wave_functions,
    )

    for test in tests:
        np.testing.assert_array_equal(
            test(args=(2, 1)), test(params=dict(p1=2, p2=1)))

    # test that mixing 'args' and 'params' raises TypeError
    with raises(TypeError):
        syst.hamiltonian(0, 0, *(2, 1), params=dict(p1=2, p2=1))
    with raises(TypeError):
        inf_syst.hamiltonian(0, 0, *(2, 1), params=dict(p1=2, p2=1))

    # Missing parameters raises TypeError
    with raises(TypeError):
        syst.hamiltonian(0, 0, params=dict(p1=2))
    with raises(TypeError):
        syst.hamiltonian_submatrix(params=dict(p1=2))

    # test that passing parameters without default values works, and that
    # passing parameters with default values fails
    def onsite(site, p1, p2):
        return p1 + p2

    def hopping(site, site2, p1, p2):
        return p1 - p2

    fill_syst = ft.partial(gen_fill_syst, onsite, hopping)

    syst = fill_syst(kwant.Builder())
    inf_syst = fill_syst(kwant.Builder(kwant.TranslationalSymmetry((-3,))))

    tests = (
        syst.hamiltonian_submatrix,
        inf_syst.cell_hamiltonian,
        inf_syst.inter_cell_hopping,
        inf_syst.selfenergy,
        lambda *args, **kw: inf_syst.modes(*args, **kw)[0].wave_functions,
    )

    for test in tests:
        np.testing.assert_array_equal(
            test(args=(1, 2)), test(params=dict(p1=1, p2=2)))

    # Some common, some different args for value functions
    def onsite2(site, a, b):
        return site.pos + a + b

    def hopping2(site1, site2, a, c, b):
        return a + b + c

    syst = kwant.Builder()
    syst[(chain(i) for i in range(3))] = onsite2
    syst[((chain(i), chain(i + 1)) for i in range(2))] = hopping2
    fsyst = syst.finalized()

    def expected_hamiltonian(a, b, c):
        return [[a + b, a + b + c, 0],
                [a + b + c, 1 + a + b, a + b + c],
                [0, a+ b + c, 2 + a + b]]

    params = dict(a=1, b=2, c=3)
    np.testing.assert_array_equal(
        fsyst.hamiltonian_submatrix(params=params),
        expected_hamiltonian(**params)
    )


def test_parameter_substitution():

    subs = builder._substitute_params

    def f(x, y):
        return (('x', x), ('y', y))

    # 'f' already has a parameter 'y'
    assert raises(ValueError, subs, f, dict(x='y'))

    # Swap argument names.
    g = subs(f, dict(x='y', y='x'))
    assert g(1, 2) == f(1, 2)

    # Swap again.
    h = subs(g, dict(x='y', y='x'))
    assert h(1, 2) == f(1, 2)
    # don't nest wrappers inside each other
    assert h.func is f

    # Try different names.
    g = subs(f, dict(x='a', y='b'))
    assert g(1, 2) == f(1, 2)

    # Can substitutions be used in sets/dicts?
    g = subs(f, dict(x='a'))
    h = subs(f, dict(x='a'))
    assert len(set([f, g, h])) == 2


def test_subs():

    # Simple case

    def onsite(site, a, b):
        salt = str(a) + str(b)
        return kwant.digest.uniform(site.tag, salt=salt)

    def hopping(sitea, siteb, b, c):
        salt = str(b) + str(c)
        return kwant.digest.uniform(ta.array((sitea.tag, siteb.tag)), salt=salt)

    lat = kwant.lattice.chain()

    def make_system(sym=kwant.builder.NoSymmetry(), n=3):
        syst = kwant.Builder(sym)
        syst[(lat(i) for i in range(n))] = onsite
        syst[lat.neighbors()] = hopping
        return syst

    def hamiltonian(syst, **kwargs):
        return syst.finalized().hamiltonian_submatrix(params=kwargs)

    syst = make_system()
    # substituting a paramter that doesn't exist produces a warning
    warns(RuntimeWarning, syst.substituted, fakeparam='yes')
    # name clash in value functions
    raises(ValueError, syst.substituted, b='a')
    raises(ValueError, syst.substituted, b='c')
    raises(ValueError, syst.substituted, a='site')
    raises(ValueError, syst.substituted, c='sitea')
    # cannot call 'substituted' on systems with attached leads, because
    # it is not clear whether the substitutions should propagate
    # into the leads too.
    syst = make_system()
    lead = make_system(kwant.TranslationalSymmetry((-1,)), n=1)
    syst.attach_lead(lead)
    raises(ValueError, syst.substituted, a='d')

    # test basic substitutions
    syst = make_system()
    assert syst.finalized().parameters == {'a', 'b', 'c'}
    expected = hamiltonian(syst, a=1, b=2, c=3)
    # 1 level of substitutions
    sub_syst = syst.substituted(a='d', b='e')
    assert sub_syst.finalized().parameters == {'d', 'e', 'c'}
    assert np.allclose(hamiltonian(sub_syst, d=1, e=2, c=3), expected)
    # 2 levels of substitution
    sub_sub_syst = sub_syst.substituted(d='g', c='h')
    assert np.allclose(hamiltonian(sub_sub_syst, g=1, e=2, h=3), expected)
    assert sub_sub_syst.finalized().parameters == {'g', 'e', 'h'}
    # very confusing but technically valid. 'a' does not appear in 'hopping',
    # so the signature of 'onsite' is valid.
    sub_syst = syst.substituted(a='sitea')
    assert sub_syst.finalized().parameters == {'sitea', 'b', 'c'}
    assert np.allclose(hamiltonian(sub_syst, sitea=1, b=2, c=3), expected)

    # Check that this also works for infinite systems, as their finalization
    # follows a different code path.
    lead = make_system(kwant.TranslationalSymmetry((-1,)), n=1)
    lead = lead.substituted(a='lead_a', b='lead_b', c='lead_c')
    lead = lead.finalized()
    assert lead.parameters == {'lead_a', 'lead_b', 'lead_c'}

def test_attach_stores_padding():
    lat = kwant.lattice.chain()
    syst = kwant.Builder()
    syst[lat(0)] = 0
    lead = kwant.Builder(kwant.TranslationalSymmetry(lat.prim_vecs[0]))
    lead[lat(0)] = 0
    lead[lat(1), lat(0)] = 0
    added_sites = syst.attach_lead(lead, add_cells=2)
    assert syst.leads[0].padding == sorted(added_sites)


def test_finalization_preserves_padding():
    lat = kwant.lattice.chain()
    syst = kwant.Builder()
    for i in range(10):
        syst[lat(i)] = 0

    lead = kwant.Builder(kwant.TranslationalSymmetry(lat.prim_vecs[0]))
    lead[lat(0)] = 0
    lead[lat(0), lat(1)] = 0
    # We use a low level way to provide a lead to directly check that the
    # padding is preserved. We also check that the sites that do not belong to
    # the system don't end up in the padding of the finalized system.
    padding = [lat(0), lat(3), lat(5), lat(11)]
    syst.leads.append(kwant.builder.BuilderLead(lead, [lat(0)], padding))
    syst = syst.finalized()
    # The order is guaranteed because the paddings are sorted.
    assert [syst.sites[i] for i in syst.lead_paddings[0]] == padding[:-1]
