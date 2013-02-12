# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from __future__ import division
from random import Random
from nose.tools import assert_raises
from numpy.testing import assert_equal
import tinyarray as ta
import kwant
from kwant import builder


def test_site_groups():
    sys = builder.Builder()
    sg = builder.SimpleSiteGroup()
    osg = builder.SimpleSiteGroup()
    yasg = builder.SimpleSiteGroup('another_name')

    assert_raises(KeyError, sys.__setitem__, (0, ), 7)
    sys[sg(0)] = 7
    assert_equal(sys[sg(0)], 7)
    assert_raises(KeyError, sys.__getitem__, (0, ))

    assert len(set([sg, osg, sg('a'), osg('a'), yasg])) == 3
    sys[sg(1)] = 123
    assert_equal(sys[sg(1)], 123)
    assert_equal(sys[osg(1)], 123)
    assert_raises(KeyError, sys.__getitem__, yasg(1))


class VerySimpleSymmetry(builder.Symmetry):
    def __init__(self, period):
        self.period = period

    @property
    def num_directions(self):
        return 1

    def which(self, site):
        return ta.array((site.tag[0] // self.period,), int)

    def act(self, element, a, b=None):
        shifted = lambda site, delta: site.group(*ta.add(site.tag, delta))
        delta = (self.period * element[0],) + (len(a.tag) - 1) * (0,)
        if b is None:
            return shifted(a, delta)
        else:
            return shifted(a, delta), shifted(b, delta)


# The hoppings have to form a ring.  Some other implicit assumptions are also
# made.
def check_construction_and_indexing(sites, sites_fd, hoppings, hoppings_fd,
                                    failing_hoppings, sym=None):
    gr = builder.SimpleSiteGroup()
    sys = builder.Builder(sym)
    t, V = 1.0j, 0.0
    sys[sites] = V
    for site in sites:
        sys[site] = V
    sys[hoppings] = t
    for hopping in hoppings:
        sys[hopping] = t

    for hopping in failing_hoppings:
        assert_raises(KeyError, sys.__setitem__, hopping, t)

    assert (gr(5), gr(123)) not in sys
    assert (sites[0], gr(5, 123)) not in sys
    assert (gr(7, 8), sites[0]) not in sys
    for site in sites:
        assert site in sys
        assert_equal(sys[site], V)
    for hop in hoppings:
        rev_hop = hop[1], hop[0]
        assert hop in sys
        assert rev_hop in sys
        assert_equal(sys[hop], t)
        assert_equal(sys[rev_hop], t.conjugate())

    assert_equal(sys.degree(sites[0]), 2)
    assert_equal(sorted(s for s in sys.neighbors(sites[0])),
                 sorted([sites[1], sites[-1]]))

    del sys[hoppings]
    assert_equal(list(sys.hoppings()), [])
    sys[hoppings] = t

    del sys[sites[0]]
    assert_equal(sorted(tuple(s)
                        for s in sys.sites()), sorted(sites_fd[1:]))
    assert_equal(sorted((a, b)
                        for a, b in sys.hoppings()),
                 sorted(hoppings_fd[1:-1]))

    assert_equal(sorted((tuple(site.tag), value)
                        for site, value in sys.site_value_pairs()),
                 sorted((tuple(site.tag), sys[site]) for site in sys.sites()))
    assert_equal(sorted((tuple(a.tag), tuple(b.tag), value)
                        for (a, b), value in sys.hopping_value_pairs()),
                 sorted((tuple(a.tag), tuple(b.tag), sys[a, b])
                        for a, b in sys.hoppings()))


def test_construction_and_indexing():
    # Without symmetry
    gr = builder.SimpleSiteGroup()
    sites = [gr(0, 0), gr(0, 1), gr(1, 0)]
    hoppings = [(gr(0, 0), gr(0, 1)),
                (gr(0, 1), gr(1, 0)),
                (gr(1, 0), gr(0, 0))]
    failing_hoppings = [(gr(0, 1), gr(0, 1)),
                        (gr(0, 1), gr(7, 8)),
                        (gr(12, 14), gr(0, 1))]
    check_construction_and_indexing(sites, sites, hoppings, hoppings,
                                    failing_hoppings)

    # With symmetry
    sites = [gr(0, 0), gr(1, 1), gr(2, 1), gr(4, 2)]
    sites_fd = [gr(0, 0), gr(1, 1), gr(0, 1), gr(0, 2)]
    hoppings = [(gr(0, 0), gr(1, 1)),
                (gr(1, 1), gr(2, 1)),
                (gr(2, 1), gr(4, 2)),
                (gr(4, 2), gr(0, 0))]
    hoppings_fd = [(gr(0, 0), gr(1, 1)),
                   (gr(1, 1), gr(2, 1)),
                   (gr(0, 1), gr(2, 2)),
                   (gr(0, 2), gr(-4, 0))]
    failing_hoppings = [(gr(0, 0), gr(0, 3)), (gr(0, 4), gr(0, 0)),
                        (gr(0, 0), gr(2, 3)), (gr(2, 4), gr(0, 0)),
                        (gr(4, 2), gr(6, 3)), (gr(6, 4), gr(4, 2))]
    sym = VerySimpleSymmetry(2)
    check_construction_and_indexing(sites, sites_fd, hoppings, hoppings_fd,
                                    failing_hoppings, sym)


def test_hermitian_conjugation():
    def f(i, j):
        i, j = i.tag, j.tag
        if j[0] == i[0] + 1:
            return ta.array([[1, 2j], [3 + 1j, 4j]])
        else:
            raise ValueError

    sys = builder.Builder()
    sg = builder.SimpleSiteGroup()
    sys[sg(0)] = sys[sg(1)] = ta.identity(2)

    sys[sg(0), sg(1)] = f
    assert sys[sg(0), sg(1)] is f
    assert isinstance(sys[sg(1), sg(0)], builder.HermConjOfFunc)
    assert_equal(sys[sg(1), sg(0)](sg(1), sg(0)),
                 sys[sg(0), sg(1)](sg(0), sg(1)).conjugate().transpose())
    sys[sg(0), sg(1)] = sys[sg(1), sg(0)]
    assert isinstance(sys[sg(0), sg(1)], builder.HermConjOfFunc)
    assert sys[sg(1), sg(0)] is f


def test_value_equality_and_identity():
    m = ta.array([[1, 2], [3j, 4j]])
    sys = builder.Builder()
    sg = builder.SimpleSiteGroup()

    sys[sg(0)] = m
    sys[sg(1)] = m
    assert sys[sg(1)] is m

    sys[sg(0), sg(1)] = m
    assert_equal(sys[sg(1), sg(0)], m.transpose().conjugate())
    assert sys[sg(0), sg(1)] is m

    sys[sg(1), sg(0)] = m
    assert_equal(sys[sg(0), sg(1)], m.transpose().conjugate())
    assert sys[sg(1), sg(0)] is m


def random_onsite_hamiltonian(rng):
    return 2 * rng.random() - 1


def random_hopping_integral(rng):
    return complex(2 * rng.random() - 1, 2 * rng.random() - 1)


def check_onsite(fsys, sites, subset=False, check_values=True):
    freq = {}
    for node in xrange(fsys.graph.num_nodes):
        site = fsys.site(node).tag
        freq[site] = freq.get(site, 0) + 1
        if check_values and site in sites:
            assert fsys.onsite_hamiltonians[node] is sites[site]
    if not subset:
        # Check that all sites of `fsys` are in `sites`.
        for site, n in freq.iteritems():
            assert site in sites
    # Check that all sites of `sites` are in `fsys`.
    for site in sites:
        assert_equal(freq[site], 1)


def check_hoppings(fsys, hops):
    assert_equal(fsys.graph.num_edges, 2 * len(hops))
    for edge_id, edge in enumerate(fsys.graph):
        tail, head = edge
        tail = fsys.site(tail).tag
        head = fsys.site(head).tag
        value = fsys.hoppings[edge_id]
        if value is builder.Other:
            assert (head, tail) in hops
        else:
            assert (tail, head) in hops
            assert value is hops[tail, head]


def test_finalization():
    """Test the finalization of finite and infinite systems.

    In order to exactly verify the finalization, low-level features of the
    build module are used directly.  This is not the way one would use a
    finalized system in normal code.
    """
    def set_sites(dest):
        while len(dest) < n_sites:
            site = rng.randrange(size), rng.randrange(size)
            if site not in dest:
                dest[site] = random_onsite_hamiltonian(rng)

    def set_hops(dest, sites):
        while len(dest) < n_hops:
            a, b = rng.sample(sites, 2)
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
    for i in xrange(n_hops):
        while True:
            a = rng.choice(lead_sites_list)
            b = rng.choice(possible_neighbors)
            neighbors.add(b)
            b = b[0] - size, b[1]
            if rng.randrange(2):
                a, b = b, a
            if (a, b) not in lead_hops and (b, a) not in lead_hops:
                break
        lead_hops[a, b] = random_hopping_integral(rng)
    neighbors = sorted(neighbors)

    # Build scattering region from blueprint and test it.
    sys = builder.Builder()
    sg = kwant.lattice.general(ta.identity(2))
    for site, value in sr_sites.iteritems():
        sys[sg(*site)] = value
    for hop, value in sr_hops.iteritems():
        sys[sg(*hop[0]), sg(*hop[1])] = value
    fsys = sys.finalized()
    check_onsite(fsys, sr_sites)
    check_hoppings(fsys, sr_hops)

    # Build lead from blueprint and test it.
    lead = builder.Builder(kwant.TranslationalSymmetry((size, 0)))
    for site, value in lead_sites.iteritems():
        shift = rng.randrange(-5, 6) * size
        site = site[0] + shift, site[1]
        lead[sg(*site)] = value
    for (a, b), value in lead_hops.iteritems():
        shift = rng.randrange(-5, 6) * size
        a = a[0] + shift, a[1]
        b = b[0] + shift, b[1]
        lead[sg(*a), sg(*b)] = value
    flead = lead.finalized()
    all_sites = list(lead_sites)
    all_sites.extend((x - size, y) for (x, y) in neighbors)
    check_onsite(flead, all_sites, check_values=False)
    check_onsite(flead, lead_sites, subset=True)
    check_hoppings(flead, lead_hops)

    # Attach lead to system.
    sys.leads.append(builder.BuilderLead(
            lead, (builder.Site(sg, n) for n in neighbors)))
    fsys = sys.finalized()
    assert_equal(len(fsys.lead_interfaces), 1)
    assert_equal([fsys.site(i).tag for i in fsys.lead_interfaces[0]],
                 neighbors)

    # Add a hopping to the lead which couples two next-nearest slices and check
    # whether this leads to an error.
    a = rng.choice(lead_sites_list)
    b = rng.choice(possible_neighbors)
    b = b[0] + 2 * size, b[1]
    lead[sg(*a), sg(*b)] = random_hopping_integral(rng)
    assert_raises(ValueError, lead.finalized)


def test_hamiltonian_evaluation():
    def f_onsite(site):
        return site.tag[0]

    def f_hopping(a, b):
        a, b = a.tag, b.tag
        return complex(a[0] + b[0], a[1] - b[1])

    tags = [(0, 0), (1, 1), (2, 2), (3, 3)]
    edges = [(0, 1), (0, 2), (0, 3), (1, 2)]

    sys = builder.Builder()
    sg = builder.SimpleSiteGroup()
    sites = [sg(*tag) for tag in tags]
    sys[(sg(*tag) for tag in tags)] = f_onsite
    sys[((sg(*tags[i]), sg(*tags[j])) for (i, j) in edges)] = f_hopping
    fsys = sys.finalized()

    assert_equal(fsys.graph.num_nodes, len(tags))
    assert_equal(fsys.graph.num_edges, 2 * len(edges))

    for i in range(len(tags)):
        site = fsys.site(i)
        assert site in sites
        assert_equal(fsys.hamiltonian(i, i),
                     sys[site](site))

    for t, h in fsys.graph:
        tsite = fsys.site(t)
        hsite = fsys.site(h)
        assert_equal(fsys.hamiltonian(t, h),
                     sys[tsite, hsite](tsite, hsite))


def test_dangling():
    def make_system():
        #        1
        #       / \
        #    3-0---2-4-5  6-7  8
        sys = builder.Builder()
        sg = builder.SimpleSiteGroup()
        sys[(sg(i) for i in range(9))] = None
        sys[[(sg(0), sg(1)), (sg(1), sg(2)), (sg(2), sg(0))]] = None
        sys[[(sg(0), sg(3)), (sg(2), sg(4)), (sg(4), sg(5))]] = None
        sys[sg(6), sg(7)] = None
        return sys

    sys0 = make_system()
    assert_equal(sorted(site.tag for site in sys0.dangling()),
                 sorted([(3,), (5,), (6,), (7,), (8,)]))
    sys0.eradicate_dangling()

    sys1 = make_system()
    while True:
        dangling = list(sys1.dangling())
        if not dangling:
            break
        del sys1[dangling]

    assert_equal(sorted(site.tag for site in sys0.sites()),
                 sorted([(0,), (1,), (2,)]))
    assert_equal(sorted(site.tag for site in sys0.sites()),
                 sorted(site.tag for site in sys1.sites()))


def test_builder_with_symmetry():
    g = kwant.lattice.general(ta.identity(3))
    sym = kwant.TranslationalSymmetry((0, 0, 3), (0, 2, 0))
    sys = builder.Builder(sym)

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

    sys[(a for a, b in hoppings)] = V
    sys[hoppings] = t

    # TODO: Once tinyarray supports "<" the conversion to tuple can be removed.
    assert_equal(sorted(tuple(site.tag) for site in sys.sites()),
                 sorted(set(tuple(a.tag) for a, b in hoppings_fd)))
    for sites in hoppings_fd:
        for site in sites:
            assert site in sys
            assert_equal(sys[site], V)

    # TODO: Once tinyarray supports "<" the conversion to tuple can be removed.
    assert_equal(sorted((tuple(a.tag), tuple(b.tag))
                        for a, b in sys.hoppings()),
                 sorted((tuple(a.tag), tuple(b.tag)) for a, b in hoppings_fd))
    for hop in hoppings_fd:
        rhop = hop[1], hop[0]
        assert hop in sys
        assert rhop in sys
        assert_equal(sys[hop], t)
        assert_equal(sys[rhop], t.conjugate())

    del sys[g(0, 6, -4), g(0, 11, -9)]
    assert (g(0, 1, 0), g(0, -4, 5)) not in sys

    del sys[g(0, 3, -3)]
    assert_equal(list((a.tag, b.tag) for a, b in sys.hoppings()),
                 [((0, 0, 2), (5, 0, -3))])


def test_attach_lead():
    gr = builder.SimpleSiteGroup()

    sys = builder.Builder()
    sys[gr(1)] = 0
    lead0 = builder.Builder(VerySimpleSymmetry(-2))
    assert_raises(ValueError, sys.attach_lead, lead0)

    lead0.default_site_group = gr
    lead0[gr(0)] = lead0[gr(1)] = 1

    sys2 = builder.Builder()
    sys2.default_site_group = gr
    sys2[gr(1)] = 0
    assert_raises(ValueError, sys2.attach_lead, lead0)

    lead0[gr(0), gr(1)] = lead0[gr(0), gr(2)] = 1
    assert_raises(ValueError, sys.attach_lead, lead0)

    sys[gr(0)] = 1
    assert_raises(ValueError, sys.attach_lead, lead0, gr(5))

    sys = builder.Builder()
    sys[gr(1)] = 0
    sys[gr(0)] = 1
    sys.attach_lead(lead0)
    assert_equal(len(list(sys.sites())), 3)
    assert_equal(set(sys.leads[0].interface), set([gr(-1), gr(0)]))
    sys[gr(-10)] = sys[gr(-11)] = 0
    sys.attach_lead(lead0)
    assert_equal(set(sys.leads[1].interface), set([gr(-10), gr(-11)]))
    assert_equal(len(list(sys.sites())), 5)
    sys.attach_lead(lead0, gr(-5))
    assert_equal(set(sys.leads[0].interface), set([gr(-1), gr(0)]))
    sys.finalized()


def test_neighbors_not_in_single_domain():
    sr = builder.Builder()
    lead = builder.Builder(VerySimpleSymmetry(-1))
    gr = builder.SimpleSiteGroup()
    sr[(gr(x, y) for x in range(3) for y in range(3) if x >= y)] = 0
    sr[sr.possible_hoppings((1, 0), gr, gr)] = 1
    sr[sr.possible_hoppings((0, 1), gr, gr)] = 1
    lead[(gr(0, y) for y in range(3))] = 0
    lead[((gr(0, y), gr(1, y)) for y in range(3))] = 1
    lead[((gr(0, y), gr(0, y + 1)) for y in range(2))] = 1
    sr.leads.append(builder.BuilderLead(lead, [gr(i, i) for i in range(3)]))
    assert_raises(ValueError, sr.finalized)


def test_iadd():
    lat = builder.SimpleSiteGroup()

    sys = builder.Builder()
    sys[[lat(0,), lat(1,)]] = 1
    sys[lat(0,), lat(1,)] = 1

    other_sys = builder.Builder()
    other_sys[[lat(1,), lat(2,)]] = 2
    other_sys[lat(1,), lat(2,)] = 1

    lead0 = builder.Builder(VerySimpleSymmetry(-1))
    lead0[lat(0,)] = 1
    lead0[(lat(0,), lat(1,))] = 1
    lead0 = builder.BuilderLead(lead0, [lat(0,)])
    sys.leads.append(lead0)

    lead1 = builder.Builder(VerySimpleSymmetry(1))
    lead1[lat(2,)] = 1
    lead1[(lat(2,), lat(1,))] = 1
    lead1 = builder.BuilderLead(lead1, [lat(2,)])
    other_sys.leads.append(lead1)

    sys += other_sys
    assert_equal(sys.leads, [lead0, lead1])
    expected = sorted([[(0,), 1], [(1,), 2], [(2,), 2]])
    assert_equal(sorted(((s.tag, v) for s, v in sys.site_value_pairs())),
                 expected)
    expected = sorted([[(0,), (1,), 1], [(1,), (2,), 1]])
    assert_equal(sorted(((a.tag, b.tag, v)
                         for (a, b), v in sys.hopping_value_pairs())),
                 expected)


# y=0:    y=1:
#
# hhhh    hhhh
# gggh    gggh
# hhgh    hhgh
# ghgh    hhgh
#
def test_possible_hoppings():
    g = kwant.lattice.general(ta.identity(3), name='some_lattice')
    h = kwant.lattice.general(ta.identity(3), name='another_lattice')
    sym = kwant.TranslationalSymmetry((0, 2, 0))
    sys = builder.Builder(sym)
    sys[((h if max(x, y, z) % 2 else g)(x, y, z)
         for x in range(4) for y in range(2) for z in range(4))] = None
    for delta, group_a, group_b, n in [((1, 0, 0), g, h, 4),
                                       ((1, 0, 0), h, g, 7),
                                       ((0, 1, 0), g, h, 1),
                                       ((0, 4, 0), h, h, 21),
                                       ((0, 0, 1), g, h, 4)
                                       ]:
        ph = list(sys.possible_hoppings(delta, group_a, group_b))
        assert_equal(len(ph), n)
        ph = set(ph)
        assert_equal(len(ph), n)

        ph2 = list((
                sym.to_fd(b, a) for a, b in
                sys.possible_hoppings(ta.negative(delta), group_b, group_a)))
        assert_equal(len(ph2), n)
        ph2 = set(ph2)
        assert_equal(ph2, ph)

        for a, b in ph:
            assert a.group == group_a
            assert b.group == group_b
            assert sym.to_fd(a) == a
            assert_equal(a.tag - b.tag, delta)
