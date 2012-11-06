from __future__ import division
from random import Random
from StringIO import StringIO
from nose.tools import assert_raises, assert_not_equal
from numpy.testing import assert_equal
import tinyarray as ta
import kwant
from kwant import builder


def test_site_groups():
    sys = builder.Builder()
    sg = builder.SimpleSiteGroup()
    osg = builder.SimpleSiteGroup()

    assert_raises(KeyError, sys.__setitem__, (0, ), 7)
    sys[sg(0)] = 7
    assert_equal(sys[sg(0)], 7)
    assert_raises(KeyError, sys.__getitem__, (0, ))

    sys.default_site_group = sg
    sys[1,] = 123
    assert_equal(sys[1,], 123)
    assert_equal(sys[sg(1)], 123)
    assert_raises(KeyError, sys.__getitem__, osg(1))

    assert_equal(sg(-5).shifted((-2,), osg), osg(-7))


class VerySimpleSymmetry(builder.Symmetry):
    def __init__(self, period):
        self.period = period

    @property
    def num_directions(self):
        return 1

    def which(self, site):
        return ta.array((site.tag[0] // self.period,), int)

    def act(self, element, a, b=None):
        delta = (self.period * element[0],) + (len(a.tag) - 1) * (0,)
        if b is None:
            return a.shifted(delta)
        else:
            return a.shifted(delta), b.shifted(delta)


# The hoppings have to form a ring.  Some other implicit assumptions are also
# made.
def check_construction_and_indexing(sites, sites_fd, hoppings, hoppings_fd,
                                    failing_hoppings, sym=None):
    sys = builder.Builder(sym)
    sys.default_site_group = builder.SimpleSiteGroup()
    t, V = 1.0j, 0.0
    sys[sites] = V
    for site in sites:
        sys[site] = V
    sys[hoppings] = t
    for hopping in hoppings:
        sys[hopping] = t

    for hopping in failing_hoppings:
        assert_raises(KeyError, sys.__setitem__, hopping, t)

    assert (5, 123) not in sys
    assert (sites[0], (5, 123)) not in sys
    assert ((7, 8), sites[0]) not in sys
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
    assert_equal(sorted(s.tag for s in sys.neighbors(sites[0])),
                 sorted([sites[1], sites[-1]]))

    del sys[hoppings]
    assert_equal(list(sys.hoppings()), [])
    sys[hoppings] = t

    del sys[sites[0]]
    assert_equal(sorted(tuple(s.tag)
                        for s in sys.sites()), sorted(sites_fd[1:]))
    assert_equal(sorted((tuple(a.tag), tuple(b.tag))
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
    sites = [(0, 0), (0, 1), (1, 0)]
    hoppings = [((0, 0), (0, 1)),
                ((0, 1), (1, 0)),
                ((1, 0), (0, 0))]
    failing_hoppings = [((0, 1), (7, 8)), ((12, 14), (0, 1))]
    check_construction_and_indexing(sites, sites, hoppings, hoppings,
                                    failing_hoppings)

    # With symmetry
    sites = [(0, 0), (1, 1), (2, 1), (4, 2)]
    sites_fd = [(0, 0), (1, 1), (0, 1), (0, 2)]
    hoppings = [((0, 0), (1, 1)),
                ((1, 1), (2, 1)),
                ((2, 1), (4, 2)),
                ((4, 2), (0, 0))]
    hoppings_fd = [((0, 0), (1, 1)),
                   ((1, 1), (2, 1)),
                   ((0, 1), (2, 2)),
                   ((0, 2), (-4, 0))]
    failing_hoppings = [((0, 0), (0, 3)), ((0, 4), (0, 0)),
                        ((0, 0), (2, 3)), ((2, 4), (0, 0)),
                        ((4, 2), (6, 3)), ((6, 4), (4, 2))]
    sym = VerySimpleSymmetry(2)
    check_construction_and_indexing(sites, sites_fd, hoppings, hoppings_fd,
                                    failing_hoppings, sym)


def test_hermitian_conjugation():
    def f(i, j):
        if j[0] == i[0] + 1:
            return ta.array([[1, 2j], [3 + 1j, 4j]])
        else:
            raise ValueError

    sys = builder.Builder()
    sys.default_site_group = builder.SimpleSiteGroup()
    sys[0,] = sys[1,] = ta.identity(2)

    sys[(0,), (1,)] = f
    assert sys[(0,), (1,)] is f
    assert isinstance(sys[(1,), (0,)], builder.HermConjOfFunc)
    assert_equal(sys[(1,), (0,)]((1,), (0,)),
                 sys[(0,), (1,)]((0,), (1,)).conjugate().transpose())
    sys[(0,), (1,)] = sys[(1,), (0,)]
    assert isinstance(sys[(0,), (1,)], builder.HermConjOfFunc)
    assert sys[(1,), (0,)] is f


def test_value_equality_and_identity():
    m = ta.array([[1, 2], [3j, 4j]])
    sys = builder.Builder()
    sys.default_site_group = builder.SimpleSiteGroup()

    sys[0,] = m
    sys[1,] = m
    assert sys[1,] is m

    sys[(0,), (1,)] = m
    assert_equal(sys[(1,), (0,)], m.transpose().conjugate())
    assert sys[(0,), (1,)] is m

    sys[(1,), (0,)] = m
    assert_equal(sys[(0,), (1,)], m.transpose().conjugate())
    assert sys[(1,), (0,)] is m


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
        if value is builder.other:
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
    sys.default_site_group = sg = kwant.make_lattice(ta.identity(2))
    for site, value in sr_sites.iteritems():
        sys[site] = value
    for hop, value in sr_hops.iteritems():
        sys[hop] = value
    fsys = sys.finalized()
    check_onsite(fsys, sr_sites)
    check_hoppings(fsys, sr_hops)

    # Build lead from blueprint and test it.
    lead = builder.Builder(kwant.TranslationalSymmetry([(size, 0)]))
    lead.default_site_group = sg
    for site, value in lead_sites.iteritems():
        shift = rng.randrange(-5, 6) * size
        site = site[0] + shift, site[1]
        lead[site] = value
    for (a, b), value in lead_hops.iteritems():
        shift = rng.randrange(-5, 6) * size
        a = a[0] + shift, a[1]
        b = b[0] + shift, b[1]
        lead[a, b] = value
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
    lead[a, b] = random_hopping_integral(rng)
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
    sys.default_site_group = sg = builder.SimpleSiteGroup()
    sites = [sg(*tag) for tag in tags]
    sys[tags] = f_onsite
    sys[((tags[i], tags[j]) for (i, j) in edges)] = f_hopping
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
        sys.default_site_group = builder.SimpleSiteGroup()
        sys[((i,) for i in range(9))] = None
        sys[[((0,), (1,)), ((1,), (2,)), ((2,), (0,))]] = None
        sys[[((0,), (3,)), ((2,), (4,)), ((4,), (5,))]] = None
        sys[(6,), (7,)] = None
        return sys

    sys0 = make_system()
    assert_equal(sorted(site.tag for site in sys0.dangling()),
                 sorted([(3,), (5,), (6,), (7,), (8,)]))
    sys0.eradicate_dangling()

    sys1 = make_system()
    while True:
        dangling = list(sys1.dangling())
        if not dangling: break
        del sys1[dangling]

    assert_equal(sorted(site.tag for site in sys0.sites()),
                 sorted([(0,), (1,), (2,)]))
    assert_equal(sorted(site.tag for site in sys0.sites()),
                 sorted(site.tag for site in sys1.sites()))


def test_builder_with_symmetry():
    g = kwant.make_lattice(ta.identity(3))
    sym = kwant.TranslationalSymmetry([(0, 0, 3), (0, 2, 0)])
    bob = builder.Builder(sym)
    bob.default_site_group = g

    t, V = 1.0j, 0.0
    hoppings = [((5, 0, 0), (0, 5, 0)),
                ((0, 5, 0), (0, 0, 5)),
                ((0, 0, 5), (5, 0, 0)),
                ((0, 3, 0), (0, 0, 5)),
                ((0, 7, -6), (5, 6, -6))]
    hoppings_fd = [((5, 0, 0), (0, 5, 0)),
                   ((0, 1, 0), (0, -4, 5)),
                   ((0, 0, 2), (5, 0, -3)),
                   ((0, 1, 0), (0, -2, 5)),
                   ((0, 1, 0), (5, 0, 0))]

    bob[(a for a, b in hoppings)] = V
    bob[hoppings] = t

    # TODO: Once tinyarray supports "<" the conversion to tuple can be removed.
    assert_equal(sorted(tuple(site.tag) for site in bob.sites()),
                 sorted(set(a for a, b in hoppings_fd)))
    for sites in hoppings_fd:
        for site in sites:
            assert site in bob
            assert_equal(bob[site], V)

    # TODO: Once tinyarray supports "<" the conversion to tuple can be removed.
    assert_equal(sorted((tuple(a.tag), tuple(b.tag))
                        for a, b in bob.hoppings()),
                 sorted(hoppings_fd))
    for hop in hoppings_fd:
        rhop = hop[1], hop[0]
        assert hop in bob
        assert rhop in bob
        assert_equal(bob[hop], t)
        assert_equal(bob[rhop], t.conjugate())

    del bob[(0, 6, -4), (0, 11, -9)]
    assert ((0, 1, 0), (0, -4, 5)) not in bob

    del bob[0, 3, -3]
    assert_equal(list((a.tag, b.tag) for a, b in bob.hoppings()),
                 [((0, 0, 2), (5, 0, -3))])


def test_attach_lead():
    gr = builder.SimpleSiteGroup()

    sys = builder.Builder()
    sys.default_site_group = gr
    sys[(1,)] = 0
    lead0 = builder.Builder(VerySimpleSymmetry(-2))
    assert_raises(ValueError, sys.attach_lead, lead0)

    lead0.default_site_group = gr
    lead0[(0,)] = lead0[(1,)] = 1

    sys2 = builder.Builder()
    sys2.default_site_group = gr
    sys2[(1,)] = 0
    assert_raises(ValueError, sys2.attach_lead, lead0)

    lead0[(0,), (1,)] = lead0[(0,), (2,)] = 1
    assert_raises(ValueError, sys.attach_lead, lead0)

    sys[(0,)] = 1
    assert_raises(ValueError, sys.attach_lead, lead0, gr(5))

    sys = builder.Builder()
    sys.default_site_group = gr
    sys[(1,)] = 0
    sys[(0,)] = 1
    sys.attach_lead(lead0)
    assert_equal(len(list(sys.sites())), 3)
    assert_equal(set(sys.leads[0].interface), set([gr(-1), gr(0)]))
    sys[(-10,)] = sys[(-11,)] = 0
    sys.attach_lead(lead0)
    assert_equal(set(sys.leads[1].interface), set([gr(-10), gr(-11)]))
    assert_equal(len(list(sys.sites())), 5)
    sys.attach_lead(lead0, gr(-5))
    assert_equal(set(sys.leads[0].interface), set([gr(-1), gr(0)]))
    sys.finalized()


def test_neighbors_not_in_single_domain():
    sr = builder.Builder()
    lead = builder.Builder(VerySimpleSymmetry(-1))
    lat = builder.SimpleSiteGroup()
    sr.default_site_group = lead.default_site_group = lat
    sr[((x, y) for x in range(3) for y in range(3) if x >= y)] = 0
    sr[sr.possible_hoppings((1, 0), lat, lat)] = 1
    sr[sr.possible_hoppings((0, 1), lat, lat)] = 1
    lead[((0, y) for y in range(3))] = 0
    lead[(((0, y), (1, y)) for y in range(3))] = 1
    lead[(((0, y), (0, y + 1)) for y in range(2))] = 1
    sr.leads.append(builder.BuilderLead(lead, [lat(i, i) for i in range(3)]))
    assert_raises(ValueError, sr.finalized)
