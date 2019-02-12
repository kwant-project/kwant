from collections import namedtuple, Counter
import warnings
from math import sqrt
import numpy as np
import pytest

import kwant
from ... import lattice
from ...builder import HoppingKind, Builder, NoSymmetry, Site
from .. import gauge


## Utilities

# TODO: remove in favour of 'scipy.stats.special_ortho_group' once
#       we depend on scipy 0.18
class special_ortho_group_gen:

    def rvs(self, dim):
        H = np.eye(dim)
        D = np.empty((dim,))
        for n in range(dim-1):
            x = np.random.normal(size=(dim-n,))
            D[n] = np.sign(x[0]) if x[0] != 0 else 1
            x[0] += D[n]*np.sqrt((x*x).sum())
            # Householder transformation
            Hx = (np.eye(dim-n)
                  - 2.*np.outer(x, x)/(x*x).sum())
            mat = np.eye(dim)
            mat[n:, n:] = Hx
            H = np.dot(H, mat)
        D[-1] = (-1)**(dim-1)*D[:-1].prod()
        # Equivalent to np.dot(np.diag(D), H) but faster, apparently
        H = (D*H.T).T
        return H

special_ortho_group = special_ortho_group_gen()


square_lattice = lattice.square(norbs=1, name='square')
honeycomb_lattice = lattice.honeycomb(norbs=1, name='honeycomb')
cubic_lattice = lattice.cubic(norbs=1, name='cubic')

def rectangle(W, L):
    return (
        lambda s: 0 <= s.pos[0] < L and 0 <= s.pos[1] < W,
        (L/2, W/2)
    )


def ring(r_inner, r_outer):
    return (
        lambda s: r_inner <= np.linalg.norm(s.pos) <= r_outer,
        ((r_inner + r_outer) / 2, 0)
    )

def wedge(W):
    return (
        lambda s: (0 <= s.pos[0] < W) and (0 <= s.pos[1] <= s.pos[0]),
        (0, 0)
    )


def half_ring(r_inner, r_outer):
    in_ring, _ = ring(r_inner, r_outer)
    return (
        lambda s: s.pos[0] <= 0 and in_ring(s),
        (-(r_inner + r_outer) / 2, 0)
    )


def cuboid(a, b, c):
    return (
        lambda s: 0 <= s.pos[0] < a and 0 <= s.pos[1] < b and 0 <= s.pos[2] < c,
        (a/2, b/2, c/2)
    )

def hypercube(dim, W):
    return (
        lambda s: all(0 <= x < W for x in s.pos),
        (W / 2,) * dim
    )


def circle(r):
    return (
        lambda s: np.linalg.norm(s.pos) < r,
        (0, 0)
    )

def ball(dim, r):
    return (
        lambda s: np.linalg.norm(s.pos) < r,
        (0,) * dim
    )


def model(lat, neighbors):
    syst = Builder(lattice.TranslationalSymmetry(*lat.prim_vecs))
    if hasattr(lat, 'sublattices'):
        for l in lat.sublattices:
            zv = (0,) * len(l.prim_vecs)
            syst[l(*zv)] = None
    else:
        zv = (0,) * len(l.prim_vecs)
        syst[lat(*zv)] = None
    for r in range(neighbors):
        syst[lat.neighbors(r + 1)] = None
    return syst


def check_loop_kind(loop_kind):
    (_, first_fam_a, prev_fam_b), *rest = loop_kind
    for (_, fam_a, fam_b) in rest:
        if prev_fam_b != fam_a:
            raise ValueError('Invalid loop kind: does not close')
        prev_fam_b = fam_b
    # loop closes
    net_delta = np.sum([hk.delta for hk in loop_kind])
    if first_fam_a != fam_b or np.any(net_delta != 0):
        raise ValueError('Invalid loop kind: does not close')


def available_loops(syst, loop_kind):

    def maybe_loop(site):
        loop = [site]
        a = site
        for delta, family_a, family_b in loop_kind:
            b = Site(family_b, a.tag + delta, True)
            if family_a != a.family or (a, b) not in syst:
                return None
            loop.append(b)
            a = b
        return loop

    check_loop_kind(loop_kind)

    return list(filter(None, map(maybe_loop, syst.sites())))


def loop_to_links(loop):
    return list(zip(loop, loop[1:]))


def no_symmetry(lat, neighbors):
    return NoSymmetry()


def translational_symmetry(lat, neighbors):
    return lattice.TranslationalSymmetry(int((neighbors + 1)/2) * lat.prim_vecs[0])


## Tests

# Tests that phase around a loop is equal to the flux through the loop.
# First we define the loops that we want to test, for various lattices.
# If a system does not support a particular kind of loop, they will simply
# not be generated.

Loop = namedtuple('Loop', ('path', 'flux'))

square_loops = [([HoppingKind(d, square_lattice) for d in l.path], l.flux)
        for l in [
    # 1st nearest neighbors
    Loop(path=[(1, 0), (0, 1), (-1, 0), (0, -1)], flux=1),
    # 2nd nearest neighbors
    Loop(path=[(1, 0), (0, 1), (-1, -1)], flux=0.5),
    Loop(path=[(1, 0), (-1, 1), (0, -1)], flux=0.5),
    # 3rd nearest neighbors
    Loop(path=[(2, 0), (0, 1), (-2, 0), (0, -1)], flux=2),
    Loop(path=[(2, 0), (-1, 1), (-1, 0), (0, -1)], flux=1.5),
]]

a, b = honeycomb_lattice.sublattices
honeycomb_loops = [([HoppingKind(d, a, b) for *d, a, b in l.path], l.flux)
        for l in [
    # 1st nearest neighbors
    Loop(path=[(0, 0, a, b), (-1, 1, b, a), (0, -1, a, b), (0, 0, b, a),
               (1, -1, a, b), (0, 1, b, a)],
         flux=sqrt(3)/2),
    # 2nd nearest neighbors
    Loop(path=[(-1, 1, a, a), (0, -1, a, a), (1, 0, a, a)],
         flux=sqrt(3)/4),
    Loop(path=[(-1, 0, b, b), (1, -1, b, b), (0, 1, b, b)],
         flux=sqrt(3)/4),
]]

cubic_loops = [([HoppingKind(d, cubic_lattice) for d in l.path], l.flux)
        for l in [
    # 1st nearest neighbors
    Loop(path=[(1, 0, 0), (0, 1, 0), (-1, 0, 0), (0, -1, 0)], flux=1),
    Loop(path=[(0, 1, 0), (0, 0, 1), (0, -1, 0), (0, 0, -1)], flux=0),
    Loop(path=[(1, 0, 0), (0, 0, 1), (-1, 0, 0), (0, 0, -1)], flux=0),
    # 2nd nearest neighbors
    Loop(path=[(1, 0, 0), (-1, 1, 0), (0, -1, 0)], flux=0.5),
    Loop(path=[(1, 0, 0), (0, 1, 0), (-1, -1, 0)], flux=0.5),
    Loop(path=[(1, 0, 0), (-1, 0, 1), (0, 0, -1)], flux=0),
    Loop(path=[(1, 0, 0), (0, 0, 1), (-1, 0, -1)], flux=0),
    Loop(path=[(0, 1, 0), (0, -1, 1), (0, 0, -1)], flux=0),
    Loop(path=[(0, 1, 0), (0, 0, 1), (0, -1, -1)], flux=0),
    # 3rd nearest neighbors
    Loop(path=[(1, 1, 1), (0, 0, -1), (-1, -1, 0)], flux=0),
    Loop(path=[(1, 1, 1), (-1, 0, -1), (0, -1, 0)], flux=0.5),
]]

square = (square_lattice, square_loops)
honeycomb = (honeycomb_lattice, honeycomb_loops)
cubic = (cubic_lattice, cubic_loops)


def _test_phase_loops(syst, phases, loops):
    for loop_kind, loop_flux in loops:
        for loop in available_loops(syst, loop_kind):
            loop_phase = np.prod([phases(a, b) for a, b in loop_to_links(loop)])
            expected_loop_phase = np.exp(1j * np.pi * loop_flux)
            assert np.isclose(loop_phase, expected_loop_phase)


@pytest.mark.parametrize("neighbors", [1, 2, 3])
@pytest.mark.parametrize("symmetry", [no_symmetry, translational_symmetry],
                         ids=['finite', 'infinite'])
@pytest.mark.parametrize("lattice, loops", [square, honeycomb, cubic],
                         ids=['square', 'honeycomb', 'cubic'])
def test_phases(lattice, neighbors, symmetry, loops):
    """Check that the phases around common loops are equal to the flux, for
    finite and infinite systems with uniform magnetic field.
    """
    W = 4
    dim = len(lattice.prim_vecs)
    field = np.array([0, 0, 1]) if dim == 3 else 1

    syst = Builder(symmetry(lattice, neighbors))
    syst.fill(model(lattice, neighbors), *hypercube(dim, W))

    this_gauge = gauge.magnetic_gauge(syst.finalized())
    phases = this_gauge(field)

    _test_phase_loops(syst, phases, loops)


@pytest.mark.parametrize("neighbors", [1, 2, 3])
@pytest.mark.parametrize("lat, loops", [square, honeycomb],
                         ids=['square', 'honeycomb'])
def test_phases_composite(neighbors, lat, loops):
    """Check that the phases around common loops are equal to the flux, for
    composite systems with uniform magnetic field.
    """
    W = 4
    dim = len(lat.prim_vecs)
    field = np.array([0, 0, 1]) if dim == 3 else 1

    lead = Builder(lattice.TranslationalSymmetry(-lat.prim_vecs[0]))
    lead.fill(model(lat, neighbors), *hypercube(dim, W))

    # Case where extra sites are added and fields are same in
    # scattering region and lead.
    syst = Builder()
    syst.fill(model(lat, neighbors), *ball(dim, W + 1))
    extra_sites = syst.attach_lead(lead)
    assert extra_sites  # make sure we're testing the edge case with added sites

    this_gauge = gauge.magnetic_gauge(syst.finalized())
    # same field in scattering region and lead
    phases, lead_phases = this_gauge(field, field)

    # When extra sites are added to the central region we need to select
    # the correct phase function.
    def combined_phases(a, b):
        if a in extra_sites or b in extra_sites:
            return lead_phases(a, b)
        else:
            return phases(a, b)

    _test_phase_loops(syst, combined_phases, loops)
    _test_phase_loops(lead, lead_phases, loops)


@pytest.mark.parametrize("neighbors", [1, 2])
def test_overlapping_interfaces(neighbors):
    """Test composite systems with overlapping lead interfaces."""

    lat = square_lattice

    def _make_syst(edge, W=5):

        syst = Builder()
        syst.fill(model(lat, neighbors), *rectangle(W, W))

        leadx = Builder(lattice.TranslationalSymmetry((-1, 0)))
        leadx[(lat(0, j) for j in range(edge, W - edge))] = None
        for n in range(1, neighbors + 1):
            leadx[lat.neighbors(n)] = None

        leady = Builder(lattice.TranslationalSymmetry((0, -1)))
        leady[(lat(i, 0) for i in range(edge, W - edge))] = None
        for n in range(1, neighbors + 1):
            leady[lat.neighbors(n)] = None

        assert not syst.attach_lead(leadx)  # sanity check; no sites added
        assert not syst.attach_lead(leady)  # sanity check; no sites added

        return syst, leadx, leady

    # edge == 0: lead interfaces overlap
    # edge == 1: lead interfaces partition scattering region
    #            into 2 disconnected components
    for edge in (0, 1):
        syst, leadx, leady = _make_syst(edge)
        this_gauge = gauge.magnetic_gauge(syst.finalized())
        phases, leadx_phases, leady_phases = this_gauge(1, 1, 1)
        _test_phase_loops(syst, phases, square_loops)
        _test_phase_loops(leadx, leadx_phases, square_loops)
        _test_phase_loops(leady, leady_phases, square_loops)


def _make_square_syst(sym, neighbors=1):
    lat = square_lattice
    syst = Builder(sym)
    syst[(lat(i, j) for i in (0, 1) for j in (0, 1))] = None
    for n in range(1, neighbors + 1):
        syst[lat.neighbors(n)] = None
    return syst


def test_unfixable_gauge():
    """Check error is raised when we cannot fix the gauge."""

    leadx = _make_square_syst(lattice.TranslationalSymmetry((-1, 0)))
    leady = _make_square_syst(lattice.TranslationalSymmetry((0, -1)))

    # 1x2 line with 2 leads
    syst = _make_square_syst(NoSymmetry())
    del syst[[square_lattice(1, 0), square_lattice(1, 1)]]
    syst.attach_lead(leadx)
    syst.attach_lead(leadx.reversed())
    with pytest.raises(ValueError):
        gauge.magnetic_gauge(syst.finalized())

    # 2x2 square with leads attached from all 4 sides,
    # and nearest neighbor hoppings
    syst = _make_square_syst(NoSymmetry())
    # Until we add the last lead we have enough gauge freedom
    # to specify independent fields in the scattering region
    # and each of the leads. We check that no extra sites are
    # added as a sanity check.
    assert not syst.attach_lead(leadx)
    gauge.magnetic_gauge(syst.finalized())
    assert not syst.attach_lead(leady)
    gauge.magnetic_gauge(syst.finalized())
    assert not syst.attach_lead(leadx.reversed())
    gauge.magnetic_gauge(syst.finalized())
    # Adding the last lead removes our gauge freedom.
    assert not syst.attach_lead(leady.reversed())
    with pytest.raises(ValueError):
        gauge.magnetic_gauge(syst.finalized())

    # 2x2 square with 2 leads, but 4rd nearest neighbor hoppings
    syst = _make_square_syst(NoSymmetry())
    del syst[(square_lattice(1, 0), square_lattice(1, 1))]
    leadx = _make_square_syst(lattice.TranslationalSymmetry((-1, 0)))
    leadx[square_lattice.neighbors(4)] = None
    for lead in (leadx, leadx.reversed()):
        syst.attach_lead(lead)

    with pytest.raises(ValueError):
        gauge.magnetic_gauge(syst.finalized())


def _test_disconnected(syst):
    with pytest.raises(ValueError) as excinfo:
        gauge.magnetic_gauge(syst.finalized())
        assert 'unit cell not connected' in str(excinfo.value)

def test_invalid_lead():
    """Check error is raised when a lead unit cell is not connected
       within the unit cell itself.

       In order for the algorithm to work we need to be able to close
       loops within the lead. However we only add a single lead unit
       cell, so not all paths can be closed, even if the lead is
       connected.
    """
    lat = square_lattice

    lead = _make_square_syst(lattice.TranslationalSymmetry((-1, 0)),
                             neighbors=0)
    # Truly disconnected system
    # Ignore warnings to suppress Kwant's complaint about disconnected lead
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        _test_disconnected(lead)

    # 2 disconnected chains (diagonal)
    lead[(lat(0, 0), lat(1, 1))] = None
    lead[(lat(0, 1), lat(1, 0))] = None
    _test_disconnected(lead)

    lead = _make_square_syst(lattice.TranslationalSymmetry((-1, 0)),
                             neighbors=0)
    # 2 disconnected chains (horizontal)
    lead[(lat(0, 0), lat(1, 0))] = None
    lead[(lat(0, 1), lat(1, 1))] = None
    _test_disconnected(lead)

    # System has loops, but need 3 unit cells
    # to express them.
    lead[(lat(0, 0), lat(1, 1))] = None
    lead[(lat(0, 1), lat(1, 0))] = None
    _test_disconnected(lead)

# Test internal parts of magnetic_gauge

@pytest.mark.parametrize("shape",
                         [rectangle(5, 5), circle(4),
                          half_ring(5, 10)],
                         ids=['rectangle', 'circle', 'half-ring']
    )
@pytest.mark.parametrize("lattice", [square_lattice, honeycomb_lattice],
                         ids=['square', 'honeycomb'])
@pytest.mark.parametrize("neighbors", [1, 2, 3])
def test_minimal_cycle_basis(lattice, neighbors, shape):
    """Check that for lattice models on genus 0 shapes, nearly
       all loops have the same (minimal) length. This is not an
       equality, as there may be weird loops on the edges.
    """
    syst = Builder()
    syst.fill(model(lattice, neighbors), *shape)
    syst = syst.finalized()

    loops = gauge._loops_in_finite(syst)
    loop_counts = Counter(map(len, loops))
    min_loop = min(loop_counts)
    # arbitrarily allow 1% of slightly longer loops;
    # we don't make stronger guarantees about the quality
    # of our loop basis
    assert loop_counts[min_loop] / len(loops) > 0.99, loop_counts


def random_loop(n, max_radius=10, planar=False):
    """Return a loop of 'n' points.

    The loop is in the x-y plane if 'planar is False', otherwise
    each point is given a random perturbation in the z direction
    """
    theta = np.sort(2 * np.pi * np.random.rand(n))
    r = max_radius * np.random.rand(n)
    if planar:
        z = np.zeros((n,))
    else:
        z = 2 * (max_radius / 5) * (np.random.rand(n) - 1)
    return np.array([r * np.cos(theta), r * np.sin(theta), z]).transpose()


def test_constant_surface_integral():
    field_direction = np.random.rand(3)
    field_direction /= np.linalg.norm(field_direction)
    loop = random_loop(7)

    integral = gauge._surface_integral

    I = integral(lambda r: field_direction, loop)
    assert np.isclose(I, integral(field_direction, loop))
    assert np.isclose(I, integral(lambda r: field_direction, loop, average=True))


def circular_field(r_vec):
    return np.array([r_vec[1], -r_vec[0], 0])


def test_invariant_surface_integral():
    """Surface integral should be identical if we apply a random
       rotation to loop and vector field.
    """
    integral = gauge._surface_integral
    # loop with random orientation
    orig_loop = loop = random_loop(7)
    I = integral(circular_field, loop)
    for _ in range(4):
        rot = special_ortho_group.rvs(3)
        loop = orig_loop @ rot.transpose()
        assert np.isclose(I, integral(lambda r: rot @ circular_field(rot.transpose() @ r), loop))


@pytest.fixture
def system_and_gauge():
    def hopping(a, b, peierls):
        return -1 * peierls(a, b)

    syst = Builder()
    syst[(square_lattice(i, j) for i in range(3) for j in range(10))] = 4
    syst[square_lattice.neighbors()] = hopping

    lead = Builder(lattice.TranslationalSymmetry((-1, 0)))
    lead[(square_lattice(0, j) for j in range(10))] = 4
    lead[square_lattice.neighbors()] = hopping

    syst.attach_lead(lead.substituted(peierls='peierls_left'))
    syst.attach_lead(lead.reversed().substituted(peierls='peierls_right'))

    syst = syst.finalized()

    magnetic_gauge = gauge.magnetic_gauge(syst)

    return syst, magnetic_gauge


@pytest.mark.parametrize('B',[0, 0.1, lambda r: 0.1 * np.exp(-r[1]**2)])
def test_uniform_magnetic_field(system_and_gauge, B):
    syst, gauge = system_and_gauge

    peierls, peierls_left, peierls_right = gauge(B, B, B)

    params = dict(peierls=peierls, peierls_left=peierls_left,
                  peierls_right=peierls_right)

    s = kwant.smatrix(syst, energy=0.6, params=params)
    t = s.submatrix(1, 0)

    assert t.shape > (0, 0)  # sanity check
    assert np.allclose(np.abs(t)**2, np.eye(*t.shape))


def test_phase_sign(system_and_gauge):
    syst, gauge = system_and_gauge

    peierls, peierls_left, peierls_right = gauge(0.1, 0.1, 0.1)

    params = dict(peierls=peierls, peierls_left=peierls_left,
                  peierls_right=peierls_right)

    cut = [(square_lattice(1, j), square_lattice(0, j))
            for j in range(10)]
    J = kwant.operator.Current(syst, where=cut)
    J = J.bind(params=params)

    psi = kwant.wave_function(syst, energy=0.6, params=params)(0)[0]

    # Electrons incident from the left travel along the *top*
    # edge of the Hall bar in the presence of a magnetic field
    # out of the plane
    j = J(psi)
    j_bottom = sum(j[0:5])
    j_top = sum(j[5:10])

    assert np.isclose(j_top + j_bottom, 1)  # sanity check
    assert j_top > 0.9
