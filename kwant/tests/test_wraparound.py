# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import tempfile
import itertools
import numpy as np
from numpy.testing import assert_equal
import tinyarray as ta
import pytest

import kwant
from kwant import _plotter
from kwant.wraparound import wraparound, plot_2d_bands
from kwant._common import get_parameters

if _plotter.mpl_available:
    from mpl_toolkits import mplot3d  # pragma: no flakes
    from matplotlib import pyplot  # pragma: no flakes


def _simple_syst(lat, E=0, t=1+1j, sym=None):
    """Create a builder for a simple infinite system."""
    if not sym:
        sym = kwant.TranslationalSymmetry(lat.vec((1, 0)), lat.vec((0, 1)))
    # Build system with 2d periodic BCs. This system cannot be finalized in
    # Kwant <= 1.2.
    syst = kwant.Builder(sym)
    syst[lat.shape(lambda p: True, (0, 0))] = E
    syst[lat.neighbors(1)] = t
    return syst


def test_consistence_with_bands(kx=1.9, nkys=31):
    kys = np.linspace(-np.pi, np.pi, nkys)
    for lat in [kwant.lattice.honeycomb(), kwant.lattice.square()]:
        syst = _simple_syst(lat)
        wa_keep_1 = wraparound(syst, keep=1).finalized()
        wa_keep_none = wraparound(syst).finalized()

        bands = kwant.physics.Bands(wa_keep_1, params=dict(k_x=kx))
        energies_a = [bands(ky) for ky in kys]

        energies_b = []
        for ky in kys:
            params = dict(k_x=kx, k_y=ky)
            H = wa_keep_none.hamiltonian_submatrix(params=params, sparse=False)
            evs = np.sort(np.linalg.eigvalsh(H).real)
            energies_b.append(evs)

        np.testing.assert_almost_equal(energies_a, energies_b)


def test_opposite_hoppings():
    lat = kwant.lattice.square()

    for val in [1j, lambda a, b: 1j]:
        syst = kwant.Builder(kwant.TranslationalSymmetry((1, 1)))
        syst[ (lat(x, 0) for x in [-1, 0]) ] = 0
        syst[lat(0, 0), lat(-1, 0)] = val
        syst[lat(-1, 0), lat(-1, -1)] = val

        fsyst = wraparound(syst).finalized()
        params = dict(k_x=0)
        np.testing.assert_almost_equal(
            fsyst.hamiltonian_submatrix(params=params),
            0)


def test_value_types(k=(-1.1, 0.5), E=2, t=1):
    k = dict(zip(('k_x', 'k_y', 'k_z'), k))
    sym_extents = [1, 2, 3]
    lattices = [kwant.lattice.honeycomb(), kwant.lattice.square()]
    lat_syms = [
        (lat, kwant.TranslationalSymmetry(lat.vec((n, 0)), lat.vec((0, n))))
        for n, lat in itertools.product(sym_extents, lattices)
    ]
    for lat, sym in lat_syms:
        syst = wraparound(_simple_syst(lat, E, t, sym)).finalized()
        H = syst.hamiltonian_submatrix(params=k, sparse=False)
        for E1, t1 in [(float(E), float(t)),
                       (np.array([[E]], float), np.array([[t]], float)),
                       (ta.array([[E]], float), ta.array([[t]], float))]:
            # test when Hamiltonian values do not take any extra parameters
            # (only 'k' needs to be passed)
            for E2 in [E1, lambda a: E1]:
                for t2 in [t1, lambda a, b: t1]:
                    syst = wraparound(_simple_syst(lat, E2, t2, sym)).finalized()
                    H_alt = syst.hamiltonian_submatrix(params=k, sparse=False)
                    np.testing.assert_equal(H_alt, H)
            # test when Hamiltonian value functions take extra parameters and
            # have incompatible signatures (must be passed with 'params')
            onsites = [
                lambda a, E: E,
                lambda a, E, t: E,
                lambda a, t, E: E,
            ]
            hoppings = [
                lambda a, b, t: t,
                lambda a, b, t, E: t,
                lambda a, b, E, t: t,
            ]
            params = dict(E=E1, t=t1, **k)
            for E2, t2 in itertools.product(onsites, hoppings):
                syst = wraparound(_simple_syst(lat, E2, t2, sym)).finalized()
                H_alt = syst.hamiltonian_submatrix(params=params, sparse=False)
                np.testing.assert_equal(H_alt, H)


def test_signatures():
    lat = kwant.lattice.square()
    syst = kwant.Builder(kwant.TranslationalSymmetry((-3, 0), (0, 1)))
    # onsites and hoppings that will be bound as sites
    syst[lat(-2, 0)] = 4
    syst[(lat(-2, 0), lat(-2, 1))] = -1
    #
    syst[lat(-1, 0)] = lambda a, E1: E1
    syst[(lat(-1, 0), lat(-1, 1))] = lambda a, b, t1: t1
    #
    syst[lat(0, 0)] =  lambda a, E2: E2
    syst[(lat(0, 0), lat(0, 1))] = lambda a, b, t2: t2

    # hoppings that will be bound as hoppings
    syst[(lat(-2, 0), lat(-1, 0))] = -1
    syst[(lat(-2, 0), lat(2, 0))] = -1
    #
    syst[(lat(-2, 0), lat(0, 0))] = -1
    syst[(lat(-2, 0), lat(3, 0))] = lambda a, b, t3: t3

    syst[(lat(-1, 0), lat(0, 0))] = lambda a, b, t4: t4
    syst[(lat(-1, 0), lat(3, 0))] = lambda a, b, t5: t5

    wrapped_syst = wraparound(syst)

    ## testing

    momenta = ('k_x', 'k_y')

    onsites = [
        (lat(-2, 0), momenta),
        (lat(-1, 0), ('E1', 't1') + momenta),
        (lat(0, 0), ('E2', 't2') + momenta),
    ]

    for site, params_should_be in onsites:
        params = get_parameters(wrapped_syst[site])
        assert params[1:] == params_should_be

    hoppings = [
        ((lat(-2, 0), lat(-1, 0)), momenta),
        ((lat(-2, 0), lat(0, 0)), ('t3',) + momenta),
        ((lat(-1, 0), lat(0, 0)), ('t4', 't5') + momenta),
    ]

    for hopping, params_should_be in hoppings:
        params = get_parameters(wrapped_syst[hopping])
        assert params[2:] == params_should_be


def test_symmetry():
    syst = _simple_syst(kwant.lattice.square())

    matrices = [np.random.rand(2, 2) for i in range(4)]
    laws = (matrices, [(lambda a: m) for m in matrices])
    for cl, ch, ph, tr in laws:
        syst.conservation_law = cl
        syst.chiral = ch
        syst.particle_hole = ph
        syst.time_reversal = tr

        with pytest.warns(RuntimeWarning):
            wrapped = wraparound(syst)

        assert wrapped.time_reversal is None
        assert wrapped.particle_hole is None
        for attr in ('conservation_law', 'chiral'):
            new = getattr(wrapped, attr)
            orig = getattr(syst, attr)
            if callable(orig):
                params = get_parameters(new)
                assert params[1:] == ('k_x', 'k_y')
                assert np.all(orig(None) == new(None, None, None))
            else:
                assert np.all(orig == new)


@pytest.mark.skipif(not _plotter.mpl_available, reason="Matplotlib unavailable.")
def test_plot_2d_bands():
    chain = kwant.lattice.chain()
    square = kwant.lattice.square()
    cube = kwant.lattice.general([(1, 0, 0), (0, 1, 0), (0, 0, 1)])
    hc = kwant.lattice.honeycomb()

    syst_1d = kwant.Builder(kwant.TranslationalSymmetry(*chain._prim_vecs))
    syst_1d[chain(0)] = 2
    syst_1d[chain.neighbors()] = -1

    syst_2d = _simple_syst(square, t=-1)
    syst_graphene = _simple_syst(hc, t=-1)

    syst_3d = kwant.Builder(kwant.TranslationalSymmetry(*cube._prim_vecs))
    syst_3d[cube(0, 0, 0)] = 6
    syst_3d[cube.neighbors()] = -1


    with tempfile.TemporaryFile('w+b') as out:
        # test 2D
        plot_2d_bands(wraparound(syst_2d).finalized(), k_x=11, k_y=11, file=out)
        plot_2d_bands(wraparound(syst_graphene).finalized(), k_x=11, k_y=11,
                   file=out)

    # test non-wrapped around system
    with pytest.raises(TypeError):
        plot_2d_bands(syst_1d.finalized())
    # test incompletely wrapped around system
    with pytest.raises(TypeError):
        plot_2d_bands(wraparound(syst_2d, keep=0).finalized())
    # test incorrect lattice dimention (1, 3)
    with pytest.raises(ValueError):
        plot_2d_bands(wraparound(syst_1d).finalized())
    with pytest.raises(ValueError):
        plot_2d_bands(wraparound(syst_3d).finalized())

    # test k_x and k_y differ
    with tempfile.TemporaryFile('w+b') as out:
        syst = wraparound(syst_2d).finalized()
        plot_2d_bands(syst, k_x=11, k_y=15, file=out)
        plot_2d_bands(syst, k_x=np.linspace(-np.pi, np.pi, 11), file=out)
        plot_2d_bands(syst, k_y=np.linspace(-np.pi, np.pi, 11), file=out)

        syst = wraparound(syst_graphene).finalized()
        # test extend_bbox2d
        plot_2d_bands(syst, extend_bbox=1.2, k_x=11, k_y=11, file=out)
        # test mask Brillouin zone
        plot_2d_bands(syst, mask_brillouin_zone=True, k_x=11, k_y=11, file=out)


def test_fd_mismatch():
    # The fundamental domains of the two 1D symmetries that make up T are
    # incompatible. This produced a bug where certain systems could be wrapped
    # around in all directions, but could not be wrapped around when 'keep' is
    # provided.
    sqrt3 = np.sqrt(3)
    lat = kwant.lattice.general([(sqrt3, 0), (-sqrt3/2, 1.5)])
    T = kwant.TranslationalSymmetry((sqrt3, 0), (0, 3))

    syst1 = kwant.Builder(T)
    syst1[lat(1, 1)] = syst1[lat(0, 1)] = 1
    syst1[lat(1, 1), lat(0, 1)] = 1

    syst2 = kwant.Builder(T)
    syst2[lat(0, 0)] = syst2[lat(0, -1)] = 1
    syst2[lat(0, 0), lat(0, -1)] = 1

    # combine the previous two
    syst3 = kwant.Builder(T)
    syst3.update(syst1)
    syst3.update(syst2)

    for syst in (syst1, syst2, syst3):
        wraparound(syst)
        wraparound(syst, keep=0)
        wraparound(syst, keep=1)

    ## Test that spectrum of non-trivial system (including above cases)
    ## is the same, regardless of the way in which it is wrapped around
    lat = kwant.lattice.general([(sqrt3, 0), (-sqrt3/2, 1.5)],
                                [(sqrt3 / 2, 0.5), (0, 1)])
    a, b = lat.sublattices
    T = kwant.TranslationalSymmetry((3 * sqrt3, 0), (0, 3))
    syst = kwant.Builder(T)
    syst[(l(i, j) for i in range(-1, 2) for j in range(-1, 2)
         for l in (a, b))] = 0
    syst[kwant.HoppingKind((0, 0), b, a)] = -1j
    syst[kwant.HoppingKind((1, 0), b, a)] = 2j
    syst[kwant.HoppingKind((0, 1), a, b)] = -2j

    def spectrum(syst, keep):
        syst = wraparound(syst, keep=keep).finalized()
        ks = ('k_x', 'k_y', 'k_z')
        if keep is None:
            def _(*args):
                params = dict(zip(ks, args))
                return np.linalg.eigvalsh(
                    syst.hamiltonian_submatrix(params=params))
        else:
            def _(*args):
                params = dict(zip(ks, args))
                kext = params.pop(ks[keep])
                B = kwant.physics.Bands(syst, params=params)
                return B(kext)
        return _

    spectra = [spectrum(syst, keep) for keep in (None, 0, 1)]
    # Check that spectra are the same at k=0. Checking finite k
    # is more tricky, as we would also need to transform the k vectors
    E_k = np.array([spec(0, 0) for spec in spectra]).transpose()
    assert all(np.allclose(E, E[0]) for E in E_k)

    # Test square lattice with oblique unit cell
    lat = kwant.lattice.general(np.eye(2))
    translations = kwant.lattice.TranslationalSymmetry([2, 2], [0, 2])
    syst = kwant.Builder(symmetry=translations)
    syst[lat.shape(lambda site: True, [0, 0])] = 1
    syst[lat.neighbors()] = 1
    # Check that spectra are the same at k=0.
    spectra = [spectrum(syst, keep) for keep in (None, 0, 1)]
    E_k = np.array([spec(0, 0) for spec in spectra]).transpose()
    assert all(np.allclose(E, E[0]) for E in E_k)

    # Test Rocksalt structure
    # cubic lattice that contains both sublattices
    lat = kwant.lattice.general(np.eye(3))
    # Builder with FCC translational symmetries.
    translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, 0, 1], [0, 1, 1])
    syst = kwant.Builder(symmetry=translations)
    syst[lat(0, 0, 0)] = 1
    syst[lat(0, 0, 1)] = -1
    syst[lat.neighbors()] = 1
    # Check that spectra are the same at k=0.
    spectra = [spectrum(syst, keep) for keep in (None, 0, 1, 2)]
    E_k = np.array([spec(0, 0, 0) for spec in spectra]).transpose()
    assert all(np.allclose(E, E[0]) for E in E_k)
    # Same with different translation vectors
    translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, -1, 0], [0, 1, 1])
    syst = kwant.Builder(symmetry=translations)
    syst[lat(0, 0, 0)] = 1
    syst[lat(0, 0, 1)] = -1
    syst[lat.neighbors()] = 1
    # Check that spectra are the same at k=0.
    spectra = [spectrum(syst, keep) for keep in (None, 0, 1, 2)]
    E_k = np.array([spec(0, 0, 0) for spec in spectra]).transpose()
    assert all(np.allclose(E, E[0]) for E in E_k)

    # Test that spectrum in slab geometry is identical regardless of choice of unit
    # cell in rocksalt structure
    def shape(site):
        return abs(site.tag[2]) < 4

    lat = kwant.lattice.general(np.eye(3))
    # First choice: primitive UC
    translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, -1, 0], [1, 0, 1])
    syst = kwant.Builder(symmetry=translations)
    syst[lat(0, 0, 0)] = 1
    syst[lat(0, 0, 1)] = -1
    # Set all the nearest neighbor hoppings
    for d in [[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]:
        syst[(lat(0, 0, 0), lat(*d))] = 1

    wrapped = kwant.wraparound.wraparound(syst, keep=2)
    finitewrapped = kwant.Builder()
    finitewrapped.fill(wrapped, shape, start=np.zeros(3));

    sysf = finitewrapped.finalized()
    spectrum1 = [np.linalg.eigvalsh(
                    sysf.hamiltonian_submatrix(params=dict(k_x=k, k_y=0)))
                for k in np.linspace(-np.pi, np.pi, 5)]

    # Second choice: doubled UC with third translation purely in z direction
    translations = kwant.lattice.TranslationalSymmetry([1, 1, 0], [1, -1, 0], [0, 0, 2])
    syst = kwant.Builder(symmetry=translations)
    syst[lat(0, 0, 0)] = 1
    syst[lat(0, 0, 1)] = -1
    syst[lat(1, 0, 1)] = 1
    syst[lat(-1, 0, 0)] = -1
    for s in np.array([[0, 0, 0], [1, 0, 1]]):
        for d in np.array([[1, 0, 0], [-1, 0, 0], [0, 1, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1]]):
            syst[(lat(*s), lat(*(s + d)))] = 1
    wrapped = kwant.wraparound.wraparound(syst, keep=2)
    finitewrapped = kwant.Builder()

    finitewrapped.fill(wrapped, shape, start=np.zeros(3));

    sysf = finitewrapped.finalized()
    spectrum2 = [np.linalg.eigvalsh(
                    sysf.hamiltonian_submatrix(params=dict(k_x=k, k_y=0)))
                for k in np.linspace(-np.pi, np.pi, 5)]

    assert np.allclose(spectrum1, spectrum2)


# There seems no more specific way to only filter KwantDeprecationWarning.
@pytest.mark.filterwarnings('ignore')
def test_args_params_equivalence():
    for lat in [kwant.lattice.square(), kwant.lattice.honeycomb(),
                kwant.lattice.kagome()]:
        syst = kwant.Builder(kwant.TranslationalSymmetry(*lat.prim_vecs))
        syst[lat.shape((lambda pos: True), (0, 0))] = 1
        syst[lat.neighbors(1)] = 0.1
        syst[lat.neighbors(2)] = lambda a, b, param: 0.01
        syst = wraparound(syst).finalized()
        shs = syst.hamiltonian_submatrix
        assert_equal(shs(args=["bla", 1, 2]),
                     shs(params=dict(param="bla", k_x=1, k_y=2)))
