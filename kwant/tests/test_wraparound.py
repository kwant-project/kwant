# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import itertools
import numpy as np
import tinyarray as ta

import kwant
from kwant.wraparound import wraparound
from kwant._common import get_parameters


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

        bands = kwant.physics.Bands(wa_keep_1, (kx,))
        energies_a = [bands(ky) for ky in kys]

        energies_b = []
        for ky in kys:
            H = wa_keep_none.hamiltonian_submatrix((kx, ky), sparse=False)
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
        np.testing.assert_almost_equal(fsyst.hamiltonian_submatrix([0]), 0)


def test_value_types(k=(-1.1, 0.5), E=2, t=1):
    sym_extents = [1, 2, 3]
    lattices = [kwant.lattice.honeycomb(), kwant.lattice.square()]
    lat_syms = [
        (lat, kwant.TranslationalSymmetry(lat.vec((n, 0)), lat.vec((0, n))))
        for n, lat in itertools.product(sym_extents, lattices)
    ]
    for lat, sym in lat_syms:
        syst = wraparound(_simple_syst(lat, E, t, sym)).finalized()
        H = syst.hamiltonian_submatrix(k, sparse=False)
        for E1, t1 in [(float(E), float(t)),
                       (np.array([[E]], float), np.array([[t]], float)),
                       (ta.array([[E]], float), ta.array([[t]], float))]:
            # test when Hamiltonian values do not take any extra parameters
            # (only 'k' needs to be passed)
            for E2 in [E1, lambda a: E1]:
                for t2 in [t1, lambda a, b: t1]:
                    syst = wraparound(_simple_syst(lat, E2, t2, sym)).finalized()
                    H_alt = syst.hamiltonian_submatrix(k, sparse=False)
                    np.testing.assert_equal(H_alt, H)
            # test when Hamiltonian value functions take extra parameters and
            # have compatible signatures (can be passed with 'args')
            onsites = [
                lambda a, E, t: E,
                lambda a, E, *args: E,
                lambda a, *args: args[0],
            ]
            hoppings = [
                lambda a, b, E, t: t,
                lambda a, b, E, *args: args[0],
                lambda a, b, *args: args[1],
            ]
            args = (E1, t1) + k
            for E2, t2 in itertools.product(onsites, hoppings):
                    syst = wraparound(_simple_syst(lat, E2, t2, sym)).finalized()
                    H_alt = syst.hamiltonian_submatrix(args, sparse=False)
                    np.testing.assert_equal(H_alt, H)
            # test when hamiltonian value functions take extra parameters and
            # have incompatible signaures (must be passed with 'params')
            onsites = [
                lambda a, E: E,
                lambda a, **kwargs: kwargs['E'],
                lambda a, *, E: E,
            ]
            hoppings = [
                lambda a, b, t: t,
                lambda a, b, **kwargs: kwargs['t'],
                lambda a, b, *, t: t,
            ]
            params = dict(E=E1, t=t1, **dict(zip(['k_x', 'k_y'], k)))
            for E2, t2 in itertools.product(onsites, hoppings):
                syst = wraparound(_simple_syst(lat, E2, t2, sym)).finalized()
                H_alt = syst.hamiltonian_submatrix(params=params,
                                                   sparse=False)
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
    syst[lat(0, 0)] =  lambda a, E2, **kwargs: E2
    syst[(lat(0, 0), lat(0, 1))] = lambda a, b, t2, **kwargs: t2

    # hoppings that will be bound as hoppings
    syst[(lat(-2, 0), lat(-1, 0))] = -1
    syst[(lat(-2, 0), lat(2, 0))] = -1
    #
    syst[(lat(-2, 0), lat(0, 0))] = -1
    syst[(lat(-2, 0), lat(3, 0))] = lambda a, b, t3: t3

    syst[(lat(-1, 0), lat(0, 0))] = lambda a, b, t4, **kwargs: t4
    syst[(lat(-1, 0), lat(3, 0))] = lambda a, b, t5: t5

    wrapped_syst = wraparound(syst)

    ## testing

    momenta = ['k_x', 'k_y']

    onsites = [
        (lat(-2, 0), momenta, False),
        (lat(-1, 0), ['E1', 't1'] + momenta, False),
        (lat(0, 0), ['E2', 't2'] + momenta, True),
    ]

    for site, params_should_be, should_take_kwargs in onsites:
        params, takes_kwargs = get_parameters(wrapped_syst[site])
        assert params[1:] == params_should_be
        assert takes_kwargs == should_take_kwargs

    hoppings = [
        ((lat(-2, 0), lat(-1, 0)), momenta, False),
        ((lat(-2, 0), lat(0, 0)), ['t3'] + momenta, False),
        ((lat(-1, 0), lat(0, 0)), ['t4', 't5'] + momenta, True),
    ]

    for hopping, params_should_be, should_take_kwargs in hoppings:
        params, takes_kwargs = get_parameters(wrapped_syst[hopping])
        assert params[2:] == params_should_be
        assert takes_kwargs == should_take_kwargs


def test_symmetry():
    syst = _simple_syst(kwant.lattice.square())

    matrices = [np.random.rand(2, 2) for i in range(4)]
    laws = (matrices, [(lambda a: m) for m in matrices])
    for cl, ch, ph, tr in laws:
        syst.conservation_law = cl
        syst.chiral = ch
        syst.particle_hole = ph
        syst.time_reversal = tr

        wrapped = wraparound(syst)

        assert wrapped.time_reversal is None
        assert wrapped.particle_hole is None
        for attr in ('conservation_law', 'chiral'):
            new = getattr(wrapped, attr)
            orig = getattr(syst, attr)
            if callable(orig):
                params, _ = get_parameters(new)
                assert params[1:] == ['k_x', 'k_y']
                assert np.all(orig(None) == new(None, None, None))
            else:
                assert np.all(orig == new)
