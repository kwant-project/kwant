# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np
import tinyarray as ta

import kwant
from kwant.wraparound import wraparound


def _simple_syst(lat, E=0, t=1+1j):
    """Create a builder for a simple infinite system."""
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


def test_value_types(k=(-1.1, 0.5), E=0, t=1):
    for lat in [kwant.lattice.honeycomb(), kwant.lattice.square()]:
        syst = wraparound(_simple_syst(lat, E, t)).finalized()
        H = syst.hamiltonian_submatrix(k, sparse=False)
        for E1, t1 in [(float(E), float(t)),
                       (np.array([[E]], float), np.array([[1]], float)),
                       (ta.array([[E]], float), ta.array([[1]], float))]:
            for E2 in [E1, lambda a: E1]:
                for t2 in [t1, lambda a, b: t1]:
                    syst = wraparound(_simple_syst(lat, E2, t2)).finalized()
                    H_alt = syst.hamiltonian_submatrix(k, sparse=False)
                    np.testing.assert_equal(H_alt, H)
