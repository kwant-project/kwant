# Copyright 2011-2019 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from math import sqrt

import numpy as np
import sympy
import pytest
import itertools

import kwant.builder
import kwant.lattice

from .._common import position_operators, momentum_operators, sympify
from ..landau_levels import (
    to_landau_basis,
    discretize_landau,
    _ladder_term,
    _evaluate_ladder_term,
    ladder_lower,
    ladder_raise,
    LandauLattice,
)

x, y, z = position_operators
k_x, k_y, k_z = momentum_operators
B = sympy.symbols("B")
V = sympy.symbols("V", cls=sympy.Function)
a, a_dag = ladder_lower, ladder_raise


def test_ladder_term():
    assert _ladder_term(a ** 2 * a_dag) == (-2, 1)
    assert _ladder_term(a_dag ** 5 * a ** 3 * a_dag) == (5, -3, 1)
    # non-ladder operators give a shift of 0
    assert _ladder_term(B * a ** 2) == (0, -2)


def test_evaluate_ladder_term():
    assert np.isclose(_evaluate_ladder_term((+1, -1), 1, +1), 1)
    assert np.isclose(_evaluate_ladder_term((+1, -1), 2, +1), 2)
    assert np.isclose(_evaluate_ladder_term((-2, +3, -2), 5, +1), 4 * 5 * 6 * sqrt(5))
    # annihilating |0> is always 0
    assert _evaluate_ladder_term((-1,), 0, +1) == 0
    assert _evaluate_ladder_term((-2,), 1, +1) == 0
    assert _evaluate_ladder_term((-3,), 1, +1) == 0
    assert _evaluate_ladder_term((+3, -2), 1, +1) == 0
    assert _evaluate_ladder_term((-3, -2, +3), 1, +1) == 0
    # negative B swaps creation and annihilation operators
    assert _evaluate_ladder_term((+1, -1), 2, +1) == _evaluate_ladder_term(
        (-1, +1), 2, -1
    )
    assert _evaluate_ladder_term((-2, +3, -2), 5, +1) == _evaluate_ladder_term(
        (+2, -3, +2), 5, -1
    )


def test_to_landau_basis():
    # test basic usage
    ham, momenta, normal_coord = to_landau_basis("k_x**2 + k_y**2")
    assert sympy.expand(ham) == abs(B) * a * a_dag + abs(B) * a_dag * a
    assert momenta == (k_x, k_y)
    assert normal_coord == z

    # test that hamiltonian can be specified as a sympy expression
    ham, momenta, normal_coord = to_landau_basis(sympify("k_x**2 + k_y**2"))
    assert sympy.expand(ham) == abs(B) * a * a_dag + abs(B) * a_dag * a
    assert momenta == (k_x, k_y)
    assert normal_coord == z

    # test that
    ham, momenta, normal_coord = to_landau_basis("k_x**2 + k_y**2 + k_z**2 + V(z)")
    assert sympy.expand(ham) == (
        abs(B) * a * a_dag + abs(B) * a_dag * a + k_z ** 2 + V(z)
    )
    assert momenta == (k_x, k_y)
    assert normal_coord == z

    # test for momenta explicitly specified
    ham, momenta, normal_coord = to_landau_basis(
        "k_x**2 + k_y**2 + k_z**2 + k_x*k_y", momenta=("k_z", "k_y")
    )
    assert sympy.expand(ham) == (
        abs(B) * a * a_dag
        + abs(B) * a_dag * a
        + k_x ** 2
        + sympy.I * sympy.sqrt(abs(B) / 2) * k_x * a
        - sympy.I * sympy.sqrt(abs(B) / 2) * k_x * a_dag
    )
    assert momenta == (k_z, k_y)
    assert normal_coord == x


def test_discretize_landau():
    n_levels = 10
    magnetic_field = 1 / 3  # a suitably arbitrary value
    # Ensure that we can handle products of ladder operators by truncating
    # several levels higher than the number of levels we actually want.
    a = np.diag(np.sqrt(np.arange(1, n_levels + 5)), k=1)
    a_dag = a.conjugate().transpose()
    k_x = sqrt(magnetic_field / 2) * (a + a_dag)
    k_y = 1j * sqrt(magnetic_field / 2) * (a - a_dag)
    sigma_0 = np.eye(2)
    sigma_x = np.array([[0, 1], [1, 0]])
    sigma_y = np.array([[0, -1j], [1j, 0]])

    # test that errors are raised on invalid input
    with pytest.raises(ValueError):
        discretize_landau("k_x**2 + k_y**2", N=0)
    with pytest.raises(ValueError):
        discretize_landau("k_x**2 + k_y**2", N=-1)

    # test a basic Hamiltonian with no normal coordinate dependence
    syst = discretize_landau("k_x**2 + k_y**2", N=n_levels)
    lat = LandauLattice(1, norbs=1)
    assert isinstance(syst.symmetry, kwant.builder.NoSymmetry)
    syst = syst.finalized()
    assert set(syst.sites) == {lat(0, j) for j in range(n_levels)}
    assert np.allclose(
        syst.hamiltonian_submatrix(params=dict(B=0)), np.zeros((n_levels, n_levels))
    )
    should_be = k_x @ k_x + k_y @ k_y
    assert np.allclose(
        syst.hamiltonian_submatrix(params=dict(B=magnetic_field)),
        should_be[:n_levels, :n_levels],
    )

    # test negative magnetic field
    assert np.allclose(
        syst.hamiltonian_submatrix(params=dict(B=-magnetic_field)),
        should_be[:n_levels, :n_levels],
    )

    # test hamiltonian with no onsite elements
    syst = discretize_landau("k_x", N=n_levels)
    syst = syst.finalized()
    assert np.allclose(
        syst.hamiltonian_submatrix(params=dict(B=magnetic_field)),
        k_x[:n_levels, :n_levels],
    )

    # test a basic Hamiltonian with normal coordinate dependence
    grid = 1 / 7  # a suitably arbitrary value
    syst = discretize_landau(
        "k_x**2 + k_y**2 + k_z**2 + V(z)", N=n_levels, grid_spacing=grid
    )
    assert isinstance(syst.symmetry, kwant.lattice.TranslationalSymmetry)
    syst = syst.finalized()
    zero_potential = syst.cell_hamiltonian(params=dict(B=magnetic_field, V=lambda z: 0))
    with_potential = syst.cell_hamiltonian(params=dict(B=magnetic_field, V=lambda z: 1))
    # extra +2/grid**2 from onsite kinetic energy
    assert np.allclose(
        zero_potential,
        should_be[:n_levels, :n_levels] + (2 / grid ** 2) * np.eye(n_levels),
    )
    # V(z) just adds the same energy to each onsite
    assert np.allclose(with_potential - zero_potential, np.eye(n_levels))
    # hopping matrix does not exchange landau levels
    assert np.allclose(
        syst.inter_cell_hopping(params=dict(B=magnetic_field, V=lambda z: 0)),
        -np.eye(n_levels) / grid ** 2,
    )

    # test a Hamiltonian with mixing between Landau levels
    # and spatial degrees of freedom.
    syst = discretize_landau("k_x**2 + k_y**2 + k_x*k_z", N=n_levels)
    syst = syst.finalized()
    assert np.allclose(
        syst.inter_cell_hopping(params=dict(B=magnetic_field)),
        -1j * k_x[:n_levels, :n_levels] / 2,
    )

    # test a Hamiltonian with extra degrees of freedom
    syst = discretize_landau("sigma_0 * k_x**2 + sigma_x * k_y**2", N=n_levels)
    syst = syst.finalized()
    assert syst.sites[0].family.norbs == 2
    should_be = np.kron(k_x @ k_x, sigma_0) + np.kron(k_y @ k_y, sigma_x)
    assert np.allclose(
        syst.hamiltonian_submatrix(params=dict(B=magnetic_field)),
        should_be[: 2 * n_levels, : 2 * n_levels],
    )

    # test a linear Hamiltonian
    syst = discretize_landau("sigma_y * k_x - sigma_x * k_y", N=n_levels)
    syst = syst.finalized()
    should_be = np.kron(k_x, sigma_y) - np.kron(k_y, sigma_x)
    assert np.allclose(
        syst.hamiltonian_submatrix(params=dict(B=magnetic_field)),
        should_be[: 2 * n_levels, : 2 * n_levels],
    )
    assert np.allclose(
        syst.hamiltonian_submatrix(params=dict(B=magnetic_field)),
        syst.hamiltonian_submatrix(params=dict(B=-magnetic_field)),
    )


def test_analytical_spectrum():
    hamiltonian = """(k_x**2 + k_y**2) * sigma_0 +
                    alpha * (k_x * sigma_y - k_y * sigma_x) +
                    g * B * sigma_z"""

    def exact_Es(n, B, alpha, g):
        # See e.g. R. Winkler (2003), section 8.4.1
        sign_B = np.sign(B)
        B = np.abs(B)
        Ep = 2*B*(n+1) - 0.5*np.sqrt((2*B - sign_B*2*g*B)**2 + 8*B*alpha**2*(n+1))
        Em = 2*B*n + 0.5*np.sqrt((2*B - sign_B*2*g*B)**2 + 8*B*alpha**2*n)
        return Ep, Em

    N = 20
    syst = discretize_landau(hamiltonian, N)
    syst = syst.finalized()
    params = dict(alpha=0.07, g=0.04)
    for _ in range(5):
        B = 0.01 + np.random.rand()*3
        params['B'] = B
        exact = [exact_Es(n, B, params['alpha'], params['g']) for n in range(N)]
        # We only check the first N levels - the SOI couples adjacent levels,
        # so the higher ones are not necessarily determined accurately in the
        # discretization
        exact = np.sort([energy for energies in exact for energy in energies])[:N]
        ll_spect = np.linalg.eigvalsh(syst.hamiltonian_submatrix(params=params))[:len(exact)]
        assert np.allclose(ll_spect, exact)



def test_fill():

    def shape(site, lower, upper):
        (z, )= site.pos
        return lower <= z < upper

    hamiltonian = "k_x**2 + k_y**2 + k_z**2"
    N = 6
    template = discretize_landau(hamiltonian, N)

    syst = kwant.Builder()
    width = 4
    syst.fill(template, lambda site: shape(site, 0, width), (0, ));

    correct_tags = [(coordinate, ll_index) for coordinate, ll_index
                    in itertools.product(range(width), range(N))]

    syst_tags = [site.tag for site in syst.sites()]

    assert len(syst_tags) == len(correct_tags)
    assert all(tag in correct_tags for tag in syst_tags)
