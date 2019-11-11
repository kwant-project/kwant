# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from collections import OrderedDict

import numpy as np

from pytest import importorskip

sympy = importorskip('sympy')

qsymm = importorskip('qsymm')
from qsymm.symmetry_finder import symmetries
from qsymm.hamiltonian_generator import bloch_family, hamiltonian_from_family
from qsymm.groups import (hexagonal, PointGroupElement, spin_matrices,
                          spin_rotation, ContinuousGroupGenerator)
from qsymm.model import Model, BlochModel, BlochCoeff
from qsymm.linalg import allclose

import kwant
from kwant._common import ensure_rng
from kwant.lattice import TranslationalSymmetry
from kwant.qsymm import (builder_to_model, model_to_builder,
                         _get_builder_symmetries, find_builder_symmetries)


def test_honeycomb():
    lat = kwant.lattice.honeycomb(norbs=1)

    # Test simple honeycomb model with constant terms
    # Add discrete symmetries to the kwant builder as well, to check that they are
    # returned as well.
    syst = kwant.Builder(symmetry=TranslationalSymmetry(*lat.prim_vecs))
    syst[lat.a(0, 0)] = 1
    syst[lat.b(0, 0)] = 1
    syst[lat.neighbors(1)] = -1

    H = builder_to_model(syst)
    sg, cs = symmetries(H, hexagonal(sympy_R=False))
    assert len(sg) == 24
    assert len(cs) == 0

    # Test simple honeycomb model with value functions
    syst = kwant.Builder(symmetry=TranslationalSymmetry(*lat.prim_vecs))
    syst[lat.a(0, 0)] = lambda site, ma: ma
    syst[lat.b(0, 0)] = lambda site, mb: mb
    syst[lat.neighbors(1)] = lambda site1, site2, t: t

    H = builder_to_model(syst)
    sg, cs = symmetries(H, hexagonal(sympy_R=False))
    assert len(sg) == 12
    assert len(cs) == 0


def test_get_builder_symmetries():
    syst = kwant.Builder(kwant.TranslationalSymmetry((1, 0), (0, 1)),
                         particle_hole=np.eye(2),
                         conservation_law=2*np.eye(2))
    builder_symmetries = _get_builder_symmetries(syst)

    assert len(builder_symmetries) == 2
    P = builder_symmetries['particle_hole']
    assert isinstance(P, PointGroupElement)
    assert allclose(P.U, np.eye(2))
    assert P.conjugate and P.antisymmetry
    assert allclose(P.R, np.eye(2))

    cons = builder_symmetries['conservation_law']
    assert isinstance(cons, ContinuousGroupGenerator)
    assert allclose(cons.U, 2*np.eye(2))
    assert cons.R is None

    syst = kwant.Builder()
    builder_symmetries = _get_builder_symmetries(syst)
    assert len(builder_symmetries) == 0


def test_higher_dim():
    # Test 0D finite system
    lat = kwant.lattice.cubic(norbs=1)
    syst = kwant.Builder()
    syst[lat(0, 0, 0)] = 1
    syst[lat(1, 1, 0)] = 1
    syst[lat(0, 1, 1)] = 1
    syst[lat(1, 0, -1)] = 1
    syst[lat(0, 0, 0), lat(1, 1, 0)] = -1
    syst[lat(0, 0, 0), lat(0, 1, 1)] = -1
    syst[lat(0, 0, 0), lat(1, 0, -1)] = -1

    H = builder_to_model(syst)
    sg, cs = symmetries(H)
    assert len(sg) == 2
    assert len(cs) == 5

    # Test triangular lattice system embedded in 3D
    sym = TranslationalSymmetry([1, 1, 0], [0, 1, 1])
    lat = kwant.lattice.cubic(norbs=1)
    syst = kwant.Builder(symmetry=sym)
    syst[lat(0, 0, 0)] = 1
    syst[lat(0, 0, 0), lat(1, 1, 0)] = -1
    syst[lat(0, 0, 0), lat(0, 1, 1)] = -1
    syst[lat(0, 0, 0), lat(1, 0, -1)] = -1

    H = builder_to_model(syst)
    sg, cs = symmetries(H, hexagonal(sympy_R=False))
    assert len(sg) == 24
    assert len(cs) == 0


def test_graphene_to_kwant():

    norbs = OrderedDict([('A', 1), ('B', 1)])  # A and B atom per unit cell, one orbital each
    hopping_vectors = [('A', 'B', [1, 0])] # Hopping between neighbouring A and B atoms
    # Atomic coordinates within the unit cell
    atom_coords = [(0, 0), (1, 0)]
    # We set the interatom distance to 1, so the lattice vectors have length sqrt(3)
    lat_vecs = [(3/2, np.sqrt(3)/2), (3/2, -np.sqrt(3)/2)]

    # Time reversal
    TR = PointGroupElement(sympy.eye(2), True, False, np.eye(2))
    # Chiral symmetry
    C = PointGroupElement(sympy.eye(2), False, True, np.array([[1, 0], [0, -1]]))
    # Atom A rotates into A, B into B.
    sphi = 2*sympy.pi/3
    RC3 = sympy.Matrix([[sympy.cos(sphi), -sympy.sin(sphi)],
                      [sympy.sin(sphi), sympy.cos(sphi)]])
    C3 = PointGroupElement(RC3, False, False, np.eye(2))

    # Generate graphene Hamiltonian in Kwant from qsymm
    symmetries = [C, TR, C3]
    # Generate using a family
    family = bloch_family(hopping_vectors, symmetries, norbs)
    syst_from_family = model_to_builder(family, norbs, lat_vecs, atom_coords, coeffs=None)
    # Generate using a single Model object
    g = sympy.Symbol('g')
    # tosympy=False to return a BlochModel
    ham = hamiltonian_from_family(family, coeffs=[g], tosympy=False)
    syst_from_model = model_to_builder(ham, norbs, lat_vecs, atom_coords)

    # Make the graphene Hamiltonian using kwant only
    atoms, orbs = zip(*[(atom, norb) for atom, norb in
                        norbs.items()])
    # Make the kwant lattice
    lat = kwant.lattice.general(lat_vecs,
                                atom_coords,
                                norbs=orbs)
    # Store sublattices by name
    sublattices = {atom: sublat for atom, sublat in
                   zip(atoms, lat.sublattices)}

    sym = kwant.TranslationalSymmetry(*lat_vecs)
    bulk = kwant.Builder(sym)

    bulk[ [sublattices['A'](0, 0), sublattices['B'](0, 0)] ] = 0

    def hop(site1, site2, c0):
        return c0

    bulk[lat.neighbors()] = hop

    fsyst_family = kwant.wraparound.wraparound(syst_from_family).finalized()
    fsyst_model = kwant.wraparound.wraparound(syst_from_model).finalized()
    fsyst_kwant = kwant.wraparound.wraparound(bulk).finalized()

    # Check that the energies are identical at random points in the Brillouin zone
    coeff = 0.5 + np.random.rand()
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeff, k_x=kx, k_y=ky)
        hamiltonian1 = fsyst_kwant.hamiltonian_submatrix(params=params, sparse=False)
        hamiltonian2 = fsyst_family.hamiltonian_submatrix(params=params, sparse=False)
        assert allclose(hamiltonian1, hamiltonian2)
        params = dict(g=coeff, k_x=kx, k_y=ky)
        hamiltonian3 = fsyst_model.hamiltonian_submatrix(params=params, sparse=False)
        assert allclose(hamiltonian2, hamiltonian3)

    # Include random onsites as well
    one = sympy.numbers.One()
    onsites = [Model({one: np.array([[1, 0], [0, 0]])}, momenta=family[0].momenta),
               Model({one: np.array([[0, 0], [0, 1]])}, momenta=family[0].momenta)]
    family = family + onsites
    syst_from_family = model_to_builder(family, norbs, lat_vecs, atom_coords, coeffs=None)
    gs = list(sympy.symbols('g0:3'))
    # tosympy=False to return a BlochModel
    ham = hamiltonian_from_family(family, coeffs=gs, tosympy=False)
    syst_from_model = model_to_builder(ham, norbs, lat_vecs, atom_coords)

    def onsite_A(site, c1):
        return c1

    def onsite_B(site, c2):
        return c2

    bulk[[sublattices['A'](0, 0)]] = onsite_A
    bulk[[sublattices['B'](0, 0)]] = onsite_B

    fsyst_family = kwant.wraparound.wraparound(syst_from_family).finalized()
    fsyst_model = kwant.wraparound.wraparound(syst_from_model).finalized()
    fsyst_kwant = kwant.wraparound.wraparound(bulk).finalized()

    # Check equivalence of the Hamiltonian at random points in the BZ
    coeffs = 0.5 + np.random.rand(3)
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeffs[0], c1=coeffs[1], c2=coeffs[2], k_x=kx, k_y=ky)
        hamiltonian1 = fsyst_kwant.hamiltonian_submatrix(params=params, sparse=False)
        hamiltonian2 = fsyst_family.hamiltonian_submatrix(params=params, sparse=False)
        assert allclose(hamiltonian1, hamiltonian2)
        params = dict(g0=coeffs[0], g1=coeffs[1], g2=coeffs[2], k_x=kx, k_y=ky)
        hamiltonian3 = fsyst_model.hamiltonian_submatrix(params=params, sparse=False)
        assert allclose(hamiltonian2, hamiltonian3)


def test_wraparound_convention():
    # Test that it matches exactly kwant.wraparound convention
    # Make the graphene Hamiltonian using kwant only
    norbs = OrderedDict([('A', 1), ('B', 1)])  # A and B atom per unit cell, one orbital each
    atoms, orbs = zip(*[(atom, norb) for atom, norb in
                        norbs.items()])
    # Atomic coordinates within the unit cell
    atom_coords = [(0, 0), (1, 0)]
    # We set the interatom distance to 1, so the lattice vectors have length sqrt(3)
    lat_vecs = [(3/2, np.sqrt(3)/2), (3/2, -np.sqrt(3)/2)]
    # Make the kwant lattice
    lat = kwant.lattice.general(lat_vecs,
                                atom_coords,
                                norbs=orbs)
    # Store sublattices by name
    sublattices = {atom: sublat for atom, sublat in
                   zip(atoms, lat.sublattices)}

    sym = kwant.TranslationalSymmetry(*lat_vecs)
    bulk = kwant.Builder(sym)

    bulk[ [sublattices['A'](0, 0), sublattices['B'](0, 0)] ] = 0

    def hop(site1, site2, c0):
        return c0

    bulk[lat.neighbors()] = hop

    wrapped = kwant.wraparound.wraparound(bulk).finalized()
    ham2 = builder_to_model(bulk, real_space=False)
    # Check that the Hamiltonians are identical at random points in the Brillouin zone
    H1 = wrapped.hamiltonian_submatrix
    H2 = ham2.lambdify()
    coeffs = 0.5 + np.random.rand(1)
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeffs[0], k_x=kx, k_y=ky)
        h1, h2 = H1(params=params), H2(**params)
        assert allclose(h1, h2), (h1, h2)



def test_inverse_transform():
    # Define family on square lattice
    s = spin_matrices(1/2)
    # Time reversal
    TR = PointGroupElement(np.eye(2), True, False,
                           spin_rotation(2 * np.pi * np.array([0, 1/2, 0]), s))
    # Mirror symmetry
    Mx = PointGroupElement(np.array([[-1, 0], [0, 1]]), False, False,
                           spin_rotation(2 * np.pi * np.array([1/2, 0, 0]), s))
    # Fourfold
    C4 = PointGroupElement(np.array([[0, 1], [-1, 0]]), False, False,
                           spin_rotation(2 * np.pi * np.array([0, 0, 1/4]), s))
    symmetries = [TR, Mx, C4]

    # One site per unit cell
    norbs = OrderedDict([('A', 2)])
    # Hopping to a neighbouring atom one primitive lattice vector away
    hopping_vectors = [('A', 'A', [1, 0])]
    # Make family
    family = bloch_family(hopping_vectors, symmetries, norbs, bloch_model=True)
    fam = hamiltonian_from_family(family, tosympy=False)
    # Atomic coordinates within the unit cell
    atom_coords = [(0, 0)]
    lat_vecs = [(1, 0), (0, 1)]
    syst = model_to_builder(fam, norbs, lat_vecs, atom_coords)
    # Convert it back
    ham2 = builder_to_model(syst)
    # Check that it's the same as the original
    assert fam.allclose(ham2)

    # Check that the Hamiltonians are identical at random points in the Brillouin zone
    sysw = kwant.wraparound.wraparound(syst).finalized()
    H1 = sysw.hamiltonian_submatrix
    H2 = ham2.lambdify()
    H3 = fam.lambdify()
    coeffs = 0.5 + np.random.rand(3)
    for _ in range(20):
        kx, ky = 3*np.pi*(np.random.rand(2) - 0.5)
        params = dict(c0=coeffs[0], c1=coeffs[1], c2=coeffs[2], k_x=kx, k_y=ky)
        assert allclose(H1(params=params), H2(**params))
        assert allclose(H1(params=params), H3(**params))


def test_consistency_kwant():
    """Make a random 1D Model, convert it to a builder, and compare
    the Bloch representation of the Model with that which Kwant uses
    in wraparound and in Bands. Then, convert the builder back to a Model
    and compare with the original Model.
    For comparison, we also make the system using Kwant only.
    """
    orbs = 4
    T = np.random.rand(2*orbs, 2*orbs) + 1j*np.random.rand(2*orbs, 2*orbs)
    H = np.random.rand(2*orbs, 2*orbs) + 1j*np.random.rand(2*orbs, 2*orbs)
    H += H.T.conj()

    # Make the 1D Model manually using only qsymm features.
    c0, c1 = sympy.symbols('c0 c1')

    Ham = BlochModel({BlochCoeff(np.array([-1]), c0): T}, momenta=['k_x'])
    Ham += Ham.T().conj()
    Ham += BlochModel({BlochCoeff(np.array([0]), c1): H}, momenta=['k_x'])

    # Two superimposed atoms, same number of orbitals on each
    norbs = OrderedDict([('A', orbs), ('B', orbs)])
    atom_coords = [(0.3, ), (0.3, )]
    lat_vecs = [(1, )] # Lattice vector

    # Make a Kwant builder out of the qsymm Model
    model_syst = model_to_builder(Ham, norbs, lat_vecs, atom_coords)
    fmodel_syst = model_syst.finalized()

    # Make the same system manually using only Kwant features.
    lat = kwant.lattice.general(np.array([[1.]]),
                            [(0., )],
                            norbs=2*orbs)
    kwant_syst = kwant.Builder(kwant.TranslationalSymmetry(*lat.prim_vecs))

    def onsite(site, c1):
        return c1*H

    def hopping(site1, site2, c0):
        return c0*T

    sublat = lat.sublattices[0]
    kwant_syst[sublat(0,)] = onsite
    hopp = kwant.builder.HoppingKind((1, ), sublat)
    kwant_syst[hopp] = hopping
    fkwant_syst = kwant_syst.finalized()

    # Make sure we are consistent with bands calculations in kwant
    # The Bloch Hamiltonian used in Kwant for the bands computation
    # is h(k) = exp(-i*k)*hop + onsite + exp(i*k)*hop.T.conj.
    # We also check that all is consistent with wraparound
    coeffs = (0.7, 1.2)
    params = dict(c0 = coeffs[0], c1 = coeffs[1])
    kwant_hop = fkwant_syst.inter_cell_hopping(params=params)
    kwant_onsite = fkwant_syst.cell_hamiltonian(params=params)
    model_kwant_hop = fmodel_syst.inter_cell_hopping(params=params)
    model_kwant_onsite = fmodel_syst.cell_hamiltonian(params=params)

    assert allclose(model_kwant_hop, coeffs[0]*T)
    assert allclose(model_kwant_hop, kwant_hop)
    assert allclose(model_kwant_onsite, kwant_onsite)

    h_model_kwant = (lambda k: np.exp(-1j*k)*model_kwant_hop + model_kwant_onsite +
                     np.exp(1j*k)*model_kwant_hop.T.conj()) # As in kwant.Bands
    h_model = Ham.lambdify()
    wsyst = kwant.wraparound.wraparound(model_syst).finalized()
    ks = np.linspace(-np.pi, np.pi, 21)
    for k in ks:
        assert allclose(h_model_kwant(k), h_model(coeffs[0], coeffs[1], k))
        params['k_x'] = k
        h_wrap = wsyst.hamiltonian_submatrix(params=params)
        assert allclose(h_model(coeffs[0], coeffs[1], k), h_wrap)

    # Get the model back from the builder
    # From the Kwant builder based on original Model
    Ham1 = builder_to_model(model_syst, momenta=Ham.momenta)
    # From the pure Kwant builder
    Ham2 = builder_to_model(kwant_syst, momenta=Ham.momenta)
    assert Ham.allclose(Ham1)
    assert Ham.allclose(Ham2)


def test_find_builder_discrete_symmetries():
    symm_class = ['AI', 'D', 'AIII', 'BDI']
    class_dict = {'AI': ['time_reversal'],
                'D': ['particle_hole'],
                'AIII': ['chiral'],
                'BDI': ['time_reversal', 'particle_hole', 'chiral']}
    sym_dict = {'time_reversal': qsymm.PointGroupElement(np.eye(2), True, False, None),
                'particle_hole': qsymm.PointGroupElement(np.eye(2), True, True, None),
                'chiral': qsymm.PointGroupElement(np.eye(2), False, True, None)}
    n = 4
    rng = 11
    for sym in symm_class:
        # Random Hamiltonian in the symmetry class
        h_ons = kwant.rmt.gaussian(n, sym, rng=rng)
        h_hop = 10 * kwant.rmt.gaussian(2*n, sym, rng=rng)[:n, n:]
        # Make a Kwant builder in the symmetry class and find its symmetries
        lat = kwant.lattice.square(norbs=n)
        bulk = kwant.Builder(TranslationalSymmetry([1, 0], [0, 1]))
        bulk[lat(0, 0)] = h_ons
        bulk[kwant.builder.HoppingKind((1, 0), lat)] = h_hop
        bulk[kwant.builder.HoppingKind((0, 1), lat)] = h_hop

        # We need to specify 'prettify=True' here to ensure that we do not end up with
        # an overcomplete set of symmetries. In some badly conditioned cases sparse=True
        # or sparse=False may affect how many symmetries are found.
        builder_symmetries_default = find_builder_symmetries(bulk, spatial_symmetries=True,
                                                             prettify=True)
        builder_symmetries_sparse = find_builder_symmetries(bulk, spatial_symmetries=True,
                                                            prettify=True, sparse=True)
        builder_symmetries_dense = find_builder_symmetries(bulk, spatial_symmetries=True,
                                                           prettify=True, sparse=False)

        assert len(builder_symmetries_default) == len(builder_symmetries_sparse)
        assert len(builder_symmetries_default) == len(builder_symmetries_dense)

        # Equality of symmetries ignores unitary part
        fourfold_rotation = qsymm.PointGroupElement(np.array([[0, 1],[1, 0]]), False, False, None)
        assert fourfold_rotation in builder_symmetries_default
        assert fourfold_rotation in builder_symmetries_sparse
        assert fourfold_rotation in builder_symmetries_dense
        class_symmetries = class_dict[sym]
        for class_symmetry in class_symmetries:
            assert sym_dict[class_symmetry] in builder_symmetries_default
            assert sym_dict[class_symmetry] in builder_symmetries_sparse
            assert sym_dict[class_symmetry] in builder_symmetries_dense


def test_real_space_basis():
    lat = kwant.lattice.honeycomb(norbs=[1, 1])
    sym = kwant.TranslationalSymmetry(lat.vec((1, 0)), lat.vec((0, 1)))
    bulk = kwant.Builder(sym)
    bulk[[lat.a(0, 0), lat.b(0, 0)]] = 0
    bulk[lat.neighbors()] = 1

    # Including real space symmetries
    symmetries = find_builder_symmetries(bulk)
    hex_group_2D = hexagonal()
    hex_group_2D = set(PointGroupElement(np.array(s.R).astype(float),
                                         s.conjugate, s.antisymmetry, None)
                for s in hex_group_2D)
    assert len(symmetries) == len(hex_group_2D)
    assert all([s1 in symmetries and s2 in hex_group_2D
                for s1, s2 in zip(hex_group_2D, symmetries)])

    # Only onsite discrete symmetries
    symmetries = find_builder_symmetries(bulk, spatial_symmetries=False)
    onsites = [PointGroupElement(np.eye(2), True, False, None),  # T
               PointGroupElement(np.eye(2), True, True, None),   # P
               PointGroupElement(np.eye(2), False, True, None),  # C
               PointGroupElement(np.eye(2), False, False, None)] # I
    assert len(symmetries) == len(onsites)
    assert all([s1 in symmetries and s2 in onsites
                for s1, s2 in zip(onsites, symmetries)])


def random_onsite_hop(n, rng=0):
    rng = ensure_rng(rng)
    onsite = rng.randn(n, n) + 1j * rng.randn(n, n)
    onsite = onsite + onsite.T.conj()
    hop = rng.rand(n, n) + 1j * rng.rand(n, n)
    return onsite, hop


def test_find_cons_law():
    sy = np.array([[0, -1j], [1j, 0]])

    n = 3
    lat = kwant.lattice.chain(norbs=2*n)
    syst = kwant.Builder()
    rng = 1337
    ons, hop = random_onsite_hop(n, rng=rng)

    syst[lat(0)] = np.kron(sy, ons)
    syst[lat(1)] = np.kron(sy, ons)
    syst[lat(1), lat(0)] = np.kron(sy, hop)

    builder_symmetries = find_builder_symmetries(syst, spatial_symmetries=False)
    onsites = [symm for symm in builder_symmetries if
               isinstance(symm, qsymm.ContinuousGroupGenerator) and symm.R is None]
    mham = builder_to_model(syst)
    assert all([symm.apply(mham).allclose(0, atol=1e-6) for symm in onsites])


def test_basis_ordering():
    symm_class = ['AI', 'D', 'AIII', 'BDI']
    n = 2
    rng = 12
    for sym in symm_class:
        # Make a small finite system in the symmetry class, finalize it and check
        # that the basis is consistent.
        h_ons = kwant.rmt.gaussian(n, sym, rng=rng)
        h_hop = 10 * kwant.rmt.gaussian(2*n, sym, rng=rng)[:n, n:]
        lat = kwant.lattice.square(norbs=n)
        bulk = kwant.Builder(TranslationalSymmetry([1, 0], [0, 1]))
        bulk[lat(0, 0)] = h_ons
        bulk[kwant.builder.HoppingKind((1, 0), lat)] = h_hop
        bulk[kwant.builder.HoppingKind((0, 1), lat)] = h_hop

        def rect(site):
            x, y = site.pos
            return (0 <= x < 2) and (0 <= y < 3)

        square = kwant.Builder()
        square.fill(bulk, lambda site: rect(site), (0, 0),
                    max_sites=float('inf'))

        # Find the symmetries of the square
        builder_symmetries = find_builder_symmetries(square,
                                                     spatial_symmetries=False)
        # Finalize the square, extract Hamiltonian
        fsquare = square.finalized()
        ham = fsquare.hamiltonian_submatrix()

        # Check manually that the found symmetries are in the same basis as the
        # finalized system
        for symmetry in builder_symmetries:
            U = symmetry.U
            if isinstance(symmetry, qsymm.ContinuousGroupGenerator):
                assert symmetry.R is None
                assert allclose(U.dot(ham), ham.dot(U))
            else:
                if symmetry.conjugate:
                    left = U.dot(ham.conj())
                else:
                    left = U.dot(ham)
                if symmetry.antisymmetry:
                    assert allclose(left, -ham.dot(U))
                else:
                    assert allclose(left, ham.dot(U))
