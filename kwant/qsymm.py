# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.


__all__ = ['builder_to_model', 'model_to_builder', 'find_builder_symmetries']

import itertools as it
from collections import OrderedDict, defaultdict

import numpy as np
import tinyarray as ta
import scipy.linalg as la

try:
    import sympy
    import qsymm
    from qsymm.model import Model, BlochModel, BlochCoeff
    from qsymm.groups import PointGroupElement, ContinuousGroupGenerator
    from qsymm.symmetry_finder import bravais_point_group
    from qsymm.linalg import allclose
    from qsymm.hamiltonian_generator import hamiltonian_from_family
except ImportError as error:
    msg = ("'kwant.qsymm' is not available because one or more of its "
           "dependencies is not installed.")
    raise ImportError(msg) from error

from kwant import lattice, builder
from kwant._common import get_parameters


def builder_to_model(syst, momenta=None, real_space=True,
                     params=None):
    """Make a qsymm.BlochModel out of a `~kwant.builder.Builder`.

    Parameters
    ----------
    syst : `~kwant.builder.Builder`
        May have translational symmetries.
    momenta : list of strings or None
        Names of momentum variables. If None, 'k_x', 'k_y', ... is used.
    real_space : bool (default True)
        If False, use the unit cell convention for Bloch basis, the
        exponential has the difference in the unit cell coordinates and
        k is expressed in the reciprocal lattice basis. This is consistent
        with `kwant.wraparound`.
        If True, the difference in the real space coordinates is used
        and k is given in an absolute basis.
        Only the default choice guarantees that qsymm is able to find
        nonsymmorphic symmetries.
    params : dict, optional
        Dictionary of parameter names and their values; used when
        evaluating the Hamiltonian matrix elements.

    Returns
    -------
    model : qsymm.BlochModel
        Model representing the tight-binding Hamiltonian.

    Notes
    -----
    The sites in the the builder are in lexicographical order, i.e. ordered
    first by their family and then by their tag. This is the same ordering that
    is used in finalized kwant systems.
    """
    def term_to_model(d, par, matrix):
        if allclose(matrix, 0):
            result = BlochModel({}, shape=matrix.shape, format=np.ndarray)
        else:
            result = BlochModel({BlochCoeff(d, qsymm.sympify(par)): matrix},
                                momenta=momenta)
        return result

    def hopping_to_model(hop, value, proj, params):
        site1, site2 = hop
        if real_space:
            d = proj @ np.array(site2.pos - site1.pos)
        else:
            # site in the FD
            d = np.array(syst.symmetry.which(site2))

        slice1, slice2 = slices[to_fd(site1)], slices[to_fd(site2)]
        if callable(value):
            return sum(term_to_model(d, par, set_block(slice1, slice2, val))
                       for par, val in function_to_terms(hop, value, params))
        else:
            matrix = set_block(slice1, slice2, value)
            return term_to_model(d, '1', matrix)

    def onsite_to_model(site, value, params):
        d = np.zeros((dim, ))
        slice1 = slices[to_fd(site)]
        if callable(value):
            return sum(term_to_model(d, par, set_block(slice1, slice1, val))
                       for par, val in function_to_terms(site, value, params))
        else:
            return term_to_model(d, '1', set_block(slice1, slice1, value))

    def function_to_terms(site_or_hop, value, fixed_params):
        assert callable(value)
        parameters = get_parameters(value)
        # remove site or site1, site2 parameters
        if isinstance(site_or_hop, builder.Site):
            parameters = parameters[1:]
            site_or_hop = (site_or_hop,)
        else:
            parameters = parameters[2:]
        free_parameters = (par for par in parameters
                           if par not in fixed_params.keys())
        # first set all free parameters to 0
        args = ((fixed_params[par] if par in fixed_params.keys() else 0)
                for par in parameters)
        h_0 = value(*site_or_hop, *args)
        # set one of the free parameters to 1 at a time, the rest 0
        terms = []
        for p in free_parameters:
            args = ((fixed_params[par] if par in fixed_params.keys() else
                     (1 if par == p else 0)) for par in parameters)
            terms.append((p, value(*site_or_hop, *args) - h_0))
        return terms + [('1', h_0)]

    def orbital_slices(syst):
        orbital_slices = {}
        start_orb = 0

        for site in sorted(syst.sites()):
            n = site.family.norbs
            if n is None:
                raise ValueError('norbs must be provided for every lattice.')
            orbital_slices[site] = slice(start_orb, start_orb + n)
            start_orb += n
        return orbital_slices, start_orb

    def set_block(slice1, slice2, val):
        matrix = np.zeros((N, N), dtype=complex)
        matrix[slice1, slice2] = val
        return matrix

    if params is None:
        params = dict()

    periods = np.array(syst.symmetry.periods)
    dim = len(periods)
    to_fd = syst.symmetry.to_fd
    if momenta is None:
        momenta = ['k_x', 'k_y', 'k_z'][:dim]
    # If the system is higher dimensional than the number of translation
    # vectors, we need to project onto the subspace spanned by the
    # translation vectors.
    if dim == 0:
        proj = np.empty((0, len(list(syst.sites())[0].pos)))
    elif dim < len(list(syst.sites())[0].pos):
        proj, r = la.qr(np.array(periods).T, mode='economic')
        sign = np.diag(np.diag(np.sign(r)))
        proj = sign @ proj.T
    else:
        proj = np.eye(dim)

    slices, N = orbital_slices(syst)

    one_way_hoppings = [hopping_to_model(hop, value, proj, params)
                        for hop, value in syst.hopping_value_pairs()]
    other_way_hoppings = [term.T().conj() for term in one_way_hoppings]
    hoppings = one_way_hoppings + other_way_hoppings

    onsites = [onsite_to_model(site, value, params)
               for site, value in syst.site_value_pairs()]

    result = sum(onsites) + sum(hoppings)

    return result


def model_to_builder(model, norbs, lat_vecs, atom_coords, *, coeffs=None):
    """Make a `~kwant.builder.Builder` out of qsymm.Models or qsymm.BlochModels.

    Parameters
    ----------
    model : qsymm.Model, qsymm.BlochModel, or an iterable thereof
        The Hamiltonian (or terms of the Hamiltonian) to convert to a
        Builder.
    norbs : OrderedDict or sequence of pairs
        Maps sites to the number of orbitals per site in a unit cell.
    lat_vecs : list of arrays
        Lattice vectors of the underlying tight binding lattice.
    atom_coords : list of arrays
        Positions of the sites (or atoms) within a unit cell.
        The ordering of the atoms is the same as in norbs.
    coeffs : list of sympy.Symbol, default None.
        Constant prefactors for the individual terms in model, if model
        is a list of multiple objects. If model is a single Model or BlochModel
        object, this argument is ignored. By default assigns the coefficient
        c_n to element model[n].

    Returns
    -------
    syst : `~kwant.builder.Builder`
        The unfinalized Kwant system representing the qsymm Model(s).

    Notes
    -----
    Onsite terms that are not provided in the input model are set
    to zero by default.

    The input model(s) representing the tight binding Hamiltonian in
    Bloch form should follow the convention where the difference in the real
    space atomic positions appear in the Bloch factors.
    """

    def make_int(R):
        # If close to an integer array convert to integer tinyarray, else
        # return None
        R_int = ta.array(np.round(R), int)
        if qsymm.linalg.allclose(R, R_int):
            return R_int
        else:
            return None

    def term_onsite(onsites_dict, hopping_dict, hop_mat, atoms,
                    sublattices, coords_dict):
        """Find the Kwant onsites and hoppings in a qsymm.BlochModel term
        that has no lattice translation in the Bloch factor.
        """
        for atom1, atom2 in it.product(atoms, atoms):
            # Subblock within the same sublattice is onsite
            hop = hop_mat[ranges[atom1], ranges[atom2]]
            if sublattices[atom1] == sublattices[atom2]:
                onsites_dict[atom1] += Model({coeff: hop}, momenta=momenta)
            # Blocks between sublattices are hoppings between sublattices
            # at the same position.
            # Only include nonzero hoppings
            elif not allclose(hop, 0):
                if not allclose(np.array(coords_dict[atom1]),
                                np.array(coords_dict[atom2])):
                    raise ValueError(
                        "Position of sites not compatible with qsymm model.")
                lat_basis = np.array(zer)
                hop = Model({coeff: hop}, momenta=momenta)
                hop_dir = builder.HoppingKind(-lat_basis, sublattices[atom1],
                                              sublattices[atom2])
                hopping_dict[hop_dir] += hop
        return onsites_dict, hopping_dict

    def term_hopping(hopping_dict, hop_mat, atoms,
                     sublattices, coords_dict):
        """Find Kwant hoppings in a qsymm.BlochModel term that has a lattice
        translation in the Bloch factor.
        """
        # Iterate over combinations of atoms, set hoppings between each
        for atom1, atom2 in it.product(atoms, atoms):
            # Take the block from atom1 to atom2
            hop = hop_mat[ranges[atom1], ranges[atom2]]
            # Only include nonzero hoppings
            if allclose(hop, 0):
                continue
            # Adjust hopping vector to Bloch form basis
            r_lattice = (
                r_vec
                + np.array(coords_dict[atom1])
                - np.array(coords_dict[atom2])
            )
            # Bring vector to basis of lattice vectors
            lat_basis = np.linalg.solve(np.vstack(lat_vecs).T, r_lattice)
            lat_basis = make_int(lat_basis)
            # Should only have hoppings that are integer multiples of
            # lattice vectors
            if lat_basis is not None:
                hop_dir = builder.HoppingKind(-lat_basis,
                                              sublattices[atom1],
                                              sublattices[atom2])
                # Set the hopping as the matrix times the hopping amplitude
                hopping_dict[hop_dir] += Model({coeff: hop}, momenta=momenta)
            else:
                raise RuntimeError('A nonzero hopping not matching a '
                                   'lattice vector was found.')
        return hopping_dict

    # Disambiguate single model instances from iterables thereof. Because
    # Model is itself iterable (subclasses dict) this is a bit cumbersome.
    if isinstance(model, Model):
        # BlochModel can't yet handle getting a Blochmodel as input
        if not isinstance(model, BlochModel):
            model = BlochModel(model)
    else:
        model = BlochModel(hamiltonian_from_family(
            model, coeffs=coeffs, nsimplify=False, tosympy=False))


    # 'momentum' and 'zer' are used in the closures defined above, so don't
    # move these declarations down.
    momenta = model.momenta
    if len(momenta) != len(lat_vecs):
        raise ValueError("Dimension of the lattice and number of "
                         "momenta do not match.")
    zer = [0] * len(momenta)


    # Subblocks of the Hamiltonian for different atoms.
    N = 0
    if not any([isinstance(norbs, OrderedDict), isinstance(norbs, list),
                isinstance(norbs, tuple)]):
        raise ValueError('norbs must be OrderedDict, tuple, or list.')
    else:
        norbs = OrderedDict(norbs)
    ranges = dict()
    for a, n in norbs.items():
        ranges[a] = slice(N, N + n)
        N += n

    # Extract atoms and number of orbitals per atom,
    # store the position of each atom
    atoms, orbs = zip(*norbs.items())
    coords_dict = dict(zip(atoms, atom_coords))

    # Make the kwant lattice
    lat = lattice.general(lat_vecs, atom_coords, norbs=orbs)
    # Store sublattices by name
    sublattices = dict(zip(atoms, lat.sublattices))

    # Keep track of the hoppings and onsites by storing those
    # which have already been set.
    hopping_dict = defaultdict(lambda: 0)
    onsites_dict = defaultdict(lambda: 0)

    # Iterate over all terms in the model.
    for key, hop_mat in model.items():
        # Determine whether this term is an onsite or a hopping, extract
        # overall symbolic coefficient if any, extract the exponential
        # part describing the hopping if present.
        r_vec, coeff = key
        # Onsite term; modifies onsites_dict and hopping_dict in-place
        if allclose(r_vec, 0):
            term_onsite(
                onsites_dict, hopping_dict, hop_mat,
                atoms, sublattices, coords_dict)
        # Hopping term; modifies hopping_dict in-place
        else:
            term_hopping(hopping_dict, hop_mat, atoms,
                         sublattices, coords_dict)

    # If some onsite terms are not set, we set them to zero.
    for atom in atoms:
        if atom not in onsites_dict:
            onsites_dict[atom] = Model(
                {sympy.numbers.One(): np.zeros((norbs[atom], norbs[atom]))},
                momenta=momenta)

    # Make the Kwant system, and set all onsites and hoppings.

    sym = lattice.TranslationalSymmetry(*lat_vecs)
    syst = builder.Builder(sym)

    # Iterate over all onsites and set them
    for atom, onsite in onsites_dict.items():
        syst[sublattices[atom](*zer)] = onsite.lambdify(onsite=True)

    # Finally, iterate over all the hoppings and set them
    for direction, hopping in hopping_dict.items():
        syst[direction] = hopping.lambdify(hopping=True)

    return syst


# This may be useful in the future, so we'll keep it as internal for now,
# and can make it part of the API in the future if we wish.
def _get_builder_symmetries(builder):
    """Extract the declared symmetries of a Kwant builder.

    Parameters
    ----------
    builder : `~kwant.builder.Builder`

    Returns
    -------
    builder_symmetries : dict
        Dictionary of the discrete symmetries that the builder has.
        The symmetries can be particle-hole, time-reversal or chiral,
        which are returned as qsymm.PointGroupElements, or
        a conservation law, which is returned as a
        qsymm.ContinuousGroupGenerators.
    """

    dim = len(np.array(builder.symmetry.periods))
    symmetry_names = ['time_reversal', 'particle_hole', 'chiral',
                      'conservation_law']
    builder_symmetries = {name: getattr(builder, name)
                          for name in symmetry_names
                          if getattr(builder, name) is not None}
    for name, symmetry in builder_symmetries.items():
        if name == 'time_reversal':
            builder_symmetries[name] = PointGroupElement(np.eye(dim),
                                                         True, False, symmetry)
        elif name == 'particle_hole':
            builder_symmetries[name] = PointGroupElement(np.eye(dim),
                                                         True, True, symmetry)
        elif name == 'chiral':
            builder_symmetries[name] = PointGroupElement(np.eye(dim),
                                                         False, True, symmetry)
        elif name == 'conservation_law':
            builder_symmetries[name] = ContinuousGroupGenerator(R=None,
                                                                U=symmetry)
        else:
            raise ValueError("Invalid symmetry name.")
    return builder_symmetries


def find_builder_symmetries(builder, momenta=None, params=None,
                            spatial_symmetries=True, prettify=True,
                            sparse=None):
    """Finds the symmetries of a Kwant system using qsymm.

    Parameters
    ----------
    builder : `~kwant.builder.Builder`
    momenta : list of strings or None
        Names of momentum variables, if None 'k_x', 'k_y', ... is used.
    params : dict, optional
        Dictionary of parameter names and their values; used when
        evaluating the Hamiltonian matrix elements.
    spatial_symmetries : bool (default True)
        If True, search for all symmetries.
        If False, only searches for the symmetries that are declarable in
        `~kwant.builder.Builder` objects, i.e. time-reversal symmetry,
        particle-hole symmetry, chiral symmetry, or conservation laws.
        This can save computation time.
    prettify : bool (default True)
        Whether to carry out sparsification of the continuous symmetry
        generators, in general an arbitrary linear combination of the
        symmetry generators is returned.
    sparse : bool, or None (default None)
        Whether to use sparse linear algebra in the calculation.
        Can give large performance gain in large systems.
        If None, uses sparse or dense computation depending on
        the size of the Hamiltonian.


    Returns
    -------
    symmetries : list of qsymm.PointGroupElements and/or qsymm.ContinuousGroupElement
        The symmetries of the Kwant system.
    """

    if params is None:
        params = dict()

    ham = builder_to_model(builder, momenta=momenta,
                           real_space=spatial_symmetries, params=params)

    # Use dense or sparse computation depending on Hamiltonian size
    if sparse is None:
        sparse = list(ham.values())[0].shape[0] > 20

    dim = len(np.array(builder.symmetry.periods))

    if spatial_symmetries:
        candidates = bravais_point_group(builder.symmetry.periods, tr=True,
                                         ph=True, generators=False,
                                         verbose=False)
    else:
        candidates = [
            qsymm.PointGroupElement(np.eye(dim), True, False, None),  # T
            qsymm.PointGroupElement(np.eye(dim), True, True, None),   # P
            qsymm.PointGroupElement(np.eye(dim), False, True, None)]  # C
    sg, cg = qsymm.symmetries(ham, candidates, prettify=prettify,
                              continuous_rotations=False,
                              sparse_linalg=sparse)
    return list(sg) + list(cg)
