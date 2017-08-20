# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['TranslationalSymmetry', 'general', 'Polyatomic', 'Monatomic',
           'chain', 'square', 'cubic', 'triangular', 'honeycomb', 'kagome']

from math import sqrt
from itertools import product
import numpy as np
import tinyarray as ta
from . import builder
from .linalg import lll
from ._common import ensure_isinstance


def general(prim_vecs, basis=None, name='', norbs=None):
    """
    Create a Bravais lattice of any dimensionality, with any number of sites.

    Parameters
    ----------
    prim_vecs : 2d array-like of floats
        The primitive vectors of the Bravais lattice
    basis : 2d array-like of floats
        The coordinates of the basis sites inside the unit cell
    name : string or sequence of strings
        Name of the lattice, or sequence of names of all of the sublattices.
        If the name of the lattice is given, the names of sublattices (if any)
        are obtained by appending their number to the name of the lattice.
    norbs : int or sequence of ints, optional
        The number of orbitals per site on the lattice, or a sequence
        of the number of orbitals of sites on each of the sublattices.

    Returns
    -------
    lattice : either `Monatomic` or `Polyatomic`
        Resulting lattice.

    Notes
    -----
    This function is largely an alias to the constructors of corresponding
    lattices.
    """
    if basis is None:
        return Monatomic(prim_vecs, name=name, norbs=norbs)
    else:
        return Polyatomic(prim_vecs, basis, name=name, norbs=norbs)


def _check_prim_vecs(prim_vecs):
    """Check constraints to ensure that prim_vecs is correct."""
    if prim_vecs.ndim != 2:
        raise ValueError('``prim_vecs`` must be a 2d array-like object.')

    if prim_vecs.shape[0] > prim_vecs.shape[1]:
        raise ValueError('Number of primitive vectors exceeds '
                         'the space dimensionality.')

    if np.linalg.matrix_rank(prim_vecs) < len(prim_vecs):
        raise ValueError('"prim_vecs" must be linearly independent.')


class Polyatomic:
    """
    A Bravais lattice with an arbitrary number of sites in the basis.

    Contains `Monatomic` sublattices.  Note that an instance of ``Polyatomic`` is
    not itself a `~kwant.builder.SiteFamily`, only its sublattices are.

    Parameters
    ----------
    prim_vecs : 2d array-like of floats
        The primitive vectors of the Bravais lattice
    basis : 2d array-like of floats
        The coordinates of the basis sites inside the unit cell.
    name : string or sequence of strings, optional
        The name of the lattice, or a sequence of the names of all the
        sublattices.  If the name of the lattice is given, the names of
        sublattices are obtained by appending their number to the name of the
        lattice.
    norbs : int or sequence of ints, optional
        The number of orbitals per site on the lattice, or a sequence
        of the number of orbitals of sites on each of the sublattices.

    Raises
    ------
    ValueError
        If dimensionalities do not match.
    """
    def __init__(self, prim_vecs, basis, name='', norbs=None):
        prim_vecs = ta.array(prim_vecs, float)
        _check_prim_vecs(prim_vecs)

        dim = prim_vecs.shape[1]
        if name is None:
            name = ''
        if isinstance(name, str):
            name = [name + str(i) for i in range(len(basis))]

        basis = ta.array(basis, float)
        if basis.ndim != 2:
            raise ValueError('`basis` must be a 2d array-like object.')
        if basis.shape[1] != dim:
            raise ValueError('Basis dimensionality does not match '
                             'the space dimensionality.')

        try:
            norbs = list(norbs)
            if len(norbs) != len(basis):
                raise ValueError('Length of `norbs` is not the same as '
                                 'the number of basis vectors')
        except TypeError:
            norbs = [norbs] * len(basis)

        self.sublattices = [Monatomic(prim_vecs, offset, sname, norb)
                            for offset, sname, norb in zip(basis, name, norbs)]
        # Sequence of primitive vectors of the lattice.
        self._prim_vecs = prim_vecs
        # Precalculation of auxiliary arrays for real space calculations.
        self.reduced_vecs, self.transf = lll.lll(prim_vecs)
        self.voronoi = ta.dot(lll.voronoi(self.reduced_vecs), self.transf)

    def __str__(self):
        sl_names = ', '.join(str(sl.name) for sl in self.sublattices)
        return '<Polyatomic lattice with sublattices {0}>'.format(sl_names)

    def shape(self, function, start):
        """Return a key for all the lattice sites inside a given shape.

        The object returned by this method is primarily meant to be used as a
        key for indexing `~kwant.builder.Builder` instances.  See example below.

        Parameters
        ----------
        function : callable
            A function of real space coordinates that returns a truth value:
            true for coordinates inside the shape, and false otherwise.
        start : 1d array-like
            The real-space origin for the flood-fill algorithm.

        Returns
        -------
        shape_sites : function

        Notes
        -----
        When the function returned by this method is called, a flood-fill
        algorithm finds and yields all the lattice sites inside the specified
        shape starting from the specified position.

        A `~kwant.builder.Symmetry` or `~kwant.builder.Builder` may be passed as
        sole argument when calling the function returned by this method.  This
        will restrict the flood-fill to the fundamental domain of the symmetry
        (or the builder's symmetry).  Note that unless the shape function has
        that symmetry itself, the result may be unexpected.

        Examples
        --------
        >>> def circle(pos):
        ...     x, y = pos
        ...     return x**2 + y**2 < 100
        ...
        >>> lat = kwant.lattice.honeycomb()
        >>> syst = kwant.Builder()
        >>> syst[lat.shape(circle, (0, 0))] = 0
        >>> syst[lat.neighbors()] = 1
        """
        def shape_sites(symmetry=None):
            Site = builder.Site

            if symmetry is None:
                symmetry = builder.NoSymmetry()
            elif not isinstance(symmetry, builder.Symmetry):
                symmetry = symmetry.symmetry

            def fd_site(lat, tag):
                return symmetry.to_fd(Site(lat, tag, True))

            dim = len(start)
            if dim != self._prim_vecs.shape[1]:
                raise ValueError('Dimensionality of start position does not '
                                 'match the space dimensionality.')
            lats = self.sublattices
            deltas = list(self.voronoi)

            #### Flood-fill ####
            sites = []
            for tag in set(lat.closest(start) for lat in lats):
                for lat in lats:
                    site = fd_site(lat, tag)
                    if function(site.pos):
                        sites.append(site)
            if not sites:
                msg = 'No sites close to {0} are inside the desired shape.'
                raise ValueError(msg.format(start))
            tags = set(s.tag for s in sites)

            while sites:
                old_tags = tags
                tags = set()
                for site in sites:
                    yield site
                    tags.add(site.tag)
                old_tags |= tags

                new_tags = set()
                for tag in tags:
                    for delta in deltas:
                        new_tag = tag + delta
                        if new_tag not in old_tags:
                            new_tags.add(new_tag)

                sites = set()
                for tag in new_tags:
                    for lat in lats:
                        site = fd_site(lat, tag)
                        if site.tag not in old_tags and function(site.pos):
                            sites.add(site)

        return shape_sites

    def wire(self, center, radius):
        """Return a key for all the lattice sites inside an infinite cylinder.

        This method makes it easy to define cylindrical (2d: rectangular) leads
        that point in any direction.  The object returned by this method is
        primarily meant to be used as a key for indexing `~kwant.builder.Builder`
        instances.  See example below.

        Parameters
        ----------
        center : 1d array-like of floats
            A point belonging to the axis of the cylinder.
        radius : float
            The radius of the cylinder.

        Notes
        -----
        The function returned by this method is to be called with a
        `~kwant.builder.TranslationalSymmetry` instance (or a
        `~kwant.builder.Builder` instance whose symmetry is used then) as sole
        argument.  All the lattice sites (in the fundamental domain of the
        symmetry) inside the specified infinite cylinder are yielded.  The
        direction of the cylinder is determined by the symmetry.

        Examples
        --------
        >>> lat = kwant.lattice.honeycomb()
        >>> sym = kwant.TranslationalSymmetry(lat.a.vec((-2, 1)))
        >>> lead = kwant.Builder(sym)
        >>> lead[lat.wire((0, -5), 5)] = 0
        >>> lead[lat.neighbors()] = 1
        """
        center = ta.array(center, float)

        def wire_sites(sym):
            if not isinstance(sym, builder.Symmetry):
                sym = sym.symmetry
            if not isinstance(sym, TranslationalSymmetry):
                raise ValueError('wire shape only works with '
                                 'translational symmetry.')
            if len(sym.periods) != 1:
                raise ValueError('wire shape only works with one-dimensional '
                                 'translational symmetry.')
            period = np.array(sym.periods)[0]
            direction = ta.array(period / np.linalg.norm(period))
            r_squared = radius**2

            def wire_shape(pos):
                rel_pos = pos - center
                projection = rel_pos - direction * ta.dot(direction, rel_pos)
                return sum(projection * projection) <= r_squared

            return self.shape(wire_shape, center)(sym)

        return wire_sites

    def neighbors(self, n=1, eps=1e-8):
        """Return n-th nearest neighbor hoppings.

        Parameters
        ----------
        n : integer
            Order of the hoppings to return. Note that the zeroth neighbor is
            the site itself or any other sites with the same position.
        eps : float
            Tolerance relative to the length of the shortest lattice vector for
            when to consider lengths to be approximately equal.

        Returns
        -------
        hoppings : list of kwant.builder.HopplingKind objects
            The n-th nearest neighbor hoppings.

        Notes
        -----
        The hoppings are ordered lexicographically according to sublattice from
        which they originate, sublattice on which they end, and their lattice
        coordinates. Out of the two equivalent hoppings (a hopping and its
        reverse) only the lexicographically larger one is returned.
        """
        # This algorithm is not designed to be fast and can be improved,
        # however there is no real need.
        Site = builder.Site
        sls = self.sublattices
        shortest_hopping = sls[0].n_closest(
            sls[0].pos(([0] * sls[0].lattice_dim)), 2)[-1]
        rtol = eps
        eps *= np.linalg.norm(self.vec(shortest_hopping))
        nvec = len(self._prim_vecs)
        sublat_pairs = [(i, j) for (i, j) in product(sls, sls)
                        if sls.index(j) >= sls.index(i)]
        def first_nonnegative(tag):
            for i in tag:
                if i < 0:
                    return False
                elif i > 0:
                    return True
                else:
                    continue
            return True

        # Find the `n` closest neighbors (with multiplicity) for each
        # pair of lattices, this surely includes the `n` closest neighbors overall.
        sites = []
        for i, j in sublat_pairs:
            origin = Site(j, ta.zeros(nvec)).pos
            tags = i.n_closest(origin, n=n+1, group_by_length=True, rtol=rtol)
            ij_dist = [np.linalg.norm(Site(i, tag).pos - origin)
                          for tag in tags]
            sites.append((tags, (j, i), ij_dist))
        distances = np.r_[tuple((i[2] for i in sites))]
        distances = np.sort(distances)
        group_boundaries = np.where(np.diff(distances) > eps)[0]
        # Find distance in `n`-th group.
        if len(group_boundaries) == n:
            n_dist = distances[-1]
        else:
            n_dist = distances[group_boundaries[n]]

        # We now have all the required sites, select the ones in `n`-th group.
        result = []
        for group in sites:
            tags, distance = group[0], group[2]
            i, j = group[1]
            tags = np.array([tag for tag, dist in zip(tags, distance)
                             if abs(dist - n_dist) < eps])
            if len(tags):
                # Sort the tags.
                tags = tags[np.lexsort(tags.T[::-1])][::-1]
                # Throw away equivalent hoppings if
                # two sublattices are the same.
                if i == j and len(tags) > 1:
                    tags = tags[: len(tags) // 2]
                for tag in tags:
                    result.append(builder.HoppingKind(tag, j, i))
        return result

    @property
    def prim_vecs(self):
        """(sequence of vectors) Primitive vectors

        `prim_vecs[i]`` is the `i`-th primitive basis vector of the lattice
        displacement of the lattice origin from the real space coordinates
        origin.
        """
        return np.array(self._prim_vecs)

    def vec(self, int_vec):
        """
        Return the coordinates of a Bravais lattice vector in real space.

        Parameters
        ----------
        vec : integer vector

        Returns
        -------
        output : real vector
        """
        return ta.dot(int_vec, self._prim_vecs)


def short_array_repr(array):
    full = ' '.join([i.lstrip() for i in repr(array).split('\n')])
    return full[6 : -1]


def short_array_str(array):
    full = ', '.join([i.lstrip() for i in str(array).split('\n')])
    return full[1 : -1]


class Monatomic(builder.SiteFamily, Polyatomic):
    """
    A Bravais lattice with a single site in the basis.

    Instances of this class provide the `~kwant.builder.SiteFamily` interface.
    Site tags (see `~kwant.builder.SiteFamily`) are sequences of integers and
    describe the lattice coordinates of a site.

    ``Monatomic`` instances are used as site families on their own or as
    sublattices of `Polyatomic` lattices.

    Parameters
    ----------
    prim_vecs : 2d array-like of floats
        Primitive vectors of the Bravais lattice.

    offset : vector of floats
        Displacement of the lattice origin from the real space
        coordinates origin.

    Attributes
    ----------
    ``offset`` : vector
        Displacement of the lattice origin from the real space coordinates origin
    """

    def __init__(self, prim_vecs, offset=None, name='', norbs=None):
        prim_vecs = ta.array(prim_vecs, float)
        _check_prim_vecs(prim_vecs)

        dim = prim_vecs.shape[1]
        if name is None:
            name = ''

        if offset is None:
            offset = ta.zeros(dim)
        else:
            offset = ta.array(offset, float)
            if offset.shape != (dim,):
                raise ValueError('Dimensionality of offset does not match '
                                 'that of the space.')

        msg = '{0}({1}, {2}, {3}, {4})'
        cl = self.__module__ + '.' + self.__class__.__name__
        canonical_repr = msg.format(cl, short_array_repr(prim_vecs),
                                    short_array_repr(offset),
                                    repr(name), repr(norbs))
        super().__init__(canonical_repr, name, norbs)

        self.sublattices = [self]
        self._prim_vecs = prim_vecs
        self.inv_pv = ta.array(np.linalg.pinv(prim_vecs))
        self.offset = offset

        # Precalculation of auxiliary arrays for real space calculations.
        self.reduced_vecs, self.transf = lll.lll(prim_vecs)
        self.voronoi = ta.dot(lll.voronoi(self.reduced_vecs), self.transf)

        self.dim = dim
        self.lattice_dim = len(prim_vecs)

        if name != '':
            msg = "<Monatomic lattice {0}{1}>"
            orbs = ' with {0} orbitals'.format(self.norbs) if self.norbs else ''
            self.cached_str = msg.format(name, orbs)
        else:
            msg = "<unnamed Monatomic lattice, vectors {0}, origin [{1}]{2}>"
            orbs = ', with {0} orbitals'.format(norbs) if norbs else ''
            self.cached_str = msg.format(short_array_str(self._prim_vecs),
                                         short_array_str(self.offset), orbs)

    def __str__(self):
        return self.cached_str

    def normalize_tag(self, tag):
        tag = ta.array(tag, int)
        if len(tag) != self.lattice_dim:
            raise ValueError("Dimensionality mismatch.")
        return tag

    def n_closest(self, pos, n=1, group_by_length=False, rtol=1e-9):
        """Find n sites closest to position `pos`.

        Returns
        -------
        sites : numpy array
            An array with sites coordinates.
        """
        # TODO (Anton): transform to tinyarrays, once ta indexing is better.
        return np.dot(lll.cvp(pos - self.offset, self.reduced_vecs,
                              n=n, group_by_length=group_by_length, rtol=rtol),
                      self.transf.T)

    def closest(self, pos):
        """
        Find the lattice coordinates of the site closest to position ``pos``.
        """
        return ta.array(self.n_closest(pos)[0])

    def pos(self, tag):
        """Return the real-space position of the site with a given tag."""
        return ta.dot(tag, self._prim_vecs) + self.offset


# The following class is designed such that it should avoid floating
# point precision issues.

class TranslationalSymmetry(builder.Symmetry):
    """A translational symmetry defined in real space.

    An alias exists for this common name: ``kwant.TranslationalSymmetry``.

    Group elements of this symmetry are integer tuples of appropriate length.

    Parameters
    ----------
    p0, p1, p2, ... : sequences of real numbers
        The symmetry periods in real space.

    Notes
    -----
    This symmetry automatically chooses the fundamental domain for each new
    `SiteFamily` it encounters. If this site family does not correspond to a
    Bravais lattice, or if it does not have a commensurate period, an error is
    produced. A certain flexibility in choice of the fundamental domain can be
    achieved by calling manually the `add_site_family` method and providing it
    the `other_vectors` parameter.

    The fundamental domain for hoppings are all hoppings ``(a, b)`` with site
    `a` in fundamental domain of sites.
    """
    def __init__(self, *periods):
        self.periods = ta.array(periods)
        if self.periods.ndim != 2:
            # TODO: remove the second part of the following message once
            # everybody got used to it.
            msg = ("TranslationalSymmetry takes 1d sequences as parameters.\n"
                   "See What's new in Kwant 0.2 in the documentation.")
            raise ValueError(msg)
        if np.linalg.matrix_rank(periods) < len(periods):
            raise ValueError("Translational symmetry periods must be "
                             "linearly independent")
        # A dictionary containing cached data required for applying the
        # symmetry to different site families.
        self.site_family_data = {}
        self.is_reversed = False

    def subgroup(self, *generators):
        """Return the subgroup generated by a sequence of group elements.

        Parameters
        ----------
        *generators: sequence of int
            Each generator must have length ``self.num_directions``.
        """
        generators = ta.array(generators)
        if generators.dtype != int:
            raise ValueError('Generators must be sequences of integers.')
        return TranslationalSymmetry(*ta.dot(generators, self.periods))

    def has_subgroup(self, other):
        if isinstance(other, builder.NoSymmetry):
            return True
        elif not isinstance(other, TranslationalSymmetry):
            raise ValueError("Unknown symmetry type.")

        if other.periods.shape[1] != self.periods.shape[1]:
            return False  # Mismatch of spatial dimensionalities.

        inv = np.linalg.pinv(self.periods)
        factors = np.dot(other.periods, inv)
        # Absolute tolerance is correct in the following since we want an error
        # relative to the closest integer.
        return (np.allclose(factors, np.round(factors), rtol=0, atol=1e-8) and
                np.allclose(ta.dot(factors, self.periods), other.periods))

    def add_site_family(self, fam, other_vectors=None):
        """
        Select a fundamental domain for site family and cache associated data.

        Parameters
        ----------
        fam : `SiteFamily`
            the site family which has to be processed.  Be sure to delete the
            previously processed site families from `site_family_data` if you
            want to modify the cache.

        other_vectors : 2d array-like of integers
            Bravais lattice vectors used to complement the periods in forming
            a basis. The fundamental domain consists of all the lattice sites
            for which the zero coefficients corresponding to the symmetry
            periods in the basis formed by the symmetry periods and
            `other_vectors`. If an insufficient number of `other_vectors` is
            provided to form a basis, the missing ones are selected
            automatically.

        Raises
        ------
        KeyError
            If `fam` is already stored in `site_family_data`.
        ValueError
            If lattice `fam` is incompatible with given periods.
        """
        ensure_isinstance(fam, Monatomic)

        dim = self.periods.shape[1]
        if fam in self.site_family_data:
            raise KeyError('Family already processed, delete it from '
                           'site_family_data first.')
        inv = np.linalg.pinv(fam.prim_vecs)
        try:
            bravais_periods = np.dot(self.periods, inv)
        except ValueError:
            fam_space_dim = fam.prim_vecs.shape[1]
            if dim == fam_space_dim:
                raise
            msg = ("{0}-d-embedded lattice is incompatible with "
                   "{1}-d translational symmetry.")
            raise ValueError(msg.format(fam_space_dim, dim))
        # Absolute tolerance is correct in the following since we want an error
        # relative to the closest integer.
        if (not np.allclose(bravais_periods, np.round(bravais_periods),
                            rtol=0, atol=1e-8) or
            not np.allclose([fam.vec(i) for i in bravais_periods],
                            self.periods)):
            msg = ('Site family {0} does not have commensurate periods with '
                   'symmetry {1}.')
            raise ValueError(msg.format(fam, self))
        bravais_periods = np.array(np.round(bravais_periods), dtype='int')
        (num_dir, lat_dim) = bravais_periods.shape
        if other_vectors is None:
            other_vectors = np.zeros((0, lat_dim), dtype=int)
        else:
            other_vectors = np.array(other_vectors)
            if other_vectors.ndim != 2:
                raise ValueError(
                    '`other_vectors` must be a 2d array-like object.')
            if np.any(np.round(other_vectors) - other_vectors):
                raise ValueError('Only integer other_vectors are allowed.')
            other_vectors = np.array(np.round(other_vectors), dtype=int)

        m = np.zeros((lat_dim, lat_dim), dtype=int)

        m.T[:num_dir] = bravais_periods
        num_vec = num_dir + len(other_vectors)
        m.T[num_dir:num_vec] = other_vectors

        if np.linalg.matrix_rank(m) < num_vec:
            raise ValueError('other_vectors and symmetry periods are not '
                             'linearly independent.')

        # To define the fundamental domain of the new site family we now need to
        # choose `lat_dim - num_vec` extra lattice vectors that are not
        # linearly dependent on the vectors we already have. To do so we
        # continuously add the lattice basis vectors one by one such that they
        # are not linearly dependent on the existent vectors
        while num_vec < lat_dim:
            vh = np.linalg.svd(np.dot(m[:, :num_vec].T, fam.prim_vecs),
                               full_matrices=False)[2]
            projector = np.identity(dim) - np.dot(vh.T, vh)

            residuals = np.dot(fam.prim_vecs, projector)
            residuals = np.apply_along_axis(np.linalg.norm, 1, residuals)
            m[np.argmax(residuals), num_vec] = 1
            num_vec += 1

        det_m = int(round(np.linalg.det(m)))
        if det_m == 0:
            raise RuntimeError('Adding site family failed.')

        det_x_inv_m = np.array(np.round(det_m * np.linalg.inv(m)), dtype=int)
        assert (np.dot(m, det_x_inv_m) // det_m == np.identity(lat_dim)).all()

        det_x_inv_m_part = det_x_inv_m[:num_dir, :]
        m_part = m[:, :num_dir]
        self.site_family_data[fam] = (ta.array(m_part),
                                     ta.array(det_x_inv_m_part), det_m)

    @property
    def num_directions(self):
        return len(self.periods)

    def _get_site_family_data(self, family):
        try:
            return self.site_family_data[family]
        except KeyError:
            self.add_site_family(family)
            return self.site_family_data[family]

    def which(self, site):
        det_x_inv_m_part, det_m = self._get_site_family_data(site.family)[-2:]
        result = ta.dot(det_x_inv_m_part, site.tag) // det_m
        return -result if self.is_reversed else result

    def act(self, element, a, b=None):
        element = ta.array(element)
        if element.dtype is not int:
            raise ValueError("group element must be a tuple of integers")
        m_part = self._get_site_family_data(a.family)[0]
        try:
            delta = ta.dot(m_part, element)
        except ValueError:
            msg = 'Expecting a {0}-tuple group element, but got `{1}` instead.'
            raise ValueError(msg.format(self.num_directions, element))
        if self.is_reversed:
            delta = -delta
        if b is None:
            return builder.Site(a.family, a.tag + delta, True)
        elif b.family == a.family:
            return (builder.Site(a.family, a.tag + delta, True),
                    builder.Site(b.family, b.tag + delta, True))
        else:
            m_part = self._get_site_family_data(b.family)[0]
            try:
                delta2 = ta.dot(m_part, element)
            except ValueError:
                msg = ('Expecting a {0}-tuple group element, '
                       'but got `{1}` instead.')
                raise ValueError(msg.format(self.num_directions, element))
            if self.is_reversed:
                delta2 = -delta2
            return (builder.Site(a.family, a.tag + delta, True),
                    builder.Site(b.family, b.tag + delta2, True))

    def reversed(self):
        """Return a reversed copy of the symmetry.

        The resulting symmetry has all the period vectors opposite to the
        original and an identical fundamental domain.
        """
        result = TranslationalSymmetry(*self.periods)
        result.site_family_data = self.site_family_data
        result.is_reversed = not self.is_reversed
        result.periods = -self.periods
        return result



################ Library of lattices

def chain(a=1, name='', norbs=None):
    """Make a one-dimensional lattice."""
    return Monatomic(((a,),), name=name, norbs=norbs)


def square(a=1, name='', norbs=None):
    """Make a square lattice."""
    return Monatomic(((a, 0), (0, a)), name=name, norbs=norbs)


def cubic(a=1, name='', norbs=None):
    """Make a cubic lattice."""
    return Monatomic(((a, 0, 0), (0, a, 0), (0, 0, a)),
                     name=name, norbs=norbs)


tri = ta.array(((1, 0), (0.5, 0.5 * sqrt(3))))

def triangular(a=1, name='', norbs=None):
    """Make a triangular lattice."""
    return Monatomic(a * tri, name=name, norbs=norbs)


def honeycomb(a=1, name='', norbs=None):
    """Make a honeycomb lattice."""
    lat = Polyatomic(a * tri, ((0, 0), (0, a / sqrt(3))),
                     name=name, norbs=norbs)
    lat.a, lat.b = lat.sublattices
    return lat


def kagome(a=1, name='', norbs=None):
    """Make a kagome lattice."""
    lat = Polyatomic(a * tri, ((0, 0),) + tuple(0.5 * a * tri),
                     name=name, norbs=norbs)
    lat.a, lat.b, lat.c = lat.sublattices
    return lat


# TODO (Anton): unhide _prim_vecs, once tinyarray supports indexing of
# sub-arrays.
