# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from __future__ import division

__all__ = ['TranslationalSymmetry', 'general', 'Polyatomic', 'Monatomic']

from math import sqrt
import numpy as np
import tinyarray as ta
from . import builder


def general(prim_vecs, basis=None, name=''):
    """
    Create a Bravais lattice of any dimensionality, with any number of sites.

    Parameters
    ----------
    prim_vecs : sequence of sequences of floats
        The primitive vectors of the Bravais lattice.
    basis : sequence of floats
        The coordinates of the basis sites inside the unit cell.
    name : string or sequence of strings
        Name of the lattice, or the list of names of all of the sublattices.
        If the name of the lattice is given, the names of sublattices (if any)
        are obtained by appending their number to the name of the lattice.

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
        return Monatomic(prim_vecs, name=name)
    else:
        return Polyatomic(prim_vecs, basis, name=name)


class Polyatomic(object):
    """
    A Bravais lattice with an arbitrary number of sites in the basis.

    Contains `Monatomic` sublattices.

    Parameters
    ----------
    prim_vecs : sequence of floats
        The primitive vectors of a Bravais lattice.
    basis : sequence of floats
        The coordinates of the basis sites inside the unit cell.
    name : string or sequence of strings
        The name of the lattice, or a list of the names of all the sublattices.
        If the name of the lattice is given, the names of sublattices are
        obtained by appending their number to the name of the lattice.

    Instance Variables
    ------------------
    sublattices : list of `Monatomic`
        Sublattices belonging to this lattice.

    Raises
    ------
    ValueError
        If dimensionalities do not match.

    Notes
    -----

    """
    def __init__(self, prim_vecs, basis, name=''):
        prim_vecs = ta.array(prim_vecs, float)
        dim = prim_vecs.shape[1]
        if name is None:
            name = ''
        if isinstance(name, str):
            name = [name + str(i) for i in range(len(basis))]
        if prim_vecs.shape[0] > dim:
            raise ValueError('Number of primitive vectors exceeds '
                             'the space dimensionality.')
        basis = ta.array(basis, float)
        if basis.shape[1] != dim:
            raise ValueError('Basis dimensionality does not match '
                             'the space dimensionality.')
        self.sublattices = [Monatomic(prim_vecs, offset, sname)
                            for offset, sname in zip(basis, name)]
        # Sequence of primitive vectors of the lattice.
        self.prim_vecs = prim_vecs

    def shape(self, function, start):
        """
        Yield all the lattice sites which belong to a certain shape.

        Parameters
        ----------
        function : a boolean function of real-space coordinates
            A function which evaluates to True inside the desired shape.
        start : real-valued vector
            The starting point to the flood-fill algorithm.  If the site
            nearest to `start` is not inside the shape, no sites are returned.

        Returns
        -------
        sites : sequence of `Site` objects
            all the sites that belong to the lattice and fit inside the shape.
        """
        Site = builder.Site

        dim = len(start)
        num_vecs = len(self.prim_vecs)
        if dim != self.prim_vecs.shape[1]:
            raise ValueError('Dimensionality of start position does not match'
                             ' the space dimensionality.')
        sls = self.sublattices
        deltas = [ta.array(i * (0,) + (1,) + (num_vecs - 1 - i) * (0,))
                  for i in xrange(num_vecs)]
        deltas += [-delta for delta in deltas]

        # Check if no sites are going to be added, to catch a common error.
        empty = True
        for sl in sls:
            if function(sl(*sl.closest(start)).pos):
                empty = False
        if empty:
            msg = 'No sites close to {0} are inside the desired shape.'
            raise ValueError(msg.format(start))

        # Continue to flood fill.
        outer_shell = set(sl.closest(start) for sl in sls)
        inner_shell = set()
        while outer_shell:
            tmp = set()
            for tag in outer_shell:
                vec = ta.dot(tag, self.prim_vecs)
                any_hits = False
                for sl in sls:
                    if not function(vec + sl.offset):
                        continue
                    yield Site(sl, tag, True)
                    any_hits = True
                if not any_hits:
                    continue
                for shift in deltas:
                    new_tag = tag + shift
                    if new_tag not in inner_shell and \
                       new_tag not in outer_shell:
                        tmp.add(new_tag)
            inner_shell = outer_shell
            outer_shell = tmp

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
        return ta.dot(int_vec, self.prim_vecs)


class Monatomic(builder.SiteGroup, Polyatomic):
    """
    A Bravais lattice with a single site in the basis.  Also a site group.

    Used on its own and as sublattices of `Polyatomic` lattices.

    Parameters
    ----------
    prim_vecs : sequence of floats
        Primitive vectors of the Bravais lattice.
    offset : vector of floats
        Displacement of the lattice origin from the real space
        coordinates origin.
    """

    def __init__(self, prim_vecs, offset=None, name=''):
        prim_vecs = ta.array(prim_vecs, float)
        dim = prim_vecs.shape[1]
        if name is None:
            name = ''
        self.name = name
        if prim_vecs.shape[0] > dim:
            raise ValueError('Number of primitive vectors exceeds '
                             'the space dimensionality.')
        if offset is None:
            offset = ta.zeros(dim)
        else:
            offset = ta.array(offset, float)
            if offset.shape != (dim,):
                raise ValueError('Dimensionality of offset does not match '
                                 'that of the space.')
        self.sublattices = [self]
        self.prim_vecs = prim_vecs
        self.inv_pv = ta.array(np.linalg.pinv(prim_vecs))
        self.offset = offset

        def short_array_repr(array):
            full = ' '.join([i.lstrip() for i in repr(array).split('\n')])
            return full[6 : -1]

        msg = '{0}({1}, {2}, {3})'
        cl = self.__module__ + '.' + self.__class__.__name__
        self.canonical_repr = msg.format(cl, short_array_repr(self.prim_vecs),
                                         short_array_repr(self.offset),
                                         repr(self.name))
        intern(self.canonical_repr)
        self.dim = dim
        self._lattice_dim = len(prim_vecs)

        def short_array_str(array):
            full = ', '.join([i.lstrip() for i in str(array).split('\n')])
            return full[1 : -1]

        if self.name != '':
            msg = "Monatomic lattice {0}, vectors {1}, origin {2}"
            self.cached_str = msg.format(self.name,
                                         short_array_str(self.prim_vecs),
                                         short_array_str(self.offset))
        else:
            msg = "unnamed Monatomic lattice, vectors {0}, origin [{1}]"
            self.cached_str = msg.format(short_array_str(self.prim_vecs),
                                         short_array_str(self.offset))

    def __str__(self):
        return self.cached_str

    def normalize_tag(self, tag):
        tag = ta.array(tag, int)
        if len(tag) != self._lattice_dim:
            raise ValueError("Dimensionality mismatch.")
        return tag

    def closest(self, pos):
        """Find the site closest to position `pos`."""
        return ta.array(ta.round(ta.dot(pos - self.offset, self.inv_pv)), int)

    def pos(self, tag):
        """Return the real-space position of the site with a given tag."""
        return ta.dot(tag, self.prim_vecs) + self.offset


# The following class is designed such that it should avoid floating
# point precision issues.

class TranslationalSymmetry(builder.Symmetry):
    """
    A translational symmetry defined in real space.

    Group elements of this symmetry are integer tuples of appropriate length.

    Parameters
    ----------
    p0, p1, p2, ... : sequences of real numbers
        The symmetry periods in real space.

    Notes
    -----
    This symmetry automatically chooses the fundamental domain for each new
    `SiteGroup` it encounters. If this site group does not correspond to a
    Bravais lattice, or if it does not have a commensurate period, an error is
    produced. A certain flexibility in choice of the fundamental domain can be
    achieved by calling manually the `add_site_group` method and providing it
    the `other_vectors` parameter.

    The fundamental domain for hoppings are all hoppings ``(a, b)`` with site
    `a` in fundamental domain of sites.
    """
    def __init__(self, *periods):
        self.periods = ta.array(periods)
        self._periods = self.periods
        if self.periods.ndim != 2:
            # TODO: remove the second part of the following message once
            # everybody got used to it.
            msg = "TranslationalSymmetry takes 1d sequences as parameters.\n" \
                "See What's new in kwant 0.2 in the documentation."
            raise ValueError(msg)
        if np.linalg.matrix_rank(periods) < len(periods):
            raise ValueError("Translational symmetry periods must be "
                             "linearly independent")
        # A dictionary containing cached data required for applying the
        # symmetry to different site groups.
        self.site_group_data = {}
        self.is_reversed = False

    def add_site_group(self, gr, other_vectors=None):
        """
        Select a fundamental domain for site group and cache associated data.

        Parameters
        ----------
        gr : `SiteGroup`
            the site group which has to be processed.  Be sure to delete the
            previously processed site groups from `site_group_data` if you want
            to modify the cache.

        other_vectors : list of lists of integers
            Bravais lattice vectors used to complement the periods in forming
            a basis. The fundamental domain belongs to the linear space
            spanned by these vectors.

        Raises
        ------
        KeyError
            If `gr` is already stored in `site_group_data`.
        ValueError
            If lattice shape of `gr` cannot have the given `periods`.
        """
        if gr.canonical_repr in self.site_group_data:
            raise KeyError('Group already processed, delete it from '
                           'site_group_data first.')
        inv = np.linalg.pinv(gr.prim_vecs)
        bravais_periods = [np.dot(i, inv) for i in self._periods]
        if not np.allclose(bravais_periods, np.round(bravais_periods),
                           rtol=0, atol=1e-8) or \
           not np.allclose([gr.vec(i) for i in bravais_periods],
                           self._periods):
            msg = 'Site group {0} does not have commensurate periods with ' +\
                  'symmetry {1}.'
            raise ValueError(msg.format(gr, self))
        bravais_periods = np.array(np.round(bravais_periods), dtype='int')
        (num_dir, dim) = bravais_periods.shape
        if other_vectors is None:
            other_vectors = []
        for vec in other_vectors:
            for a in vec:
                if not isinstance(a, int):
                    raise ValueError('Only integer other_vectors are allowed.')
        m = np.zeros((dim, dim), dtype=int)

        m.T[: num_dir] = bravais_periods
        num_vec = num_dir + len(other_vectors)
        if len(other_vectors) != 0:
            m.T[num_dir:num_vec] = other_vectors
        norms = np.apply_along_axis(np.linalg.norm, 1, m)
        indices = np.argsort(norms)
        for coord in zip(indices, range(num_vec, dim)):
            m[coord] = 1

        det_m = int(round(np.linalg.det(m)))
        if det_m == 0:
            raise ValueError('Singular symmetry matrix.')

        det_x_inv_m = \
            np.array(np.round(det_m * np.linalg.inv(m)), dtype=int)
        assert (np.dot(m, det_x_inv_m) // det_m == np.identity(dim)).all()

        det_x_inv_m_part = det_x_inv_m[:num_dir, :]
        m_part = m[:, :num_dir]
        self.site_group_data[gr.canonical_repr] = (ta.array(m_part),
                                    ta.array(det_x_inv_m_part), det_m)

    @property
    def num_directions(self):
        return len(self.periods)

    def _get_site_group_data(self, group):
        try:
            return self.site_group_data[group.canonical_repr]
        except KeyError:
            self.add_site_group(group)
            return self.site_group_data[group.canonical_repr]

    def which(self, site):
        det_x_inv_m_part, det_m = self._get_site_group_data(site.group)[-2:]
        result = ta.dot(det_x_inv_m_part, site.tag) // det_m
        return -result if self.is_reversed else result

    def act(self, element, a, b=None):
        m_part = self._get_site_group_data(a.group)[0]
        try:
            delta = ta.dot(m_part, element)
        except ValueError:
            msg = 'Expecting a {0}-tuple group element, but got `{1}` instead.'
            raise ValueError(msg.format(self.num_directions, element))
        if self.is_reversed:
            delta *= -1
        if b is None:
            return builder.Site(a.group, a.tag + delta, True)
        elif b.group == a.group:
            return builder.Site(a.group, a.tag + delta, True), \
                builder.Site(b.group, b.tag + delta, True)
        else:
            m_part = self._get_site_group_data(b.group)[0]
            try:
                delta2 = ta.dot(m_part, element)
            except ValueError:
                msg = 'Expecting a {0}-tuple group element, ' + \
                      'but got `{1}` instead.'
                raise ValueError(msg.format(self.num_directions, element))
            if self.is_reversed:
                delta2 *= -1
            return builder.Site(a.group, a.tag + delta, True), \
                builder.Site(b.group, b.tag + delta2, True)

    def to_fd(self, a, b=None):
        return self.act(-self.which(a), a, b)

    def reversed(self):
        """Return a reversed copy of the symmetry.

        The resulting symmetry has all the period vectors opposite to the
        original and an identical fundamental domain.
        """
        result = TranslationalSymmetry(*self._periods)
        result.site_group_data = self.site_group_data
        result.is_reversed = not self.is_reversed
        result.periods = -self.periods
        return result


################ Library of lattices (to be extended)

def chain(a=1, name=''):
    """Create a one-dimensional lattice."""
    lat = Monatomic(((a,),), name=name)
    lat.nearest = [((1,), lat, lat)]
    return lat


def square(a=1, name=''):
    """Create a square lattice."""
    lat = Monatomic(((a, 0), (0, a)), name=name)
    lat.nearest = [((1, 0), lat, lat),
                    ((0, 1), lat, lat)]
    return lat


def honeycomb(a=1, name=''):
    """Create a honeycomb lattice."""
    lat = Polyatomic(((a, 0), (0.5 * a, 0.5 * a * sqrt(3))),
                            ((0, 0), (0, a / sqrt(3))), name=name)
    lat.a, lat.b = lat.sublattices
    lat.nearest = [((0, 0), lat.a, lat.b),
                    ((0, 1), lat.a, lat.b),
                    ((-1, 1), lat.a, lat.b)]
    return lat
