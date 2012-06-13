from __future__ import division

__all__ = ['make_lattice', 'TranslationalSymmetry',
           'PolyatomicLattice', 'MonatomicLattice']

import struct
from math import sqrt
from itertools import izip, chain
import numpy as np
from . import builder


def make_lattice(prim_vecs, basis=None):
    """
    Create a Bravais lattice, which may have more than one basis site.

    Parameters
    ----------
    prim_vecs : sequence of sequences of floats
        Primitive vectors of a Bravais lattice.
    basis : sequence of floats
        Coordinates of the basis sites inside the unit cell.

    Returns
    -------
    lattice : either `MonatomicLattice` or `PolyatomicLattice`
        Resulting lattice.

    Notes
    -----
    This function is largely an alias to the constructors of corresponding
    lattices.
    """
    if basis is None:
        return MonatomicLattice(prim_vecs)
    else:
        return PolyatomicLattice(prim_vecs, basis)


class PolyatomicLattice(object):
    """
    Bravais lattice with a basis containing more than one site.

    Contains monatomic sublattices.

    Parameters
    ----------
    prim_vecs : sequence of floats
        Primitive vectors of a Bravais lattice.
    basis : sequence of floats
        Coordinates of the basis sites inside the unit cell.

    Instance Variables
    ------------------
    sublattices : list of `MonatomicLattice`
        Sublattices belonging to this lattice.

    Raises
    ------
    ValueError
        If dimensionalities do not match.

    Notes
    -----

    """
    def __init__(self, prim_vecs, basis):
        prim_vecs = np.asarray(prim_vecs, dtype=float)
        dim = prim_vecs.shape[1]
        if prim_vecs.shape[0] > dim:
            raise ValueError('Number of primitive vectors exceeds '
                             'the space dimensionality.')
        basis = np.asarray(basis, dtype=float)
        if basis.shape[1] != dim:
            raise ValueError('Basis dimensionality does not match '
                             'the space dimensionality.')
        self.sublattices = [MonatomicLattice(prim_vecs, offset)
                            for offset in basis]
        # Sequence of primitive vectors of the lattice.
        self.prim_vecs = prim_vecs

    # TODO (Anton): Currently speed of shape does not seem to cause problem,
    # but memory usage is excessive. This function might be changed to work
    # with Builder, so that examined sites are already stored in Builder, and
    # not in an additional list.
    def shape(self, function, start):
        """
        Yield all the lattice site which belong to a certain shape.

        Parameters
        ----------
        function : a boolean function of real space coordinates
            A function which evaluates to True inside the desired shape.
        start : real-valued vector
            The starting point to the flood-fill algorithm.  If the site
            nearest to `start` is not inside the shape, no sites are returned.

        Returns
        -------
        sites : sequence of `Site` objects
            all the sites that belong to the lattice and fit inside the shape.
        """
        dim = len(start)
        if dim != self.prim_vecs.shape[1]:
            raise ValueError('Dimensionality of start position does not match'
                             ' the space dimensionality.')
        sls = self.sublattices

        # Check if no sites are going to be added, to catch a common error.
        empty = True
        for sl in sls:
            if function(sl(*sl.closest(start)).pos):
                empty = False
        if empty:
            msg = 'No sites close to {0} are inside the desired shape.'
            raise ValueError(msg.format(start))

        # Continue to flood fill.
        pending = [sl.closest(start) for sl in sls]
        examined = set([])
        while pending:
            tag = pending.pop()
            if tag in examined: continue
            examined.add(tag)

            vec = np.dot(tag, self.prim_vecs)
            any_hits = False
            for sl in sls:
                if not function(vec + sl.offset): continue
                yield sl(*tag)
                any_hits = True

            if not any_hits: continue
            tag = list(tag)
            for i in xrange(dim):
                tag[i] += 1
                pending.append(tuple(tag))
                tag[i] -= 2
                pending.append(tuple(tag))
                tag[i] += 1

    def vec(self, int_vec):
        """
        Return coordinates of a Bravais lattice vector in real space.

        Parameters
        ----------
        vec : integer vector

        Returns
        -------
        output : real vector
        """
        return np.dot(int_vec, self.prim_vecs)


class MonatomicLattice(PolyatomicLattice, builder.SiteGroup):
    """
    A site group of sites belonging to a Bravais lattice.

    Parameters
    ----------
    prim_vecs : sequence of floats
        Primitive vectors of a Bravais lattice.
    offset : vector of floats
        Displacement of the lattice origin from the real space
        coordinates origin.
    """
    dim_end = 100
    pack_fmt_prefix = '='
    pack_letter = 'i'
    _pack_item_size = struct.calcsize(pack_fmt_prefix + pack_letter)
    _pack_fmts = [pack_fmt_prefix + pack_letter * i
                  for i in xrange(dim_end)]
    del dim_end, pack_fmt_prefix, pack_letter, i

    def __init__(self, prim_vecs, offset=None):
        prim_vecs = np.asarray(prim_vecs, dtype=float)
        dim = prim_vecs.shape[1]
        if prim_vecs.shape[0] > dim:
            raise ValueError('Number of primitive vectors exceeds '
                             'the space dimensionality.')
        if offset is None:
            offset = np.zeros(dim)
        else:
            offset = np.asarray(offset, dtype=float)
            if offset.shape != (dim,):
                raise ValueError('Dimensionality of offset does not match '
                                 'that of the space.')
        self.sublattices = [self]
        self.prim_vecs = prim_vecs
        self.inv_pv = np.linalg.pinv(prim_vecs)
        self.offset = offset

        assert 0 < dim < len(self._pack_fmts)
        builder.SiteGroup.__init__(self)
        self.dim = dim

    def pack_tag(self, tag):
        assert len(tag) == self.dim
        return struct.pack(self._pack_fmts[len(tag)], *tag)

    def verify_tag(self, tag):
        try:
            l = len(tag)
        except:
            raise ValueError('The tag must be a sequence.')
        if l != self.dim:
            return False
        elif not [isinstance(i, int) for i in tag]:
            return False
        else:
            return True

    def unpack_tag(self, ptag):
        d = len(ptag)
        assert(d % self._pack_item_size == 0)
        d //= self._pack_item_size
        return struct.unpack(self._pack_fmts[d], ptag)

    def closest(self, pos):
        """Find the site closest to position `pos`."""
        return tuple(np.asarray(
                np.round(np.dot(pos - self.offset, self.inv_pv)),
                dtype=int))

    def pos(self, tag):
        """Return the real space position of the site with a given tag."""
        return np.dot(tag, self.prim_vecs) + self.offset


# The following class is designed such that it should avoid floating
# point precision issues.

class TranslationalSymmetry(builder.Symmetry):
    """
    A translational symmetry defined in real space.

    Group elements of this symmetry are integer tuples of appropriate length.

    Parameters
    ----------
    periods : list of lists of real-valued variables
        list of symmetry periods in real space.

    Notes
    -----
    This symmetry automatically chooses the fundamental domain for each new
    `SiteGroup` it encounters. If this site group does not correspond to a
    Bravais lattice, or if it does not have a commensurate period, an error is
    produced. A certain flexibility in choice of the fundamental domain can be
    achieved by calling manually the `add_site_group` method and providing it
    the `other_vectors` parameter.
    """
    def __init__(self, periods):
        self.periods = np.array(periods)
        # A dictionary containing cached data required for applying the
        # symmetry to different site groups.
        self.site_group_data = {}

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
        if gr in self.site_group_data:
            raise KeyError('Group already processed, delete it from '
                           'site_group_data first.')
        inv = gr.prim_vecs.copy()
        inv = np.linalg.pinv(inv)
        bravais_periods = [np.dot(i, inv) for i in self.periods]
        if not np.allclose(bravais_periods, np.round(bravais_periods),
                           rtol=0, atol=1e-8) or \
           not np.allclose([gr.vec(i) for i in bravais_periods],
                           self.periods):
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
            m.T[num_dir : num_vec] = other_vectors
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
        self.site_group_data[gr] = (det_x_inv_m_part, m_part, det_m)

    @property
    def num_directions(self):
        return len(self.periods)

    def which(self, site):
        try:
            det_x_inv_m_part, m_part, det_m = self.site_group_data[site.group]
        except KeyError:
            self.add_site_group(site.group)
            return self.which(site)
        return np.dot(det_x_inv_m_part, site.tag) // det_m

    def act(self, element, a, b=None):
        try:
            det_x_inv_m_part, m_part, det_m = self.site_group_data[a.group]
        except KeyError:
            self.add_site_group(gr)
            return self.act(element, a, b)
        try:
            delta = np.dot(m_part, element)
        except ValueError:
            msg = 'Expecting a {0}-tuple group element, but got `{1}` instead.'
            raise ValueError(msg.format(self.num_directions, element))
        if b is None:
            return a.shifted(delta)
        else:
            return a.shifted(delta), b.shifted(delta)

    def to_fd(self, a, b=None):
        return self.act(-self.which(a), a, b)

    def reversed(self):
        """Return a reversed copy of the symmetry.

        The result is identical to creating a new symmetry with all the
        period vectors opposite to the original but with the same fundamental
        domain.
        """
        periods = [[-i for i in j] for j in self.periods]
        result = TranslationalSymmetry(periods)
        for gr in self.site_group_data:
            det_x_inv_m_part, m_part, det_m = self.site_group_data[gr]
            if self.num_directions % 2:
                det_m = -det_m
            else:
                det_x_inv_m_part = -det_x_inv_m_part
            m_part = -m_part
            result.site_group_data[gr] = (det_x_inv_m_part.copy(), m_part,
                                          det_m)
        return result


################ Library of lattices (to be extended)

class Chain(MonatomicLattice):
    def __init__(self, a=1):
        MonatomicLattice.__init__(self, ((a,),))
        self.nearest = [((1,), self, self)]


class Square(MonatomicLattice):
    def __init__(self, a=1):
        MonatomicLattice.__init__(self, ((a, 0), (0, a)))
        self.nearest = [((1, 0), self, self),
                        ((0, 1), self, self)]


class Honeycomb(PolyatomicLattice):
    def __init__(self, a=1):
        PolyatomicLattice.__init__(
            self,
            ((a, 0), (0.5 * a, 0.5 * a * sqrt(3))),
            ((0, 0), (0, a / sqrt(3))))
        self.a, self.b = self.sublattices
        self.nearest = [((0, 0), self.b, self.a),
                        ((0, 1), self.b, self.a),
                        ((-1, 1), self.b, self.a)]
