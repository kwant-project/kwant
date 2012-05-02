from __future__ import division

__all__ = ['Builder', 'Site', 'SiteGroup', 'SimpleSiteGroup', 'Symmetry',
           'Lead', 'BuilderLead', 'SelfEnergy']

import struct, abc
from itertools import izip, islice, chain
from collections import Iterable, Sequence
import numpy as np
from kwant import graph
from . import system


################ Sites and site groups

class Site(object):
    """A site, member of a `SiteGroup`.

    Sites are the vertices of the graph which describes the tight binding
    system in a `Builder`.

    A site is uniquely identified by its group and its tag.

    Parameters
    ----------
    group : an instance of `SiteGroup`
        the 'type' of the site.
    tag : a hashable python object
        the personal idenifier of the site e. g. its number.

    Attributes
    ----------
    pos : sequence of numbers
        The real space position of the site.  Used for plotting, for example.

    Raises
    ------
    ValueError
        If ``tag`` is not a proper tag for ``group``.

    Notes
    -----
    For convenience, ``group(*tag)`` can be used instead of ``Site(group,
    tag)`` to create a site.

    The parameters of the constructor (see above) are stored as instance
    variables under the same names.  Given a site ``site``, common things to
    query are thus ``site.group``, ``site.tag``, and ``site.pos``.
    """
    __slots__ = ['group', 'tag']

    def __init__(self, group, tag):
        self.group = group
        if not group.verify_tag(tag):
            msg = 'Tag {0} is not an allowed tag for site group {1}.'
            raise ValueError(msg.format(repr(tag), repr(group)))
        self.tag = tag

    def packed(self):
        """Create a string storing all the site data."""
        group = self.group
        return group.packed_group_id + group.pack_tag(self.tag)

    def shifted(self, delta, group=None):
        """Return a copy of the site, displaced by delta.

        Parameters
        ----------
        delta : sequence of integers
            The vector by which to displace the site.
        group : `SiteGroup`
            Site group of the returned site.  If no site group is provided, the
            original one is kept.

        Returns
        -------
        new_site : `Site`
            A site shifted by `delta` with site group optionally set to
            `group`.

        Notes
        -----
        This method *works* only if the site for which it is called has a tag
        which is a sequences of integers.  It *make sense* only when this sites
        lives on a regular lattice, like one provided by `kwant.lattice`.
        """
        if group is None:
            group = self.group
        tag = self.tag
        if len(tag) != len(delta):
            raise ValueError('Dimensionality mismatch')
        return group(*tuple(a + b for a, b in izip(tag, delta)))

    def __hash__(self):
        return self.group.group_id ^ hash(self.tag)

    def __eq__(self, other):
        return self.group is other.group and self.tag == other.tag

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'Site({0}, {1})'.format(repr(self.group), repr(self.tag))

    @property
    def pos(self):
        """Real space position of the site."""
        return self.group.pos(self.tag)


# Counter used to give each newly created group an unique id.
next_group_id = 0


class SiteGroup(object):
    """
    Abstract base class for site groups.

    A site group is a 'type' of sites.  All the site groups must inherit from
    this basic one.  They have to define the following methods `pack_tag`,
    `unpack_tag` and `verify_tag`.

    Site groups which are intended for use with plotting should also provide a
    method `pos(tag)`, which returns a vector with real space coordinates of
    the site belonging to this group with a given tag.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        global next_group_id
        self.group_id = next_group_id
        next_group_id += 1
        self.packed_group_id = struct.pack(gid_pack_fmt, self.group_id)

    def __repr__(self):
        return '<{0} object: Site group {1}>'.format(
            self.__class__.__name__, self.group_id)

    @abc.abstractmethod
    def pack_tag(self, tag):
        """Return a string storing tag data."""
        pass

    @abc.abstractmethod
    def unpack_tag(self, ptag):
        """Create a tag given a string with its packed content."""
        pass

    @abc.abstractmethod
    def verify_tag(self, tag):
        """Verify if the tag is a legitimate tag for this site group."""
        pass

    def __call__(self, *tag):
        """
        A convenience function.

        This function allows to write sg(1, 2) instead of Site(sg, (1, 2)).
        """
        # Catch a likely and difficult to find mistake.
        if tag and isinstance(tag[0], tuple):
            raise ValueError('Use site_group(1, 2) instead of '
                             'site_group((1, 2))!')
        return Site(self, tag)


class SimpleSiteGroup(SiteGroup):
    """A site group used as an example and for testing.

    A group of sites tagged by any python objects where object satisfied
    condition ``object == eval(repr(object))``.

    It exists to provide a basic site group that can be used for testing the
    builder module without other dependencies.  It can be also used to tag
    sites with non-numeric objects like strings should this every be useful.

    Due to its low storage efficiency for numbers it is not recommended to use
    `SimpleSiteGroup` when `kwant.lattice.MonatomicLattice` would also work.
    """
    def pack_tag(self, tag):
        return repr(tag)

    def unpack_tag(self, ptag):
        return eval(ptag)

    def verify_tag(self, tag):
        return eval(repr(tag)) == tag


# This is used for packing and unpacking group ids (gids).
gid_pack_fmt = '@P'
gid_pack_size = len(struct.pack(gid_pack_fmt, 0))


# The reason why this is a global function and not a method of Builder is that
# this functionality is also needed by finalized systems.
def unpack(psite, group_by_pgid):
    """Unpack a complete site (packed site group + packed tag).

    This function is for internal use in `builder` module.
    """
    pgid = psite[:gid_pack_size]
    try:
        group = group_by_pgid[pgid]
    except:
        raise RuntimeError('Unknown site group id.')
    return Site(group, group.unpack_tag(psite[gid_pack_size:]))


class SequenceOfSites(Sequence):
    """An immutable sequence of sites.

    The sites are stored packed, but this is invisible to the user.
    """
    def __init__(self, sites):
        self.group_by_pgid = {}
        self.psites = []
        for site in sites:
            psite = site.packed()
            pgid = psite[:gid_pack_size]
            if pgid not in self.group_by_pgid:
                self.group_by_pgid[pgid] = site.group
            self.psites.append(psite)

    def __getitem__(self, index):
        return unpack(self.psites[index], self.group_by_pgid)

    def __len__(self):
        return len(self.psites)


################ Symmetries

class Symmetry(object):
    """Abstract base class for spatial symmetries.

    Many physical systems possess discrete spatial symmetry, which results in
    special properties of these systems.  This class is a standard tool to
    describe discrete spatial symmetries in kwant, where the symmetry of a
    `Builder` is specified at its creation.  The most important kind of the
    symmetry is translational symmetry, used to define scattering leads.  This
    class is designed with translational symmetry in mind, and will possibly be
    modified/extended in future.

    Each symmetry has a fundamental domain -- a set of sites and hoppings,
    generating all the possible sites and hoppings upon action of symmetry
    group elements.  The class derived from `Symmetry` has to implement mapping
    of any site or hopping (a tuple of two sites) into the fundamental domain,
    applying a symmetry group element to a site or a hopping, and a method
    `which` to determine the group element bringing some site from the
    fundamental domain to the requested one.  Additionally, it has to have a
    property `num_directions` returning the number of independent symmetry
    group generators (number of elementary periods for translational symmetry).
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractproperty
    def num_directions(self):
        """Number of elementary periods of the symmetry."""
        pass

    @abc.abstractmethod
    def which(self, site):
        """Calculate the domain of the site.

        Return the group element whose action on a certain site from the
        fundamental domain will result in the given `site`.
        """
        pass

    @abc.abstractmethod
    def act(self, element, a, b=None):
        """Act with a symmetry group element on a site or hopping."""
        pass

    def to_fd(self, a, b=None):
        """Map a site or hopping to the fundamental domain.

        If `b` is None, return a site equivalent to `a` within the fundamental
        domain.  Otherwise, return a hopping equivalent to `(a, b)` but where
        the first element belongs to the fundamental domain.

        This default implementation works but may be not efficient.
        """
        group_element = tuple(-x for x in self.which(a))
        return self.act(group_element, a, b)

    def in_fd(self, site):
        """Tell whether `site` lies within the fundamental domain."""
        for d in self.which(site):
            if d != 0:
                return False
        return True


class NoSymmetry(Symmetry):
    """A symmetry with a trivial symmetry group."""

    def __repr__(self):
        return 'NoSymmetry()'

    @property
    def num_directions(self):
        return 0

    def which(self, site):
        return ()

    def act(self, element, a, b=None):
        if element:
            raise ValueError('`element` must be empty for NoSymmetry.')

    def to_fd(self, a, b=None):
        return a if b is None else (a, b)

    def in_fd(self, site):
        return True


################ In-place modifiable graph

def edges(seq):
    iter = izip(islice(seq, 0, None, 2), islice(seq, 1, None, 2))
    next(iter)                  # Skip the special loop edge.
    return iter


class Graph(object):
    """A directed graph optimized for efficient querying and modification.

    The nodes are labeled by python objects which must be usable as dictionary
    keys.  Each edge, specified by a ``(tail, head)`` pair of nodes, holds an
    object as a value.  Likewise, each tail which occurs in the graph also
    holds a value.  (Nodes which only occur as heads are not required to have
    values.)

    This class is made for internal use.
    """
    def __init__(self):
        # The graph is stored in this dictionary.  For a given `tail`,
        # self.hvhv_by_tail[tail] is a list alternately storing heads and
        # values.  (The heads occupy even locations followed by the values at
        # odd locations.)  Each pair of entries thus describes a single
        # directed edge of the graph.
        #
        # The first pair of entries in each list is special: it always
        # corresponds to a loop edge.  (The head is equal to the tail.)  This
        # special edge has two purposes: It is used to store the value
        # associated with the tail node itself, and it is necessary for the
        # method getkey_tail which helps to conserve memory by storing equal
        # node label only once.
        self.hvhv_by_tail = {}

    def __nonzero__(self):
        return bool(self.hvhv_by_tail)

    def getkey_tail(self, tail):
        """Return the object equal to `tail` which is already referenced.

        This method can be used to save memory by avoiding storing several
        copies of equal node labels."""
        return self.hvhv_by_tail[tail][0]

    def getitem_tail(self, tail):
        """Return the value of a tail."""
        return self.hvhv_by_tail[tail][1]

    def getitem_edge(self, edge):
        """Return the value of an edge."""
        tail, head = edge
        for h, value in edges(self.hvhv_by_tail[tail]):
            if h == head:
                return value
        raise KeyError(edge)

    def setitem_tail(self, tail, value):
        """Set the value of a tail."""
        hvhv = self.hvhv_by_tail.setdefault(tail, [])
        if hvhv:
            hvhv[1] = value
        else:
            hvhv[:] = [tail, value]

    def setitem_edge(self, edge, value):
        """Set the value of an edge."""
        tail, head = edge
        hvhv = self.hvhv_by_tail[tail]
        heads = hvhv[2::2]
        try:
            i = 2 + 2 * heads.index(head)
        except ValueError:
            hvhv.append(head)
            hvhv.append(value)
        else:
            hvhv[i] = head
            hvhv[i + 1] = value

    def delitem_tail(self, tail):
        """Delete a tail."""
        del self.hvhv_by_tail[tail]

    def pop_tail(self, tail):
        """Delete a tail and return its value."""
        return self.hvhv_by_tail.pop(tail)[1]

    def delitem_edge(self, edge):
        """Delete an edge."""
        tail, head = edge
        hvhv = self.hvhv_by_tail[tail]
        heads = hvhv[2::2]
        try:
            i = 2 + 2 * heads.index(head)
        except ValueError:
            raise KeyError(edge)
        del hvhv[i : i + 2]

    def pop_edge(self, edge):
        """Delete an edge and return its value."""
        tail, head = edge
        hvhv = self.hvhv_by_tail[tail]
        heads = hvhv[2::2]
        try:
            i = 2 + 2 * heads.index(head)
        except ValueError:
            raise KeyError(edge)
        value = hvhv[i + 1]
        del hvhv[i : i + 2]
        return value

    def num_edges(self):
        return sum(len(hvhv) - 2
                   for hvhv in self.hvhv_by_tail.itervalues()) // 2

    def edges(self):
        """Return an iterator over all edges."""
        for tail, hvhv in self.hvhv_by_tail.iteritems():
            for head in islice(hvhv, 2, None, 2):
                yield tail, head

    def edge_value_pairs(self):
        """Return an iterator over all ``(edge, value)`` pairs."""
        for tail, hvhv in self.hvhv_by_tail.iteritems():
            for head, value in edges(hvhv):
                yield (tail, head), value

    def tails(self):
        """
        Return a view of (python2: iterator over) all the tails of the graph.
        """
        try:
            return self.hvhv_by_tail.viewkeys()
        except AttributeError:
            return iter(self.hvhv_by_tail)

    def tail_value_pairs(self):
        """Return an iterator over all ``(tails, value)`` pairs. """
        for tail, hvhv in self.hvhv_by_tail.iteritems():
            yield tail, hvhv[1]

    def has_tail(self, tail):
        """Return whether the graph contains a certain tail."""
        return tail in self.hvhv_by_tail

    def has_edge(self, edge):
        """Return whether the graph contains a certain edge."""
        tail, head = edge
        hvhv = self.hvhv_by_tail.get(tail, [])
        return head in islice(hvhv, 2, None, 2)

    def out_neighbors(self, tail):
        hvhv = self.hvhv_by_tail.get(tail, [])
        return islice(hvhv, 2, None, 2)

    def out_degree(self, tail):
        hvhv = self.hvhv_by_tail.get(tail, [])
        return len(hvhv) // 2 - 1


################ Support for Hermitian conjugation

def herm_conj(value):
    """
    Calculate the hermitian conjugate of a python object.

    If the object is neither a complex number nor a matrix, the original value
    is returned.  In the context of this module, this is the correct behavior
    for functions.
    """
    if hasattr(value, 'conjugate'):
        value = value.conjugate()
        if hasattr(value, 'transpose'):
            value = value.transpose()
    return value


class HermConjOfFunc(object):
    """Proxy returning the hermitian conjugate of the original result."""
    __slots__ = ('function')
    def __init__(self, function):
        self.function = function

    def __call__(self, i, j):
        return herm_conj(self.function(j, i))


################ Leads

class Lead(object):
    """Abstract base class for leads that can be attached to a `Builder`.

    Instance Variables
    ------------------
    neighbors : sequence of sites
    """
    __metaclass__ = abc.ABCMeta

    def check_neighbors(self):
        if len(self.neighbors) == 0:
            raise ValueError('Lead is not connected (no neighbors).')

    @abc.abstractmethod
    def finalized():
        """Return a finalized version of the lead.

        Returns
        -------
        finalized_lead

        Notes
        -----
        The finalized lead must at least have a single method
        ``self_energy(energy)`` but it can be a full
        `kwant.system.InfiniteSystem` as well.

        The method ``self_energy`` of the finalized lead must return a square
        matrix of appropriate size.

        The order of neighbors is assumed to be preserved during finalization.
        """
        pass


class BuilderLead(Lead):
    """A lead made from a `Builder` with a spatial symmetry.

    Parameters
    ----------
    builder : `Builder`
        The tight-binding system of a lead. It has to possess appropriate
        symmetry, and it may not contain hoppings between further than
        neighboring lead slices.
    neighbors : sequence of `Site` instances
        Sequence of sites in the scattering region to which the lead is
        attached.

    Notes
    -----
    The hopping from the scattering region to the lead is assumed to be
    equal to the hopping from a lead slice to the next one in the direction of
    the symmetry vector (i.e. the lead is 'leaving' the system and starts
    with a hopping).

    The given order of neighbors is preserved throughout finalization.

    Every system has an attribute `leads`, which stores a list of
    `BuilderLead` objects with all the information about the leads that are
    attached.
    """
    def __init__(self, builder, neighbors):
        self.builder = builder
        self.neighbors = SequenceOfSites(neighbors)
        self.check_neighbors()

    def finalized(self):
        """Return a `kwant.system.InfiniteSystem` corresponding to the
        compressed lead.

        The order of neighbors is kept during finalization.
        """
        return self.builder._finalized_infinite(self.neighbors)


class SelfEnergy(Lead):
    """A general lead defined by its self energy.

    Parameters
    ----------
    self_energy_func : function
        Function which returns the self energy matrix for the neighbors given
        the energy.
    neighbors : sequence of `Site` instances
    """
    def __init__(self, self_energy_func, neighbors):
        self.self_energy_func = self_energy_func
        self.neighbors = SequenceOfSites(neighbors)
        self.check_neighbors()

    def finalized(self):
        """Trivial finalization: the object is returned itself."""
        return self

    def self_energy(self, energy):
        return self.self_energy_func(energy)


################ Builder class

def is_sitelike(key):
    """Determine whether key is similar to a site.

    Returns True if `key` is potentially sitelike, False if `key` is
    potentially hoppinglike, None if it is neither."""
    if isinstance(key, Site):
        return True
    if not isinstance(key, tuple):
        return None
    if not key:
        raise KeyError(key)
    first = key[0]
    return not (isinstance(first, Site) or isinstance(first, tuple))


def for_each_in_key(key, f_site, f_hopp):
    """Perform an operation on each site or hopping in key.

    Key may be
    * a single sitelike or hoppinglike object,
    * a non-tuple iterable of sitelike objects,
    * a non-tuple iterable of hoppinglike objects.
    """
    isl = is_sitelike(key)
    if isl is not None:
        if isl:
            f_site(key)
        else:
            f_hopp(key)
    elif isinstance(key, Iterable) and not isinstance(key, tuple):
        ikey = iter(key)
        try:
            first = next(ikey)
        except StopIteration:
            return
        isl = is_sitelike(first)
        if isl is None:
            raise KeyError(first)
        else:
            if isl:
                f_site(first)
                for sitelike in ikey:
                    f_site(sitelike)
            else:
                f_hopp(first)
                for hoppinglike in ikey:
                    f_hopp(hoppinglike)
    else:
        raise KeyError(key)


# Marker which means for hopping (i, j): this value is given by the Hermitian
# conjugate the value of the hopping (j, i).  Used by Builder and System.
other = []


class Builder(object):
    """A tight binding system defined on a graph.

    This is one of the central types in kwant.  It is used to construct tight
    binding systems in a flexible way.

    The nodes of the graph are `Site` instances.  The edges, i.e. the hoppings,
    are pairs (2-tuples) of sites.  Each node and each edge has a value
    associated with it.  That value can be in fact any python object, but
    currently the only *useful* values are matrices and numbers or functions
    returning them.  The values associated with nodes are interpreted as
    on-site Hamiltonians, the ones associated with edges as hopping integrals.

    To make the graph accessible in a way that is natural within the python
    language it is exposed as a *mapping* (much like a built-in Python
    dictionary).  Keys are sites or pairs of them.  Possible values are 2d
    NumPy arrays, numbers (interpreted as 1 by 1 matrices), or functions.
    Functions receive the site or the hopping (passed to the function as two
    sites) and are expected to return a valid value.

    Builder instances can be made to automatically respect a `Symmetry` that is
    passed to them during creation.

    Parameters
    ----------
    symmetry : `Symmetry` or `None`
        The symmetry of the system.

    Instance Variables
    ------------------
    default_site_group : `SiteGroup` or `None`
        Defaults falue is `None`

    Notes
    -----
    The instance variable `default_site_group` can be set to a `SiteGroup`
    instance.  Then, whenever a `Site` would have been acceptable as parameter
    to the methods of the builder, a non-site ``tag`` object will be also
    accepted.  The ``tag`` will be converted into a site in the following way:
    ``Site(default_site_group, tag)``.

    Builder instances automatically ensure that every hopping is Hermitian, so
    that if ``builder[a, b]`` has been set, there is no need to set
    ``builder[b, a]``.

    Values which are functions allow to define systems quickly, to modify them
    without reconstructing, and to save memory for many-orbital models.

    The behavior of builders with a symmetry is slightly more sophisticated.
    First of all, it is implicitly assumed throughout kwant that **every**
    function assigned as a value to a builder with a symmetry possesses the
    same symmetry.  Secondly, all keys are mapped to the fundamental before
    storing them.  This may produce confusing results when neighbors of a site
    are queried.

    The methods `possible_hoppings` and `attach_lead` *work* only if the sites
    affected by them have tags which are sequences of integers.  They *make
    sense* only when these sites live on a regular lattice, like one provided
    by `kwant.lattice`.

    .. warning::

        If functions are used to set values in a builder with a symmetry, then
        they must satisfy the same symmetry.  There is (currently) no check and
        wrong results will be the consequence of a misbehaving function.

    Examples
    --------
    Define a site.

    >>> builder[site] = value

    Print the value of a site.

    >>> print builder[site]

    Define a hopping.

    >>> builder[site1, site2] = value

    Delete a site.

    >>> del builder[site3]

    """
    def __init__(self, symmetry=None):
        if symmetry is None:
            symmetry = NoSymmetry()
        self.symmetry = symmetry
        self.default_site_group = None
        self.leads = []
        self._ham = Graph()
        self._group_by_pgid = {}

    def reversed(self):
        """Return a shallow copy of the builder with the symmetry reversed.

        This method can be used to attach the same infinite system as lead from
        two opposite sides.  It requires a builder to which an infinite
        symmetry is associated.
        """
        result = object.__new__(Builder)
        result.symmetry = self.symmetry.reversed()
        result.default_site_group = self.default_site_group
        if self.leads:
            raise ValueError('System to be reversed may not have leads.')
        result.leads = []
        result._ham = self._ham
        result._group_by_pgid = self._group_by_pgid
        return result

    def _to_site(self, sitelike):
        """Convert `sitelike` to a site.

        Sitelike can be
        * a site, (It is returned unmodified.)
        * a tag. (Works only if self.default_site_group is not None.)
        """
        if isinstance(sitelike, Site):
            return sitelike
        dsg = self.default_site_group
        if dsg is not None:
            return Site(dsg, sitelike)
        raise KeyError(sitelike)

    def __nonzero__(self):
        return bool(self._ham)

    def _get_site(self, sitelike):
        psite = self.symmetry.to_fd(self._to_site(sitelike)).packed()
        try:
            return self._ham.getitem_tail(psite)
        except KeyError:
            raise KeyError(sitelike)

    def _get_hopping(self, hoppinglike):
        ts = self._to_site
        sym = self.symmetry
        try:
            a, b = hoppinglike
        except:
            raise KeyError(hoppinglike)
        try:
            a, b = sym.to_fd(ts(a), ts(b))
            pa, pb = a.packed(), b.packed()
            value = self._ham.getitem_edge((pa, pb))
        except KeyError:
            raise KeyError(hoppinglike)
        if value is other:
            if not sym.in_fd(b):
                b, a = sym.to_fd(b, a)
                assert not sym.in_fd(a)
                pb, pa = b.packed(), a.packed()
            value = self._ham.getitem_edge((pb, pa))
            if hasattr(value, '__call__'):
                assert not isinstance(value, HermConjOfFunc)
                value = HermConjOfFunc(value)
            else:
                value = herm_conj(value)
        return value

    def __getitem__(self, key):
        """Get the value of a single site or hopping."""
        isl = is_sitelike(key)
        if isl is None:
            raise KeyError(key)
        if isl:
            return self._get_site(key)
        else:
            return self._get_hopping(key)

    def __contains__(self, key):
        """Tell whether the system contains a site or hopping."""
        isl = is_sitelike(key)
        if isl is None:
            raise KeyError(key)
        if isl:
            site = self.symmetry.to_fd(self._to_site(key))
            return self._ham.has_tail(site.packed())
        else:
            ts = self._to_site
            a, b = key
            a, b = self.symmetry.to_fd(ts(a), ts(b))
            return self._ham.has_edge((a.packed(), b.packed()))

    def _set_site(self, sitelike, value):
        """Set a single site."""
        site = self.symmetry.to_fd(self._to_site(sitelike))
        psite = site.packed()
        pgid = psite[:gid_pack_size]
        if pgid not in self._group_by_pgid:
            self._group_by_pgid[pgid] = site.group
        self._ham.setitem_tail(psite, value)

    def _set_hopping(self, hoppinglike, value):
        """Set a single hopping."""
        # Avoid nested HermConjOfFunc instances.
        try:
            a, b = hoppinglike
        except:
            raise KeyError(hoppinglike)
        if isinstance(value, HermConjOfFunc):
            a, b = b, a
            value = value.function

        ham = self._ham
        gkt = ham.getkey_tail
        ts = self._to_site
        sym = self.symmetry

        try:
            a, b = sym.to_fd(ts(a), ts(b))
            if sym.in_fd(b):
                pa, pb = gkt(a.packed()), gkt(b.packed())
                ham.setitem_edge((pa, pb), value)
                ham.setitem_edge((pb, pa), other)
            else:
                pa, pb = gkt(a.packed()), b.packed()
                ham.setitem_edge((pa, pb), value)
                b, a = sym.to_fd(b, a)
                assert not sym.in_fd(a)
                pb, pa = gkt(b.packed()), a.packed()
                ham.setitem_edge((pb, pa), other)
        except KeyError:
            raise KeyError(hoppinglike)

    def __setitem__(self, key, value):
        """Set a single site/hopping or an iterable of them."""
        for_each_in_key(key,
                        lambda s: self._set_site(s, value),
                        lambda h: self._set_hopping(h, value))

    def _del_site(self, sitelike):
        """Delete a single site and all associated hoppings."""
        tfd = self.symmetry.to_fd
        ham = self._ham
        site = tfd(self._to_site(sitelike))
        psite = site.packed()
        try:
            for pneighbor in ham.out_neighbors(psite):
                if ham.has_tail(pneighbor):
                    ham.delitem_edge((pneighbor, psite))
                else:
                    neighbor = unpack(pneighbor, self._group_by_pgid)
                    assert not self.symmetry.in_fd(neighbor)
                    a, b = tfd(neighbor, site)
                    ham.delitem_edge((a.packed(), b.packed()))
        except KeyError:
            raise KeyError(sitelike)
        ham.delitem_tail(psite)

    def _del_hopping(self, hoppinglike):
        """Delete a single hopping."""
        ham = self._ham
        gkt = ham.getkey_tail
        ts = self._to_site
        sym = self.symmetry

        try:
            a, b = hoppinglike
        except:
            raise KeyError(hoppinglike)
        try:
            a, b = sym.to_fd(ts(a), ts(b))
            if sym.in_fd(b):
                pa, pb = a.packed(), b.packed()
                ham.delitem_edge((pa, pb))
                ham.delitem_edge((pb, pa))
            else:
                pa, pb = a.packed(), b.packed()
                ham.delitem_edge((pa, pb))
                b, a = sym.to_fd(b, a)
                assert not sym.in_fd(a)
                pb, pa = b.packed(), a.packed()
                ham.delitem_edge((pb, pa))
        except KeyError:
            raise KeyError(hoppinglike)

    def __delitem__(self, key):
        """Delete a single site/hopping or an iterable of them."""
        for_each_in_key(key,
                        lambda s: self._del_site(s),
                        lambda h: self._del_hopping(h))

    def eradicate_dangling(self):
        """Keep deleting dangling sites until none are left."""
        ham = self._ham
        psites = list(psite for psite in ham.tails()
                      if ham.out_degree(psite) < 2)
        for psite in psites:
            if not ham.has_tail(psite): continue
            while psite:
                pneighbors = tuple(ham.out_neighbors(psite))
                if pneighbors:
                    assert len(pneighbors) == 1
                    pneighbor = pneighbors[0]
                    ham.delitem_edge((pneighbor, psite))
                    if ham.out_degree(pneighbor) > 1:
                        pneighbor = False
                else:
                    pneighbor = False
                ham.delitem_tail(psite)
                psite = pneighbor

    def __iter__(self):
        """Return an iterator over all sites and hoppings."""
        return chain(self.sites(), self.hoppings())

    def sites(self):
        """Return an iterator over all sites."""
        for psite in self._ham.tails():
            yield unpack(psite, self._group_by_pgid)

    def site_value_pairs(self):
        """Return an iterator over all (site, value) pairs."""
        for psite, value in self._ham.tail_value_pairs():
            yield unpack(psite, self._group_by_pgid), value

    def hoppings(self):
        """Return an iterator over all hoppings."""
        gbp = self._group_by_pgid
        for phopp, value in self._ham.edge_value_pairs():
            if value is other: continue
            pa, pb = phopp
            yield (unpack(pa, gbp), unpack(pb, gbp))

    def hopping_value_pairs(self):
        """Return an iterator over all (hopping, value) pairs."""
        gbp = self._group_by_pgid
        for phopp, value in self._ham.edge_value_pairs():
            if value is other: continue
            pa, pb = phopp
            yield (unpack(pa, gbp), unpack(pb, gbp)), value

    def dangling(self):
        """Return an iterator over all dangling sites."""
        ham = self._ham
        for psite in ham.tails():
            if ham.out_degree(psite) < 2:
                yield unpack(psite, self._group_by_pgid)

    def degree(self, sitelike):
        """Return the number of neighbors of a site."""
        psite = self.symmetry.to_fd(self._to_site(sitelike)).packed()
        return self._ham.out_degree(psite)

    def neighbors(self, sitelike):
        """Return an iterator over all neighbors of a site."""
        gbp = self._group_by_pgid
        pa = self.symmetry.to_fd(self._to_site(sitelike)).packed()
        return (unpack(pb, gbp)
                for pb in self._ham.out_neighbors(pa))

    def __iadd__(self, other_sys):
        """Add `other_sys` to the system.

        Sites and hoppings present in both systems are overwritten by those in
        `other_sys`.  The leads of `other_sys` are appended to the leads of the
        system being extended.
        """
        raise NotImplementedError()

    def possible_hoppings(self, delta, group_b, group_a):
        """Return all matching possible hoppings between existing sites.

        A hopping ``(a, b)`` matches precisely when the site group of ``a`` is
        `group_a` and that of ``b`` is `group_b` and ``(a.tag - b.tag)``
        (interpreted as vectors) equals to `delta`.

        Parameters
        ----------
        delta : Sequence of integers
            The sequence is interpreted as a vector with integer elements.
        group_a : `~kwant.builder.SiteGroup`
        grpup_b : `~kwant.builder.SiteGroup`

        Returns
        -------
        hoppings : Iterator over hoppings
           All matching possible hoppings
        """
        d = tuple(-x for x in delta)
        for site0 in self.sites():
            if site0.group is not group_a:
                continue
            site1 = site0.shifted(d, group_b)
            if site1 in self:
                yield site0, site1

    def attach_lead(self, lead_builder, origin=None):
        """Attach a lead to the builder, possibly adding missing sites.

        Parameters
        ----------
        lead_builder : `Builder` with 1D translational symmetry
            Builder of the lead which has to be attached.
        origin : `Site`
            The site which should belong to a domain where the lead should
            begin. It is used to attach a lead inside the system, e.g. to an
            inner radius of a ring.

        Raises
        ------
        ValueError
            If `lead_builder` does not have proper symmetry, has hoppings with
            range of more than one slice, or if it is not completely
            interrupted by the system.

        Notes
        -----
        This method is not fool-proof, i.e. if it returns an error, there is
        no guarantee that the system stayed unaltered.
        """
        sym = lead_builder.symmetry

        if sym.num_directions != 1:
            raise ValueError('Only builders with a 1D symmetry are allowed.')
        for hopping in lead_builder.hoppings():
            if not -1 <= sym.which(hopping[1])[0] <= 1:
                msg = 'Hopping {0} connects non-neighboring slices. Only ' +\
                      'nearest-slice hoppings are allowed ' +\
                      '(consider increasing the lead period).'
                raise ValueError(msg.format(hopping))
        try:
            lead_builder.sites().next()
        except StopIteration:
            raise ValueError('Lead to be attached contains no sites.')

        # Check if site groups of the lead are present in the system (catches
        # a common and a hard to find bug).
        groups = set(self._group_by_pgid.values())
        for site in lead_builder.sites():
            if site.group not in groups:
                msg = 'Sites with site group {0} do not appear in the ' +\
                      'system, hence the system does not interrupt the ' +\
                      'lead. Note that different lattice instances with ' +\
                      'the same parameters are different site groups. ' +\
                      'See tutorial for more details.'
                raise ValueError(msg.format(site.group))

        all_doms = list(sym.which(site)[0]
                        for site in self.sites() if site in lead_builder)
        if origin is not None:
            orig_dom = sym.which(origin)[0]
            all_doms = [dom for dom in all_doms if dom <= orig_dom]
        if len(all_doms) == 0:
            raise ValueError('Builder does not intersect with the lead,'
                             ' this lead cannot be attached.')
        max_dom = max(all_doms)
        min_dom = min(all_doms)
        del all_doms

        neighbors = set()
        added = set()
        # Initialize flood-fill: create the outermost sites.
        for site in lead_builder.sites():
            for neighbor in lead_builder.neighbors(site):
                neighbor = sym.act((max_dom + 1,), neighbor)
                if sym.which(neighbor)[0] == max_dom:
                    if neighbor not in self:
                        self[neighbor] = lead_builder[neighbor]
                        added.add(neighbor)
                    neighbors.add(neighbor)

        # Do flood-fill.
        covered = True
        while covered:
            covered = False
            added2 = set()
            for site in added:
                site_dom = sym.which(site)
                move = lambda x: sym.act(site_dom, x)
                for site_new in lead_builder.neighbors(site):
                    site_new = move(site_new)
                    new_dom = sym.which(site_new)[0]
                    if new_dom == max_dom + 1:
                        continue
                    elif new_dom < min_dom:
                        raise ValueError('Builder does not interrupt the lead,'
                                         ' this lead cannot be attached.')
                    if site_new not in self \
                       and sym.which(site_new)[0] != max_dom + 1:
                        self[site_new] = lead_builder[site_new]
                        added2.add(site_new)
                        covered = True
                    self[site_new, site] = lead_builder[site_new, site]
            added = added2

        self.leads.append(BuilderLead(lead_builder, list(neighbors)))
        return len(self.leads) - 1

    def finalized(self):
        """Return a finalized (=usable with solvers) copy of the system.

        Returns
        -------
        finalized_system : `kwant.system.FiniteSystem`
            If there is no symmetry.
        finalized_system : `kwant.system.InfiniteSystem`
            If a symmetry is present.

        Notes
        -----
        This method does not modify the Builder instance for which it is
        called.

        Attached leads are also finalized and will be present in the finalized
        system to be returned.

        Currently, only Builder instances without or with a 1D translational
        `Symmetry` can be finalized.
        """
        if self.symmetry.num_directions == 0:
            return self._finalized_finite()
        elif self.symmetry.num_directions == 1:
            return self._finalized_infinite()
        else:
            raise ValueError('Currently, only builders without or with a 1D '
                             'translational symmetry can be finalized.')

    def _finalized_finite(self):
        assert self.symmetry.num_directions == 0
        ham = self._ham

        #### Make translation tables.
        id_by_psite = {}
        psites = []
        for psite in ham.tails(): # Loop over all packed sites.
            psite_id = len(psites)
            psites.append(psite)
            id_by_psite[psite] = psite_id

        #### Make graph.
        g = graph.Graph()
        g.num_nodes = len(psites) # Some sites could not appear in any edge.
        for tail, head in ham.edges():
            if tail == head: continue
            g.add_edge(id_by_psite[tail], id_by_psite[head])
        g = g.compressed()

        #### Connect leads.
        finalized_leads = []
        lead_neighbor_seqs = []
        for lead_nr, lead in enumerate(self.leads):
            try:
                finalized_leads.append(lead.finalized())
            except ValueError, e:
                msg = 'Problem finalizing lead {0}:'
                e.args = (' '.join((msg.format(lead_nr),) + e.args),)
                raise
            lns = [id_by_psite[neighbor.packed()]
                   for neighbor in lead.neighbors]
            lead_neighbor_seqs.append(np.array(lns))

        #### Assemble and return result.
        result = FiniteSystem()
        result.graph = g
        result.psites_idxs = np.cumsum([0] + [len(psite) for psite in psites])
        result.psites = "".join(psites)
        result.group_by_pgid = self._group_by_pgid
        result.leads = finalized_leads
        result.hoppings = [ham.getitem_edge((psites[tail], psites[head]))
                           for tail, head in g]
        result.onsite_hamiltonians = [ham.getitem_tail(psite)
                                      for psite in psites]
        result.lead_neighbor_seqs = lead_neighbor_seqs
        result.symmetry = self.symmetry
        return result

    def _finalized_infinite(self, order_of_neighbors=None):
        """
        Finalize this builder instance which has to have exactly a single
        symmetry direction.

        If order_of_neighbors is not set, the order of the neighbors in the
        finalized system will be arbitrary.  If order_of_neighbors is set to a
        sequence of neighbor sites, this order will be kept.
        """
        ham = self._ham
        sym = self.symmetry
        gbp = self._group_by_pgid
        assert sym.num_directions == 1

        #### For each site of the fundamental domain, determine whether it has
        #### neighbors or not.
        plsites_with = []    # Fund. domain sites with neighbors in prev. dom
        plsites_without = [] # Remaining sites of the fundamental domain
        for ptail in ham.tails(): # Loop over all sites of the fund. domain.
            for phead in ham.out_neighbors(ptail):
                head = unpack(phead, gbp)
                fd = sym.which(head)[0]
                if fd == 1:
                    # Tail belongs to fund. domain, head to the next domain.
                    plsites_with.append(ptail)
                    break
            else:
                # Tail is a fund. domain site not connected to prev. domain.
                plsites_without.append(ptail)
        slice_size = len(plsites_with) + len(plsites_without)

        if not plsites_with:
            raise ValueError('Infinite system with disconnected slices.')

        ### Create list of packed sites `psites` and a lookup table
        if order_of_neighbors is None:
            pneighbors = [sym.act((-1,), unpack(s, gbp)).packed()
                          for s in plsites_with]
        else:
            shift = (-sym.which(order_of_neighbors[0])[0] - 1,)
            plsites_with_set = set(plsites_with)
            plsites_with = []
            pneighbors = []
            for out_of_place_neighbor in order_of_neighbors:
                # Shift the neighbor domain before the fundamental domain.
                # That's the right place for the neighbors of a lead to be, but
                # the neighbors in order_of_neighbors might live in a different
                # domain.
                neighbor = sym.act(shift, out_of_place_neighbor)
                pneighbor = neighbor.packed()
                plsite = sym.act((1,), neighbor).packed()

                try:
                    plsites_with_set.remove(plsite)
                except KeyError:
                    if (-sym.which(out_of_place_neighbor)[0] - 1,) != shift:
                        raise ValueError(
                            'The sites in order_of_neighbors do not all '
                            'belong to the same lead slice.')
                    else:
                        raise ValueError('A site in order_of_neighbors is '
                                         'not a neighbor:\n' + str(neighbor))
                pneighbors.append(pneighbor)
                plsites_with.append(plsite)
            if plsites_with_set:
                raise ValueError(
                    'order_of_neighbors did not contain all neighbors.')
            del plsites_with_set

        psites = plsites_with + plsites_without + pneighbors
        del plsites_with
        del plsites_without
        del pneighbors
        id_by_psite = {}
        for site_id, psite in enumerate(psites):
            id_by_psite[psite] = site_id

        #### Make graph and extract onsite Hamiltonians.
        g = graph.Graph()
        onsite_hamiltonians = []
        for tail_id, ptail in enumerate(psites[:slice_size]):
            onsite_hamiltonians.append(ham.getitem_tail(ptail))
            for phead in ham.out_neighbors(ptail):
                head_id = id_by_psite.get(phead)
                if head_id is None:
                    # Head belongs neither to the fundamental domain nor to the
                    # previous domain.  Check that it belongs to the next
                    # domain and ignore it otherwise as an edge corresponding
                    # to this one has been added already or will be added.
                    head = unpack(phead, gbp)
                    fd = sym.which(head)[0]
                    if fd != 1:
                        tail = unpack(ptail, gbp)
                        msg = 'Further-than-nearest-neighbor slices ' \
                              'are connected by hopping\n{0}.'
                        raise ValueError(msg.format((tail, head)))
                    continue
                if head_id >= slice_size:
                    # Head belongs to previous domain.  The edge added here
                    # correspond to one left out just above.
                    g.add_edge(head_id, tail_id)
                g.add_edge(tail_id, head_id)
        del id_by_psite
        g = g.compressed()

        #### Extract hoppings.
        hoppings = []
        for tail_id, head_id in g:
            ptail = psites[tail_id]
            phead = psites[head_id]
            if tail_id >= slice_size:
                # The tail belongs to the previous domain.  Find the
                # corresponding hopping with the tail in the fund. domain.
                t, h = sym.to_fd(unpack(ptail, gbp), unpack(phead, gbp))
                ptail = t.packed()
                phead = h.packed()
            hoppings.append(ham.getitem_edge((ptail, phead)))

        #### Assemble and return result.
        result = InfiniteSystem()
        result.slice_size = slice_size
        result.psites_idxs = np.cumsum([0] + [len(psite) for psite in psites])
        result.psites = "".join(psites)
        result.group_by_pgid = self._group_by_pgid
        result.graph = g
        result.hoppings = hoppings
        result.onsite_hamiltonians = onsite_hamiltonians
        result.symmetry = self.symmetry
        return result


################ Finalized systems


class FiniteSystem(system.FiniteSystem):
    """
    Finalized `Builder` with leads.

    Usable as input for the solvers in `kwant.solvers`.
    """
    def hamiltonian(self, i, j):
        if i == j:
            value = self.onsite_hamiltonians[i]
            if hasattr(value, '__call__'):
                value = value(self.symmetry.to_fd(self.site(i)))
            return value
        else:
            edge_id = self.graph.first_edge_id(i, j)
            value = self.hoppings[edge_id]
            conj = value is other
            if conj:
                i, j = j, i
                edge_id = self.graph.first_edge_id(i, j)
                value = self.hoppings[edge_id]
            if hasattr(value, '__call__'):
                site_i = self.site(i)
                site_j = self.site(j)
                value = value(*self.symmetry.to_fd(site_i, site_j))
            if conj:
                value = herm_conj(value)
            return value

    def site(self, i):
        a, b = self.psites_idxs[i : i + 2]
        return unpack(self.psites[a : b], self.group_by_pgid)

    def pos(self, i):
        return self.site(i).pos


class InfiniteSystem(system.InfiniteSystem):
    """Finalized infinite system, extracted from a `Builder`."""
    def hamiltonian(self, i, j):
        if i == j:
            if i >= self.slice_size:
                i -= self.slice_size
            value = self.onsite_hamiltonians[i]
            if hasattr(value, '__call__'):
                value = value(self.symmetry.to_fd(self.site(i)))
            return value
        else:
            edge_id = self.graph.first_edge_id(i, j)
            value = self.hoppings[edge_id]
            conj = value is other
            if conj:
                i, j = j, i
                edge_id = self.graph.first_edge_id(i, j)
                value = self.hoppings[edge_id]
            if hasattr(value, '__call__'):
                site_i = self.site(i)
                site_j = self.site(j)
                value = value(*self.symmetry.to_fd(site_i, site_j))
            if conj:
                value = herm_conj(value)
            return value

    def site(self, i):
        a, b = self.psites_idxs[i : i + 2]
        return unpack(self.psites[a : b], self.group_by_pgid)

    def pos(self, i):
        return self.site(i).pos
