# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import abc
import warnings
import operator
import collections
import copy
from functools import total_ordering, wraps, update_wrapper
from itertools import islice, chain
import inspect
import tinyarray as ta
import numpy as np
from scipy import sparse
from . import system, graph, KwantDeprecationWarning, UserCodeError
from .linalg import lll
from .operator import Density
from .physics import DiscreteSymmetry
from ._common import (ensure_isinstance, get_parameters, reraise_warnings,
                      interleave, deprecate_args)


__all__ = ['Builder', 'Site', 'SiteFamily', 'SimpleSiteFamily', 'Symmetry',
           'HoppingKind', 'Lead', 'BuilderLead', 'SelfEnergyLead', 'ModesLead']


################ Sites and site families

class Site(tuple):
    """A site, member of a `SiteFamily`.

    Sites are the vertices of the graph which describes the tight binding
    system in a `Builder`.

    A site is uniquely identified by its family and its tag.

    Parameters
    ----------
    family : an instance of `SiteFamily`
        The 'type' of the site.
    tag : a hashable python object
        The unique identifier of the site within the site family, typically a
        vector of integers.

    Raises
    ------
    ValueError
        If `tag` is not a proper tag for `family`.

    Notes
    -----
    For convenience, ``family(*tag)`` can be used instead of ``Site(family,
    tag)`` to create a site.

    The parameters of the constructor (see above) are stored as instance
    variables under the same names.  Given a site ``site``, common things to
    query are thus ``site.family``, ``site.tag``, and ``site.pos``.
    """
    __slots__ = ()

    family = property(operator.itemgetter(0),
                      doc="The site family to which the site belongs.")
    tag = property(operator.itemgetter(1), doc="The tag of the site.")


    def __new__(cls, family, tag, _i_know_what_i_do=False):
        if _i_know_what_i_do:
            return tuple.__new__(cls, (family, tag))
        try:
            tag = family.normalize_tag(tag)
        except (TypeError, ValueError) as e:
            msg = 'Tag {0} is not allowed for site family {1}: {2}'
            raise type(e)(msg.format(repr(tag), repr(family), e.args[0]))
        return tuple.__new__(cls, (family, tag))

    def __repr__(self):
        return 'Site({0}, {1})'.format(repr(self.family), repr(self.tag))

    def __str__(self):
        sf = self.family
        return '<Site {0} of {1}>'.format(self.tag, sf.name if sf.name else sf)

    def __getnewargs__(self):
        return (self.family, self.tag, True)

    @property
    def pos(self):
        """Real space position of the site.

        This relies on ``family`` having a ``pos`` method (see `SiteFamily`).
        """
        return self.family.pos(self.tag)


@total_ordering
class SiteFamily(metaclass=abc.ABCMeta):
    """Abstract base class for site families.

    Site families are the 'type' of `Site` objects.  Within a family, individual
    sites are uniquely identified by tags.  Valid tags must be hashable Python
    objects, further details are up to the family.

    Site families must be immutable and fully defined by their initial
    arguments.  They must inherit from this abstract base class and call its
    __init__ function providing it with two arguments: a canonical
    representation and a name.  The canonical representation will be returned as
    the objects representation and must uniquely identify the site family
    instance.  The name is a string used to distinguish otherwise identical site
    families.  It may be empty. ``norbs`` defines the number of orbitals
    on sites associated with this site family; it may be `None`, in which case
    the number of orbitals is not specified.


    All site families must define the method `normalize_tag` which brings a tag
    to the standard format for this site family.

    Site families that are intended for use with plotting should also provide a
    method `pos(tag)`, which returns a vector with real-space coordinates of the
    site belonging to this family with a given tag.

    If the ``norbs`` of a site family are provided, and sites of this family
    are used to populate a `~kwant.builder.Builder`, then the associated
    Hamiltonian values must have the correct shape. That is, if a site family
    has ``norbs = 2``, then any on-site terms for sites belonging to this
    family should be 2x2 matrices. Similarly, any hoppings to/from sites
    belonging to this family must have a matrix structure where there are two
    rows/columns. This condition applies equally to Hamiltonian values that
    are given by functions. If this condition is not satisfied, an error will
    be raised.
    """

    def __init__(self, canonical_repr, name, norbs):
        self.canonical_repr = canonical_repr
        self.hash = hash(canonical_repr)
        self.name = name
        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError('The norbs parameter must be an integer > 0.')
            norbs = int(norbs)
        self.norbs = norbs

    def __repr__(self):
        return self.canonical_repr

    def __str__(self):
        if self.name:
            msg = '<{0} site family {1}{2}>'
        else:
            msg = '<unnamed {0} site family{2}>'
        orbs = ' with {0} orbitals'.format(self.norbs) if self.norbs else ''
        return msg.format(self.__class__.__name__, self.name, orbs)

    def __hash__(self):
        return self.hash

    def __eq__(self, other):
        try:
            return self.canonical_repr == other.canonical_repr
        except AttributeError:
            return False

    def __ne__(self, other):
        try:
            return self.canonical_repr != other.canonical_repr
        except AttributeError:
            return True

    def __lt__(self, other):
        # If this raises an AttributeError, we were trying
        # to compare it to something non-comparable anyway.
        return self.canonical_repr < other.canonical_repr

    @abc.abstractmethod
    def normalize_tag(self, tag):
        """Return a normalized version of the tag.

        Raises TypeError or ValueError if the tag is not acceptable.
        """
        pass

    def __call__(self, *tag):
        """
        A convenience function.

        This function allows to write fam(1, 2) instead of Site(fam, (1, 2)).
        """
        # Catch a likely and difficult to find mistake.
        if tag and isinstance(tag[0], tuple):
            raise ValueError('Use site_family(1, 2) instead of '
                             'site_family((1, 2))!')
        return Site(self, tag)


class SimpleSiteFamily(SiteFamily):
    """A site family used as an example and for testing.

    A family of sites tagged by any python objects where object satisfied
    condition ``object == eval(repr(object))``.

    It exists to provide a basic site family that can be used for testing the
    builder module without other dependencies.  It can be also used to tag
    sites with non-numeric objects like strings should this every be useful.

    Due to its low storage efficiency for numbers it is not recommended to use
    `SimpleSiteFamily` when `kwant.lattice.Monatomic` would also work.
    """

    def __init__(self, name=None, norbs=None):
        canonical_repr = '{0}({1}, {2})'.format(self.__class__, repr(name),
                                                repr(norbs))
        super().__init__(canonical_repr, name, norbs)

    def normalize_tag(self, tag):
        tag = tuple(tag)
        try:
            if eval(repr(tag)) != tag:
                raise RuntimeError()
        except:
            raise TypeError('It must be possible to recreate the tag from '
                            'its representation.')
        return tag


def validate_hopping(hopping):
    """Verify that the argument is a valid hopping."""

    # This check is essential to maintaining the requirement that hoppings must
    # be tuples.  Without it, list "hoppings" would work in some cases
    # (e.g. with Builder.__contains__).
    if not isinstance(hopping, tuple):
        raise TypeError("{0} object is not a valid key.".format(
            type(hopping).__name__))

    # The following check is not strictly necessary (there would be an error
    # anyway), but the error message would be confusing.
    try:
        a, b = hopping
    except:
        raise IndexError("Only length 2 tuples (=hoppings) are valid keys.")

    # This check is essential for Builder.__contains__ - without it a builder
    # would simply "not contain" such invalid hoppings.  In other cases it
    # provides a nicer error message.
    for site in hopping:
        if not isinstance(site, Site):
            raise TypeError("Hopping elements must be Site objects, not {0}."
                            .format(type(site).__name__))

    # This again is an essential check.  Without it, builders would accept loop
    # hoppings.
    if a == b:
        raise ValueError("A hopping connects the following site to itself:\n"
                         "{0}".format(a))



################ Symmetries

class Symmetry(metaclass=abc.ABCMeta):
    """Abstract base class for spatial symmetries.

    Many physical systems possess a discrete spatial symmetry, which results in
    special properties of these systems.  This class is the standard way to
    describe discrete spatial symmetries in Kwant.  An instance of this class
    can be passed to a `Builder` instance at its creation.  The most important
    kind of symmetry is translational symmetry, used to define scattering
    leads.

    Each symmetry has a fundamental domain -- a set of sites and hoppings,
    generating all the possible sites and hoppings upon action of symmetry
    group elements.  A class derived from `Symmetry` has to implement mapping
    of any site or hopping into the fundamental domain, applying a symmetry
    group element to a site or a hopping, and a method `which` to determine the
    group element bringing some site from the fundamental domain to the
    requested one.  Additionally, it has to have a property `num_directions`
    returning the number of independent symmetry group generators (number of
    elementary periods for translational symmetry).

    A ``ValueError`` must be raised by the symmetry class whenever a symmetry
    is used together with sites whose site family is not compatible with it.  A
    typical example of this is when the vector defining a translational
    symmetry is not a lattice vector.

    The type of the domain objects as handled by the methods of this class is
    not specified.  The only requirement is that it must support the unary
    minus operation.  The reference implementation of `to_fd()` is hence
    `self.act(-self.which(a), a, b)`.
    """

    @abc.abstractproperty
    def num_directions(self):
        """Number of elementary periods of the symmetry."""
        pass

    @abc.abstractmethod
    def which(self, site):
        """Calculate the domain of the site.

        Return the group element whose action on a certain site from the
        fundamental domain will result in the given ``site``.
        """
        pass

    @abc.abstractmethod
    def act(self, element, a, b=None):
        """Act with a symmetry group element on a site or hopping."""
        pass

    def to_fd(self, a, b=None):
        """Map a site or hopping to the fundamental domain.

        If ``b`` is None, return a site equivalent to ``a`` within the
        fundamental domain.  Otherwise, return a hopping equivalent to ``(a,
        b)`` but where the first element belongs to the fundamental domain.

        Equivalent to `self.act(-self.which(a), a, b)`.
        """
        return self.act(-self.which(a), a, b)

    def in_fd(self, site):
        """Tell whether ``site`` lies within the fundamental domain."""
        for d in self.which(site):
            if d != 0:
                return False
        return True

    @abc.abstractmethod
    def subgroup(self, *generators):
        """Return the subgroup generated by a sequence of group elements."""
        pass

    @abc.abstractmethod
    def has_subgroup(self, other):
        """Test whether `self` has the subgroup `other`...

        or, in other words, whether `other` is a subgroup of `self`.  The
        reason why this is the abstract method (and not `is_subgroup`) is that
        in general it's not possible for a subgroup to know its supergroups.

        """
        pass


class NoSymmetry(Symmetry):
    """A symmetry with a trivial symmetry group."""

    def __eq__(self, other):
        return isinstance(other, NoSymmetry)

    def __ne__(self, other):
        return not self.__eq__(other)

    def __repr__(self):
        return 'NoSymmetry()'

    @property
    def num_directions(self):
        return 0

    periods = ()

    _empty_array = ta.array((), int)

    def which(self, site):
        return self._empty_array

    def act(self, element, a, b=None):
        if element:
            raise ValueError('`element` must be empty for NoSymmetry.')
        return a if b is None else (a, b)

    def to_fd(self, a, b=None):
        return a if b is None else (a, b)

    def in_fd(self, site):
        return True

    def subgroup(self, *generators):
        if any(generators):
            raise ValueError('Generators must be empty for NoSymmetry.')
        return NoSymmetry(generators)

    def has_subgroup(self, other):
        return isinstance(other, NoSymmetry)



################ Hopping kinds

class HoppingKind(tuple):
    """A pattern for matching hoppings.

    An alias exists for this common name: ``kwant.HoppingKind``.

    A hopping ``(a, b)`` matches precisely when the site family of ``a`` equals
    `family_a` and that of ``b`` equals `family_b` and ``(a.tag - b.tag)`` is
    equal to `delta`.  In other words, the matching hoppings have the form:
    ``(family_a(x + delta), family_b(x))``

    Parameters
    ----------
    delta : Sequence of integers
        The sequence is interpreted as a vector with integer elements.
    family_a : `~kwant.builder.SiteFamily`
    family_b : `~kwant.builder.SiteFamily` or ``None`` (default)
        The default value means: use the same family as `family_a`.

    Notes
    -----
    A ``HoppingKind`` is a callable object: When called with a
    `~kwant.builder.Builder` as sole argument, an instance of this class will
    return an iterator over all possible matching hoppings whose sites are
    already present in the system.  The hoppings do *not* have to be already
    present in the system.  For example::

        kind = kwant.builder.HoppingKind((1, 0), lat)
        syst[kind(syst)] = 1

    Because a `~kwant.builder.Builder` can be indexed with functions or
    iterables of functions, ``HoppingKind`` instances (or any non-tuple
    iterables of them, e.g. a list) can be used directly as "wildcards" when
    setting or deleting hoppings::

        kinds = [kwant.builder.HoppingKind(v, lat) for v in [(1, 0), (0, 1)]]
        syst[kinds] = 1
    """
    __slots__ = ()

    delta = property(operator.itemgetter(0),
                     doc="The difference between the tags of the hopping's sites")
    family_a = property(operator.itemgetter(1),
                        doc="The family of the first site in the hopping")
    family_b = property(operator.itemgetter(2),
                        doc="The family of the second site in the hopping")

    def __new__(cls, delta, family_a, family_b=None):
        delta = ta.array(delta, int)
        ensure_isinstance(family_a, SiteFamily)
        if family_b is None:
            family_b = family_a
        else:
            ensure_isinstance(family_b, SiteFamily)
            family_b = family_b

        try:
            Site(family_b, family_a.normalize_tag(delta) - delta)
        except Exception as e:
            same_fams = family_b is family_a
            msg = (str(family_a),
                   'and {} are'.format(family_b) if not same_fams else ' is',
                   'not compatible with delta={}'.format(delta),
                  )
            raise ValueError(' '.join(msg)) from e

        return tuple.__new__(cls, (delta, family_a, family_b))

    def __call__(self, builder):
        delta = self.delta
        family_a = self.family_a
        family_b = self.family_b
        H = builder.H
        symtofd = builder.symmetry.to_fd

        for a in H:
            if a.family != family_a:
                continue
            b = Site(family_b, a.tag - delta, True)
            if symtofd(b) in H:
                yield a, b

    def __repr__(self):
        return '{0}({1}, {2}{3})'.format(
            self.__class__.__name__, repr(tuple(self.delta)),
            repr(self.family_a),
            ', ' + repr(self.family_b) if self.family_a != self.family_b else '')

    def __str__(self):
        return '{0}({1}, {2}{3})'.format(
            self.__class__.__name__, tuple(self.delta),
            self.family_a,
            ', ' + str(self.family_b) if self.family_a != self.family_b else '')



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


class HermConjOfFunc:
    """Proxy returning the hermitian conjugate of the original result."""
    __slots__ = ('function')

    def __init__(self, function):
        self.function = function

    def __call__(self, i, j, *args, **kwargs):
        return herm_conj(self.function(j, i, *args, **kwargs))

    @property
    def __signature__(self):
        return inspect.signature(self.function)


################ Leads

class Lead(metaclass=abc.ABCMeta):
    """Abstract base class for leads that can be attached to a `Builder`.

    To attach a lead to a builder, append it to the builder's `~Builder.leads`
    instance variable.  See the documentation of `kwant.builder` for the
    concrete classes of leads derived from this one.

    Attributes
    ----------
    interface : sequence of sites

    """

    @abc.abstractmethod
    def finalized(self):
        """Return a finalized version of the lead.

        Returns
        -------
        finalized_lead

        Notes
        -----
        The finalized lead must be an object that can be used as a lead
        in a `kwant.system.FiniteSystem`, i.e. an instance of
        `kwant.system.InfiniteSystem`.  Typically it will be a finalized
        builder: `kwant.builder.InfiniteSystem`.

        The order of sites for the finalized lead must be the one specified in
        `interface`.
        """
        pass


class BuilderLead(Lead):
    """A lead made from a `Builder` with a spatial symmetry.

    Parameters
    ----------
    builder : `Builder`
        The tight-binding system of a lead. It has to possess appropriate
        symmetry, and it may not contain hoppings between further than
        neighboring images of the fundamental domain.
    interface : sequence of `Site` instances
        Sequence of sites in the scattering region to which the lead is
        attached.

    Attributes
    ----------
    builder : `Builder`
        The tight-binding system of a lead.
    interface : list of `Site` instances
        A sorted list of interface sites.
    padding : list of `Site` instances
        A sorted list of sites that originate from the lead, have the same
        onsite Hamiltonian, and are connected by the same hoppings as the lead
        sites.

    Notes
    -----
    The hopping from the scattering region to the lead is assumed to be equal to
    the hopping from a lead unit cell to the next one in the direction of the
    symmetry vector (i.e. the lead is 'leaving' the system and starts with a
    hopping).

    Every system has an attribute `leads`, which stores a list of
    `BuilderLead` objects with all the information about the leads that are
    attached.
    """
    def __init__(self, builder, interface, padding=None):
        self.builder = builder
        self.interface = sorted(interface)
        self.padding = sorted(padding) if padding is not None else []

    def finalized(self):
        """Return a `kwant.builder.InfiniteSystem` corresponding to the
        compressed lead.

        The order of interface sites is kept during finalization.
        """
        return InfiniteSystem(self.builder, self.interface)


def _ensure_signature(func):
    """
    Ensure that a modes/selfenergy function has a keyword-only parameter
    ``params``, or takes ``**kwargs`` by potentially wrapping it.
    """
    parameters = inspect.signature(func).parameters
    has_params = bool(parameters.get('params'))
    has_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD
                     for p in parameters.values())
    if has_params or has_kwargs:
        return func

    # function conforming to old API: needs wrapping
    @deprecate_args
    def wrapper(energy, args=(), *, params=None):
        return func(energy, args)

    return wrapper


class SelfEnergyLead(Lead):
    """A general lead defined by its self energy.

    Parameters
    ----------
    selfenergy_func : function
        Has the same signature as `selfenergy` (without the ``self``
        parameter) and returns the self energy matrix for the interface sites.
    interface : sequence of `Site` instances
    parameters : sequence of strings
        The parameters on which the lead depends.
    """
    def __init__(self, selfenergy_func, interface, parameters):
        self.interface = tuple(interface)
        # we changed the API of 'selfenergy_func' to have a keyword-only
        # parameter 'params', but we still need to support the old API
        # XXX: remove this when releasing Kwant 2.0
        self.selfenergy_func = _ensure_signature(selfenergy_func)
        self.parameters = frozenset(parameters)

    def finalized(self):
        """Trivial finalization: the object is returned itself."""
        return self

    @deprecate_args
    def selfenergy(self, energy, args=(), *, params=None):
        return self.selfenergy_func(energy, args, params=params)


class ModesLead(Lead):
    """A general lead defined by its modes wave functions.

    Parameters
    ----------
    modes_func : function
        Has the same signature as `modes` (without the ``self`` parameter)
        and returns the modes of the lead as a tuple of
        `~kwant.physics.PropagatingModes` and `~kwant.physics.StabilizedModes`.
    interface :
        sequence of `Site` instances
    parameters : sequence of strings
        The parameters on which the lead depends.
    """
    def __init__(self, modes_func, interface, parameters):
        self.interface = tuple(interface)
        # we changed the API of 'selfenergy_func' to have a keyword-only
        # parameter 'params', but we still need to support the old API
        # XXX: remove this when releasing Kwant 2.0
        self.modes_func = _ensure_signature(modes_func)
        self.parameters = frozenset(parameters)

    def finalized(self):
        """Trivial finalization: the object is returned itself."""
        return self

    @deprecate_args
    def modes(self, energy, args=(), *, params=None):
        return self.modes_func(energy, args, params=params)

    @deprecate_args
    def selfenergy(self, energy, args=(), *, params=None):
        stabilized = self.modes(energy, args, params=params)[1]
        return stabilized.selfenergy()



################ Builder class

# A marker, meaning for hopping (i, j): this value is given by the Hermitian
# conjugate the value of the hopping (j, i).  Used by Builder and System.
class Other:
    pass


def edges(seq):
    result = interleave(seq)
    next(result)  # Skip the special loop edge.
    return result


def _site_ranges(sites):
    """Return a sequence of ranges for ``sites``.

    Here, "ranges" are defined as sequences of sites that have the same site
    family. Because site families now have a fixed number of orbitals,
    this coincides with the definition given in `~kwant.system.System`.
    """
    # we shall start a new range of different `SiteFamily`s separately,
    # even if they happen to contain the same number of orbitals.
    total_norbs = 0
    current_fam = None
    site_ranges = []
    for idx, fam in enumerate(s.family for s in sites):
        if not fam.norbs:
            # can't provide site_ranges if norbs not given
            return None
        if fam != current_fam:  # start a new run
            current_fam = fam
            current_norbs = fam.norbs
            site_ranges.append((idx, current_norbs, total_norbs))
        total_norbs += current_norbs
    # add sentinel to the end
    site_ranges.append((len(sites), 0, total_norbs))
    return site_ranges


class _Substituted:
    """Proxy that renames function parameters."""

    def __init__(self, func, params):
        self.func = func
        self.params = params
        update_wrapper(self, func)

    def __eq__(self, other):
        if not isinstance(other, _Substituted):
            return False
        return (self.func == other.func and self.params == other.params)

    def __hash__(self):
        return hash((self.func, self.params))

    @property
    def __signature__(self):
        return inspect.Signature(
            [inspect.Parameter(name, inspect.Parameter.POSITIONAL_ONLY)
             for name in self.params])

    def __call__(self, *args):
        return self.func(*args)


def _substitute_params(func, subs):
    """Substitute 'params' from 'subs' into 'func'."""
    assert callable(func)

    if isinstance(func, _Substituted):
        old_params = func.params
        old_func = func.func
    else:
        old_params = get_parameters(func)
        old_func = func

    params = tuple(subs.get(p, p) for p in old_params)

    duplicates = [p for p, count in collections.Counter(params).items()
                  if count > 1]
    if duplicates:
        msg = ('Cannot rename parameters ',
               ','.join('"{}"'.format(d) for d in duplicates),
               ': parameters with the same name exist')
        raise ValueError(''.join(msg))

    if params == old_params:
        return func
    else:
        return _Substituted(old_func, params)


class Builder:
    """A tight binding system defined on a graph.

    An alias exists for this common name: ``kwant.Builder``.

    This is one of the central types in Kwant.  It is used to construct tight
    binding systems in a flexible way.

    The nodes of the graph are `Site` instances.  The edges, i.e. the hoppings,
    are pairs (2-tuples) of sites.  Each node and each edge has a value
    associated with it.  The values associated with nodes are interpreted as
    on-site Hamiltonians, the ones associated with edges as hopping integrals.

    To make the graph accessible in a way that is natural within the Python
    language it is exposed as a *mapping* (much like a built-in Python
    dictionary).  Keys are sites or hoppings.  Values are 2d arrays
    (e.g. NumPy or Tinyarray) or numbers (interpreted as 1 by 1 matrices).

    Parameters
    ----------
    symmetry : `Symmetry` or `None`
        The spatial symmetry of the system.
    conservation_law : 2D array, dictionary, function, or `None`
        An onsite operator with integer eigenvalues that commutes with the
        Hamiltonian.  The ascending order of eigenvalues corresponds to the
        selected ordering of the Hamiltonian subblocks.  If a dict is
        given, it maps from site families to such matrices. If a function is
        given it must take the same arguments as the onsite Hamiltonian
        functions of the system and return the onsite matrix.
    time_reversal : scalar, 2D array, dictionary, function, or `None`
        The unitary part of the onsite time-reversal symmetry operator.
        Same format as that of `conservation_law`.
    particle_hole : scalar, 2D array, dictionary, function, or `None`
        The unitary part of the onsite particle-hole symmetry operator.
        Same format as that of `conservation_law`.
    chiral : 2D array, dictionary, function or `None`
        The unitary part of the onsite chiral symmetry operator.
        Same format as that of `conservation_law`.

    Notes
    -----

    Values can be also functions that receive the site or the hopping (passed
    to the function as two sites) and possibly additional arguments and are
    expected to return a valid value.  This allows to define systems quickly,
    to modify them without reconstructing, and to save memory for many-orbital
    models.

    In addition to simple keys (single sites and hoppings) more powerful keys
    are possible as well that allow to manipulate multiple sites/hoppings in a
    single operation.  Such keys are internally expanded into a sequence of
    simple keys by using the method `Builder.expand`.  For example,
    ``syst[general_key] = value`` is equivalent to ::

        for simple_key in syst.expand(general_key):
            syst[simple_key] = value

    Builder instances automatically ensure that every hopping is Hermitian, so
    that if ``builder[a, b]`` has been set, there is no need to set
    ``builder[b, a]``.

    Builder instances can be made to automatically respect a `Symmetry` that is
    passed to them during creation.  The behavior of builders with a symmetry
    is slightly more sophisticated: all keys are mapped to the fundamental
    domain of the symmetry before storing them.  This may produce confusing
    results when neighbors of a site are queried.

    The method `attach_lead` *works* only if the sites affected by them have
    tags which are sequences of integers.  It *makes sense* only when these
    sites live on a regular lattice, like the ones provided by `kwant.lattice`.

    Attaching a lead manually (without the use of `~Builder.attach_lead`)
    amounts to creating a `Lead` object and appending it to the list of leads
    accessbile as the `~Builder.leads` attribute.

    `conservation_law`, `time_reversal`, `particle_hole`, and `chiral`
    affect the basis in which scattering modes derived from the builder
    are expressed - see `~kwant.physics.DiscreteSymmetry` for details.

    .. warning::

        If functions are used to set values in a builder with a symmetry, then
        they must satisfy the same symmetry.  There is (currently) no check and
        wrong results will be the consequence of a misbehaving function.

    Attributes
    ----------
    leads : list of `Lead` instances
        The leads that are attached to the system.

    Examples
    --------
    Define a site.

    >>> builder[site] = value

    Print the value of a site.

    >>> print(builder[site])

    Define a hopping.

    >>> builder[site1, site2] = value

    Delete a site.

    >>> del builder[site3]

    Detach the last lead.  (This does not remove the sites that were added to
    the scattering region by `~Builder.attach_lead`.)

    >>> del builder.leads[-1]

    """

    def __init__(self, symmetry=None, *, conservation_law=None, time_reversal=None,
                 particle_hole=None, chiral=None):
        if symmetry is None:
            symmetry = NoSymmetry()
        else:
            ensure_isinstance(symmetry, Symmetry)
        self.symmetry = symmetry
        self.conservation_law = conservation_law
        self.time_reversal = time_reversal
        self.particle_hole = particle_hole
        self.chiral = chiral
        self.leads = []
        self.H = {}

    #### Note on H ####
    #
    # This dictionary stores a directed graph optimized for efficient querying
    # and modification.  The nodes are instances of `Site`.
    #
    # Each edge, specified by a ``(tail, head)`` pair of nodes, holds an object
    # as a value.  Likewise, each tail which occurs in the graph also holds a
    # value.  (Nodes which only occur as heads are not required to have
    # values.) Every tail node has to be in the fundamental domain of the
    # builder's symmetry.
    #
    # For a given `tail` site, H[tail] is a list alternately storing
    # heads and values.  (The heads occupy even locations followed by the
    # values at odd locations.)  Each pair of entries thus describes a single
    # directed edge of the graph.
    #
    # The first pair of entries in each list is special: it always
    # corresponds to a loop edge.  (The head is equal to the tail.)  This
    # special edge has two purposes: It is used to store the value
    # associated with the tail node itself, and it is necessary for the
    # method getkey_tail which helps to conserve memory by storing equal
    # node label only once.

    def _get_edge(self, tail, head):
        for h, value in edges(self.H[tail]):
            if h == head:
                return value

        # (tail, head) is not present in the system, but tail is.
        if head in self.H:
            raise KeyError((tail, head))
        else:
            # If already head is missing, we only report this.  This way the
            # behavior is symmetric with regard to tail and head.
            raise KeyError(head)

    def _set_edge(self, tail, head, value):
        hvhv = self.H[tail]
        heads = hvhv[2::2]
        if head in heads:
            i = 2 + 2 * heads.index(head)
            hvhv[i] = head
            hvhv[i + 1] = value
        else:
            hvhv.append(head)
            hvhv.append(value)

    def _del_edge(self, tail, head):
        hvhv = self.H[tail]
        heads = hvhv[2::2]

        try:
            i = 2 + 2 * heads.index(head)
        except ValueError:
            # (tail, head) is not present in the system, but tail is.
            if head in self.H:
                raise KeyError((tail, head))
            else:
                # If already head is missing, we only report this.  This way the
                # behavior is symmetric with regard to tail and head.
                raise KeyError(head)

        del hvhv[i : i + 2]

    def _out_neighbors(self, tail):
        hvhv = self.H[tail]
        return islice(hvhv, 2, None, 2)

    def _out_degree(self, tail):
        hvhv = self.H[tail]
        return len(hvhv) // 2 - 1

    def __copy__(self):
        """Shallow copy"""
        result = object.__new__(Builder)
        result.symmetry = self.symmetry
        result.conservation_law = self.conservation_law
        result.time_reversal = self.time_reversal
        result.particle_hole = self.particle_hole
        result.chiral = self.chiral
        result.leads = self.leads
        result.H = self.H
        return result

    # TODO: write a test for this method.
    def reversed(self):
        """Return a shallow copy of the builder with the symmetry reversed.

        This method can be used to attach the same infinite system as lead from
        two opposite sides.  It requires a builder to which an infinite
        symmetry is associated.
        """
        if self.leads:
            raise ValueError('System to be reversed may not have leads.')
        result = copy.copy(self)
        # if we don't assign a new list we will inadvertantly add leads to
        # the reversed system if we add leads to *this* system
        # (because we only shallow copy)
        result.leads = []
        result.symmetry = self.symmetry.reversed()
        return result

    def __bool__(self):
        return bool(self.H)

    def expand(self, key):
        """
        Expand a general key into an iterator over simple keys.

        Parameters
        ----------
        key : builder key (see notes)
            The key to be expanded

        Notes
        -----
        Keys are (recursively):
            * Simple keys: sites or 2-tuples of sites (=hoppings).
            * Any (non-tuple) iterable of keys, e.g. a list or a generator
              expression.
            * Any function that returns a key when passed a builder as sole
              argument, e.g. a `HoppingKind` instance or the function returned
              by `~kwant.lattice.Polyatomic.shape`.

        This method is internally used to expand the keys when getting or
        deleting items of a builder (i.e. ``syst[key] = value`` or ``del
        syst[key]``).

        """
        itr = iter((key,))
        iter_stack = [None]
        while iter_stack:
            for key in itr:
                while callable(key):
                    key = key(self)
                if isinstance(key, tuple):
                    # Site instances are also tuples.
                    yield key
                else:
                    iter_stack.append(itr)
                    try:
                        itr = iter(key)
                    except TypeError:
                        raise TypeError("{0} object is not a valid key."
                                        .format(type(key).__name__))
                    break
            else:
                itr = iter_stack.pop()

    def __getitem__(self, key):
        """Get the value of a single site or hopping."""
        if isinstance(key, Site):
            site = self.symmetry.to_fd(key)
            return self.H[site][1]

        sym = self.symmetry
        validate_hopping(key)
        a, b = sym.to_fd(*key)
        value = self._get_edge(a, b)
        if value is Other:
            if not sym.in_fd(b):
                b, a = sym.to_fd(b, a)
                assert not sym.in_fd(a)
            value = self._get_edge(b, a)
            if callable(value):
                assert not isinstance(value, HermConjOfFunc)
                value = HermConjOfFunc(value)
            else:
                value = herm_conj(value)
        return value

    def __contains__(self, key):
        """Tell whether the system contains a site or hopping."""
        if isinstance(key, Site):
            key = self.symmetry.to_fd(key)
            return key in self.H

        validate_hopping(key)
        a, b = self.symmetry.to_fd(*key)
        hvhv = self.H.get(a, ())
        return b in islice(hvhv, 2, None, 2)

    def _set_site(self, site, value):
        """Set a single site."""
        if not isinstance(site, Site):
            raise TypeError('Expecting a site, got {0} instead.'.format(type(site).__name__))
        site = self.symmetry.to_fd(site)
        hvhv = self.H.setdefault(site, [])
        if hvhv:
            hvhv[1] = value
        else:
            hvhv[:] = [site, value]

    def _set_hopping(self, hopping, value):
        """Set a single hopping."""
        sym = self.symmetry
        validate_hopping(hopping)
        a, b = sym.to_fd(*hopping)

        if sym.in_fd(b):
            # Make sure that we do not waste space by storing multiple instances
            # of identical sites.
            a2 = a = self.H[a][0]
            b2 = b = self.H[b][0]
        else:
            b2, a2 = sym.to_fd(b, a)
            assert not sym.in_fd(a2)
            if b2 not in self.H:
                raise KeyError(b)

        # It's important that we have verified that b2 belongs to the system.
        # Otherwise, we risk ending up with a half-added hopping.
        if isinstance(value, HermConjOfFunc):
            # Avoid nested HermConjOfFunc instances.
            self._set_edge(a, b, Other)            # May raise KeyError(a).
            self._set_edge(b2, a2, value.function) # Must succeed.
        else:
            self._set_edge(a, b, value)            # May raise KeyError(a).
            self._set_edge(b2, a2, Other)          # Must succeed.

    def __setitem__(self, key, value):
        """Set a single site/hopping or a bunch of them."""
        func = None
        for sh in self.expand(key):
            if func is None:
                func = (self._set_site if isinstance(sh, Site)
                        else self._set_hopping)
            func(sh, value)

    def _del_site(self, site):
        """Delete a single site and all associated hoppings."""
        if not isinstance(site, Site):
            raise TypeError('Expecting a site, got {0} instead.'.format(
                type(site).__name__))

        tfd = self.symmetry.to_fd
        site = tfd(site)

        out_neighbors = self._out_neighbors(site)

        for neighbor in out_neighbors:
            if neighbor in self.H:
                self._del_edge(neighbor, site)
            else:
                assert not self.symmetry.in_fd(neighbor)
                self._del_edge(*tfd(neighbor, site))

        del self.H[site]

    def _del_hopping(self, hopping):
        """Delete a single hopping."""
        sym = self.symmetry
        validate_hopping(hopping)
        a, b = sym.to_fd(*hopping)
        self._del_edge(a, b)

        if sym.in_fd(b):
            self._del_edge(b, a)
        else:
            b, a = sym.to_fd(b, a)
            assert not sym.in_fd(a)
            self._del_edge(b, a)

    def __delitem__(self, key):
        """Delete a single site/hopping or bunch of them."""
        func = None
        for sh in self.expand(key):
            if func is None:
                func = (self._del_site if isinstance(sh, Site)
                        else self._del_hopping)
            func(sh)

    def eradicate_dangling(self):
        """Keep deleting dangling sites until none are left.

        Sites are considered as dangling when less than two hoppings
        lead to them.
        """
        sites = list(site for site in self.H
                     if self._out_degree(site) < 2)
        for site in sites:
            if site not in self.H:
                continue
            while site:
                neighbors = tuple(self._out_neighbors(site))
                if neighbors:
                    assert len(neighbors) == 1
                    neighbor = neighbors[0]
                    self._del_edge(neighbor, site)
                    if self._out_degree(neighbor) > 1:
                        neighbor = False
                else:
                    neighbor = False
                del self.H[site]
                site = neighbor

    def __iter__(self):
        """Return an iterator over all sites and hoppings."""
        return chain(self.H, self.hoppings())

    def sites(self):
        """Return a read-only set over all sites.

        The sites that are returned belong to the fundamental domain of the
        `Builder` symmetry, and are not necessarily the ones that were set
        initially (but always the equivalent ones).
        """
        try:
            return self.H.keys()
        except AttributeError:
            return frozenset(self.H)

    def site_value_pairs(self):
        """Return an iterator over all (site, value) pairs."""
        for site, hvhv in self.H.items():
            yield site, hvhv[1]

    def hoppings(self):
        """Return an iterator over all Builder hoppings.

        The hoppings that are returned belong to the fundamental domain of the
        `Builder` symmetry, and are not necessarily the ones that were set
        initially (but always the equivalent ones).
        """
        for tail, hvhv in self.H.items():
            for head, value in edges(hvhv):
                if value is Other:
                    continue
                yield tail, head

    def hopping_value_pairs(self):
        """Return an iterator over all (hopping, value) pairs."""
        for tail, hvhv in self.H.items():
            for head, value in edges(hvhv):
                if value is Other:
                    continue
                yield (tail, head), value

    def dangling(self):
        """Return an iterator over all dangling sites.

        Sites are considered as dangling when less than two hoppings
        lead to them.
        """
        for site in self.H:
            if self._out_degree(site) < 2:
                yield site

    def degree(self, site):
        """Return the number of neighbors of a site."""
        if not isinstance(site, Site):
            raise TypeError('Expecting a site, got {0} instead.'.format(
                type(site).__name__))
        site = self.symmetry.to_fd(site)
        return self._out_degree(site)

    def neighbors(self, site):
        """Return an iterator over all neighbors of a site.

        Technical note: This method respects the symmetry of the builder,
        i.e. the returned sites are really connected to the given site (and not
        to its image in the fundamental domain).
        """
        if not isinstance(site, Site):
            raise TypeError('Expecting a site, got {0} instead.'.format(
                type(site).__name__))
        sym = self.symmetry
        if isinstance(sym, NoSymmetry):
            # Optimization for common case.
            yield from self._out_neighbors(site)
            return
        shift = sym.which(site)
        site = sym.act(-shift, site)
        for neighbor in self._out_neighbors(site):
            yield sym.act(shift, neighbor)

    def closest(self, pos):
        """Return the site that is closest to the given position.

        This function takes into account the symmetry of the builder.  It is
        assumed that the symmetry is a translational symmetry.

        This function executes in a time proportional to the number of sites,
        so it is not efficient for large builders.  It is especially slow for
        builders with a symmetry, but such systems often contain only a limited
        number of sites.

        """
        errmsg = ("Builder.closest() requires site families that provide "
                  "pos().\nThe following one does not:\n")
        sym = self.symmetry
        n = sym.num_directions

        if n:
            # Determine basis in real space from first site.  (The result from
            # any site would do.)
            I = ta.identity(n, int)
            site = next(iter(self.H))
            space_basis = [sym.act(element, site).pos - site.pos
                           for element in I]
            space_basis, transf = lll.lll(space_basis)
            transf = ta.array(transf.T, int)

        tag_basis_cache = {}
        dist = float('inf')
        result = None
        for site in self.H:
            try:
                site_pos = site.pos
            except AttributeError:
                raise AttributeError(errmsg + str(site.family))
            if n:
                fam = site.family
                tag_basis = tag_basis_cache.get(fam)
                if tag_basis is None:
                    zero_site = Site(fam, ta.zeros(len(site.tag), int))
                    tag_basis = [sym.act(element, zero_site).tag
                                 for element in I]
                    tag_basis = ta.dot(transf, tag_basis)
                    tag_basis_cache[fam] = tag_basis
                shift = lll.cvp(pos - site_pos, space_basis, 1)[0]
                site = Site(fam, ta.dot(shift, tag_basis) + site.tag)
                site_pos = site.pos
            d = site_pos - pos
            d = ta.dot(d, d)
            if d < dist:
                dist = d
                result = site
        return result

    def update(self, other):
        """Update builder from `other`.

        All sites and hoppings of `other`, together with their values, are
        written to `self`, overwriting already existing sites and hoppings.
        The leads of `other` are appended to the leads of the system being
        updated.

        This method requires that both builders share the same symmetry.
        """
        if (not self.symmetry.has_subgroup(other.symmetry)
            or not other.symmetry.has_subgroup(self.symmetry)):
            raise ValueError("Both builders involved in update() must have "
                             "equal symmetries.")
        for site, value in other.site_value_pairs():
            self[site] = value
        for hop, value in other.hopping_value_pairs():
            self[hop] = value
        self.leads.extend(other.leads)

    def __iadd__(self, other):
        warnings.warn("The += operator of builders is deprecated. Use "
                      "'Builder.update()' instead.", KwantDeprecationWarning,
                      stacklevel=2)
        self.update(other)
        return self

    def substituted(self, **subs):
        """Return a copy of this Builder with modified parameter names.
        """
        # Construct the a copy of the system with new value functions.
        if self.leads:
            raise ValueError("For simplicity, 'subsituted' is limited "
                             "to builders without leads. Use 'substituted' "
                             "before attaching leads to avoid this error.")

        # Get value *functions* only
        onsites = list(set(
            onsite for _, onsite in self.site_value_pairs()
            if callable(onsite)))
        hoppings = list(set(
            hop for _, hop in self.hopping_value_pairs()
            if callable(hop)))

        flatten = chain.from_iterable

        # Get parameter names to be substituted for each function,
        # without the 'site' parameter(s)
        onsite_params = [get_parameters(v)[1:] for v in onsites]
        hopping_params = [get_parameters(v)[2:] for v in hoppings]

        system_params = set(flatten(chain(onsite_params, hopping_params)))
        nonexistant_params = set(subs.keys()).difference(system_params)
        if nonexistant_params:
            msg = ('Parameters {} are not used by any onsite or hopping '
                   'value function in this system.'
                  ).format(nonexistant_params)
            warnings.warn(msg, RuntimeWarning, stacklevel=2)

        # Precompute map from old onsite/hopping value functions to ones
        # with substituted parameters.
        value_map = {value: _substitute_params(value, subs)
                     for value in chain(onsites, hoppings)}

        result = copy.copy(self)
        # if we don't assign a new list we will inadvertantly add leads to
        # the reversed system if we add leads to *this* system
        # (because we only shallow copy)
        result.leads = []
        # Copy the 'H' dictionary, mapping old values to new ones using
        # 'value_map'. If a value does not appear in the map then it means
        # that the old value should be used.
        result.H = {}
        for tail, hvhv in self.H.items():
            result.H[tail] = list(flatten(
                (head, value_map.get(value, value))
                for head, value in interleave(hvhv)))

        return result

    def fill(self, template, shape, start, *, max_sites=10**7):
        """Populate builder using another one as a template.

        Starting from one or multiple sites, traverse the graph of the template
        builder and copy sites and hoppings to the target builder.  The
        traversal stops at sites that are already present in the target and on
        sites that are not inside the provided shape.

        This function takes into account translational symmetry.  As such,
        typically the template will have a higher symmetry than the target.

        Newly added sites are connected by hoppings to sites that were already
        present.  This facilitates construction of a system by a series of
        calls to 'fill'.

        Parameters
        ----------
        template : `Builder` instance
            The builder used as the template. The symmetry of the target builder
            must be a subgroup of the symmetry of the template.
        shape : callable
            A boolean function of site returning whether the site should be
            added to the target builder or not. The shape must be compatible
            with the symmetry of the target builder.
        start : `Site` instance or iterable thereof or iterable of numbers
            The site(s) at which the the flood-fill starts.  If start is an
            iterable of numbers, the starting site will be
            ``template.closest(start)``.
        max_sites : positive number
            The maximal number of sites that may be added before
            ``RuntimeError`` is raised.  Used to prevent using up all memory.

        Returns
        -------
        added_sites : list of `Site` objects that were added to the system.

        Raises
        ------
        ValueError
            If the symmetry of the target isn't a subgroup of the template
            symmetry.
        RuntimeError
            If more than `max_sites` sites are to be added.  The target builder
            will be left in an unusable state.

        Notes
        -----
        This function uses a flood-fill algorithm.  If the template builder
        consists of disconnected parts, the fill will stop at their boundaries.

        """
        if not max_sites > 0:
            raise ValueError("max_sites must be positive.")

        to_fd = self.symmetry.to_fd
        H = self.H
        templ_sym = template.symmetry

        # Check that symmetries are commensurate.
        if not templ_sym.has_subgroup(self.symmetry):
            raise ValueError("Builder symmetry is not a subgroup of the "
                             "template symmetry")

        if isinstance(start, Site):
            start = [start]
        else:
            start = list(start)
            if start and not isinstance(start[0], Site):
                start = [template.closest(start)]

        if any(s not in template for s in start):
            warnings.warn("fill(): Some of the starting sites are "
                          "not in the template builder.",
                          RuntimeWarning, stacklevel=2)
        start = [s for s in start if s in template]
        if not start:
            return []

        try:
            # "Active" are sites (mapped to the target's FD) that have been
            # verified to lie inside the shape, have been added to the target
            # (with `None` as value), but yet without their hoppings.
            active = set()
            congested = True
            for s in start:
                s = to_fd(s)
                if s not in H:
                    congested = False
                    if shape(s):
                        active.add(s)
                        H.setdefault(s, [s, None])

            if not active:
                if congested:
                    warnings.warn("fill(): The target builder already contains "
                                  "all starting sites.", RuntimeWarning,
                                  stacklevel=2)
                else:
                    warnings.warn("fill(): None of the starting sites is in "
                                  "the desired shape",
                                  RuntimeWarning, stacklevel=2)
                return []

            done = []
            old_active = set()
            new_active = set()

            # Flood-fill on the graph.  We work site by site, writing all the
            # outgoing edges.
            while active:
                old_active.update(active)

                for tail in active:
                    done.append(tail)
                    if len(done) > max_sites:
                        raise RuntimeError("Maximal number of sites (max_sites "
                                           "parameter of fill()) added.")

                    # Make an iterator over head-value-pairs.
                    shift = templ_sym.which(tail)
                    templ_hvhv = template.H[templ_sym.act(-shift, tail)]
                    templ_hvhv = iter(templ_hvhv)
                    templ_hvhv = iter(zip(templ_hvhv, templ_hvhv))

                    hvhv = H[tail]
                    hvhv[1] = next(templ_hvhv)[1]
                    old_heads = hvhv[2::2]

                    # The remaining pairs are the heads and their associated
                    # values.
                    for head, value in templ_hvhv:
                        head = templ_sym.act(shift, head)
                        head_fd = to_fd(head)

                        if (head_fd not in old_active
                            and head_fd not in new_active):
                            # The 'head' site has not been filled yet.
                            if head_fd in H:
                                # The 'head' site exists.  (It doesn't matter
                                # whether it's in the shape or not.)  Fill the
                                # incoming edge as well to balance the hopping.
                                other_value = template._get_edge(
                                    *templ_sym.to_fd(head, tail))
                                self._set_edge(*to_fd(head, tail)
                                               + (other_value,))
                            else:
                                if not shape(head_fd):
                                    # There is no site at 'head' and it's
                                    # outside the shape.
                                    continue
                                new_active.add(head_fd)
                                H.setdefault(head_fd, [head_fd, None])

                        # Fill the outgoing edge.
                        if head in old_heads:
                            i = 2 + 2 * old_heads.index(head)
                            hvhv[i] = head
                            hvhv[i + 1] = value
                        else:
                            hvhv.extend((head, value))

                old_active, active, new_active = active, new_active, old_active
                new_active.clear()
        except Exception as e:
            # The graph has unbalanced edges: delete it.
            self.H = {}
            # Re-raise the exception with an additional message.
            msg = ("All sites of this builder have been deleted because an "
                   "exception\noccurred during the execution of fill(): "
                   "see above.")
            raise RuntimeError(msg) from e

        return done

    def attach_lead(self, lead_builder, origin=None, add_cells=0):
        """Attach a lead to the builder, possibly adding missing sites.

        This method first adds sites from 'lead_builder' until the interface
        where the lead will attach is "smooth".  Then it appends the
        'lead_builder' and the interface sites as a
        `~kwant.builder.BuilderLead` to the 'leads' of this builder.

        Parameters
        ----------
        lead_builder : `Builder` with 1D translational symmetry
            Builder of the lead which has to be attached.
        origin : `Site`
            The site which should belong to a domain where the lead should
            begin. It is used to attach a lead inside the system, e.g. to an
            inner radius of a ring.
        add_cells : int
            Number of complete unit cells of the lead to be added to the system
            *after* the missing sites have been added.

        Returns
        -------
        added_sites : list of `Site` objects that were added to the system.

        Raises
        ------
        ValueError
            If `lead_builder` does not have proper symmetry, has hoppings with
            range of more than one lead unit cell, or if it is not completely
            interrupted by the system.

        Notes
        -----
        This method is not fool-proof, i.e. if it raises an error, there is
        no guarantee that the system stayed unaltered.

        The system must "interrupt" the lead that is being attached. This means
        that for each site in the lead that has a hopping to a neighbnoring
        unit cell there must be at least one site in the system that is an
        image of the lead site under the action of the lead's translational
        symmetry.  In order to interrupt the lead, the system must contain
        sites from the same site family as the sites in the lead.  Below are
        three examples of leads being attached to systems::

            Successful           Successful          Unsuccessful
            ----------           ----------          ------------
            Lead   System        Lead   System        Lead  System
            x-  …      o         x   …                x-  …
            |          |         |                    |
            x-  …    o-o         x-  …    o-o         x-  …  o-o
            |        | |         |        | |         |      | |
            x-  …  o-o-o         x-  …  o-o-o         x-  …  o-o

        The second case succeeds, as even though the top site has no image in
        the system, because the top site has no hoppings to sites in other unit
        cells.

        Sites may be added to the system when the lead is attached, so that the
        interface to the lead is "smooth". Below we show the system after
        having attached a lead. The 'x' symbols in the system indicate the
        added sites::

            Lead   System        Lead   System
            x-  …  x-x-o         x   …  x
            |      | | |         |      |
            x-  …  x-o-o         x-  …  x-o-o
            |      | | |         |      | | |
            x-  …  o-o-o         x-  …  o-o-o
        """
        if self.symmetry.num_directions:
            raise ValueError("Leads can only be attached to finite systems.")

        if add_cells < 0 or int(add_cells) != add_cells:
            raise ValueError('add_cells must be an integer >= 0.')

        sym = lead_builder.symmetry
        H = lead_builder.H

        if sym.num_directions != 1:
            raise ValueError('Only builders with a 1D symmetry are allowed.')

        try:
            hop_range = max(abs(sym.which(hopping[1])[0])
                            for hopping in lead_builder.hoppings())
        except ValueError:  # if there are no hoppings max() will raise
            hop_range = 0

        if hop_range > 1:
            # Automatically increase the period, potentially warn the user.
            new_lead = Builder(
                sym.subgroup((hop_range,)),
                conservation_law=lead_builder.conservation_law,
                time_reversal=lead_builder.time_reversal,
                particle_hole=lead_builder.particle_hole,
                chiral=lead_builder.chiral,
            )
            with reraise_warnings():
                new_lead.fill(lead_builder, lambda site: True,
                              lead_builder.sites(), max_sites=float('inf'))
            lead_builder = new_lead
            sym = lead_builder.symmetry
            H = lead_builder.H

        if not H:
            raise ValueError('Lead to be attached contains no sites.')

        # Check if site families of the lead are present in the system (catches
        # a common and a hard to find bug).
        families = set(site.family for site in H)
        lead_only_families = families.copy()
        for site in self.H:
            lead_only_families.discard(site.family)
            if not lead_only_families:
                break
        else:
            msg = ('Sites with site families {0} do not appear in the system, '
                   'hence the system does not interrupt the lead.')
            raise ValueError(msg.format(tuple(lead_only_families)))

        all_doms = set()
        for site in self.H:
            if site.family not in families:
                continue
            ge = sym.which(site)
            if sym.act(-ge, site) in H:
                all_doms.add(ge[0])

        if origin is not None:
            orig_dom = sym.which(origin)[0]
            all_doms = [dom for dom in all_doms if dom <= orig_dom]
        if len(all_doms) == 0:
            raise ValueError('Builder does not intersect with the lead,'
                             ' this lead cannot be attached.')
        max_dom = max(all_doms) + add_cells
        min_dom = min(all_doms)
        del all_doms

        def shape(site):
            domain, = sym.which(site)
            if domain < min_dom:
                raise ValueError('Builder does not interrupt the lead,'
                                 ' this lead cannot be attached.')
            return domain <= max_dom + 1

        # We start flood-fill from the first domain that doesn't belong to the
        # system (this one is guaranteed to contain a complete unit cell of the
        # lead). After flood-fill we remove that domain.
        start = {sym.act((max_dom + 1,), site) for site in H}
        with reraise_warnings():
            all_added = self.fill(lead_builder, shape, start,
                                  max_sites=float('inf'))
        all_added = [site for site in all_added if site not in start]
        del self[start]

        # Calculate the interface.
        interface = set()
        for site in H:
            for neighbor in lead_builder.neighbors(site):
                neighbor = sym.act((max_dom + 1,), neighbor)
                if sym.which(neighbor)[0] == max_dom:
                    interface.add(neighbor)

        self.leads.append(BuilderLead(lead_builder, interface, all_added))
        return all_added

    def finalized(self):
        """Return a finalized (=usable with solvers) copy of the system.

        Returns
        -------
        finalized_system : `kwant.builder.FiniteSystem`
            If there is no symmetry.
        finalized_system : `kwant.builder.InfiniteSystem`
            If a symmetry is present.

        Notes
        -----
        This method does not modify the Builder instance for which it is
        called.

        Upon finalization, it is implicitly assumed that **every** function
        assigned as a value to a builder with a symmetry possesses the same
        symmetry.

        Attached leads are also finalized and will be present in the finalized
        system to be returned.

        Currently, only Builder instances without or with a 1D translational
        `Symmetry` can be finalized.
        """
        if self.symmetry.num_directions == 0:
            return FiniteSystem(self)
        elif self.symmetry.num_directions == 1:
            return InfiniteSystem(self)
        else:
            raise ValueError('Currently, only builders without or with a 1D '
                             'translational symmetry can be finalized.')

    # Protect novice users from confusing error messages if they
    # forget to finalize their Builder.

    @staticmethod
    def _require_system(*args, **kwargs):
        """You need a finalized system; Use Builder.finalized() first."""
        raise TypeError('You need a finalized system; '
                        'use Builder.finalized() first.')

    hamiltonian = hamiltonian_submatrix = modes = selfenergy = \
    inter_cell_hopping = cell_hamiltonian = precalculated = \
    _require_system


################ Finalized systems

def _raise_user_error(exc, func):
    msg = ('Error occurred in user-supplied value function "{0}".\n'
           'See the upper part of the above backtrace for more information.')
    raise UserCodeError(msg.format(func.__name__)) from exc


def _translate_cons_law(cons_law):
    """Translate a conservation law from builder format to something that can
    be used to initialize operator.Density.
    """
    if callable(cons_law):
        @wraps(cons_law)
        def vals(site, *args, **kwargs):
            if site.family.norbs == 1:
                return cons_law(site, *args, **kwargs)
            return np.diag(np.linalg.eigvalsh(cons_law(site, *args, **kwargs)))

        @wraps(cons_law)
        def vecs(site, *args, **kwargs):
            if site.family.norbs == 1:
                return 1
            return np.linalg.eigh(cons_law(site, *args, **kwargs))[1]

    elif isinstance(cons_law, collections.abc.Mapping):
        vals = {family: (value if family.norbs == 1 else
                         ta.array(np.diag(np.linalg.eigvalsh(value))))
                for family, value in cons_law.items()}
        vecs = {family: (1 if family.norbs == 1 else
                         ta.array(np.linalg.eigh(value)[1]))
                for family, value in cons_law.items()}

    else:
        try:
            vals, vecs = np.linalg.eigh(cons_law)
            vals = np.diag(vals)
        except np.linalg.LinAlgError as e:
            if '0-dimensional' not in e.args[0]:
                raise e  # skip coverage
            vals, vecs = cons_law, 1

    return vals, vecs


class _FinalizedBuilderMixin:
    """Common functionality for all finalized builders"""

    def _init_discrete_symmetries(self, builder):
        def operator(op):
            return Density(self, op, check_hermiticity=False)

        if builder.conservation_law is None:
            self._cons_law = None
        else:
            a = _translate_cons_law(builder.conservation_law)
            self._cons_law = tuple(map(operator, a))
        self._symmetries = tuple(None if op is None else operator(op)
                                 for op in [builder.time_reversal,
                                            builder.particle_hole,
                                            builder.chiral])

    def hamiltonian(self, i, j, *args, params=None):
        if args and params:
            raise TypeError("'args' and 'params' are mutually exclusive.")
        if i == j:
            value, param_names = self.onsites[i]
            if param_names is not None:  # 'value' is callable
                site = self.symmetry.to_fd(self.sites[i])
                if params:
                    # See body of _value_params_pair_cache().
                    if isinstance(param_names, Exception):
                        raise param_names
                    args = map(params.__getitem__, param_names)
                try:
                    value = value(site, *args)
                except Exception as exc:
                    if isinstance(exc, KeyError) and params:
                        missing = [p for p in param_names if p not in params]
                        if missing:
                            msg = ('System is missing required arguments: ',
                                   ', '.join(map('"{}"'.format, missing)))
                            raise TypeError(''.join(msg))
                    _raise_user_error(exc, value)
        else:
            edge_id = self.graph.first_edge_id(i, j)
            value, param_names = self.hoppings[edge_id]
            conj = value is Other
            if conj:
                i, j = j, i
                edge_id = self.graph.first_edge_id(i, j)
                value, param_names = self.hoppings[edge_id]
            if param_names is not None:  # 'value' is callable
                sites = self.sites
                site_i, site_j = self.symmetry.to_fd(sites[i], sites[j])
                if params:
                    # See body of _value_params_pair_cache().
                    if isinstance(param_names, Exception):
                        raise param_names
                    args = map(params.__getitem__, param_names)
                try:
                    value = value(site_i, site_j, *args)
                except Exception as exc:
                    if isinstance(exc, KeyError) and params:
                        missing = [p for p in param_names if p not in params]
                        if missing:
                            msg = ('System is missing required arguments: ',
                                   ', '.join(map('"{}"'.format, missing)))
                            raise TypeError(''.join(msg))
                    _raise_user_error(exc, value)
            if conj:
                value = herm_conj(value)
        return value

    @deprecate_args
    def discrete_symmetry(self, args=(), *, params=None):
        if self._cons_law is not None:
            eigvals, eigvecs = self._cons_law
            eigvals = eigvals.tocoo(args, params=params)
            if not np.allclose(eigvals.data, np.round(eigvals.data)):
                raise ValueError("Conservation law must have integer"
                                 " eigenvalues.")
            eigvals = np.round(eigvals).tocsr()
            # Avoid appearance of zero eigenvalues
            eigvals = eigvals + 0.5 * sparse.identity(eigvals.shape[0])
            eigvals.eliminate_zeros()
            eigvecs = eigvecs.tocoo(args, params=params)
            projectors = [eigvecs.dot(eigvals == val)
                          for val in sorted(np.unique(eigvals.data))]
        else:
            projectors = None

        def evaluate(op):
            return None if op is None else op.tocoo(args, params=params)

        return DiscreteSymmetry(projectors, *(evaluate(symm) for symm in
                                              self._symmetries))


# The same (value, parameters) pair will be used for many sites/hoppings,
# so we cache it to avoid wasting extra memory.
def _value_params_pair_cache(nstrip):
    def get(value):
        entry = cache.get(id(value))
        if entry is None:
            if isinstance(value, _Substituted):
                entry = value.func, value.params[nstrip:]
            elif callable(value):
                try:
                    param_names = get_parameters(value)
                except ValueError as ex:
                    # The parameter names are determined and stored in advance
                    # for future use.  This has failed, but it will only turn
                    # into a problem if user code ever uses the 'params'
                    # mechanism.  To maintain backwards compatibility, we catch
                    # and store the exception so that it can be raised whenever
                    # appropriate.
                    entry = value, ex
                else:
                    entry = value, param_names[nstrip:]
            else:
                # None means: value is not callable. (That's faster to check.)
                entry = value, None
            cache[id(value)] = entry
        return entry
    assert nstrip in [1, 2]
    cache = {}
    return get


class FiniteSystem(_FinalizedBuilderMixin, system.FiniteSystem):
    """Finalized `Builder` with leads.

    Usable as input for the solvers in `kwant.solvers`.

    Attributes
    ----------
    sites : sequence
        ``sites[i]`` is the `~kwant.builder.Site` instance that corresponds
        to the integer-labeled site ``i`` of the low-level system. The sites
        are ordered first by their family and then by their tag.
    id_by_site : dict
        The inverse of ``sites``; maps high-level `~kwant.builder.Site`
        instances to their integer label.
        Satisfies ``id_by_site[sites[i]] == i``.
    """

    def __init__(self, builder):
        assert builder.symmetry.num_directions == 0

        #### Make translation tables.
        sites = tuple(sorted(builder.H))
        id_by_site = {}
        for site_id, site in enumerate(sites):
            id_by_site[site] = site_id

        #### Make graph.
        g = graph.Graph()
        g.num_nodes = len(sites)  # Some sites could not appear in any edge.
        for tail, hvhv in builder.H.items():
            for head in islice(hvhv, 2, None, 2):
                if tail == head:
                    continue
                g.add_edge(id_by_site[tail], id_by_site[head])
        g = g.compressed()

        #### Connect leads.
        finalized_leads = []
        lead_interfaces = []
        lead_paddings = []
        for lead_nr, lead in enumerate(builder.leads):
            try:
                with warnings.catch_warnings(record=True) as ws:
                    warnings.simplefilter("always")
                    # The following line is the whole "payload" of the entire
                    # try-block.
                    finalized_leads.append(lead.finalized())
                for w in ws:
                    # Re-raise any warnings with an additional message and the
                    # proper stacklevel.
                    w = w.message
                    msg = 'When finalizing lead {0}:'.format(lead_nr)
                    warnings.warn(w.__class__(' '.join((msg,) + w.args)),
                                  stacklevel=3)
            except ValueError as e:
                # Re-raise the exception with an additional message.
                msg = 'Problem finalizing lead {0}:'.format(lead_nr)
                e.args = (' '.join((msg,) + e.args),)
                raise
            try:
                interface = [id_by_site[isite] for isite in lead.interface]
            except KeyError as e:
                msg = ("Lead {0} is attached to a site that does not "
                       "belong to the scattering region:\n {1}")
                raise ValueError(msg.format(lead_nr, e.args[0]))

            lead_interfaces.append(np.array(interface))

            padding = getattr(lead, 'padding', [])
            # Some padding sites might have been removed after the lead was
            # attached. Unlike in the case of the interface, this is not a
            # problem.
            finalized_padding = [
                id_by_site[isite] for isite in padding if isite in id_by_site
            ]

            lead_paddings.append(np.array(finalized_padding))

        # Because many onsites/hoppings share the same (value, parameter)
        # pairs, we keep them in a cache so that we only store a given pair
        # in memory *once*. This is a similar idea to interning strings.
        cache = _value_params_pair_cache(1)
        onsites = [cache(builder.H[site][1]) for site in sites]
        cache = _value_params_pair_cache(2)
        hoppings = [cache(builder._get_edge(sites[tail], sites[head]))
                    for tail, head in g]

        # Compute the union of the parameters of onsites and hoppings.  Here,
        # 'onsites' and 'hoppings' are pairs whose second element is one of
        # three things:
        #
        # * a tuple of parameter names when the matrix element is a function,
        # * 'None' when it is a constant,
        # * an exception when the parameter names could not have been
        #   determined (See body of _value_params_pair_cache()).
        parameters = []
        for _, names in chain(onsites, hoppings):
            if isinstance(names, Exception):
                parameters = None
                break
            if names:
                parameters.extend(names)
        else:
            parameters = frozenset(parameters)

        self.graph = g
        self.sites = sites
        self.site_ranges = _site_ranges(sites)
        self.id_by_site = id_by_site
        self.hoppings = hoppings
        self.onsites = onsites
        self.parameters = parameters
        self.symmetry = builder.symmetry
        self.leads = finalized_leads
        self.lead_interfaces = lead_interfaces
        self.lead_paddings = lead_paddings
        self._init_discrete_symmetries(builder)

    def pos(self, i):
        return self.sites[i].pos


class InfiniteSystem(_FinalizedBuilderMixin, system.InfiniteSystem):
    """Finalized infinite system, extracted from a `Builder`.

    Attributes
    ----------
    sites : sequence
        ``sites[i]`` is the `~kwant.builder.Site` instance that corresponds
        to the integer-labeled site ``i`` of the low-level system.
    id_by_site : dict
        The inverse of ``sites``; maps high-level `~kwant.builder.Site`
        instances to their integer label.
        Satisfies ``id_by_site[sites[i]] == i``.

    Notes
    -----
    In infinite systems ``sites`` consists of 3 parts: sites in the fundamental
    domain (FD) with hoppings to neighboring cells, sites in the FD with no
    hoppings to neighboring cells, and sites in FD+1 attached to the FD by
    hoppings. Each of these three subsequences is individually sorted.
    """

    def __init__(self, builder, interface_order=None):
        """
        Finalize a builder instance which has to have exactly a single
        symmetry direction.

        If interface_order is not set, the order of the interface sites in the
        finalized system will be arbitrary.  If interface_order is set to a
        sequence of interface sites, this order will be kept.
        """
        sym = builder.symmetry
        assert sym.num_directions == 1

        #### For each site of the fundamental domain, determine whether it has
        #### neighbors in the previous domain or not.
        lsites_with = []       # Fund. domain sites with neighbors in prev. dom
        lsites_without = []    # Remaining sites of the fundamental domain
        for tail in builder.H: # Loop over all sites of the fund. domain.
            for head in builder._out_neighbors(tail):
                fd = sym.which(head)[0]
                if fd == 1:
                    # Tail belongs to fund. domain, head to the next domain.
                    lsites_with.append(tail)
                    break
            else:
                # Tail is a fund. domain site not connected to prev. domain.
                lsites_without.append(tail)
        cell_size = len(lsites_with) + len(lsites_without)

        if not lsites_with:
            warnings.warn('Infinite system with disconnected cells.',
                          RuntimeWarning, stacklevel=3)

        ### Create list of sites and a lookup table
        minus_one = ta.array((-1,))
        plus_one = ta.array((1,))
        if interface_order is None:
            # interface must be sorted
            interface = [sym.act(minus_one, s) for s in lsites_with]
            interface.sort()
        else:
            lsites_with_set = set(lsites_with)
            lsites_with = []
            interface = []
            if interface_order:
                shift = ta.array((-sym.which(interface_order[0])[0] - 1,))
            for shifted_iface_site in interface_order:
                # Shift the interface domain before the fundamental domain.
                # That's the right place for the interface of a lead to be, but
                # the sites of interface_order might live in a different
                # domain.
                iface_site = sym.act(shift, shifted_iface_site)
                lsite = sym.act(plus_one, iface_site)

                try:
                    lsites_with_set.remove(lsite)
                except KeyError:
                    if (-sym.which(shifted_iface_site)[0] - 1,) != shift:
                        raise ValueError(
                            'The sites in interface_order do not all '
                            'belong to the same lead cell.')
                    else:
                        raise ValueError('A site in interface_order is not an '
                                         'interface site:\n' + str(iface_site))
                interface.append(iface_site)
                lsites_with.append(lsite)
            if lsites_with_set:
                raise ValueError(
                    'interface_order did not contain all interface sites.')
            # `interface_order` *must* be sorted, hence `interface` should also
            if interface != sorted(interface):
                raise ValueError('Interface sites must be sorted.')
            del lsites_with_set

        # we previously sorted the interface, so don't sort it again
        sites = sorted(lsites_with) + sorted(lsites_without) + interface
        del lsites_with
        del lsites_without
        del interface
        id_by_site = {}
        for site_id, site in enumerate(sites):
            id_by_site[site] = site_id

        # In the following, because many onsites/hoppings share the same
        # (value, parameter) pairs, we keep them in 'cache' so that we only
        # store a given pair in memory *once*. This is like interning strings.

        #### Make graph and extract onsite Hamiltonians.
        cache = _value_params_pair_cache(1)
        g = graph.Graph()
        g.num_nodes = len(sites)  # Some sites could not appear in any edge.
        onsites = []
        for tail_id, tail in enumerate(sites[:cell_size]):
            onsites.append(cache(builder.H[tail][1]))
            for head in builder._out_neighbors(tail):
                head_id = id_by_site.get(head)
                if head_id is None:
                    # Head belongs neither to the fundamental domain nor to the
                    # previous domain.  Check that it belongs to the next
                    # domain and ignore it otherwise as an edge corresponding
                    # to this one has been added already or will be added.
                    fd = sym.which(head)[0]
                    if fd != 1:
                        msg = ('Further-than-nearest-neighbor cells '
                               'are connected by hopping\n{0}.')
                        raise ValueError(msg.format((tail, head)))
                    continue
                if head_id >= cell_size:
                    # Head belongs to previous domain.  The edge added here
                    # correspond to one left out just above.
                    g.add_edge(head_id, tail_id)
                g.add_edge(tail_id, head_id)
        g = g.compressed()

        #### Extract hoppings.
        cache = _value_params_pair_cache(2)
        hoppings = []
        for tail_id, head_id in g:
            tail = sites[tail_id]
            head = sites[head_id]
            if tail_id >= cell_size:
                # The tail belongs to the previous domain.  Find the
                # corresponding hopping with the tail in the fund. domain.
                tail, head = sym.to_fd(tail, head)
            hoppings.append(cache(builder._get_edge(tail, head)))

        # Compute the union of the parameters of onsites and hoppings.  Here,
        # 'onsites' and 'hoppings' are pairs whose second element is one of
        # three things:
        #
        # * a tuple of parameter names when the matrix element is a function,
        # * 'None' when it is a constant,
        # * an exception when the parameter names could not have been
        #   determined (See body of _value_params_pair_cache()).
        parameters = []
        for _, names in chain(onsites, hoppings):
            if isinstance(names, Exception):
                parameters = None
                break
            if names:
                parameters.extend(names)
        else:
            parameters = frozenset(parameters)

        self.graph = g
        self.sites = sites
        self.site_ranges = _site_ranges(sites)
        self.id_by_site = id_by_site
        self.hoppings = hoppings
        self.onsites = onsites
        self.parameters = parameters
        self.symmetry = builder.symmetry
        self.cell_size = cell_size
        self._init_discrete_symmetries(builder)


    def hamiltonian(self, i, j, *args, params=None):
        cs = self.cell_size
        if i == j >= cs:
            i -= cs
            j -= cs
        return super().hamiltonian(i, j, *args, params=params)

    def pos(self, i):
        return self.sites[i].pos
