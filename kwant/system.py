# Copyright 2011-2019 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Low-level interface of systems"""

__all__ = [
    'Site', 'SiteArray', 'SiteFamily',
    'System', 'VectorizedSystem', 'FiniteSystem', 'FiniteVectorizedSystem',
    'InfiniteSystem', 'InfiniteVectorizedSystem',
    'is_finite', 'is_infinite', 'is_vectorized',
]

import abc
import warnings
import operator
from copy import copy
from collections import namedtuple
from functools import total_ordering, lru_cache
import numpy as np
from . import _system
from ._common  import deprecate_args, KwantDeprecationWarning



################ Sites and Site families

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


class SiteArray:
    """An array of sites, members of a `SiteFamily`.

    Parameters
    ----------
    family : an instance of `SiteFamily`
        The 'type' of the sites.
    tags : a sequence of python objects
        Sequence of unique identifiers of the sites within the
        site array family, typically vectors of integers.

    Raises
    ------
    ValueError
        If `tags` are not proper tags for `family`.

    See Also
    --------
    kwant.system.Site
    """

    def __init__(self, family, tags):
        self.family = family
        try:
            tags = family.normalize_tags(tags)
        except (TypeError, ValueError) as e:
            msg = 'Tags {0} are not allowed for site family {1}: {2}'
            raise type(e)(msg.format(repr(tags), repr(family), e.args[0]))
        self.tags = tags

    def __repr__(self):
        return 'SiteArray({0}, {1})'.format(repr(self.family), repr(self.tags))

    def __str__(self):
        sf = self.family
        return ('<SiteArray {0} of {1}>'
                .format(self.tags, sf.name if sf.name else sf))

    def __len__(self):
        return len(self.tags)

    def __eq__(self, other):
        if not isinstance(other, SiteArray):
            raise NotImplementedError()
        return self.family == other.family and np.all(self.tags == other.tags)

    def positions(self):
        """Real space position of the site.

        This relies on ``family`` having a ``pos`` method (see `SiteFamily`).
        """
        return self.family.positions(self.tags)


@total_ordering
class SiteFamily:
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


    All site families must define either 'normalize_tag' or 'normalize_tags',
    which brings a tag (or, in the latter case, a sequence of tags) to the
    standard format for this site family.

    Site families may also implement methods ``pos(tag)`` and
    ``positions(tags)``, which return a vector of realspace coordinates or an
    array of vectors of realspace coordinates of the site(s) belonging to this
    family with the given tag(s). These methods are used in plotting routines.
    ``positions(tags)`` should return an array with shape ``(N, M)`` where
    ``N`` is the length of ``tags``, and ``M`` is the realspace dimension.

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
        if norbs is None:
            warnings.warn("Not specfying norbs is deprecated. Always specify "
                          "norbs when creating site families.",
                          KwantDeprecationWarning, stacklevel=3)
        if norbs is not None:
            if int(norbs) != norbs or norbs <= 0:
                raise ValueError('The norbs parameter must be an integer > 0.')
            norbs = int(norbs)
        self.norbs = norbs

    def __init_subclass__(cls, **kwargs):
        super().__init_subclass__(**kwargs)
        if (cls.normalize_tag is SiteFamily.normalize_tag
            and cls.normalize_tags is SiteFamily.normalize_tags):
            raise TypeError("Must redefine either 'normalize_tag' or "
                            "'normalize_tags'")

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

    def normalize_tag(self, tag):
        """Return a normalized version of the tag.

        Raises TypeError or ValueError if the tag is not acceptable.
        """
        tag, = self.normalize_tags([tag])
        return tag

    def normalize_tags(self, tags):
        """Return a normalized version of the tags.

        Raises TypeError or ValueError if the tags are not acceptable.
        """
        return np.array([self.normalize_tag(tag) for tag in tags])

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



################ Systems


class System(metaclass=abc.ABCMeta):
    """Abstract general low-level system.

    Attributes
    ----------
    graph : kwant.graph.CGraph
        The system graph.
    site_ranges : None or sorted sequence of triples of integers
        If provided, encodes ranges of sites that have the same number of
        orbitals. Each triple consists of ``(first_site, norbs, orb_offset)``:
        the first site in the range, the number of orbitals on each site in the
        range, and the offset of the first orbital of the first site in the
        range.  In addition, the final triple should have the form
        ``(len(graph.num_nodes), 0, tot_norbs)`` where ``tot_norbs`` is the
        total number of orbitals in the system.
    parameters : frozenset of strings
        The names of the parameters on which the system depends. This attribute
        is provisional and may be changed in a future version of Kwant

    Notes
    -----
    The sites of the system are indexed by integers ranging from 0 to
    ``self.graph.num_nodes - 1``.

    Optionally, a class derived from ``System`` can provide a method ``pos`` which
    is assumed to return the real-space position of a site given its index.

    Due to the ordering semantics of sequences, and the fact that a given
    ``first_site`` can only appear *at most once* in ``site_ranges``,
    ``site_ranges`` is ordered according to ``first_site``.

    Consecutive elements in ``site_ranges`` are not required to have different
    numbers of orbitals.
    """
    @abc.abstractmethod
    def hamiltonian(self, i, j, *args, params=None):
        """Return the hamiltonian matrix element for sites ``i`` and ``j``.

        If ``i == j``, return the on-site Hamiltonian of site ``i``.

        if ``i != j``, return the hopping between site ``i`` and ``j``.

        Hamiltonians may depend (optionally) on positional and
        keyword arguments.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        pass

    @deprecate_args
    def discrete_symmetry(self, args, *, params=None):
        """Return the discrete symmetry of the system.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        # Avoid the circular import.
        from .physics import DiscreteSymmetry
        return DiscreteSymmetry()


    def __str__(self):
        items = [
            # (format, extractor, skip if info not present)
            ('{} sites', self.graph.num_nodes, False),
            ('{} hoppings', self.graph.num_edges, False),
            ('parameters: {}', tuple(self.parameters), True),
        ]
        # Skip some information when it's not present (parameters)
        details = [fmt.format(info) for fmt, info, skip in items
                   if (info or not skip)]
        details = ', and '.join((', '.join(details[:-1]), details[-1]))
        return '<{} with {}>'.format(self.__class__.__name__, details)

    hamiltonian_submatrix = _system.hamiltonian_submatrix


Term = namedtuple(
    "Term",
    ["subgraph", "hermitian", "parameters"],
)


class VectorizedSystem(System, metaclass=abc.ABCMeta):
    """Abstract general low-level system with support for vectorization.

    Attributes
    ----------
    graph : kwant.graph.CGraph
        The system graph.
    subgraphs : sequence of tuples
        Each subgraph has the form '((idx1, idx2), (offsets1, offsets2))'
        where 'offsets1' and 'offsets2' index sites within the site arrays
        indexed by 'idx1' and 'idx2'.
    terms : sequence of tuples
        Each tuple has the following structure:
        (subgraph: int, hermitian: bool, parameters: List(str))
        'subgraph' indexes 'subgraphs' and supplies the to/from sites of this
        term. 'hermitian' is 'True' if the term needs its Hermitian
        conjugate to be added when evaluating the Hamiltonian, and 'parameters'
        contains a list of parameter names used when evaluating this term.
    site_arrays : sequence of SiteArray
        The sites of the system. The family of each site array must have
        ``norbs`` specified.
    site_ranges : Nx3 integer array
        Has 1 row per site array, plus one extra row.  Each row consists
        of ``(first_site, norbs, orb_offset)``: the index of the first
        site in the site array, the number of orbitals on each site in
        the site array, and the offset of the first orbital of the first
        site in the site array.  In addition, the final row has the form
        ``(len(graph.num_nodes), 0, tot_norbs)`` where ``tot_norbs`` is the
        total number of orbitals in the system. Note 'site_ranges'
        is directly computable from 'site_arrays'.
    parameters : frozenset of strings
        The names of the parameters on which the system depends. This attribute
        is provisional and may be changed in a future version of Kwant

    Notes
    -----
    The sites of the system are indexed by integers ranging from 0 to
    ``self.graph.num_nodes - 1``.

    Optionally, a class derived from ``System`` can provide a method ``pos`` which
    is assumed to return the real-space position of a site given its index.
    """
    @abc.abstractmethod
    def hamiltonian_term(self, term_number, selector=slice(None),
                         args=(), params=None):
        """Return the Hamiltonians for hamiltonian term number k.

        Parameters
        ----------
        term_number : int
            The number of the term to evaluate.
        selector : slice or sequence of int, default: slice(None)
            The elements of the term to evaluate.
        args : tuple
            Positional arguments to the term. (Deprecated)
        params : dict
            Keyword parameters to the term

        Returns
        -------
        hamiltonian : 3d complex array
            Has shape ``(N, P, Q)`` where ``N`` is the number of matrix
            elements in this term (or the number selected by 'selector'
            if provided), ``P`` and ``Q`` are the number of orbitals in the
            'to' and 'from' site arrays associated with this term.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
    @property
    @lru_cache(1)
    def site_ranges(self):
        site_offsets = np.cumsum([0] + [len(arr) for arr in self.site_arrays])
        norbs = [arr.family.norbs for arr in self.site_arrays] + [0]
        orb_offsets = np.cumsum(
            [0] + [len(arr) * arr.family.norbs for arr in self.site_arrays]
        )
        return np.array([site_offsets, norbs, orb_offsets]).transpose()

    hamiltonian_submatrix = _system.vectorized_hamiltonian_submatrix


class FiniteSystemMixin(metaclass=abc.ABCMeta):
    """Abstract finite low-level system, possibly with leads.

    Attributes
    ----------
    leads : sequence of leads
        Each lead has to provide a method ``selfenergy`` that has
        the same signature as `InfiniteSystem.selfenergy` (without the
        ``self`` parameter), and must have property ``parameters``:
        a collection of strings that name the system parameters (
        though this requirement is provisional and may be removed in
        a future version of Kwant).
        It may also provide ``modes`` that has the
        same signature as `InfiniteSystem.modes` (without the ``self``
        parameter).
    lead_interfaces : sequence of sequences of integers
        Each sub-sequence contains the indices of the system sites
        to which the lead is connected.
    lead_paddings : sequence of sequences of integers
        Each sub-sequence contains the indices of the system sites
        that belong to the lead, and therefore have the same onsite as the lead
        sites, and are connected by the same hoppings as the lead sites.
    parameters : frozenset of strings
        The names of the parameters on which the system depends. This does
        not include the parameters for any leads. This attribute
        is provisional and may be changed in a future version of Kwant

    Notes
    -----
    The length of ``leads`` must be equal to the length of ``lead_interfaces``
    and ``lead_paddings``.

    For lead ``n``, the method leads[n].selfenergy must return a square matrix
    whose size is ``sum(len(self.hamiltonian(site, site)) for site in
    self.lead_interfaces[n])``. The output of ``leads[n].modes`` has to be a
    tuple of `~kwant.physics.PropagatingModes`, `~kwant.physics.StabilizedModes`.

    Often, the elements of `leads` will be instances of `InfiniteSystem`.  If
    this is the case for lead ``n``, the sites ``lead_interfaces[n]`` match
    the first ``len(lead_interfaces[n])`` sites of the InfiniteSystem.

    """

    @deprecate_args
    def precalculate(self, energy=0, args=(), leads=None,
                     what='modes', *, params=None):
        """
        Precalculate modes or self-energies in the leads.

        Construct a copy of the system, with the lead modes precalculated,
        which may significantly speed up calculations where only the system
        is changing.

        Parameters
        ----------
        energy : float
            Energy at which the modes or self-energies have to be
            evaluated.
        args : sequence
            Additional parameters required for calculating the Hamiltionians.
            Deprecated in favor of 'params' (and mutually exclusive with it).
        leads : sequence of integers or None
            Numbers of the leads to be precalculated. If ``None``, all are
            precalculated.
        what : 'modes', 'selfenergy', 'all'
            The quantitity to precompute. 'all' will compute both
            modes and self-energies. Defaults to 'modes'.
        params : dict, optional
            Dictionary of parameter names and their values. Mutually exclusive
            with 'args'.

        Returns
        -------
        syst : FiniteSystem
            A copy of the original system with some leads precalculated.

        Notes
        -----
        If the leads are precalculated at certain `energy` or `args` values,
        they might give wrong results if used to solve the system with
        different parameter values. Use this function with caution.
        """

        if what not in ('modes', 'selfenergy', 'all'):
            raise ValueError("Invalid value of argument 'what': "
                             "{0}".format(what))

        result = copy(self)
        if leads is None:
            leads = list(range(len(self.leads)))
        new_leads = []
        for nr, lead in enumerate(self.leads):
            if nr not in leads:
                new_leads.append(lead)
                continue
            modes, selfenergy = None, None
            if what in ('modes', 'all'):
                modes = lead.modes(energy, args, params=params)
            if what in ('selfenergy', 'all'):
                if modes:
                    selfenergy = modes[1].selfenergy()
                else:
                    selfenergy = lead.selfenergy(energy, args, params=params)
            new_leads.append(PrecalculatedLead(modes, selfenergy))
        result.leads = new_leads
        return result

    @deprecate_args
    def validate_symmetries(self, args=(), *, params=None):
        """Check that the Hamiltonian satisfies discrete symmetries.

        Applies `~kwant.physics.DiscreteSymmetry.validate` to the
        Hamiltonian, see its documentation for details on the return
        format.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        symmetries = self.discrete_symmetry(args=args, params=params)
        ham = self.hamiltonian_submatrix(args, sparse=True, params=params)
        return symmetries.validate(ham)


class FiniteSystem(System, FiniteSystemMixin, metaclass=abc.ABCMeta):
    pass


class FiniteVectorizedSystem(VectorizedSystem, FiniteSystemMixin, metaclass=abc.ABCMeta):
    pass


def is_finite(syst):
    return isinstance(syst, (FiniteSystem, FiniteVectorizedSystem))


class InfiniteSystemMixin(metaclass=abc.ABCMeta):
    """Abstract infinite low-level system.

    An infinite system consists of an infinite series of identical cells.
    Adjacent cells are connected by identical inter-cell hoppings.

    Attributes
    ----------
    cell_size : integer
        The number of sites in a single cell of the system.

    Notes
    -----
    The system graph of an infinite systems contains a single cell, as well as
    the part of the previous cell which is connected to it.  The first
    `cell_size` sites form one complete single cell.  The remaining ``N`` sites
    of the graph (``N`` equals ``graph.num_nodes - cell_size``) belong to the
    previous cell.  They are included so that hoppings between cells can be
    represented.  The N sites of the previous cell correspond to the first
    ``N`` sites of the fully included cell.  When an ``InfiniteSystem`` is used
    as a lead, ``N`` acts also as the number of interface sites to which it
    must be connected.

    The drawing shows three cells of an infinite system.  Each cell consists
    of three sites.  Numbers denote sites which are included into the system
    graph.  Stars denote sites which are not included.  Hoppings are included
    in the graph if and only if they occur between two sites which are part of
    the graph::

            * 2 *
        ... | | | ...
            * 0 3
            |/|/|
            *-1-4

        <-- order of cells

    The numbering of sites in the drawing is one of the two valid ones for that
    infinite system.  The other scheme has the numbers of site 0 and 1
    exchanged, as well as of site 3 and 4.

    Sites in the fundamental domain cell must belong to a different site array
    than the sites in the previous cell. In the above example this means that
    sites '(0, 1, 2)' and '(3, 4)' must belong to different site arrays.
    """
    @deprecate_args
    def modes(self, energy=0, args=(), *, params=None):
        """Return mode decomposition of the lead

        See documentation of `~kwant.physics.PropagatingModes` and
        `~kwant.physics.StabilizedModes` for the return format details.

        The wave functions of the returned modes are defined over the
        *unit cell* of the system, which corresponds to the degrees of
        freedom on the first ``cell_sites`` sites of the system
        (recall that infinite systems store first the sites in the unit
        cell, then connected sites in the neighboring unit cell).

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        from . import physics   # Putting this here avoids a circular import.
        ham = self.cell_hamiltonian(args, params=params)
        hop = self.inter_cell_hopping(args, params=params)
        symmetries = self.discrete_symmetry(args, params=params)
        # Check whether each symmetry is broken.
        # If a symmetry is broken, it is ignored in the computation.
        broken = set(symmetries.validate(ham) + symmetries.validate(hop))
        attribute_names = {'Conservation law': 'projectors',
                          'Time reversal': 'time_reversal',
                          'Particle-hole': 'particle-hole',
                          'Chiral': 'chiral'}
        for name in broken:
            warnings.warn('Hamiltonian breaks ' + name +
                          ', ignoring the symmetry in the computation.')
            assert name in attribute_names, 'Inconsistent naming of symmetries'
            setattr(symmetries, attribute_names[name], None)

        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy

        # Particle-hole and chiral symmetries only apply at zero energy.
        if energy:
            symmetries.particle_hole = symmetries.chiral = None
        return physics.modes(ham, hop, discrete_symmetry=symmetries)

    @deprecate_args
    def selfenergy(self, energy=0, args=(), *, params=None):
        """Return self-energy of a lead.

        The returned matrix has the shape (s, s), where s is
        ``sum(len(self.hamiltonian(i, i)) for i in range(self.graph.num_nodes -
        self.cell_size))``.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        from . import physics   # Putting this here avoids a circular import.
        ham = self.cell_hamiltonian(args, params=params)
        shape = ham.shape
        assert len(shape) == 2
        assert shape[0] == shape[1]
        # Subtract energy from the diagonal.
        ham.flat[::ham.shape[0] + 1] -= energy
        return physics.selfenergy(ham,
                                  self.inter_cell_hopping(args, params=params))

    @deprecate_args
    def validate_symmetries(self, args=(), *, params=None):
        """Check that the Hamiltonian satisfies discrete symmetries.

        Returns `~kwant.physics.DiscreteSymmetry.validate` applied
        to the onsite matrix and the hopping. See its documentation for
        details on the return format.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        symmetries = self.discrete_symmetry(args=args, params=params)
        ham = self.cell_hamiltonian(args=args, sparse=True, params=params)
        hop = self.inter_cell_hopping(args=args, sparse=True, params=params)
        broken = set(symmetries.validate(ham) + symmetries.validate(hop))
        return list(broken)


class InfiniteSystem(System, InfiniteSystemMixin, metaclass=abc.ABCMeta):

    @deprecate_args
    def cell_hamiltonian(self, args=(), sparse=False, *, params=None):
        """Hamiltonian of a single cell of the infinite system.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        cell_sites = range(self.cell_size)
        return self.hamiltonian_submatrix(args, cell_sites, cell_sites,
                                          sparse=sparse, params=params)

    @deprecate_args
    def inter_cell_hopping(self, args=(), sparse=False, *, params=None):
        """Hopping Hamiltonian between two cells of the infinite system.

        Providing positional arguments via 'args' is deprecated,
        instead, provide named parameters as a dictionary via 'params'.
        """
        cell_sites = range(self.cell_size)
        interface_sites = range(self.cell_size, self.graph.num_nodes)
        return self.hamiltonian_submatrix(args, cell_sites, interface_sites,
                                          sparse=sparse, params=params)


class InfiniteVectorizedSystem(VectorizedSystem, InfiniteSystemMixin, metaclass=abc.ABCMeta):
    cell_hamiltonian = _system.vectorized_cell_hamiltonian
    inter_cell_hopping = _system.vectorized_inter_cell_hopping


def is_infinite(syst):
    return isinstance(syst, (InfiniteSystem, InfiniteVectorizedSystem))


def is_vectorized(syst):
    return isinstance(syst, (FiniteVectorizedSystem, InfiniteVectorizedSystem))


def _normalize_matrix_blocks(matrix_blocks, expected_length):
    """Normalize a sequence of matrices into a single 3D numpy array

    Parameters
    ----------
    matrix_blocks : sequence of complex array-like
    expected_length : int
    """
    try:
        matrix_blocks = np.asarray(matrix_blocks, dtype=complex)
    except TypeError:
        raise ValueError(
            "Matrix elements declared with incompatible shapes."
        ) from None
    # Upgrade to vector of matrices
    if len(matrix_blocks.shape) == 1:
        matrix_blocks = matrix_blocks[:, np.newaxis, np.newaxis]
    if len(matrix_blocks.shape) != 3:
        msg = (
            "Vectorized value functions must return an array of"
            "scalars or an array of matrices."
        )
        raise ValueError(msg)
    if matrix_blocks.shape[0] != expected_length:
        raise ValueError("Value functions must return a single value per "
                         "onsite/hopping.")
    return matrix_blocks



class PrecalculatedLead:
    def __init__(self, modes=None, selfenergy=None):
        """A general lead defined by its self energy.

        Parameters
        ----------
        modes : (kwant.physics.PropagatingModes, kwant.physics.StabilizedModes)
            Modes of the lead.
        selfenergy : numpy array
            Lead self-energy.

        Notes
        -----
        At least one of ``modes`` and ``selfenergy`` must be provided.
        """
        if modes is None and selfenergy is None:
            raise ValueError("No precalculated values provided.")
        self._modes = modes
        self._selfenergy = selfenergy
        # Modes/Self-energy have already been evaluated, so there
        # is no parametric dependence anymore
        self.parameters = frozenset()

    @deprecate_args
    def modes(self, energy=0, args=(), *, params=None):
        if self._modes is not None:
            return self._modes
        else:
            raise ValueError("No precalculated modes were provided. "
                             "Consider using precalculate() with "
                             "what='modes' or what='all'")

    @deprecate_args
    def selfenergy(self, energy=0, args=(), *, params=None):
        if self._selfenergy is not None:
            return self._selfenergy
        else:
            raise ValueError("No precalculated selfenergy was provided. "
                             "Consider using precalculate() with "
                             "what='selfenergy' or what='all'")
