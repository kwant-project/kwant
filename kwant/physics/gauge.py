# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.
"""Functions for fixing the magnetic gauge automatically in a Kwant system.


The "gauge" module has been included in Kwant on a provisional basis.
Backwards incompatible changes (up to and including removal of the package)
may occur if deemed necessary by the core developers.
"""

import bisect
from functools import partial
from itertools import permutations

import numpy as np
import scipy
from scipy.integrate import dblquad
from scipy.sparse import csgraph

from .. import system, builder

from ..graph.dijkstra import dijkstra_directed

__all__ = ['magnetic_gauge']


### Integation

# Integrate vector field over triangle, for internal use by '_surface_integral'
# Triangle is (origin, origin + v1, origin + v2), 'n' is np.cross(v1, v2)

def _quad_triangle(f, origin, v1, v2, n, tol):
    if np.dot(n, n) < tol**2:  # does triangle have significant area?
        return 0

    def g(x, y):
        return np.dot(n, f(origin + x * v1 + y * v2))

    result, *_ = dblquad(g, 0, 1, lambda x: 0, lambda x: 1-x)
    return result.real


def _const_triangle(f, origin, v1, v2, n, tol):
    return np.dot(f, n) / 2


def _average_triangle(f, origin, v1, v2, n, tol):
    return np.dot(n, f(origin + 1/3 * (v1 + v2))) / 2


def _surface_integral(f, loop, tol=1e-8, average=False):
    """Calculate the surface integral of 'f' over a surface enclosed by 'loop'.

    This function only works for *divergence free* vector fields, where the
    surface integral depends only on the boundary.

    Parameters
    ----------
    f : callable or real n-vector
        The vector field for which to calculate the surface integral.
        If callable, takes a real n-vector as argument and returns a
        real n-vector.
    loop : sequence of vectors
        Ordered sequence of real n-vectors (positions) that define the
        vertices of the polygon that encloses the surface to integrate over.
    tol : float, default: 1e-8
        Error tolerance on the result.
    average : bool, default: False
        If True, approximate the integral over each triangle using a
        single function evaluation at the centre of the triangle.
    """
    if callable(f):
        integrator = _average_triangle if average else _quad_triangle
    else:
        integrator = _const_triangle

    origin, *points = loop
    integral = 0
    # Split loop into triangles with 1 vertex on 'origin', evaluate
    # the integral over each triangle and sum the result
    for p1, p2 in zip(points, points[1:]):
        v1 = p1 - origin
        v2 = p2 - origin
        n = np.cross(v1, v2)
        integral += integrator(f, origin, v1, v2, n, tol)
    return integral


### Loop finding graph algorithm

def _find_loops(graph, subgraph):
    """
    Parameters
    ----------
    graph : COO matrix
        The complete undirected graph, where the values of the matrix are
        the weights of the corresponding graph links.
    subgraph : COO matrix
        An subgraph of 'graph', with missing edges denoted by infinities.
        Must have the same sparsity structure as 'graph'.

    Returns
    -------
    A sequence of paths which are partially contained in the subgraph.
    The loop is formed by adding a link between the first and last node.

    The ordering is such that the paths are made of links that belong to
    the subgraph or to the previous closed loops.
    """
    # For each link we do 1 update of 'subgraph' and a call to
    # 'csgraph.shortest_path'. It is cheaper to update the CSR
    # matrix rather than convert to LIL and back every iteration.
    subgraph = subgraph.tocsr()
    graph = graph.tocsr()
    assert _same_sparsity_structure(subgraph, graph)

    # Links in graph, but not in subgraph.
    links_to_find = scipy.sparse.triu(graph - subgraph).tocoo()
    links_to_find = np.vstack((links_to_find.row, links_to_find.col)).transpose()

    links_to_find, min_length = _order_links(subgraph, links_to_find)

    # Find shortest path between each link in turn, updating the subgraph with
    # the links as we go.
    loops = []
    while len(links_to_find) > 0:
        frm, to = links_to_find[0]
        (path,), (path_length,) = dijkstra_directed(subgraph,
                                                    sources=np.array([frm]),
                                                    targets=np.array([to]))

        # Reorder links that are still to find based on the loop length in
        # the updated graph. We only reorder when the path length for *this*
        # link is a "little bit" longer that the perviously determined minimum.
        # The "little bit" is needed so we don't needlessly re-order the links
        # on amorphous lattices.
        if path_length > min_length * 1.1:
            links_to_find, min_length = _order_links(subgraph, links_to_find)
        else:
            # Assumes that 'graph' and 'subgraph' have the same sparsity structure.
            _assign_csr(subgraph, graph, (frm, to))
            _assign_csr(subgraph, graph, (to, frm))
            loops.append(path)
            links_to_find = links_to_find[1:]

    return loops


def _order_links(subgraph, links_to_find):
    if len(links_to_find) == 0:
        return [], None
    # Order 'links_to_find' by length of shortest path between the nodes of the link
    path_lengths = dijkstra_directed(subgraph,
                                     sources=links_to_find[:, 0],
                                     targets=links_to_find[:, 1],
                                     return_paths=False)
    idxs = np.argsort(path_lengths)
    return links_to_find[idxs], path_lengths[idxs[0]]


### Generic sparse matrix utilities

def _assign_csr(a, b, element):
    """Assign a single element from a CSR matrix to another.

    Parameters
    ----------
    a : CSR matrix
    b : CSR matrix or scalar
        If a CSR matrix, must have the same sparsity structure
        as 'a'. If a scalar, must be the same dtype as 'a'.
    element: (int, int)
        Row and column indices of the element to set.
    """
    assert isinstance(a, scipy.sparse.csr_matrix)
    row, col = element
    for j in range(a.indptr[row], a.indptr[row + 1]):
        if a.indices[j] == col:
            break
    else:
        raise ValueError('{} not in sparse matrix'.format(element))
    if isinstance(b, scipy.sparse.csr_matrix):
        a.data[j] = b.data[j]
    else:
        a.data[j] = b


def _same_sparsity_structure(a, b):
    a = a.tocsr().sorted_indices()
    b = b.tocsr().sorted_indices()
    return (np.array_equal(a.indices, b.indices)
            and np.array_equal(a.indptr, b.indptr))


def _add_coo_matrices(*mats, shape):
    """Add a sequence of COO matrices by appending their constituent arrays."""
    values = np.hstack([mat.data for mat in mats])
    rows = np.hstack([mat.row for mat in mats])
    cols = np.hstack([mat.col for mat in mats])
    return scipy.sparse.coo_matrix((values, (rows, cols)), shape=shape)


def _shift_diagonally(mat, shift, shape):
    """Shift the row/column indices of a COO matrix."""
    return scipy.sparse.coo_matrix(
        (mat.data, (mat.row + shift, mat.col + shift)),
        shape=shape)


def _distance_matrix(links, pos, shape):
    """Return the distances between the provided links as a COO matrix.

    Parameters
    ----------
    links : sequence of pairs of int
        The links for which to find the lengths.
    pos : callable: int -> vector
        Map from link ends (integers) to realspace position.
    shape : tuple
    """
    if len(links) == 0:  # numpy does not like 'if array'
        return scipy.sparse.coo_matrix(shape)
    links = np.array(links)
    distances = np.array([pos(i) - pos(j) for i, j in links])
    distances = np.linalg.norm(distances, axis=1)
    return scipy.sparse.coo_matrix((distances, links.T), shape=shape)


### Loop finding
#
# All of these functions take a finalized Kwant system and return
# a sequence of loops. Each loop is a sequence of sites (integers)
# that one visits when traversing the loop. The first and final sites
# are assumed to be linked, which closes the loop. The links that one
# traverses when going round a loop is thus:
#
#    list(zip(loop, loop[1:])) + [(loop[-1], loop[0])]
#
# These loops are later used to fix the magnetic gauge in the system.
# All of the links except the final one are assumed to have their gauge
# fixed (i.e. the phase across them is known), and gauge of the final
# link is the one to be determined.


def _loops_in_finite(syst):
    """Find the loops in a finite system with no leads.

    The site indices in the returned loops are those of the system,
    so they may be used as indices to 'syst.sites', or with 'syst.pos'.
    """
    assert isinstance(syst, system.FiniteSystem) and syst.leads == []
    nsites = len(syst.sites)

    # Fix the gauge across the minimum spanning tree of the system graph.
    graph = _distance_matrix(list(syst.graph),
                             pos=syst.pos, shape=(nsites, nsites))
    spanning_tree = _shortest_distance_forest(graph)
    return _find_loops(graph, spanning_tree)


def _shortest_distance_forest(graph):
    # Grow a forest of minimum distance trees for all connected components of the graph
    graph = graph.tocsr()
    tree = graph.copy()
    # set every entry in tree to infinity
    tree.data[:] = np.inf
    unvisited = set(range(graph.shape[0]))
    # set the target node to be greater than any node in the graph.
    # This way we explore the whole graph.
    end = np.array([graph.shape[0] + 1], dtype=np.int32)

    while unvisited:
        # Choose an arbitrary element as the root
        root = unvisited.pop()
        root = np.array([root], dtype=np.int32)
        _, pred = dijkstra_directed(graph, sources=root, targets=end,
                                    return_predecessors=True, return_paths=False)
        for i, p in enumerate(pred):
            # -1 if node 'i' has no predecessor. Either it is the root node,
            # or it was not reached.
            if p != -1:
                unvisited.remove(i)
                _assign_csr(tree, graph, (i, p))
                _assign_csr(tree, graph, (p, i))
    return tree


def _loops_in_infinite(syst):
    """Find the loops in an infinite system.

    Returns
    -------
    loops : sequence of sequences of integers
        The sites in the returned loops belong to two adjacent unit
        cells. The first 'syst.cell_size' sites are in the first
        unit cell, and the next 'sys.cell_size' are in the next
        (in the direction of the translational symmetry).
    extended_sites : callable : int -> Site
        Given a site index in the extended system consisting of
        two unit cells, returns the associated high-level
        `kwant.builder.Site`.
    """
    assert isinstance(syst, system.InfiniteSystem)
    _check_infinite_syst(syst)

    cell_size = syst.cell_size

    unit_cell_links = [(i, j) for i, j in syst.graph
                       if i < cell_size and j < cell_size]
    unit_cell_graph = _distance_matrix(unit_cell_links,
                                       pos=syst.pos,
                                       shape=(cell_size, cell_size))

    # Loops in the interior of the unit cell
    spanning_tree = _shortest_distance_forest(unit_cell_graph)
    loops = _find_loops(unit_cell_graph, spanning_tree)

    # Construct an extended graph consisting of 2 unit cells connected
    # by the inter-cell links.
    extended_shape = (2 * cell_size, 2 * cell_size)
    uc1 = _shift_diagonally(unit_cell_graph, 0, shape=extended_shape)
    uc2 = _shift_diagonally(unit_cell_graph, cell_size, shape=extended_shape)
    hop_links = [(i, j) for i, j in syst.graph if j >= cell_size]
    hop = _distance_matrix(hop_links,
                           pos=syst.pos,
                           shape=extended_shape)
    graph = _add_coo_matrices(uc1, uc2, hop, hop.T,
                              shape=extended_shape)

    # Construct a subgraph where only the shortest link between the
    # 2 unit cells is added. The other links are added with infinite
    # values, so that the subgraph has the same sparsity structure
    # as 'graph'.
    idx = np.argmin(hop.data)
    data = np.full_like(hop.data, np.inf)
    data[idx] = hop.data[idx]
    smallest_edge = scipy.sparse.coo_matrix(
        (data, (hop.row, hop.col)),
        shape=extended_shape)
    subgraph = _add_coo_matrices(uc1, uc2, smallest_edge, smallest_edge.T,
                                 shape=extended_shape)

    # Use these two graphs to find the loops between unit cells.
    loops.extend(_find_loops(graph, subgraph))

    def extended_sites(i):
        unit_cell = np.array([i // cell_size])
        site = syst.sites[i % cell_size]
        return syst.symmetry.act(-unit_cell, site)

    return loops, extended_sites


def _loops_in_composite(syst):
    """Find the loops in finite system with leads.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem

    Returns
    -------
    loops : sequence of sequences of integers
        The sites in each loop belong to the extended scattering region
        (see notes). The first and last site in each loop are guaranteed
        to be in 'syst'.
    which_patch : callable : int -> int
        Given a site index in the extended scattering region (see notes),
        returns the lead patch (see notes) to which the site belongs. Returns
        -1 if the site is part of the reduced scattering region (see notes).
    extended_sites : callable : int -> Site
        Given a site index in the extended scattering region (see notes),
        returns the associated high-level `kwant.builder.Site`.

    Notes
    -----
    extended scattering region
        The scattering region with a single lead unit cell attached at
        each interface. This unit cell is added so that we can "see" any
        loops formed with sites in the lead (see 'check_infinite_syst'
        for details). The sites for each lead are added in the same
        order as the leads, and within a given added unit cell the sites
        are ordered in the same way as the associated lead.
    lead patch
        Sites in the extended scattering region that belong to the added
        unit cell for a given lead, or the lead padding for a given lead
        are said to be in the "lead patch" for that lead.
    reduced scattering region
        The sites of the extended scattering region that are not
        in a lead patch.
    """
    # Check that we can consistently fix the gauge in the scattering region,
    # given that we have independently fixed gauges in the leads.
    _check_composite_system(syst)

    # Get distance matrix for the extended scattering region,
    # a function that maps sites to their lead patches (-1 for sites
    # in the reduced scattering region), and a function that maps sites
    # to high-level 'kwant.builder.Site' objects.
    distance_matrix, which_patch, extended_sites =\
        _extended_scattering_region(syst)

    spanning_tree = _spanning_tree_composite(distance_matrix, which_patch).tocsr()

    # Fill in all links with at least 1 site in a lead patch;
    # their gauge is fixed by the lead gauge.
    for i, j, v in zip(distance_matrix.row, distance_matrix.col,
                       distance_matrix.data):
        if which_patch(i) > -1 or which_patch(j) > -1:
            _assign_csr(spanning_tree, v, (i, j))
            _assign_csr(spanning_tree, v, (j, i))

    loops = _find_loops(distance_matrix, spanning_tree)

    return loops, which_patch, extended_sites


def _extended_scattering_region(syst):
    """Return the distance matrix of a finite system with 1 unit cell
       added to each lead interface.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem

    Returns
    -------
    extended_scattering_region: COO matrix
        Distance matrix between connected sites in the extended
        scattering region.
    which_patch : callable : int -> int
        Given a site index in the extended scattering region, returns
        the lead patch to which the site belongs. Returns
        -1 if the site is part of the reduced scattering region.
    extended_sites : callable : int -> Site
        Given a site index in the extended scattering region, returns
        the associated high-level `kwant.builder.Site`.

    Notes
    -----
    Definitions of the terms 'extended scatteringr region',
    'lead patch' and 'reduced scattering region' are given
    in the notes for `kwant.physics.gauge._loops_in_composite`.
    """
    extended_size = (syst.graph.num_nodes
                     + sum(l.cell_size for l in syst.leads))
    extended_shape = (extended_size, extended_size)

    added_unit_cells = []
    first_lead_site = syst.graph.num_nodes
    for lead, interface in zip(syst.leads, syst.lead_interfaces):
        # Here we assume that the distance between sites in the added
        # unit cell and sites in the interface is the same as between sites
        # in neighboring unit cells.
        uc = _distance_matrix(list(lead.graph),
                              pos=lead.pos, shape=extended_shape)
        # Map unit cell lead sites to their indices in the extended scattering,
        # region and sites in next unit cell to their interface sites.
        hop_from_syst = uc.row >= lead.cell_size
        uc.row[~hop_from_syst] = uc.row[~hop_from_syst] + first_lead_site
        uc.row[hop_from_syst] = interface[uc.row[hop_from_syst] - lead.cell_size]
        # Same for columns
        hop_to_syst = uc.col >= lead.cell_size
        uc.col[~hop_to_syst] = uc.col[~hop_to_syst] + first_lead_site
        uc.col[hop_to_syst] = interface[uc.col[hop_to_syst] - lead.cell_size]

        added_unit_cells.append(uc)
        first_lead_site += lead.cell_size

    scattering_region = _distance_matrix(list(syst.graph),
                                         pos=syst.pos, shape=extended_shape)

    extended_scattering_region = _add_coo_matrices(scattering_region,
                                                   *added_unit_cells,
                                                   shape=extended_shape)

    lead_starts = np.cumsum([syst.graph.num_nodes,
                             *[lead.cell_size for lead in syst.leads]])
    # Frozenset to quickly check 'is this site in the lead padding?'
    extra_sites = [frozenset(sites) for sites in syst.lead_paddings]


    def which_patch(i):
        if i < len(syst.sites):
            # In scattering region
            for patch_num, sites in enumerate(extra_sites):
                if i in sites:
                    return patch_num
            # If not in 'extra_sites' it is in the reduced scattering region.
            return -1
        else:
            # Otherwise it's in an attached lead cell
            which_lead = bisect.bisect(lead_starts, i) - 1
            assert which_lead > -1
            return which_lead


    # Here we use the fact that all the sites in a lead interface belong
    # to the same symmetry domain.
    interface_domains = [lead.symmetry.which(syst.sites[interface[0]])
                         for lead, interface in
                         zip(syst.leads, syst.lead_interfaces)]

    def extended_sites(i):
        if i < len(syst.sites):
            # In scattering region
            return syst.sites[i]
        else:
            # Otherwise it's in an attached lead cell
            which_lead = bisect.bisect(lead_starts, i) - 1
            assert which_lead > -1
            lead = syst.leads[which_lead]
            domain = interface_domains[which_lead] + 1
            # Map extended scattering region site index to site index in lead.
            i = i - lead_starts[which_lead]
            return lead.symmetry.act(domain, lead.sites[i])

    return extended_scattering_region, which_patch, extended_sites


def _interior_links(distance_matrix, which_patch):
    """Return the indices of the links in 'distance_matrix' that
       connect interface sites of the scattering region to other
       sites (interface and non-interface) in the scattering region.
    """

    def _is_in_lead(i):
        return which_patch(i) > -1

    # Sites that connect to/from sites in a lead patch
    interface_sites = {
        (i if not _is_in_lead(i) else j)
        for i, j in zip(distance_matrix.row, distance_matrix.col)
        if _is_in_lead(i) ^ _is_in_lead(j)
    }

    def _we_want(i, j):
        return i in interface_sites and not _is_in_lead(j)

    # Links that connect interface sites to the rest of the scattering region.
    return np.array([
        k
        for k, (i, j) in enumerate(zip(distance_matrix.row, distance_matrix.col))
        if _we_want(i, j) or _we_want(j, i)
    ])


def _make_metatree(graph, links_to_delete):
    """Make a tree of the components of 'graph' that are
       disconnected by deleting 'links'. The values of
       the returned tree are indices of edges in 'graph'
       that connect components.
    """
    # Partition the graph into disconnected components
    dl = partial(np.delete, obj=links_to_delete)
    partitioned_graph = scipy.sparse.coo_matrix(
        (dl(graph.data), (dl(graph.row), dl(graph.col)))
    )
    # Construct the "metagraph", where each component is reduced to
    # a single node, and a representative (smallest) edge is chosen
    # among the edges that connected the components in the original graph.
    ncc, labels = csgraph.connected_components(partitioned_graph)
    metagraph = scipy.sparse.dok_matrix((ncc, ncc), int)
    for k in links_to_delete:
        i, j = labels[graph.row[k]], labels[graph.col[k]]
        if i == j:
            continue  # Discard loop edges
        # Add a representative (smallest) edge from each graph component.
        if graph.data[k] < metagraph.get((i, j), np.inf):
            metagraph[i, j] = k
            metagraph[j, i] = k

    return csgraph.minimum_spanning_tree(metagraph).astype(int)


def _spanning_tree_composite(distance_matrix, which_patch):
    """Find a spanning tree for a composite system.

    We cannot use a simple minimum-distance spanning tree because
    we have the additional constraint that all links with at least
    one end in a lead patch have their gauge fixed. See the notes
    for details.

    Parameters
    ----------
    distance_matrix : COO matrix
        Distance matrix between connected sites in the extended
        scattering region.
    which_patch : callable : int -> int
        Given a site index in the extended scattering region (see notes),
        returns the lead patch (see notes) to which the site belongs. Returns
        -1 if the site is part of the reduced scattering region (see notes).
    Returns
    -------
    spanning_tree : CSR matrix
        A spanning tree with the same sparsity structure as 'distance_matrix',
        where missing links are denoted with infinite weights.

    Notes
    -----
    Definitions of the terms 'extended scattering region', 'lead patch'
    and 'reduced scattering region' are given in the notes for
    `kwant.physics.gauge._loops_in_composite`.

    We cannot use a simple minimum-distance spanning tree because
    we have the additional constraint that all links with at least
    one end in a lead patch have their gauge fixed.
    Consider the following case using a minimum-distance tree
    where 'x' are sites in the lead patch::

        o-o-x      o-o-x
        | | |  -->   | |
        o-o-x      o-o x

    The removed link on the lower right comes from the lead, and hence
    is gauge-fixed, however the vertical link in the center is not in
    the lead, but *is* in the tree, which means that we will fix its
    gauge to 0. The loop on the right would thus not have the correct
    gauge on all links.

    Instead we first cut all links between *interface* sites and
    sites in the scattering region (including other interface sites).
    We then construct a minimum distance forest for these disconnected
    graphs. Finally we add back links from the ones that were cut,
    ensuring that we do not form any loops; we do this by contructing
    a tree of representative links from the "metagraph" of components
    that were disconnected by the link cutting.
    """
    # Links that connect interface sites to other sites in the
    # scattering region (including other interface sites)
    links_to_delete = _interior_links(distance_matrix, which_patch)
    # Make a shortest distance tree for each of the components
    # obtained by cutting the links.
    cut_syst = distance_matrix.copy()
    cut_syst.data[links_to_delete] = np.inf
    forest = _shortest_distance_forest(cut_syst)
    # Connect the forest back up with representative links until
    # we have a single tree (if the original system was not connected,
    # we get a forest).
    metatree = _make_metatree(distance_matrix, links_to_delete)
    for k in np.unique(metatree.data):
        value = distance_matrix.data[k]
        i, j = distance_matrix.row[k], distance_matrix.col[k]
        _assign_csr(forest, value, (i, j))
        _assign_csr(forest, value, (j, i))

    return forest


def _check_infinite_syst(syst):
    r"""Check that the unit cell is a connected graph.

    If the unit cell is not connected then we cannot be sure whether
    there are loops or not just by inspecting the unit cell graph
    (this may be a solved problem, but we could not find an algorithm
    to do this).

    To illustrate this, consider the following unit cell consisting
    of 3 sites and 4 hoppings::

        o-
         \
        o
         \
        o-

    None of the sites are connected within the unit cell, however if we repeat
    a few unit cells::

        o-o-o-o
         \ \ \
        o o o o
         \ \ \
        o-o-o-o

    we see that there is a loop crossing 4 unit cells. A connected unit cell
    is a sufficient condition that all the loops can be found by inspecting
    the graph consisting of two unit cells glued together.
    """
    assert isinstance(syst, system.InfiniteSystem)
    n = syst.cell_size
    rows, cols = np.array([(i, j) for i, j in syst.graph
                            if i < n and j < n]).transpose()
    data = np.ones(len(rows))
    graph = scipy.sparse.coo_matrix((data, (rows, cols)), shape=(n, n))
    if csgraph.connected_components(graph, return_labels=False) > 1:
        raise ValueError(
            'Infinite system unit cell is not connected: we cannot determine '
            'if there are any loops in the system\n\n'
            'If there are, then you must define your unit cell so that it is '
            'connected. If there are not, then you must add zero-magnitude '
            'hoppings to your system.'
        )


def _check_composite_system(syst):
    """Check that we can consistently fix the gauge in a system with leads.

    If not, raise an exception with an informative error message.
    """
    assert isinstance(syst, system.FiniteSystem) and syst.leads
    # Frozenset to quickly check 'is this site in the lead padding?'
    extras = [frozenset(sites) for sites in syst.lead_paddings]
    interfaces = [set(iface) for iface in syst.lead_interfaces]
    # Make interfaces between lead patches and the reduced scattering region.
    for interface, extra in zip(interfaces, extras):
        extra_interface = set()
        if extra:
            extra_interface = set()
            for i, j in syst.graph:
                if i in extra and j not in extra:
                    extra_interface.add(j)
            interface -= extra
            interface |= extra_interface
        assert not extra.intersection(interface)

    pre_msg = (
        'Attaching leads results in gauge-fixed loops in the extended '
        'scattering region (scattering region plus one lead unit cell '
        'from every lead). This does not allow consistent gauge-fixing.\n\n'
    )
    solution_msg = (
        'To avoid this error, attach leads further away from each other.\n\n'
        'Note: calling `attach_lead()` with `add_cells > 0` will not fix '
        'this problem, as the added sites inherit the gauge from the lead. '
        'To extend the scattering region, you must manually add sites '
        'making sure that they use the scattering region gauge.'
    )

    # Check that there is at most one overlapping site between
    # reduced interface of one lead and extra sites of another
    num_leads = len(syst.leads)
    metagraph = scipy.sparse.lil_matrix((num_leads, num_leads))
    for i, j in permutations(range(num_leads), 2):
        intersection = len(interfaces[i] & (interfaces[j] | extras[j]))
        if intersection > 1:
            raise ValueError(
                pre_msg
                + ('There is at least one gauge-fixed loop in the overlap '
                   'of leads {} and {}.\n\n'.format(i, j))
                + solution_msg
            )
        elif intersection == 1:
            metagraph[i, j] = 1
    # Check that there is no loop formed by gauge-fixed bonds of multiple leads.
    num_components = scipy.sparse.csgraph.connected_components(metagraph, return_labels=False)
    if metagraph.nnz // 2 + num_components != num_leads:
        raise ValueError(
            pre_msg
            + ('There is at least one gauge-fixed loop formed by more than 2 leads. '
               ' The connectivity matrix of the leads is:\n\n'
               '{}\n\n'.format(metagraph.A))
            + solution_msg
        )

### Phase calculation

def _calculate_phases(loops, pos, previous_phase, flux):
    """Calculate the phase across the terminal links of a set of loops

    Parameters
    ----------
    loops : sequence of sequences of int
        The loops over which to calculate the flux. We wish to find the phase
        over the link connecting the first and last sites in each loop.
        The phase across all other links in a given loop is assumed known.
    pos : callable : int -> ndarray
        A map from site (integer) to realspace position.
    previous_phase : callable
        Takes a dict that maps from links to phases, and a loop and returns
        the product of the phases across each link in the loop,
        *except* the link between the first and last site in the loop.
    flux : callable
        Takes a sequence of positions and returns the magnetic flux through the
        surface defined by the provided loop.

    Returns
    -------
    phases : dict : (int, int) -> float
        A map from links to the phase across those links.
    """
    phases = dict()
    for loop in loops:
        tail, head = loop[-1], loop[0]
        integral = flux([pos(p) for p in loop])
        phase = np.exp(1j * np.pi * integral)
        phases[tail, head] = phase / previous_phase(phases, loop)
    return phases


# These functions are to be used with '_calculate_phases'.
# 'phases' always stores *either* the phase across (i, j) *or*
# (j, i), and never both. If a phase is not present it is assumed to
# be zero.


def _previous_phase_finite(phases, path):
    previous_phase = 1
    for i, j in zip(path, path[1:]):
        previous_phase *= phases.get((i, j), 1)
        previous_phase /= phases.get((j, i), 1)
    return previous_phase


def _previous_phase_infinite(cell_size, phases, path):
    previous_phase = 1
    for i, j in zip(path, path[1:]):
        # i and j are only in the fundamental unit cell (0 <= i < cell_size)
        # or the next one (cell_size <= i < 2 * cell_size).
        if i >= cell_size and j >= cell_size:
            assert i // cell_size == j // cell_size
            i = i % cell_size
            j = j % cell_size
        previous_phase *= phases.get((i, j), 1)
        previous_phase /= phases.get((j, i), 1)
    return previous_phase


def _previous_phase_composite(which_patch, extended_sites, lead_phases,
                              phases, path):
    previous_phase = 1
    for i, j in zip(path, path[1:]):
        patch_i = which_patch(i)
        patch_j = which_patch(j)
        if patch_i == -1 and patch_j == -1:
            # Both sites in reduced scattering region.
            previous_phase *= phases.get((i, j), 1)
            previous_phase /= phases.get((j, i), 1)
        else:
            # At least one site in a lead patch; use the phase from the
            # associated lead. Check that if both are in a patch, they
            # are in the same patch.
            assert patch_i * patch_j <= 0 or patch_i == patch_j
            patch = max(patch_i, patch_j)
            a, b = extended_sites(i), extended_sites(j)
            previous_phase *= lead_phases[patch](a, b)

    return previous_phase


### High-level interface
#
# These functions glue all the above functionality together.

# Wrapper for phase dict that takes high-level sites
def _finite_wrapper(syst, phases, a, b):
    i = syst.id_by_site[a]
    j = syst.id_by_site[b]
    # We only store *either* (i, j) *or* (j, i). If not present
    # then the phase is unity by definition.
    if (i, j) in phases:
        return phases[i, j]
    elif (j, i) in phases:
        return phases[j, i].conjugate()
    else:
        return 1


def _infinite_wrapper(syst, phases, a, b):
    sym = syst.symmetry
    # Bring link to fundamental domain consistently with how
    # we store the phases.
    t = max(sym.which(a), sym.which(b))
    a, b = sym.act(-t, a, b)
    i = syst.id_by_site[a]
    j = syst.id_by_site[b]
    # We only store *either* (i, j) *or* (j, i). If not present
    # then the phase is unity by definition.
    if (i, j) in phases:
        return phases[i, j]
    elif (j, i) in phases:
        return phases[j, i].conjugate()
    else:
        return 1


def _peierls_finite(syst, loops, syst_field, tol, average):
    integrate = partial(_surface_integral, syst_field,
                        tol=tol, average=average)
    phases = _calculate_phases(
        loops,
        syst.pos,
        _previous_phase_finite,
        integrate,
    )
    return partial(_finite_wrapper, syst, phases)


def _peierls_infinite(syst, loops, extended_sites, syst_field, tol, average):
    integrate = partial(_surface_integral, syst_field,
                        tol=tol, average=average)
    phases = _calculate_phases(
        loops,
        lambda i: extended_sites(i).pos,
        partial(_previous_phase_infinite, syst.cell_size),
        integrate,
    )
    return partial(_infinite_wrapper, syst, phases)


def _peierls_composite(syst, loops, which_patch, extended_sites, lead_gauges,
                       syst_field, *lead_fields, tol, average):
    if len(lead_fields) != len(syst.leads):
        raise ValueError('Magnetic fields must be provided for all leads.')

    lead_phases = [gauge(B, tol=tol, average=average)
                   for gauge, B in zip(lead_gauges, lead_fields)]

    flux = partial(_surface_integral, syst_field, tol=tol, average=average)

    # NOTE: uses the scattering region magnetic field to set the phase
    # of the inteface hoppings this choice is somewhat arbitrary,
    # but it is consistent with the position defined in the scattering
    # region coordinate system. the integrate functions for the leads
    # may be defined far from the interface.
    phases = _calculate_phases(
        loops,
        lambda i: extended_sites(i).pos,
        partial(_previous_phase_composite,
                which_patch, extended_sites, lead_phases),
        flux,
    )

    return (partial(_finite_wrapper, syst, phases), *lead_phases)


# This class is essentially a closure, but documenting closures is a pain.
# To emphasise the lack of manipulatable or inspectable state, we name the
# class as we would for a function.

class magnetic_gauge:
    """Fix the magnetic gauge for a finalized system.

    This can be used to later calculate the Peierls phases that
    should be applied to each hopping, given a magnetic field.

    This API is currently provisional. Refer to the documentation
    for details.

    Parameters
    ----------
    syst : `kwant.builder.FiniteSystem` or `kwant.builder.InfiniteSystem`

    Examples
    --------
    The following illustrates basic usage for a scattering region with
    a single lead attached:

    >>> import numpy as np
    >>> import kwant
    >>>
    >>> def hopping(a, b, t, peierls):
    >>>     return -t * peierls(a, b)
    >>>
    >>> syst = make_system(hopping)
    >>> lead = make_lead(hopping)
    >>> lead.substituted(peierls='peierls_lead')
    >>> syst.attach_lead(lead)
    >>> syst = syst.finalized()
    >>>
    >>> gauge = kwant.physics.magnetic_gauge(syst)
    >>>
    >>> def B_syst(pos):
    >>>    return np.exp(-np.sum(pos * pos))
    >>>
    >>> peierls_syst, peierls_lead = gauge(B_syst, 0)
    >>>
    >>> params = dict(t=1, peierls=peierls_syst, peierls_lead=peierls_lead)
    >>> kwant.hamiltonian_submatrix(syst, params=params)
    """

    def __init__(self, syst):
        if isinstance(syst, builder.FiniteSystem):
            if syst.leads:
                loops, which_patch, extended_sites = _loops_in_composite(syst)
                lead_gauges = [magnetic_gauge(lead) for lead in syst.leads]
                self._peierls = partial(_peierls_composite, syst,
                                        loops, which_patch,
                                        extended_sites, lead_gauges)
            else:
                loops = _loops_in_finite(syst)
                self._peierls = partial(_peierls_finite, syst, loops)
        elif isinstance(syst, builder.InfiniteSystem):
            loops, extended_sites = _loops_in_infinite(syst)
            self._peierls = partial(_peierls_infinite, syst,
                                    loops, extended_sites)
        else:
            raise TypeError('Expected a finalized Builder')

    def __call__(self, syst_field, *lead_fields, tol=1E-8, average=False):
        """Return the Peierls phase for a magnetic field configuration.

        Parameters
        ----------
        syst_field : scalar, vector or callable
            The magnetic field to apply to the scattering region.
            If callable, takes a position and returns the
            magnetic field at that position. Can be a scalar if
            the system is 1D or 2D, otherwise must be a vector.
            Magnetic field is expressed in units :math:`φ₀ / l²`,
            where :math:`φ₀` is the magnetic flux quantum and
            :math:`l` is the unit of length.
        *lead_fields : scalar, vector or callable
            The magnetic fields to apply to each of the leads, in
            the same format as 'syst_field'. In addition, if a callable
            is provided, then the magnetic field must have the symmetry
            of the associated lead.
        tol : float, default: 1E-8
            The tolerance to which to calculate the flux through each
            hopping loop in the system.
        average : bool, default: False
            If True, estimate the magnetic flux through each hopping loop
            in the system by evaluating the magnetic field at a single
            position inside the loop and multiplying it by the area of the
            loop. If False, then ``scipy.integrate.quad`` is used to integrate
            the magnetic field. This parameter is only used when 'syst_field'
            or 'lead_fields' are callable.

        Returns
        -------
        phases : callable, or sequence of callables
            The first callable computes the Peierls phase in the scattering
            region and the remaining callables compute the Peierls phases
            in each of the leads. Each callable takes a pair of
            `~kwant.builder.Site` (a hopping) and returns a unit complex
            number (Peierls phase) that multiplies that hopping.
        """
        return self._peierls(syst_field, *lead_fields, tol=tol, average=False)
