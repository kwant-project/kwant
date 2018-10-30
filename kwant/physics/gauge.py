# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.
"""Functions for fixing the magnetic gauge automatically in a Kwant system."""

import functools as ft

import numpy as np
import scipy
from scipy.integrate import dblquad

from .. import system, builder

from ..graph.dijkstra import dijkstra_directed

__all__ = ['magnetic_gauge']


### Integation

# Integrate vector field over triangle, for internal use by 'surface_integral'
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


def surface_integral(f, loop, tol=1e-8, average=False):
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

def find_loops(graph, subgraph):
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
    assert same_sparsity_structure(subgraph, graph)

    # Links in graph, but not in subgraph.
    links_to_find = scipy.sparse.triu(graph - subgraph).tocoo()
    links_to_find = np.vstack((links_to_find.row, links_to_find.col)).transpose()

    links_to_find, min_length = order_links(subgraph, links_to_find)

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
            links_to_find, min_length = order_links(subgraph, links_to_find)
        else:
            # Assumes that 'graph' and 'subgraph' have the same sparsity structure.
            assign_csr(subgraph, graph, (frm, to))
            assign_csr(subgraph, graph, (to, frm))
            loops.append(path)
            links_to_find = links_to_find[1:]

    return loops


def order_links(subgraph, links_to_find):
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

def assign_csr(a, b, element):
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


def same_sparsity_structure(a, b):
    a = a.tocsr().sorted_indices()
    b = b.tocsr().sorted_indices()
    return (np.array_equal(a.indices, b.indices)
            and np.array_equal(a.indptr, b.indptr))


def add_coo_matrices(*mats, shape):
    """Add a sequence of COO matrices by appending their constituent arrays."""
    values = np.hstack([mat.data for mat in mats])
    rows = np.hstack([mat.row for mat in mats])
    cols = np.hstack([mat.col for mat in mats])
    return scipy.sparse.coo_matrix((values, (rows, cols)), shape=shape)


def shift_diagonally(mat, shift, shape):
    """Shift the row/column indices of a COO matrix."""
    return scipy.sparse.coo_matrix(
        (mat.data, (mat.row + shift, mat.col + shift)),
        shape=shape)


def distance_matrix(links, pos, shape):
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


def loops_in_finite(syst):
    """Find the loops in a finite system with no leads.

    The site indices in the returned loops are those of the system,
    so they may be used as indices to 'syst.sites', or with 'syst.pos'.
    """
    assert isinstance(syst, system.FiniteSystem) and syst.leads == []
    nsites = len(syst.sites)

    # Fix the gauge across the minimum spanning tree of the system graph.
    graph = distance_matrix(list(syst.graph),
                            pos=syst.pos, shape=(nsites, nsites))
    spanning_tree = shortest_distance_forest(graph)
    return find_loops(graph, spanning_tree)


def shortest_distance_forest(graph):
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
                assign_csr(tree, graph, (i, p))
                assign_csr(tree, graph, (p, i))
    return tree


### Phase calculation

def calculate_phases(loops, pos, previous_phase, flux):
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
        the sum of the phases across each link in the loop, *except* the link
        between the first and last site in the loop.
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
        phases[tail, head] = integral - previous_phase(phases, loop)
    return phases


# These functions are to be used with 'calculate_phases'.
# 'phases' always stores *either* the phase across (i, j) *or*
# (j, i), and never both. If a phase is not present it is assumed to
# be zero.


def _previous_phase_finite(phases, path):
    previous_phase = 0
    for i, j in zip(path, path[1:]):
        previous_phase += phases.get((i, j), 0)
        previous_phase -= phases.get((j, i), 0)
    return previous_phase


### High-level interface
#
# These functions glue all the above functionality together.

# Wrapper for phase dict that takes high-level sites
def _finite_wrapper(syst, phases, a, b):
    i = syst.id_by_site[a]
    j = syst.id_by_site[b]
    # We only store *either* (i, j) *or* (j, i). If not present
    # then the phase is zero by definition.
    return phases.get((i, j), -phases.get((j, i), 0))


def _gauge_finite(syst):
    loops = loops_in_finite(syst)

    def _gauge(syst_field, tol=1E-8, average=False):
        integrate = ft.partial(surface_integral, syst_field,
                               tol=tol, average=average)
        phases = calculate_phases(
            loops,
            syst.pos,
            _previous_phase_finite,
            integrate,
        )
        return ft.partial(_finite_wrapper, syst, phases)

    return _gauge


def magnetic_gauge(syst):
    """Fix the magnetic gauge for a finalized system.

    Fix the magnetic gauge for a Kwant system. This can
    be used to later calculate the Peierls phases that
    should be applied to each hopping, given a magnetic field.

    Parameters
    ----------
    syst : kwant.builder.FiniteSystem
        May not have leads attached (this restriction will
        be lifted in the future).

    Returns
    -------
    gauge : callable
        When called with a magnetic field as argument, returns
        another callable 'phase' that returns the Peierls phase to
        apply to a given hopping.

    Examples
    --------
    The following illustrates basic usage:

    >>> import numpy as np
    >>> import kwant
    >>>
    >>> def hopping(a, b, t, phi):
    >>>     return -t * np.exp(-1j * phi(a, b))
    >>>
    >>> syst = make_system(hopping).finalized()
    >>> gauge = kwant.physics.magnetic_gauge(syst)
    >>>
    >>> def B(pos):
    >>>    return np.exp(-np.sum(pos * pos))
    >>>
    >>> kwant.hamiltonian_submatrix(syst, params=dict(t=1, phi=gauge(B))
    """
    if isinstance(syst, builder.FiniteSystem):
        if syst.leads:
            raise ValueError('Can only fix magnetic gauge for finite systems '
                             'without leads')
        else:
            return _gauge_finite(syst)
    else:
        raise TypeError('Can only fix magnetic gauge for finite systems '
                        'without leads')
