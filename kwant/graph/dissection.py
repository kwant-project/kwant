"""Routines to compute nested dissections of graphs"""

__all__ = ['edge_dissection']

import numpy as np
from . import core, utils, scotch
from .defs import gint_dtype


def edge_dissection(gr, minimum_size, is_undirected=False):
    """Returns a nested dissection tree (represented as a nested number of
    tuples) for the graph gr, based on edge separators.

    minimum_size indicates the smallest size of a part for which the algorithm
    should still try to dissect the part.

    If the graph gr is already undirected, setting is_undirected=True avoids
    the work of making the graph undirected.
    """

    if isinstance(gr, core.Graph):
        grc = gr.compressed()
    elif isinstance(gr, core.CGraph):
        grc = gr
    else:
        raise ValueError('edge_dissection expects a Graph or CGraph!')

    # Unless the graph is undirected from the very beginning, make it
    # undirected.
    if not is_undirected:
        grc = utils.make_undirected(grc)

    return edge_bisection(grc, np.arange(grc.num_nodes, dtype=gint_dtype),
                          minimum_size)


def edge_bisection(grc, nodeids, minimum_size):
    """This function returns a nested dissection tree (represented as a nested
    number of tuples) for an undirected graph represented as a CGraph and ids
    (that go into the tree) given in nodeids.  minimum_size indicates the
    smallest size of a part for which the algorithm should still try to dissect
    the part.
    """
    parts = scotch.bisect(grc)

    # Count the number of nodes in parts 0 or 1.
    size2 = np.sum(parts)
    size1 = grc.num_nodes - size2

    # If the size of one of the parts is zero, we can't further dissect.
    if size1 == 0 or size2 == 0:
        return nodeids.tolist()

    # Now extract all nodes that are in part 0.
    sub_nodeids = nodeids[parts == 0]

    if size1 > minimum_size:
        subgr = utils.induced_subgraph(grc, parts == 0)
        left = edge_bisection(subgr, sub_nodeids, minimum_size)
    else:
        left = sub_nodeids.tolist()

    # Now extract all nodes that are in part 1.
    sub_nodeids = nodeids[parts == 1]

    if size2 > minimum_size:
        subgr = utils.induced_subgraph(grc, parts == 1)
        right = edge_bisection(subgr, sub_nodeids, minimum_size)
    else:
        right = sub_nodeids.tolist()

    return left, right
