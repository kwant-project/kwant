"""Wrapper for the graph library SCOTCH"""

__all__ = ['bisect', 'reset']

cimport libc.stdio
import numpy as np
cimport numpy as np
from kwant.graph cimport core
from . import core, defs
from kwant.graph.c_scotch cimport *

DEF SCOTCH_STRATQUALITY = 1
DEF SCOTCH_STRATSPEED = 2
DEF SCOTCH_STRATBALANCE = 4
DEF SCOTCH_STRATSAFETY = 8
DEF SCOTCH_STRATSCALABILITY = 16

def bisect(core.CGraph gr,
           double bal=0.0):
    """Compute a bisection of a CGraph using SCOTCH, minimizing the number of
    edges that are cut in the process. The bisection is returned as a numpy
    array (with a size given by the number of nodes in graph) indicating
    whether a node i is in part 0 or 1.

    The graph to be bisected must be undirected, i.e.for every edge (i,j)
    there must also be the edge (j,i). The result of applying bisect
    on a directed graph is undefined.

    The optional parameter bal defines how precise the bisection should be,
    i.e. the smaller bal, the more the two parts after the bisection are
    equally sized (however, this might affect the quality of the cut).
    """
    cdef SCOTCH_Graph graph
    cdef SCOTCH_Strat strat

    cdef np.ndarray[int, ndim=1] parts

    parts=np.empty(gr.num_nodes, dtype=defs.gint_dtype)

    SCOTCH_graphInit(&graph)

    SCOTCH_graphBuild(&graph, 0, gr.num_nodes,
                      <SCOTCH_Num *>gr.heads_idxs,
                      NULL, NULL, NULL, gr.heads_idxs[gr.num_nodes],
                      <SCOTCH_Num *>gr.heads,
                      NULL,)

    SCOTCH_stratInit(&strat)

    SCOTCH_stratGraphMapBuild(&strat,
                              SCOTCH_STRATQUALITY, 2, bal)

    SCOTCH_graphPart(&graph, 2,
                     &strat, <SCOTCH_Num *>parts.data)

    SCOTCH_stratExit(&strat)

    SCOTCH_graphExit(&graph)

    return parts

def reset():
    """Resets the internal random number generator of SCOTCH. After a reset,
    SCOTCH returns identical results for bisections, etc.
    """
    SCOTCH_randomReset()
