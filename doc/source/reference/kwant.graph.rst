:mod:`kwant.graph` -- Low-level, efficient directed graphs
==========================================================

.. module:: kwant.graph

Graphs, as handled by this module, consist of nodes (numbered by integers,
usually :math:`\geq 0`).  Pairs of nodes can be connected by edges (numbered by
integers :math:`\geq 0`).  An edge is described by a pair (tail, head) of node
numbers and is always directed.

The basic workflow is to

 (1) create an object of type `Graph`,

 (2) add edges to it using the methods `~Graph.add_edge` and
     `~Graph.add_edges`,

 (3) create a compressed copy of the graph using the method
     `~Graph.compressed`,

 (4) and use the thus created object for efficient queries.

Example:

>>> import kwant
>>> g = kwant.graph.Graph()
>>> g.add_edge(0, 1)
0
>>> g.add_edge(0, 2)
1
>>> g = g.compressed()
>>> list(g.out_neighbors(0))
[1, 2]

Node numbers can be assigned freely, but if they are not consecutive integers
starting with zero, storage space is wasted in the compressed graph.  Negative
node numbers are special and can be allowed optionally (see further).

Whenever a method returns multiple edges or nodes (via an iterator), they
appear in the order in which the edges associated with them were added to the
graph during construction.

Edge IDs are non-negative integers which identify edges unambiguously.  They
are assigned automatically when the graph is compressed.  The edge IDs of edges
with the same tail will occupy a dense interval of integers.  The IDs of edges
sharing the same tail will be assigned from lowest to highest in the order in
which these edges had been added.

The method `Graph.compressed` takes a parameter which determines whether the
graph will be one-way (the default) or two-way.  One-way graphs can be queried
for the existence of an edge and provide the nodes to which a node points
(=outgoing neighbors).  In addition, two-way graphs can be queried for the
nodes which point to a node (=incoming neighbors).

Another parameter of `Graph.compressed`, `edge_nr_translation`, determines
whether it will be possible to use the method `edge_id` of the compressed
graph.  This method returns the edge ID of an edge given the edge number that
was returned when an edge was added.

Negative node numbers can be allowed for a `Graph` (parameter
`allow_negative_nodes` of the constructor).  Edges with negative nodes are
considered to be dangling: negative nodes can be neighbors of other nodes, but
cannot be queried directly for neighbors.  Consequently, "doubly-dangling"
edges which connect two negative nodes do not make sense and are never allowed.
The range of values used for the negative node numbers does not influence the
required storage space in any way.

Compressed graphs have the read-only attributes `~kwant.CGraph.num_nodes` and
`~kwant.CGraph.num_edges`.

Graph types
-----------
.. autosummary::
   :toctree: generated/

   Graph
   CGraph

Other
-----
+----------------+------------------------------------------+
| ``gint_dtype`` | Data type used for graph nodes and edges |
+----------------+------------------------------------------+
