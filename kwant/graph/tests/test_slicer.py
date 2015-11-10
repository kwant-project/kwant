# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import kwant
from kwant.graph import slicer

def assert_sanity(graph, slices):
    # Slices must comprise all of the graph.
    slclist = [slices[j][i] for j in range(len(slices))
               for i in range(len(slices[j]))]
    slclist.sort()
    assert slclist == [i for i in range(graph.num_nodes)]

    # Nodes may only have neighbors in neighboring slices.
    for j in range(len(slices)):
        for node in slices[j]:
            for neigh in graph.out_neighbors(node):
                if j > 0 and j < len(slices) - 1:
                    assert (neigh in slices[j] or
                            neigh in slices[j+1] or
                            neigh in slices[j-1])
                elif j == 0:
                    assert (neigh in slices[j] or
                            neigh in slices[j+1])
                else:
                    assert (neigh in slices[j] or
                            neigh in slices[j-1])


def test_rectangle():
    w = 5

    for l in [1, 2, 5, 10]:
        sys = kwant.Builder()
        lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
        lat = kwant.lattice.square()
        lead[(lat(0, i) for i in range(w))] = 0
        sys[(lat(j, i) for j in range(l) for i in range(w))] = 0
        for s in [lead, sys]:
            for kind in [kwant.builder.HoppingKind((1, 0), lat),
                         kwant.builder.HoppingKind((0, 1), lat)]:
                s[kind] = -1
        sys.attach_lead(lead)
        sys.attach_lead(lead.reversed())
        fsys = sys.finalized()

        slices = slicer.slice(fsys.graph,
                              fsys.lead_interfaces[0],
                              fsys.lead_interfaces[1])

        # In the rectangle case, the slicing is very constricted and
        # we know that all slices must have the same shape.
        assert len(slices) == l

        for j in range(l):
            assert len(slices[j]) == w

        assert_sanity(fsys.graph, slices)
