# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import kwant
from kwant.graph import slicer

def assert_sanity(graph, slices):
    # Slices must comprise all of the graph.
    slclist = [slices[j][i] for j in xrange(len(slices))
               for i in xrange(len(slices[j]))]
    slclist.sort()
    assert slclist == [i for i in xrange(graph.num_nodes)]

    # Nodes may only have neighbors in neighboring slices.
    for j in xrange(len(slices)):
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
        lat = kwant.lattice.Square()
        lead[(lat(0, i) for i in xrange(w))] = 0
        sys[(lat(j, i) for j in xrange(l) for i in xrange(w))] = 0
        for s in [lead, sys]:
            for delta in [(1, 0), (0, 1)]:
                s[s.possible_hoppings(delta, lat, lat)] = -1
        sys.attach_lead(lead)
        sys.attach_lead(lead.reversed())
        fsys = sys.finalized()

        slices = slicer.slice(fsys.graph,
                              fsys.lead_interfaces[0],
                              fsys.lead_interfaces[1])

        # In the rectangle case, the slicing is very constricted and
        # we know that all slices must have the same shape.
        assert len(slices) == l

        for j in xrange(l):
            assert len(slices[j]) == w

        assert_sanity(fsys.graph, slices)
