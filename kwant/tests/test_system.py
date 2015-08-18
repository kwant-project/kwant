# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np
from scipy import sparse
from nose.tools import assert_raises
import kwant

def test_hamiltonian_submatrix():
    sys = kwant.Builder()
    gr = kwant.lattice.chain()
    for i in xrange(3):
        sys[gr(i)] = 0.5 * i
    for i in xrange(2):
        sys[gr(i), gr(i + 1)] = 1j * (i + 1)

    sys2 = sys.finalized()
    mat = sys2.hamiltonian_submatrix()
    assert mat.shape == (3, 3)
    # Sorting is required due to unknown compression order of builder.
    perm = np.argsort(sys2.onsite_hamiltonians)
    mat_should_be = np.array([[0, 1j, 0], [-1j, 0.5, 2j], [0, -2j, 1]])

    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)

    mat = sys2.hamiltonian_submatrix(sparse=True)
    assert sparse.isspmatrix_coo(mat)
    mat = mat.todense()
    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)

    mat = sys2.hamiltonian_submatrix((), perm[[0, 1]], perm[[2]])
    np.testing.assert_array_equal(mat, mat_should_be[:2, 2:3])

    mat = sys2.hamiltonian_submatrix((), perm[[0, 1]], perm[[2]], sparse=True)
    mat = mat.todense()
    np.testing.assert_array_equal(mat, mat_should_be[:2, 2:3])

    # Test for correct treatment of matrix input.
    sys = kwant.Builder()
    sys[gr(0)] = np.array([[0, 1j], [-1j, 0]])
    sys[gr(1)] = np.array([[1]])
    sys[gr(2)] = np.array([[2]])
    sys[gr(1), gr(0)] = np.array([[1, 2j]])
    sys[gr(2), gr(1)] = np.array([[3j]])
    sys2 = sys.finalized()
    mat_dense = sys2.hamiltonian_submatrix()
    mat_sp = sys2.hamiltonian_submatrix(sparse=True).todense()
    np.testing.assert_array_equal(mat_sp, mat_dense)

    # Test precalculation of modes.
    np.random.seed(5)
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[gr(0)] = np.zeros((2, 2))
    lead[gr(0), gr(1)] = np.random.randn(2, 2)
    sys.attach_lead(lead)
    sys2 = sys.finalized()
    smatrix = kwant.smatrix(sys2, .1).data
    sys3 = sys2.precalculate(.1, what='modes')
    smatrix2 = kwant.smatrix(sys3, .1).data
    np.testing.assert_almost_equal(smatrix, smatrix2)
    assert_raises(ValueError, kwant.solvers.default.greens_function, sys3, 0.2)

    # Test for shape errors.
    sys[gr(0), gr(2)] = np.array([[1, 2]])
    sys2 = sys.finalized()
    assert_raises(ValueError, sys2.hamiltonian_submatrix)
    assert_raises(ValueError, sys2.hamiltonian_submatrix, sparse=True)
    sys[gr(0), gr(2)] = 1
    sys2 = sys.finalized()
    assert_raises(ValueError, sys2.hamiltonian_submatrix)
    assert_raises(ValueError, sys2.hamiltonian_submatrix, sparse=True)

    # Test for passing parameters to hamiltonian matrix elements
    def onsite(site, p1, p2=0):
        return site.pos + p1 + p2

    def hopping(site1, site2, p1, p2=0):
        return p1 - p2

    sys = kwant.Builder()
    sys[(gr(i) for i in xrange(3))] = onsite
    sys[((gr(i), gr(i + 1)) for i in xrange(2))] = hopping
    sys2 = sys.finalized()
    mat = sys2.hamiltonian_submatrix((2, 1))
    mat_should_be = [[3, 1, 0], [1, 4, 1], [0, 1, 5]]

    # Sorting is required due to unknown compression order of builder.
    onsite_hamiltonians = mat.flat[::4]
    perm = np.argsort(onsite_hamiltonians)
    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)
