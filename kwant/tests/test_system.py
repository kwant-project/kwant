# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
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

    mat = sys2.hamiltonian_submatrix(perm[[0, 1]], perm[[2]])
    np.testing.assert_array_equal(mat, mat_should_be[:2, 2:3])

    mat = sys2.hamiltonian_submatrix(perm[[0, 1]], perm[[2]], sparse=True)
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

    # Test for shape errors.
    sys[gr(0), gr(2)] = np.array([[1, 2]])
    sys2 = sys.finalized()
    assert_raises(ValueError, sys2.hamiltonian_submatrix)
    assert_raises(ValueError, sys2.hamiltonian_submatrix, None, None, True)
    sys[gr(0), gr(2)] = 1
    sys2 = sys.finalized()
    assert_raises(ValueError, sys2.hamiltonian_submatrix)
    assert_raises(ValueError, sys2.hamiltonian_submatrix, None, None, True)
