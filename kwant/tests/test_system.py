# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np
from scipy import sparse
from nose.tools import assert_raises
import kwant

def test_hamiltonian_submatrix():
    syst = kwant.Builder()
    chain = kwant.lattice.chain()
    for i in range(3):
        syst[chain(i)] = 0.5 * i
    for i in range(2):
        syst[chain(i), chain(i + 1)] = 1j * (i + 1)

    syst2 = syst.finalized()
    mat = syst2.hamiltonian_submatrix()
    assert mat.shape == (3, 3)
    # Sorting is required due to unknown compression order of builder.
    perm = np.argsort(syst2.onsite_hamiltonians)
    mat_should_be = np.array([[0, 1j, 0], [-1j, 0.5, 2j], [0, -2j, 1]])

    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)

    mat = syst2.hamiltonian_submatrix(sparse=True)
    assert sparse.isspmatrix_coo(mat)
    mat = mat.todense()
    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)

    mat = syst2.hamiltonian_submatrix((), perm[[0, 1]], perm[[2]])
    np.testing.assert_array_equal(mat, mat_should_be[:2, 2:3])

    mat = syst2.hamiltonian_submatrix((), perm[[0, 1]], perm[[2]], sparse=True)
    mat = mat.todense()
    np.testing.assert_array_equal(mat, mat_should_be[:2, 2:3])

    # Test for correct treatment of matrix input.
    syst = kwant.Builder()
    syst[chain(0)] = np.array([[0, 1j], [-1j, 0]])
    syst[chain(1)] = np.array([[1]])
    syst[chain(2)] = np.array([[2]])
    syst[chain(1), chain(0)] = np.array([[1, 2j]])
    syst[chain(2), chain(1)] = np.array([[3j]])
    syst2 = syst.finalized()
    mat_dense = syst2.hamiltonian_submatrix()
    mat_sp = syst2.hamiltonian_submatrix(sparse=True).todense()
    np.testing.assert_array_equal(mat_sp, mat_dense)

    # Test precalculation of modes.
    np.random.seed(5)
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[chain(0)] = np.zeros((2, 2))
    lead[chain(0), chain(1)] = np.random.randn(2, 2)
    syst.attach_lead(lead)
    syst2 = syst.finalized()
    smatrix = kwant.smatrix(syst2, .1).data
    syst3 = syst2.precalculate(.1, what='modes')
    smatrix2 = kwant.smatrix(syst3, .1).data
    np.testing.assert_almost_equal(smatrix, smatrix2)
    assert_raises(ValueError, kwant.solvers.default.greens_function, syst3, 0.2)

    # Test for shape errors.
    syst[chain(0), chain(2)] = np.array([[1, 2]])
    syst2 = syst.finalized()
    assert_raises(ValueError, syst2.hamiltonian_submatrix)
    assert_raises(ValueError, syst2.hamiltonian_submatrix, sparse=True)
    syst[chain(0), chain(2)] = 1
    syst2 = syst.finalized()
    assert_raises(ValueError, syst2.hamiltonian_submatrix)
    assert_raises(ValueError, syst2.hamiltonian_submatrix, sparse=True)

    # Test for passing parameters to hamiltonian matrix elements
    def onsite(site, p1, p2=0):
        return site.pos + p1 + p2

    def hopping(site1, site2, p1, p2=0):
        return p1 - p2

    syst = kwant.Builder()
    syst[(chain(i) for i in range(3))] = onsite
    syst[((chain(i), chain(i + 1)) for i in range(2))] = hopping
    syst2 = syst.finalized()
    mat = syst2.hamiltonian_submatrix((2, 1))
    mat_should_be = [[3, 1, 0], [1, 4, 1], [0, 1, 5]]

    # Sorting is required due to unknown compression order of builder.
    onsite_hamiltonians = mat.flat[::4]
    perm = np.argsort(onsite_hamiltonians)
    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)
