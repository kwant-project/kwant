# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import pickle
import copy
from pytest import raises
import numpy as np
from scipy import sparse
import kwant
from kwant._common import ensure_rng


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
    perm = np.argsort([os[0] for os in syst2.onsites])
    mat_should_be = np.array([[0, 1j, 0], [-1j, 0.5, 2j], [0, -2j, 1]])

    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)

    mat = syst2.hamiltonian_submatrix(sparse=True)
    assert sparse.isspmatrix_coo(mat)
    mat = mat.toarray()
    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)

    mat = syst2.hamiltonian_submatrix((), perm[[0, 1]], perm[[2]])
    np.testing.assert_array_equal(mat, mat_should_be[:2, 2:3])

    mat = syst2.hamiltonian_submatrix((), perm[[0, 1]], perm[[2]], sparse=True)
    mat = mat.toarray()
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
    mat_sp = syst2.hamiltonian_submatrix(sparse=True).toarray()
    np.testing.assert_array_equal(mat_sp, mat_dense)

    # Test precalculation of modes.
    rng = ensure_rng(5)
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[chain(0)] = np.zeros((2, 2))
    lead[chain(0), chain(1)] = rng.randn(2, 2)
    syst.attach_lead(lead)
    syst2 = syst.finalized()
    smatrix = kwant.smatrix(syst2, .1).data
    syst3 = syst2.precalculate(.1, what='modes')
    smatrix2 = kwant.smatrix(syst3, .1).data
    np.testing.assert_almost_equal(smatrix, smatrix2)
    raises(ValueError, kwant.solvers.default.greens_function, syst3, 0.2)

    # Test for shape errors.
    syst[chain(0), chain(2)] = np.array([[1, 2]])
    syst2 = syst.finalized()
    raises(ValueError, syst2.hamiltonian_submatrix)
    raises(ValueError, syst2.hamiltonian_submatrix, sparse=True)
    syst[chain(0), chain(2)] = 1
    syst2 = syst.finalized()
    raises(ValueError, syst2.hamiltonian_submatrix)
    raises(ValueError, syst2.hamiltonian_submatrix, sparse=True)


def test_pickling():
    syst = kwant.Builder()
    lead = kwant.Builder(symmetry=kwant.TranslationalSymmetry([1.]))
    lat = kwant.lattice.chain()
    syst[lat(0)] = syst[lat(1)] = 0
    syst[lat(0), lat(1)] = 1
    lead[lat(0)] = syst[lat(1)] = 0
    lead[lat(0), lat(1)] = 1
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    syst_copy1 = copy.copy(syst).finalized()
    syst_copy2 = pickle.loads(pickle.dumps(syst)).finalized()
    syst = syst.finalized()
    syst_copy3 = copy.copy(syst)
    syst_copy4 = pickle.loads(pickle.dumps(syst))
    s = kwant.smatrix(syst, 0.1)
    for other in (syst_copy1, syst_copy2, syst_copy3, syst_copy4):
        assert np.all(kwant.smatrix(other, 0.1).data == s.data)
