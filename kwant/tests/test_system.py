# Copyright 2011-2019 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import pickle
import copy
import pytest
from pytest import raises
import numpy as np
from scipy import sparse
import kwant
from kwant._common import ensure_rng


@pytest.mark.parametrize("vectorize", [False, True])
def test_hamiltonian_submatrix(vectorize):
    syst = kwant.Builder(vectorize=vectorize)
    chain = kwant.lattice.chain(norbs=1)
    chain2 = kwant.lattice.chain(norbs=2)
    for i in range(3):
        syst[chain(i)] = 0.5 * i
    for i in range(2):
        syst[chain(i), chain(i + 1)] = 1j * (i + 1)

    syst2 = syst.finalized()
    mat = syst2.hamiltonian_submatrix()
    assert mat.shape == (3, 3)
    # Sorting is required due to unknown compression order of builder.
    if vectorize:
        _, (site_offsets, _) = syst2.subgraphs[0]
    else:
        site_offsets = [os[0] for os in syst2.onsites]
    perm = np.argsort(site_offsets)
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

    # Test for correct treatment of matrix input.
    syst = kwant.Builder(vectorize=vectorize)
    syst[chain2(0)] = np.array([[0, 1j], [-1j, 0]])
    syst[chain(1)] = np.array([[1]])
    syst[chain(2)] = np.array([[2]])
    syst[chain(1), chain2(0)] = np.array([[1, 2j]])
    syst[chain(2), chain(1)] = np.array([[3j]])
    syst2 = syst.finalized()
    mat_dense = syst2.hamiltonian_submatrix()
    mat_sp = syst2.hamiltonian_submatrix(sparse=True).toarray()
    np.testing.assert_array_equal(mat_sp, mat_dense)

    # Test precalculation of modes.
    rng = ensure_rng(5)
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)),
                         vectorize=vectorize)
    lead[chain2(0)] = np.zeros((2, 2))
    lead[chain2(0), chain2(1)] = rng.randn(2, 2)
    syst.attach_lead(lead)
    syst2 = syst.finalized()
    smatrix = kwant.smatrix(syst2, .1).data
    syst3 = syst2.precalculate(.1, what='modes')
    smatrix2 = kwant.smatrix(syst3, .1).data
    np.testing.assert_almost_equal(smatrix, smatrix2)
    raises(ValueError, kwant.solvers.default.greens_function, syst3, 0.2)

    # Test for shape errors.
    badly_shaped_hoppings = [
        1,
        [[1, 2]],  # shape (1, 2) instead of (2, 1)
        lambda a, b: 1,
        lambda a, b: [[1, 2]],
    ]
    for hopping in badly_shaped_hoppings:
        syst[chain2(0), chain(2)] = hopping
        with raises(ValueError):
            syst.finalized().hamiltonian_submatrix(sparse=False)
        with raises(ValueError):
            syst.finalized().hamiltonian_submatrix(sparse=True)


@pytest.mark.parametrize("vectorize", [False, True])
def test_pickling(vectorize):
    syst = kwant.Builder(vectorize=vectorize)
    lead = kwant.Builder(symmetry=kwant.TranslationalSymmetry([1.]),
                         vectorize=vectorize)
    lat = kwant.lattice.chain(norbs=1)
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
