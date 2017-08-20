# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.


import numpy as np
from kwant.linalg import lll
from kwant._common import ensure_rng

def test_lll():
    rng = ensure_rng(1)
    for i in range(50):
        x = rng.randint(4) + 1
        mat = rng.randn(x, x + rng.randint(2))
        c = 1.34 + .5 * rng.random_sample()
        reduced_mat, coefs = lll.lll(mat)
        assert lll.is_c_reduced(reduced_mat, c)
        assert np.allclose(np.dot(mat.T, coefs), reduced_mat.T)


def test_cvp():
    rng = ensure_rng(0)
    for i in range(1, 5):
        for j in range(i, 5):
            mat = rng.randn(i, j)
            mat = lll.lll(mat)[0]
            for k in range(4):
                point = 50 * rng.randn(j)
                assert np.array_equal(lll.cvp(point, mat, 10)[:3],
                                      lll.cvp(point, mat, 3))

    # Test equidistant vectors
    # Cubic lattice
    basis = np.eye(3)
    vec = np.zeros((3))
    assert len(lll.cvp(vec, basis, n=2, group_by_length=True)) == 7
    assert len(lll.cvp(vec, basis, n=3, group_by_length=True)) == 19
    vec = 0.5 * np.array([1, 1, 1])
    assert len(lll.cvp(vec, basis, group_by_length=True)) == 8
    vec = 0.5 * np.array([1, 1, 0])
    assert len(lll.cvp(vec, basis, group_by_length=True)) == 4
    vec = 0.5 * np.array([1, 0, 0])
    assert len(lll.cvp(vec, basis, group_by_length=True)) == 2
    # Square lattice with offset
    offset = np.array([0, 0, 1])
    basis = np.eye(3)[:2]
    vec = np.zeros((3)) + rng.rand() * offset
    assert len(lll.cvp(vec, basis, n=2, group_by_length=True)) == 5
    assert len(lll.cvp(vec, basis, n=3, group_by_length=True)) == 9
    vec = 0.5 * np.array([1, 1, 0]) + rng.rand() * offset
    assert len(lll.cvp(vec, basis, group_by_length=True)) == 4
    vec = 0.5 * np.array([1, 0, 0]) + rng.rand() * offset
    assert len(lll.cvp(vec, basis, group_by_length=True)) == 2
    # Hexagonal lattice
    basis = np.array([[1, 0], [-0.5, 0.5 * np.sqrt(3)]])
    vec = np.zeros((2))
    assert len(lll.cvp(vec, basis, n=2, group_by_length=True)) == 7
    assert len(lll.cvp(vec, basis, n=3, group_by_length=True)) == 13
    vec = np.array([0.5, 0.5 / np.sqrt(3)])
    assert len(lll.cvp(vec, basis, group_by_length=True)) == 3
    assert len(lll.cvp(vec, basis, n=2, group_by_length=True)) == 6
    assert len(lll.cvp(vec, basis, n=3, group_by_length=True)) == 12
