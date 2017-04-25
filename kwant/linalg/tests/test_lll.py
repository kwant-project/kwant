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
