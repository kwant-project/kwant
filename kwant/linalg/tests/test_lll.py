# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from __future__ import division
import numpy as np
from kwant.linalg import lll

def test_lll():
    np.random.seed(1)
    for i in range(50):
        x = np.random.randint(4) + 1
        mat = np.random.randn(x, x + np.random.randint(2))
        c = 1.34 + .5 * np.random.rand()
        reduced_mat, coefs = lll.lll(mat)
        assert lll.is_c_reduced(reduced_mat, c)
        assert np.allclose(np.dot(mat.T, coefs), reduced_mat.T)


def test_cvp():
    np.random.seed(0)
    for i in range(10):
        mat = np.random.randn(4, 4)
        mat = lll.lll(mat)[0]
        for j in range(4):
            point = 50 * np.random.randn(4)
            assert np.array_equal(lll.cvp(point, mat, 10)[:3],
                                  lll.cvp(point, mat, 3))
