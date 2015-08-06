# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

try:
    from kwant.linalg.mumps import MUMPSContext, schur_complement
    no_mumps = False
except ImportError:
    no_mumps = True

from kwant.lattice import honeycomb
from kwant.builder import Builder, HoppingKind
from numpy.testing.decorators import skipif
import numpy as np
import scipy.sparse as sp
from ._test_utils import _Random, assert_array_almost_equal

@skipif(no_mumps)
def test_lu_with_dense():
    def _test_lu_with_dense(dtype):
        rand = _Random()
        a = rand.randmat(5, 5, dtype)
        bmat = rand.randmat(5, 5, dtype)
        bvec = rand.randvec(5, dtype)

        ctx = MUMPSContext()
        ctx.factor(sp.coo_matrix(a))

        xvec = ctx.solve(bvec)
        xmat = ctx.solve(bmat)

        assert_array_almost_equal(dtype, np.dot(a, xmat), bmat)
        assert_array_almost_equal(dtype, np.dot(a, xvec), bvec)

        # now "sparse" right hand side

        xvec = ctx.solve(sp.csc_matrix(bvec.reshape(5,1)))
        xmat = ctx.solve(sp.csc_matrix(bmat))

        assert_array_almost_equal(dtype, np.dot(a, xmat), bmat)
        assert_array_almost_equal(dtype, np.dot(a, xvec),
                                  bvec.reshape(5,1))

    _test_lu_with_dense(np.complex128)


@skipif(no_mumps)
def test_schur_complement_with_dense():
    def _test_schur_complement_with_dense(dtype):
        rand = _Random()
        a = rand.randmat(10, 10, dtype)
        s = schur_complement(sp.coo_matrix(a), list(range(3)))
        assert_array_almost_equal(dtype, np.linalg.inv(s),
                                  np.linalg.inv(a)[:3, :3])

    _test_schur_complement_with_dense(np.complex128)


@skipif(no_mumps)
def test_error_minus_9(r=10):
    """Test if MUMPSError -9 is properly caught by increasing memory"""

    graphene = honeycomb()
    a, b = graphene.sublattices

    def circle(pos):
        x, y = pos
        return x**2 + y**2 < r**2

    sys = Builder()
    sys[graphene.shape(circle, (0,0))] = -0.0001
    for kind in [((0, 0), b, a), ((0, 1), b, a), ((-1, 1), b, a)]:
        sys[HoppingKind(*kind)] = - 1

    ham = sys.finalized().hamiltonian_submatrix(sparse=True)

    # No need to check result, it's enough if no exception is raised
    MUMPSContext().factor(ham)
