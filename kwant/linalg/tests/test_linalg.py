# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from kwant.linalg import (
    lu_factor, lu_solve, rcond_from_lu, gen_eig, schur,
    convert_r2c_schur, order_schur, evecs_from_schur, gen_schur,
    convert_r2c_gen_schur, order_gen_schur, evecs_from_gen_schur)
import numpy as np
from ._test_utils import _Random, assert_array_almost_equal

def test_gen_eig():
    def _test_gen_eig(dtype):
        rand = _Random()
        a = rand.randmat(4, 4, dtype)
        b = rand.randmat(4, 4, dtype)

        (alpha, beta, vl, vr) = gen_eig(a, b, True, True)

        assert_array_almost_equal(dtype, np.dot(np.dot(a, vr), beta),
                                  np.dot(np.dot(b, vr), alpha))
        assert_array_almost_equal(dtype,
                                  np.dot(beta, np.dot(np.conj(vl.T), a)),
                                  np.dot(alpha, np.dot(np.conj(vl.T), b)))

    _test_gen_eig(np.float32)
    _test_gen_eig(np.float64)
    _test_gen_eig(np.complex64)
    _test_gen_eig(np.complex128)
    #int should be propagated to float64
    _test_gen_eig(np.int32)

def test_lu():
    def _test_lu(dtype):
        rand = _Random()
        a = rand.randmat(4, 4, dtype)
        bmat = rand.randmat(4, 4, dtype)
        bvec = rand.randvec(4, dtype)

        lu = lu_factor(a)
        xmat = lu_solve(lu, bmat)
        xvec = lu_solve(lu, bvec)

        assert_array_almost_equal(dtype, np.dot(a, xmat), bmat)
        assert_array_almost_equal(dtype, np.dot(a, xvec), bvec)

    _test_lu(np.float32)
    _test_lu(np.float64)
    _test_lu(np.complex64)
    _test_lu(np.complex128)
    #int should be propagated to float64
    _test_lu(np.int32)

def test_rcond_from_lu():
    def _test_rcond_from_lu(dtype):
        rand = _Random()
        a = rand.randmat(10, 10, dtype)

        norm1_a = np.linalg.norm(a, 1)
        normI_a = np.linalg.norm(a, np.inf)

        lu = lu_factor(a)

        rcond1 = rcond_from_lu(lu, norm1_a, '1')
        rcondI = rcond_from_lu(lu, normI_a, 'I')

        err1 = abs(rcond1 -
                   1/(norm1_a * np.linalg.norm(np.linalg.inv(a), 1)))
        errI = abs(rcondI -
                   1/(normI_a * np.linalg.norm(np.linalg.inv(a), np.inf)))

        #rcond_from_lu returns an estimate for the reciprocal
        #condition number only; hence we shouldn't be too strict about
        #the assertions here
        #Note: in my experience the estimate is excellent for somewhat
        #larger matrices
        assert err1/rcond1 < 0.1
        assert errI/rcondI < 0.1

    _test_rcond_from_lu(np.float32)
    _test_rcond_from_lu(np.float64)
    _test_rcond_from_lu(np.complex64)
    _test_rcond_from_lu(np.complex128)
    #int should be propagated to float64
    _test_rcond_from_lu(np.int32)

def test_schur():
    def _test_schur(dtype):
        rand = _Random()
        a = rand.randmat(5, 5, dtype)

        t, q = schur(a)[:2]

        assert_array_almost_equal(dtype, np.dot(np.dot(q, t), np.conj(q.T)), a)

    _test_schur(np.float32)
    _test_schur(np.float64)
    _test_schur(np.complex64)
    _test_schur(np.complex128)
    #int should be propagated to float64
    _test_schur(np.int32)

def test_convert_r2c_schur():
    def _test_convert_r2c_schur(dtype):
        rand = _Random()
        a = rand.randmat(10, 10, dtype)

        t, q = schur(a)[:2]
        t2, q2 = convert_r2c_schur(t, q)

        assert_array_almost_equal(dtype, np.dot(np.dot(q, t), np.conj(q.T)), a)
        assert_array_almost_equal(dtype, np.dot(np.dot(q2, t2), np.conj(q2.T)),
                                  a)

    _test_convert_r2c_schur(np.float32)
    _test_convert_r2c_schur(np.float64)
    #in the complex case the function should actually just copy
    _test_convert_r2c_schur(np.complex64)
    _test_convert_r2c_schur(np.complex128)
    #int should be propagated to float64
    _test_convert_r2c_schur(np.int32)

def test_order_schur():
    def _test_order_schur(dtype):
        rand = _Random()
        a = rand.randmat(10, 10, dtype)

        t, q, ev = schur(a)

        t2, q2, ev2 = order_schur(lambda i: i>2 and i<7, t, q)

        assert_array_almost_equal(dtype, np.dot(np.dot(q, t), np.conj(q.T)), a)
        assert_array_almost_equal(dtype, np.dot(np.dot(q2, t2), np.conj(q2.T)),
                                  a)
        assert_array_almost_equal(dtype, np.sort(ev), np.sort(ev2))
        assert_array_almost_equal(dtype, np.sort(ev[3:7]), np.sort(ev2[:4]))

        sel = [False, False, 0, True, True, True, 1, False, False, False]

        t3, q3 = order_schur(sel, t, q)[:2]
        assert_array_almost_equal(dtype, np.dot(np.dot(q3, t3), np.conj(q3.T)),
                                  a)
        assert_array_almost_equal(dtype, t2, t3)
        assert_array_almost_equal(dtype, q2, q3)

    _test_order_schur(np.float32)
    _test_order_schur(np.float64)
    _test_order_schur(np.complex64)
    _test_order_schur(np.complex128)
    #int should be propagated to float64
    _test_order_schur(np.int32)

def test_evecs_from_schur():
    def _test_evecs_from_schur(dtype):
        rand = _Random()
        a = rand.randmat(5, 5, dtype)

        t, q, ev = schur(a)

        vl, vr = evecs_from_schur(t, q, select=None, left=True, right=True)

        assert_array_almost_equal(dtype, np.dot(vr, np.dot(np.diag(ev),
                                                    np.linalg.inv(vr))), a)
        assert_array_almost_equal(dtype, np.dot(np.linalg.inv(np.conj(vl.T)),
                                         np.dot(np.diag(ev), np.conj(vl.T))),
                                  a)

        select = np.array([True, True, False, False, False], dtype=bool)

        vl, vr = evecs_from_schur(t, q, select, left=True, right=True)

        assert vr.shape[1] == 2
        assert vl.shape[1] == 2
        assert_array_almost_equal(dtype, np.dot(a, vr),
                                  np.dot(vr, np.diag(ev[select])))
        assert_array_almost_equal(dtype, np.dot(vl.T.conj(), a),
                                  np.dot(np.diag(ev[select]), vl.T.conj()))

        vl, vr = evecs_from_schur(t, q, lambda i: i<2, left=True, right=True)

        assert vr.shape[1] == 2
        assert vl.shape[1] == 2
        assert_array_almost_equal(dtype, np.dot(a, vr),
                                  np.dot(vr, np.diag(ev[select])))
        assert_array_almost_equal(dtype, np.dot(vl.T.conj(), a),
                                  np.dot(np.diag(ev[select]), vl.T.conj()))

    _test_evecs_from_schur(np.float32)
    _test_evecs_from_schur(np.float64)
    _test_evecs_from_schur(np.complex64)
    _test_evecs_from_schur(np.complex128)
    #int should be propagated to float64
    _test_evecs_from_schur(np.int32)

def test_gen_schur():
    def _test_gen_schur(dtype):
        rand = _Random()
        a = rand.randmat(5, 5, dtype)
        b = rand.randmat(5, 5, dtype)

        s, t, q, z = gen_schur(a, b)[:4]

        assert_array_almost_equal(dtype, np.dot(np.dot(q, s), z.T.conj()), a)
        assert_array_almost_equal(dtype, np.dot(np.dot(q, t), z.T.conj()), b)

    _test_gen_schur(np.float32)
    _test_gen_schur(np.float64)
    _test_gen_schur(np.complex64)
    _test_gen_schur(np.complex128)
    #int should be propagated to float64
    _test_gen_schur(np.int32)

def test_convert_r2c_gen_schur():
    def _test_convert_r2c_gen_schur(dtype):
        rand = _Random()
        a = rand.randmat(10, 10, dtype)
        b = rand.randmat(10, 10, dtype)

        s, t, q, z = gen_schur(a, b)[:4]
        s2, t2, q2, z2 = convert_r2c_gen_schur(s, t, q, z)

        assert_array_almost_equal(dtype, np.dot(np.dot(q, s), z.T.conj()), a)
        assert_array_almost_equal(dtype, np.dot(np.dot(q, t), z.T.conj()), b)
        assert_array_almost_equal(dtype, np.dot(np.dot(q2, s2), z2.T.conj()),
                                  a)
        assert_array_almost_equal(dtype, np.dot(np.dot(q2, t2), z2.T.conj()),
                                  b)

    _test_convert_r2c_gen_schur(np.float32)
    _test_convert_r2c_gen_schur(np.float64)
    #in the complex case the function should actually just copy
    _test_convert_r2c_gen_schur(np.complex64)
    _test_convert_r2c_gen_schur(np.complex128)
    #int should be propagated to float64
    _test_convert_r2c_gen_schur(np.int32)

def test_order_gen_schur():
    def _test_order_gen_schur(dtype):
        rand = _Random()
        a = rand.randmat(10, 10, dtype)
        b = rand.randmat(10, 10, dtype)

        s, t, q, z, alpha, beta = gen_schur(a, b)

        s2, t2, q2, z2, alpha2, beta2 = order_gen_schur(lambda i: i>2 and i<7,
                                                        s, t, q, z)

        assert_array_almost_equal(dtype, np.dot(np.dot(q, s), z.T.conj()), a)
        assert_array_almost_equal(dtype, np.dot(np.dot(q, t), z.T.conj()), b)
        assert_array_almost_equal(dtype, np.dot(np.dot(q2, s2), z2.T.conj()),
                                  a)
        assert_array_almost_equal(dtype, np.dot(np.dot(q2, t2), z2.T.conj()),
                                  b)

        #Sorting here is a bit tricky: For real matrices we expect
        #for complex conjugated pairs identical real parts - however
        #that seems messed up (only an error on the order of machine precision)
        #in the division. The solution here is to sort and compare the real
        #and imaginary parts separately. The only error that would not be
        #catched in this comparison is if the real and imaginary parts would
        #be assembled differently in the two arrays - an error that is highly
        #unlikely.
        assert_array_almost_equal(dtype, np.sort((alpha/beta).real),
                                  np.sort((alpha2/beta2).real))
        assert_array_almost_equal(dtype, np.sort((alpha/beta).imag),
                                  np.sort((alpha2/beta2).imag))
        assert_array_almost_equal(dtype, np.sort((alpha[3:7]/beta[3:7]).real),
                                  np.sort((alpha2[:4]/beta2[:4]).real))
        assert_array_almost_equal(dtype, np.sort((alpha[3:7]/beta[3:7]).imag),
                                  np.sort((alpha2[:4]/beta2[:4]).imag))

        sel = [False, False, 0, True, True, True, 1, False, False, False]

        s3, t3, q3, z3 = order_gen_schur(sel, s, t, q, z)[:4]
        assert_array_almost_equal(dtype, np.dot(np.dot(q3, s3), z3.T.conj()),
                                  a)
        assert_array_almost_equal(dtype, np.dot(np.dot(q3, t3), z3.T.conj()),
                                  b)
        assert_array_almost_equal(dtype, s2, s3)
        assert_array_almost_equal(dtype, t2, t3)
        assert_array_almost_equal(dtype, q2, q3)
        assert_array_almost_equal(dtype, z2, z3)

    _test_order_gen_schur(np.float32)
    _test_order_gen_schur(np.float64)
    _test_order_gen_schur(np.complex64)
    _test_order_gen_schur(np.complex128)
    #int should be propagated to float64
    _test_order_gen_schur(np.int32)


def test_evecs_from_gen_schur():
    def _test_evecs_from_gen_schur(dtype):
        rand = _Random()
        a = rand.randmat(5, 5, dtype)
        b = rand.randmat(5, 5, dtype)

        s, t, q, z, alpha, beta = gen_schur(a, b)

        vl, vr = evecs_from_gen_schur(s, t, q, z , select=None,
                                      left=True, right=True)

        assert_array_almost_equal(dtype, np.dot(a, np.dot(vr, np.diag(beta))),
                                  np.dot(b, np.dot(vr, np.diag(alpha))))
        assert_array_almost_equal(dtype,
                                  np.dot(np.dot(np.diag(beta), vl.T.conj()),
                                         a),
                                  np.dot(np.dot(np.diag(alpha), vl.T.conj()),
                                         b))

        select = np.array([True, True, False, False, False], dtype=bool)

        vl, vr = evecs_from_gen_schur(s, t, q, z, select,
                                      left=True, right=True)

        assert vr.shape[1] == 2
        assert vl.shape[1] == 2
        assert_array_almost_equal(dtype,
                                  np.dot(a, np.dot(vr,
                                                   np.diag(beta[select]))),
                                  np.dot(b, np.dot(vr,
                                                   np.diag(alpha[select]))))
        assert_array_almost_equal(dtype,
                                  np.dot(np.dot(np.diag(beta[select]),
                                                vl.T.conj()),
                                         a),
                                  np.dot(np.dot(np.diag(alpha[select]),
                                                vl.T.conj()),
                                         b))

        vl, vr = evecs_from_gen_schur(s, t, q, z, lambda i: i<2, left=True,
                                      right=True)

        assert vr.shape[1] == 2
        assert vl.shape[1] == 2
        assert_array_almost_equal(dtype,
                                  np.dot(a, np.dot(vr,
                                                   np.diag(beta[select]))),
                                  np.dot(b, np.dot(vr,
                                                   np.diag(alpha[select]))))
        assert_array_almost_equal(dtype,
                                  np.dot(np.dot(np.diag(beta[select]),
                                                vl.T.conj()),
                                         a),
                                  np.dot(np.dot(np.diag(alpha[select]),
                                                vl.T.conj()),
                                         b))

    _test_evecs_from_gen_schur(np.float32)
    _test_evecs_from_gen_schur(np.float64)
    _test_evecs_from_gen_schur(np.complex64)
    _test_evecs_from_gen_schur(np.complex128)
    #int should be propagated to float64
    _test_evecs_from_gen_schur(np.int32)
