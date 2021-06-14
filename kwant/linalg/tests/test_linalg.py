# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.

import pytest
import numpy as np

from kwant.linalg import (
    lu_factor, lu_solve, rcond_from_lu, gen_eig, schur,
    convert_r2c_schur, order_schur, evecs_from_schur, gen_schur,
    convert_r2c_gen_schur, order_gen_schur, evecs_from_gen_schur)
from ._test_utils import _Random, assert_array_almost_equal


# int should always be propagated to float64
@pytest.fixture(scope='module', params=[
    np.float32, np.float64, np.complex64, np.complex128, np.int32
])
def dtype(request):
    return request.param


def test_gen_eig(dtype):
    rand = _Random()
    a = rand.randmat(4, 4, dtype)
    b = rand.randmat(4, 4, dtype)

    (alpha, beta, vl, vr) = gen_eig(a, b, True, True)

    assert_array_almost_equal(dtype, a @ vr @ beta, b @ vr @ alpha)
    assert_array_almost_equal(dtype, beta @ vl.T.conj() @ a,
                              alpha @ vl.T.conj() @ b)


def test_lu(dtype):
    rand = _Random()
    a = rand.randmat(4, 4, dtype)
    bmat = rand.randmat(4, 4, dtype)
    bvec = rand.randvec(4, dtype)

    lu = lu_factor(a)
    xmat = lu_solve(lu, bmat)
    xvec = lu_solve(lu, bvec)

    assert_array_almost_equal(dtype, a @ xmat, bmat)
    assert_array_almost_equal(dtype, a @ xvec, bvec)


def test_rcond_from_lu(dtype):
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


def test_schur(dtype):
    rand = _Random()
    a = rand.randmat(5, 5, dtype)

    t, q = schur(a)[:2]

    assert_array_almost_equal(dtype, q @ t @ q.T.conj(), a)


# in the complex case the function should actually just copy
def test_convert_r2c_schur(dtype):
    rand = _Random()
    a = rand.randmat(10, 10, dtype)

    t, q = schur(a)[:2]
    t2, q2 = convert_r2c_schur(t, q)

    assert_array_almost_equal(dtype, q @ t @ q.T.conj(), a)
    assert_array_almost_equal(dtype, q2 @ t2 @ q2.T.conj(), a)


def test_order_schur(dtype):
    rand = _Random()
    a = rand.randmat(10, 10, dtype)

    t, q, ev = schur(a)

    t2, q2, ev2 = order_schur(lambda i: i>2 and i<7, t, q)

    assert_array_almost_equal(dtype, q @ t @ q.T.conj(), a)
    assert_array_almost_equal(dtype, q2 @ t2 @ q2.T.conj(), a)
    assert_array_almost_equal(dtype, np.sort(ev), np.sort(ev2))
    assert_array_almost_equal(dtype, np.sort(ev[3:7]), np.sort(ev2[:4]))

    sel = [False, False, 0, True, True, True, 1, False, False, False]

    t3, q3 = order_schur(sel, t, q)[:2]
    assert_array_almost_equal(dtype, q3 @ t3 @ q3.T.conj(), a)
    assert_array_almost_equal(dtype, t2, t3)
    assert_array_almost_equal(dtype, q2, q3)


def test_evecs_from_schur(dtype):
    rand = _Random()
    a = rand.randmat(5, 5, dtype)

    t, q, ev = schur(a)

    vl, vr = evecs_from_schur(t, q, select=None, left=True, right=True)

    assert_array_almost_equal(dtype, vr @ np.diag(ev) @ np.linalg.inv(vr), a)
    assert_array_almost_equal(dtype, (np.linalg.inv(vl.T.conj())
                                      @ np.diag(ev) @ vl.T.conj()), a)

    select = np.array([True, True, False, False, False], dtype=bool)

    vl, vr = evecs_from_schur(t, q, select, left=True, right=True)

    assert vr.shape[1] == vl.shape[1] == 2
    assert_array_almost_equal(dtype, a @ vr, vr @ np.diag(ev[select]))
    assert_array_almost_equal(dtype, vl.T.conj() @ a,
                              np.diag(ev[select]) @ vl.T.conj())

    vl, vr = evecs_from_schur(t, q, lambda i: i<2, left=True, right=True)

    assert vr.shape[1] == vl.shape[1] == 2
    assert_array_almost_equal(dtype, a @ vr, vr @ np.diag(ev[select]))
    assert_array_almost_equal(dtype, vl.T.conj() @ a,
                              np.diag(ev[select]) @ vl.T.conj())


def test_gen_schur(dtype):
    rand = _Random()
    a = rand.randmat(5, 5, dtype)
    b = rand.randmat(5, 5, dtype)

    s, t, q, z = gen_schur(a, b)[:4]

    assert_array_almost_equal(dtype, q @ s @ z.T.conj(), a)
    assert_array_almost_equal(dtype, q @ t @ z.T.conj(), b)


# in the complex case the function should actually just copy
def test_convert_r2c_gen_schur(dtype):
    rand = _Random()
    a = rand.randmat(10, 10, dtype)
    b = rand.randmat(10, 10, dtype)

    s, t, q, z = gen_schur(a, b)[:4]
    s2, t2, q2, z2 = convert_r2c_gen_schur(s, t, q, z)

    assert_array_almost_equal(dtype, q @ s @ z.T.conj(), a)
    assert_array_almost_equal(dtype, q @ t @ z.T.conj(), b)
    assert_array_almost_equal(dtype, q2 @ s2 @ z2.T.conj(), a)
    assert_array_almost_equal(dtype, q2 @ t2 @ z2.T.conj(), b)


def test_order_gen_schur(dtype):
    rand = _Random()
    a = rand.randmat(10, 10, dtype)
    b = rand.randmat(10, 10, dtype)

    s, t, q, z, alpha, beta = gen_schur(a, b)

    s2, t2, q2, z2, alpha2, beta2 = order_gen_schur(lambda i: i>2 and i<7,
                                                    s, t, q, z)

    assert_array_almost_equal(dtype, q @ s @ z.T.conj(), a)
    assert_array_almost_equal(dtype, q @ t @ z.T.conj(), b)
    assert_array_almost_equal(dtype, q2 @ s2 @ z2.T.conj(), a)
    assert_array_almost_equal(dtype, q2 @ t2 @ z2.T.conj(), b)

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
    assert_array_almost_equal(dtype, q3 @ s3 @ z3.T.conj(), a)
    assert_array_almost_equal(dtype, q3 @ t3 @ z3.T.conj(), b)
    assert_array_almost_equal(dtype, s2, s3)
    assert_array_almost_equal(dtype, t2, t3)
    assert_array_almost_equal(dtype, q2, q3)
    assert_array_almost_equal(dtype, z2, z3)


def test_evecs_from_gen_schur(dtype):
    rand = _Random()
    a = rand.randmat(5, 5, dtype)
    b = rand.randmat(5, 5, dtype)

    s, t, q, z, alpha, beta = gen_schur(a, b)

    vl, vr = evecs_from_gen_schur(s, t, q, z , select=None,
                                    left=True, right=True)

    assert_array_almost_equal(dtype, a @ vr @ np.diag(beta),
                              b @ vr @ np.diag(alpha))
    assert_array_almost_equal(dtype, np.diag(beta) @ vl.T.conj() @ a,
                              np.diag(alpha) @ vl.T.conj() @ b)

    select = np.array([True, True, False, False, False], dtype=bool)

    vl, vr = evecs_from_gen_schur(s, t, q, z, select, left=True, right=True)

    assert vr.shape[1] == vl.shape[1] == 2
    assert_array_almost_equal(dtype, a @ vr @ np.diag(beta[select]),
                              b @ vr @ np.diag(alpha[select]))
    assert_array_almost_equal(dtype, np.diag(beta[select]) @ vl.T.conj() @ a,
                              np.diag(alpha[select]) @ vl.T.conj() @ b)

    vl, vr = evecs_from_gen_schur(s, t, q, z, lambda i: i<2, left=True,
                                    right=True)

    assert vr.shape[1] == vl.shape[1] == 2
    assert_array_almost_equal(dtype, a @ vr @ np.diag(beta[select]),
                              b @ vr @ np.diag(alpha[select]))
    assert_array_almost_equal(dtype, np.diag(beta[select]) @ vl.T.conj() @ a,
                              np.diag(alpha[select]) @ vl.T.conj() @ b)
