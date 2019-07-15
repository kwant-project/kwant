# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import functools as ft
from collections import deque
import pickle
import numpy as np
import tinyarray as ta
import numpy.linalg as la
from scipy.sparse import coo_matrix
import pytest
from pytest import raises
# needed to get round odd bug in test_mask_interpolate
from contextlib import contextmanager

import kwant
from kwant import operator as ops


sigma0 = np.array([[1, 0], [0, 1]])
sigmax = np.array([[0, 1], [1, 0]])
sigmay = np.array([[0, -1j], [1j, 0]])
sigmaz = np.array([[1, 0], [0, -1]])

opservables = (ops.Density, ops.Current, ops.Source)


def _random_square_system(N, norbs=1):

    def random_onsite(i):
        return 2 + kwant.digest.uniform(i.tag)

    def random_hopping(i, j):
        return -1 + kwant.digest.uniform(i.tag + j.tag)

    lat = kwant.lattice.square(norbs=norbs)
    syst = kwant.Builder()
    syst[(lat(i, j) for i in range(N) for j in range(N))] = random_onsite
    syst[lat.neighbors()] = random_hopping
    return lat, syst


def _perfect_lead(N, norbs=1):
    lat = kwant.lattice.square(norbs=norbs)
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    syst[(lat(0, j) for j in range(N))] = 2
    syst[lat.neighbors()] = -1
    return  lat, syst


def test_operator_construction():
    lat, syst = _random_square_system(3)
    fsyst = syst.finalized()
    N = len(fsyst.sites)

    # test construction failure if norbs not given
    latnone = kwant.lattice.chain()
    syst[latnone(0)] = 1
    for A in opservables:
        raises(ValueError, A, syst.finalized())
    del syst[latnone(0)]

    # test construction failure when dimensions of onsite do not match
    for A in opservables:
        raises(ValueError, A, fsyst, onsite=np.eye(2))

    # test that error is raised when input array is the wrong size
    for A in opservables:
        a = A(fsyst)
        kets = list(map(np.ones, [(0,), (N - 1,), (N + 1,), (N, 1)]))
        for ket in kets:
            raises(ValueError, a, ket)
            raises(ValueError, a, ket, ket)
            raises(ValueError, a.act, ket)

    # Test failure on non-hermitian
    for A in (ops.Density, ops.Current, ops.Source):
        raises(ValueError, A, fsyst, 1j)

    # Test output dtype
    ket = np.ones(len(fsyst.sites))
    for A in (ops.Density, ops.Current, ops.Source):
        a = A(fsyst)
        a_nonherm = A(fsyst, check_hermiticity=False)
        assert a(ket, ket).dtype == np.complex128
        assert a(ket).dtype == np.float64
        assert a_nonherm(ket, ket).dtype == np.complex128
        assert a_nonherm(ket).dtype == np.complex128

    # test construction with different numbers of orbitals
    lat2 = kwant.lattice.chain(norbs=2)
    extra_sites = [lat2(i) for i in range(N)]
    syst[extra_sites] = np.eye(2)
    syst[zip(fsyst.sites, extra_sites)] = ta.matrix([1, 1])
    for A in opservables:
        raises(ValueError, A, syst.finalized(), onsite=np.eye(2))
        A(syst.finalized())
        A.onsite == np.eye(2)
    del syst[extra_sites]

    check = [(ops.Density, np.arange(N).reshape(-1, 1), ta.identity(1)),
             (ops.Current, np.array(list(fsyst.graph)), ta.identity(1)),
             (ops.Source, np.arange(N).reshape(-1, 1), ta.identity(1))]

    # test basic construction
    for A, where, onsite in check:
        a = A(fsyst)
        assert np.all(np.asarray(a.where) == where)
        assert all(a.onsite == onsite for i in range(N))

    # test construction with dict `onsite`
    for A in opservables:
        B = A(fsyst, {lat: 1})
        assert all(B.onsite(i) == 1 for i in range(N))

    # test construction with a functional onsite
    for A in opservables:
        B = A(fsyst, lambda site: site.pos[0])  # x-position operator
        assert all(B.onsite(i) == fsyst.sites[i].pos[0] for i in range(N))

    # test construction with `where` given by a sequence
    where = [lat(2, 2), lat(1, 1)]
    fwhere = tuple(fsyst.id_by_site[s] for s in where)
    A = ops.Density(fsyst, where=where)
    assert np.all(np.asarray(A.where).reshape(-1) == fwhere)

    where = [(lat(2, 2), lat(1, 2)), (lat(0, 0), lat(0, 1))]
    fwhere = np.asarray([(fsyst.id_by_site[a], fsyst.id_by_site[b])
              for a, b in where])
    A = ops.Current(fsyst, where=where)
    assert np.all(np.asarray(A.where) == fwhere)

    # test construction with `where` given by a function
    tag_list = [(1, 0), (1, 1), (1, 2)]
    def where(site):
        return site.tag in tag_list
    A = ops.Density(fsyst, where=where)
    assert all(fsyst.sites[A.where[w, 0]].tag in tag_list
               for w in range(A.where.shape[0]))

    where_list = set(kwant.HoppingKind((1, 0), lat)(syst))
    fwhere_list = set((fsyst.id_by_site[a], fsyst.id_by_site[b])
                      for a, b in where_list)
    def where(a, b):
        return (a, b) in where_list
    A = ops.Current(fsyst, where=where)
    assert all((a, b) in fwhere_list for a, b in A.where)

    # test that `sum` is passed to constructors correctly
    for A in opservables:
        A(fsyst, sum=True).sum == True


def _test(A, bra, ket=None, per_el_val=None, reduced_val=None, params=None):
    if per_el_val is not None:
        val = A(bra, ket, params=params)
        assert np.allclose(val, per_el_val)
        # with bound args
        val = A.bind(params=params)(bra, ket)
        assert np.allclose(val, per_el_val)
    # test that inner products give the same thing
    ket = bra if ket is None else ket
    act_val = np.dot(bra.conj(), A.act(ket, params=params))
    inner_val = np.sum(A(bra, ket, params=params))
    # check also when sum is done internally by operator
    try:
        sum_reset = A.sum
        A.sum = True
        sum_inner_val = A(bra, ket, params=params)
        assert inner_val == sum_inner_val
    finally:
        A.sum = sum_reset

    assert np.isclose(act_val, inner_val)
    assert np.isclose(sum_inner_val, inner_val)
    if reduced_val is not None:
        assert np.isclose(inner_val, reduced_val)


def test_opservables_finite():
    lat, syst = _random_square_system(3)
    fsyst = syst.finalized()
    ev, wfs = la.eigh(fsyst.hamiltonian_submatrix())

    Q = ops.Density(fsyst)
    Qtot = ops.Density(fsyst, sum=True)
    J = ops.Current(fsyst)
    K = ops.Source(fsyst)

    for i, wf in enumerate(wfs.T):  # wfs[:, i] is i'th eigenvector
        assert np.allclose(Q.act(wf), wf)  # this operation is identity
        _test(Q, wf, reduced_val=1)  # eigenvectors are normalized
        _test(Qtot, wf, per_el_val=1)  # eigenvectors are normalized
        _test(J, wf, per_el_val=0)  # time-reversal symmetry: no current
        _test(K, wf, per_el_val=0)  # onsite commutes with hamiltonian

    # check that we get correct (complex) output
    for bra, ket in zip(wfs.T, wfs.T):
        _test(Q, bra, ket, per_el_val=(bra * ket))

    # check with get_hermiticity=False
    Qi = ops.Density(fsyst, 1j, check_hermiticity=False)
    for wf in wfs.T:  # wfs[:, i] is i'th eigenvector
        assert np.allclose(Qi.act(wf), 1j * wf)

    # test with different numbers of orbitals
    lat2 = kwant.lattice.chain(norbs=2)
    extra_sites = [lat2(i) for i in range(len(fsyst.sites))]
    syst[extra_sites] = np.eye(2)
    syst[zip(fsyst.sites, extra_sites)] = ta.matrix([1, 1])
    fsyst = syst.finalized()
    ev, wfs = la.eigh(fsyst.hamiltonian_submatrix())

    Q = ops.Density(fsyst)
    Qtot = ops.Density(fsyst, sum=True)
    J = ops.Current(fsyst)
    K = ops.Source(fsyst)

    for wf in wfs.T:  # wfs[:, i] is i'th eigenvector
        assert np.allclose(Q.act(wf), wf)  # this operation is identity
        _test(Q, wf, reduced_val=1)  # eigenvectors are normalized
        _test(Qtot, wf, per_el_val=1)  # eigenvectors are normalized
        _test(J, wf, per_el_val=0)  # time-reversal symmetry: no current
        _test(K, wf, per_el_val=0)  # onsite commutes with hamiltonian


def test_opservables_infinite():
    lat, lead = _perfect_lead(3)
    flead = lead.finalized()

    # Cannot calculate current between unit cells because the wavefunction
    # is only defined over a single unit cell
    raises(ValueError, ops.Current, flead, where=[(lat(0, j), lat(1, j))
                                                   for j in range(3)])

    transverse_kind = kwant.builder.HoppingKind((0, 1), lat)
    J_intra = ops.Current(flead, where=list(transverse_kind(lead)))

    prop, _ = flead.modes(energy=1.)
    for wf, v in zip(prop.wave_functions.T, prop.velocities):
        _test(J_intra, wf, per_el_val=0)  # no transverse current


def test_opservables_scattering():
    # Disordered system with two ordered strips on the left/right.  We check
    # that the current on the right of the disorder due to incoming mode `m` is
    # equal to Σ_n |t_nm|^2. Similarly the current on the left of the disorder
    # is checked against 1 - Σ_n |r_nm|^2

    N = 10
    lat, syst = _random_square_system(N)
    # add extra sites so we can calculate the current in a region
    # where there is no backscattering
    syst[(lat(i, j) for i in [-1, N] for j in range(N))] = 2
    syst[((lat(-1, j), lat(0, j)) for j in range(N))] = -1
    syst[((lat(N-1, j), lat(N, j)) for j in range(N))] = -1

    lat, lead = _perfect_lead(3)
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    fsyst = syst.finalized()

    # currents on the left and right of the disordered region
    right_interface = [(lat(N, j), lat(N-1, j)) for j in range(3)]
    left_interface = [(lat(0, j), lat(-1, j)) for j in range(3)]
    J_right = ops.Current(fsyst, where=right_interface)
    J_right_tot = ops.Current(fsyst, where=right_interface, sum=True)
    J_left = ops.Current(fsyst, where=left_interface)
    J_left_tot = ops.Current(fsyst, where=left_interface, sum=True)

    smatrix = kwant.smatrix(fsyst, energy=1.0)
    t = smatrix.submatrix(1, 0).T  # want to iterate over the columns
    r = smatrix.submatrix(0, 0).T  # want to iterate over the columns
    wfs = kwant.wave_function(fsyst, energy=1.0)(0)
    for rv, tv, wf in zip(r, t, wfs):
        _test(J_right, wf, reduced_val=np.sum(np.abs(tv)**2))
        _test(J_right_tot, wf, per_el_val=np.sum(np.abs(tv)**2))
        _test(J_left, wf, reduced_val=(1 - np.sum(np.abs(rv)**2)))
        _test(J_left_tot, wf, per_el_val=(1 - np.sum(np.abs(rv)**2)))


def test_opservables_spin():

    def onsite(site, B):
        return 2 * np.eye(2) + B * sigmaz

    L = 20
    lat = kwant.lattice.chain(norbs=2)
    syst = kwant.Builder()
    syst[(lat(i) for i in range(L))] = onsite
    syst[lat.neighbors()] = -1 * np.eye(2)
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = onsite
    lead[lat.neighbors()] = -1 * np.eye(2)
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    fsyst = syst.finalized()
    params = dict(B=0.1)
    down, up = kwant.wave_function(fsyst, energy=1., params=params)(0)

    x_hoppings = kwant.builder.HoppingKind((1,), lat)
    spin_current_z = ops.Current(fsyst, sigmaz, where=x_hoppings(syst))
    _test(spin_current_z, up, params=params, per_el_val=1)
    _test(spin_current_z, down, params=params, per_el_val=-1)

    # calculate spin_x torque
    spin_torque_x = ops.Source(fsyst, sigmax, where=[lat(L//2)])
    i = fsyst.id_by_site[lat(L//2)]
    psi = up[2*i:2*(i+1)] + down[2*i:2*(i+1)]
    H_ii = onsite(None, **params)
    K = np.dot(H_ii, sigmax) - np.dot(sigmax, H_ii)
    expect = 1j * ft.reduce(np.dot, (psi.conj(), K, psi))
    _test(spin_torque_x, up+down, params=params, reduced_val=expect)


def test_opservables_gauged():
    # Test that we get the same answer when we apply a random
    # gauge (unitary) transformation to each site. We also
    # adjust our definition of the current to match

    # this is to get round a bug in test_mask_interpolate (?!) that fails when
    # the random number state is altered.
    @contextmanager
    def save_random_state():
        old_state = np.random.get_state()
        yield
        np.random.set_state(old_state)

    L = 20
    with save_random_state():
        Us = deque([kwant.rmt.circular(2) for i in range(L)])
    # need these to get the coupling to the leads right
    Us.append(np.eye(2))
    Us.appendleft(np.eye(2))

    H0 = 2 * np.eye(2) + 0.1 * sigmaz  # onsite
    V0 = -1 * np.eye(2)  # hopping

    lat = kwant.lattice.chain(norbs=2)
    syst = kwant.Builder()

    for i, U in enumerate(Us):
        syst[lat(i)] = ft.reduce(np.dot, (U, H0, U.conjugate().transpose()))

    for a, b in kwant.builder.HoppingKind((1,), lat)(syst):
        i, j = a.tag[0], b.tag[0]
        syst[(a, b)] = ft.reduce(np.dot,
                                 (Us[i], V0, Us[j].conjugate().transpose()))

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1,)))
    lead[lat(0)] = H0
    lead[lat.neighbors()] = V0
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    fsyst = syst.finalized()
    down, up = kwant.wave_function(fsyst, energy=1.0)(0)

    def M_a(site):
        i = site.tag[0]
        return ft.reduce(np.dot,
                         (Us[i], sigmaz, Us[i].conjugate().transpose()))

    x_hoppings = kwant.builder.HoppingKind((1,), lat)
    spin_current_gauge = ops.Current(fsyst, M_a, where=x_hoppings(syst))
    _test(spin_current_gauge, up, per_el_val=1)
    _test(spin_current_gauge, down, per_el_val=-1)
    # check the reverse is also true
    minus_x_hoppings = kwant.builder.HoppingKind((-1,), lat)
    spin_current_gauge = ops.Current(fsyst, M_a, where=minus_x_hoppings(syst))
    _test(spin_current_gauge, up, per_el_val=-1)
    _test(spin_current_gauge, down, per_el_val=1)


def test_tocoo():
    syst = kwant.Builder()
    lat1 = kwant.lattice.chain(norbs=1)
    syst[lat1(0)] = syst[lat1(1)] = 0
    syst = syst.finalized()

    op = ops.Density(syst)
    assert isinstance(op.tocoo(), coo_matrix)

    # Constant and non-constant values.
    assert np.all(op.tocoo().toarray() == np.eye(2))
    op = ops.Density(syst, lambda site: 1)
    assert np.all(op.tocoo().toarray() == np.eye(2))

    # Correct treatment of where
    op = ops.Density(syst, where=[lat1(0)])
    assert np.all(op.tocoo().toarray() == [[1, 0], [0, 0]])

    # No accidental transpose.
    syst = kwant.Builder()
    lat2 = kwant.lattice.chain(norbs=2)
    syst[lat2(0)] = lambda site, p: np.eye(2)
    syst = syst.finalized()
    op = ops.Density(syst, [[1, 1], [0, 1]], check_hermiticity=False)
    assert np.all(op.tocoo().toarray() == [[1, 1], [0, 1]])

    op = ops.Density(syst, lambda site, p: [[1, 1], [0, 1]],
                     check_hermiticity=False)
    op = op.bind(params=dict(p=1))
    raises(ValueError, op.tocoo, [1])


# We need to keep testing 'args', but we don't want to see
# all the deprecation warnings in the test logs
@pytest.mark.filterwarnings("ignore:.*'args' parameter")
@pytest.mark.parametrize("A", opservables)
def test_arg_passing(A):
    lat1 = kwant.lattice.chain(norbs=1)

    syst = kwant.Builder()
    syst[lat1(0)] = syst[lat1(1)] = lambda s0, a, b: s0.pos + a + b
    syst[lat1.neighbors()] = lambda s0, s1, a, b: a - b
    fsyst = syst.finalized()

    wf = np.ones(len(fsyst.sites))

    # test missing params
    op = A(fsyst, onsite=lambda x, a, b: 1)
    params = dict(a=1)
    with raises(TypeError):
        op(wf, params=params)
    with raises(TypeError):
        op.act(wf, params=params)
    with raises(TypeError):
        op.bind(params=params)
    if hasattr(op, 'tocoo'):
        with raises(TypeError):
            op.tocoo(params=params)


    op = A(fsyst)
    canonical_args = (1, 2)
    params = dict(a=1, b=2)
    call_should_be = op(wf, args=canonical_args)
    act_should_be = op.act(wf, args=canonical_args)
    has_tocoo = hasattr(op, 'tocoo')
    if has_tocoo:
        tocoo_should_be = op.tocoo(args=canonical_args).toarray()

    with raises(TypeError) as exc_info:
        op(wf, args=canonical_args, params=params)
    assert 'mutually exclusive' in str(exc_info.value)
    with raises(TypeError) as exc_info:
        op.act(wf, args=canonical_args, params=params)
    assert 'mutually exclusive' in str(exc_info.value)
    with raises(TypeError) as exc_info:
        op.bind(args=canonical_args, params=params)
    assert 'mutually exclusive' in str(exc_info.value)
    if has_tocoo:
        with raises(TypeError) as exc_info:
            op.tocoo(args=canonical_args, params=params)
        assert 'mutually exclusive' in str(exc_info.value)

    np.testing.assert_array_equal(
        call_should_be, op(wf, params=params))
    np.testing.assert_array_equal(
        act_should_be, op.act(wf, params=params))
    if has_tocoo:
        np.testing.assert_array_equal(
            tocoo_should_be, op.tocoo(params=params).toarray())
    # after binding
    op2 = op.bind(params=params)
    np.testing.assert_array_equal(
        call_should_be, op2(wf))
    np.testing.assert_array_equal(
        act_should_be, op2.act(wf))
    if has_tocoo:
        np.testing.assert_array_equal(
            tocoo_should_be, op2.tocoo().toarray())

    # system and onsite having different args
    def onsite(site, flip):
        return -1 if flip else 1

    op = A(fsyst, onsite=onsite)
    params['flip'] = True
    call_should_be = -call_should_be
    act_should_be = -act_should_be
    if has_tocoo:
        tocoo_should_be = -tocoo_should_be

    np.testing.assert_array_equal(
        call_should_be, op(wf, params=params))
    np.testing.assert_array_equal(
        act_should_be, op.act(wf, params=params))
    if has_tocoo:
        np.testing.assert_array_equal(
            tocoo_should_be, op.tocoo(params=params).toarray())
    # after binding
    op2 = op.bind(params=params)
    np.testing.assert_array_equal(
        call_should_be, op2(wf))
    np.testing.assert_array_equal(
        act_should_be, op2.act(wf))
    if has_tocoo:
        np.testing.assert_array_equal(
            tocoo_should_be, op2.tocoo().toarray())


def random_onsite(i):
    return (2 + kwant.digest.uniform(i.tag)) * sigmaz


def random_hopping(i, j):
    return (-1 + kwant.digest.uniform(i.tag + j.tag)) * sigmay


def f_sigmay(i):
    return sigma0


@pytest.mark.parametrize("A", opservables)
def test_pickling(A):

    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()
    syst[(lat(i, j) for i in range(5) for j in range(5))] = random_onsite
    syst[lat.neighbors()] = random_hopping
    fsyst = syst.finalized()

    wf = np.random.rand(2 * len(fsyst.sites))

    def all_equal(it):
        it = iter(it)
        first = next(it)
        return all(np.all(first == rest) for rest in it)

    ops = [
        A(fsyst),
        A(fsyst, onsite=f_sigmay),
        A(fsyst, onsite=sigmaz),
        A(fsyst, sum=True),
        A(fsyst, onsite=sigmaz, sum=True),
        A(fsyst, onsite=f_sigmay, sum=True),
    ]
    ops += [op.bind() for op in ops]

    for op in ops:
        loaded_op = pickle.loads(pickle.dumps(op))
        assert np.all(op(wf) == loaded_op(wf))
