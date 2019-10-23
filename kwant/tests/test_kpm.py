# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from copy import copy as copy
from types import SimpleNamespace

import pytest
import numpy as np
import scipy.sparse
import scipy.sparse.linalg as sla
from scipy.integrate import simps

import kwant
from ..kpm import _rescale, LocalVectors, RandomVectors
from .._common import ensure_rng


SpectralDensity = kwant.kpm.SpectralDensity

# ### General parameters
dim = 20
p = SimpleNamespace(
    num_moments=200,
    num_vectors=5,
    )
ham = kwant.rmt.gaussian(dim)

# ### Auxiliary functions
TOL = 1e-12
TOL_SP = 1e-6
TOL_WEAK = 1e-3


def assert_allclose(arr1, arr2):
    np.testing.assert_allclose(arr1, arr2, rtol=0., atol=TOL)


def assert_allclose_sp(arr1, arr2):
    np.testing.assert_allclose(arr1, arr2, rtol=0., atol=TOL_SP)

def test_mean_SD():
    kpm_params = dict(num_vectors=1, num_moments=10, rng=0)
    sites = [np.random.randint(dim), np.random.randint(dim)]
    local_spectrum = SpectralDensity(
        ham, vector_factory=LocalVectors(ham, where=sites),
        mean=False, **kpm_params)
    spectrum = SpectralDensity(
        ham, vector_factory=LocalVectors(ham, where=sites),
        mean=True, **kpm_params)
    assert_allclose(local_spectrum.energies, spectrum.energies)
    assert_allclose(local_spectrum.densities.sum(axis=1), spectrum.densities)

def test_conductivity():
    radius = 3
    letters = [('x','x'), ('x','y'), ('y','x'), ('y','y')]
    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=1)
    syst._lattice = lat
    def circle(x):
        return np.linalg.norm(x) < radius
    syst[lat.shape(circle, (0, 0))] = 0
    syst[lat.neighbors()] = -1
    syst = syst.finalized()
    hamiltonian = syst.hamiltonian_submatrix()
    positions = np.array([s.pos for s in syst.sites])

    # results for num_moments=10, num_vectors=10
    kpm_params = dict(
        params=None, num_vectors=10, num_moments=12,
        energy_resolution=None, vector_factory=None, bounds=None,
        eps=0.05, rng=0, accumulate_vectors=False)

    # test system and no positions
    for alpha, beta in letters:
        cond = kwant.kpm.conductivity(syst, alpha=alpha, beta=beta,
                                        positions=None, **kpm_params)

    # test when 'norbs' not provided
    n = lat.norbs
    lat.norbs = None
    cond = kwant.kpm.conductivity(syst, alpha=alpha, beta=beta,
                                  positions=None, **kpm_params)
    lat.norbs = n

    # test system or hamiltonian, no positions, but velocity operators
    cond_xx = kwant.kpm.conductivity(syst, alpha='x', beta='x',
                                     positions=None, **kpm_params)
    x_op = scipy.sparse.coo_matrix(hamiltonian, copy=True)
    displacements = positions[x_op.col] - positions[x_op.row]
    x_op.data *= 1j * displacements[:, 0]
    for ham, pos in [(syst, None), (hamiltonian, positions)]:
        # test formatting of alpha and beta
        for alpha, beta in [('x',x_op), (x_op, 'x'), (x_op, x_op)]:
            cond = kwant.kpm.conductivity(ham, alpha=alpha, beta=beta,
                                          positions=pos, **kpm_params)
            assert_allclose(cond(), cond_xx())

    # test increase number of moments
    increase_moments = 90
    kpm_params['accumulate_vectors'] = True
    cond = kwant.kpm.conductivity(syst, alpha='x', beta='x', positions=None,
                                  **kpm_params)
    cond.add_moments(num_moments=increase_moments)

    # assert that it's equivalent to construct the instance directly
    kpm_params['accumulate_vectors'] = False
    kpm_params['num_moments'] = kpm_params['num_moments'] + increase_moments
    cond_100 = kwant.kpm.conductivity(syst, alpha='x', beta='x', positions=None,
                                      **kpm_params)
    assert_allclose(cond_100(), cond())

def test_where():
    radius = 3
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(norbs=1)
    syst._lattice = lat
    def circle(x):
        return np.linalg.norm(x) < radius
    syst[lat.shape(circle, (0, 0))] = 0
    syst[lat.neighbors()] = -1
    syst = syst.finalized()

    where = lambda s: np.linalg.norm(s.pos) < 2
    sites = [s for s in syst.sites if where(s)]

    kpm_params = dict(num_vectors=None, num_moments=10, rng=0)
    spectrum2 = SpectralDensity(
        syst, vector_factory=LocalVectors(syst, where=where), **kpm_params)
    spectrum1 = SpectralDensity(
        syst, vector_factory=LocalVectors(syst, where=sites), **kpm_params)
    assert_allclose(spectrum1(), spectrum2())

    # test local vectors
    cond_xy1 = kwant.kpm.conductivity(
        syst, vector_factory=LocalVectors(syst, where=sites),
        alpha='x', beta='y', **kpm_params)
    cond_xy2 = kwant.kpm.conductivity(
        syst, vector_factory=LocalVectors(syst, where=where),
        alpha='x', beta='y', **kpm_params)
    assert_allclose(cond_xy1(), cond_xy2())

    # test random factory
    kpm_params = dict(num_vectors=10, num_moments=10, rng=0)
    cond_xy1 = kwant.kpm.conductivity(
        syst, vector_factory=RandomVectors(syst, sites, kpm_params['rng']),
        alpha='x', beta='y', **kpm_params)
    cond_xy2 = kwant.kpm.conductivity(
        syst, vector_factory=RandomVectors(syst, where, kpm_params['rng']),
        alpha='x', beta='y', **kpm_params)
    assert_allclose(cond_xy1(), cond_xy2())

def test_vector_factory():

    vectors = np.identity(dim)
    def vectors2():
        count = 0
        while count < 6:
            yield 'string'[count]
            count += 1

    vf = lambda n, v: kwant.kpm._VectorFactory(num_vectors=n, vectors=v)
    this_vf = vf(None, vectors)
    assert this_vf.num_vectors == dim
    this_vf = vf(3, vectors)
    assert this_vf.num_vectors == 3
    this_vf.add_vectors(num_vectors=2)
    assert this_vf.num_vectors == 5

    spectrum = lambda n, v : SpectralDensity(ham, vector_factory=v,
                                             num_vectors=n)
    corr = lambda n, v : kwant.kpm.Correlator(ham, vector_factory=v,
                                              num_vectors=n)

    for instance in [spectrum, corr]:
        this_instance = instance(None, vectors)
        assert this_instance.num_vectors == dim
        this_instance = instance(3, vectors)
        assert this_instance.num_vectors == 3
        this_instance.add_vectors(num_vectors=2)
        assert this_instance.num_vectors == 5


def make_spectrum(ham, p, operator=None, vector_factory=None, rng=None, params=None):
    """Create an instance of SpectralDensity class."""
    return SpectralDensity(
        ham,
        operator=operator,
        vector_factory=vector_factory,
        num_moments=p.num_moments,
        num_vectors=p.num_vectors,
        rng=rng,
        params=params
        )


def make_chain(r=dim, t=-1):
    syst = kwant.Builder()
    lat = kwant.lattice.chain(norbs=1)
    for i in range(r):
        syst[lat(i)] = 0
    syst[lat.neighbors()] = t
    return syst.finalized()


def make_chain_with_params(r=10, pot=0, t=-1):
    syst = kwant.Builder()
    lat = kwant.lattice.chain(norbs=1)
    pot = lambda site, pot: pot
    hop = lambda site1, site2, t: t
    for i in range(r):
        syst[lat(i)] = pot
    syst[lat.neighbors()] = hop
    return syst.finalized()


def kpm_derivative(spectrum, e, order=1):
    """Calculate the coeficients for Chebyshev expansion and its derivates."""
    e = np.asarray(e, dtype=complex)
    rescaled_energy = (e - spectrum._b) / spectrum._a
    g_e = np.pi * np.sqrt(1 - rescaled_energy) * np.sqrt(1 + rescaled_energy)

    m = np.arange(spectrum.num_moments)
    kernel = ((spectrum.num_moments - m + 1) *
              np.cos(np.pi * m / (spectrum.num_moments + 1)) +
              np.sin(np.pi * m / (spectrum.num_moments + 1)) /
              np.tan(np.pi / (spectrum.num_moments + 1)))
    kernel = kernel / (spectrum.num_moments + 1)

    moments = np.sum(spectrum._moments_list, axis=0) /\
        (spectrum.num_vectors * spectrum.hamiltonian.shape[0])
    coef_cheb = np.zeros_like(moments)
    coef_cheb[0] = moments[0]
    coef_cheb[1:] = 2 * moments[1:] * kernel[1:]

    i = 1
    while i <= order:
        i += 1
        coef_cheb = np.polynomial.chebyshev.chebder(coef_cheb)

    return np.polynomial.chebyshev.chebval(rescaled_energy, coef_cheb)/g_e


def make_spectrum_and_peaks(ham, precise, threshold=0.005):
    """Calculate the spectrum and the peaks in the array energies."""
    spectrum = make_spectrum(ham, precise)
    derivate_array = kpm_derivative(spectrum, spectrum.energies)
    second_derivate_array = kpm_derivative(spectrum,
                                           spectrum.energies, order=2)
    # where the derivative changes sign
    zero_index = np.where(np.diff(np.sign(derivate_array)))[0]
    scale = threshold * np.max(spectrum.densities)
    # where the spectrum is avobe the threshold
    filter_index = zero_index[
        np.where(spectrum.densities[zero_index] > scale)[0]]
    # where is a local maxmimum
    filter_index2 = filter_index[
        np.where(second_derivate_array[filter_index] < 0)[0]]
    return spectrum, filter_index2


def deviation_from_eigenvalues(dim, precise):
    """Return the maxmimum difference between the peaks in the spectrum and
    the eigenvalues."""
    ham = kwant.rmt.gaussian(dim)
    spectrum, filter_index = make_spectrum_and_peaks(ham, precise)
    try:
        return 1./(2 * spectrum._a) * np.max(np.abs(
            spectrum.energies[filter_index] - np.linalg.eigh(ham)[0]))
    except ValueError:
        """This means that two eigenvalues are nearly degenerate, therefore the
        maximum deviation is the minimum difference between a peak and all the
        eigenvalues."""
        return np.max([1./(2*spectrum._a) *
                       np.min(np.abs(spectrum.energies[filter_index] -
                                     np.linalg.eigh(ham)[0][i]))
                       for i in range(len(ham))])


def find_peaks_in_an_array(array, threshold=0.005):
    """Find the peaks above threshold and return the index inside array."""
    derivate_array = np.gradient(array)
    second_derivate_array = np.gradient(derivate_array)
    # where the derivative changes sign
    zero_index = np.where(np.diff(np.sign(derivate_array)))[0]
    scale = threshold * np.max(array)
    # where the spectrum is avobe the threshold
    filter_index = zero_index[np.where(array[zero_index] > scale)[0]]
    # where is a local maxmimum
    filter_index2 = filter_index[
        np.where(second_derivate_array[filter_index] < 0)[0]]
    return filter_index2


# ### Tests for consistency in top-level API ###


def test_api_ham():
    not_a_hamiltonian = str('not a hamiltonian')
    with pytest.raises(ValueError):
        SpectralDensity(not_a_hamiltonian)


def test_api_operator():
    not_an_operator = str('not an operator')
    with pytest.raises(ValueError):
        SpectralDensity(kwant.rmt.gaussian(dim),
                        operator=not_an_operator)


def test_api_single_eigenvalue_error():
    with pytest.raises(ValueError):
        SpectralDensity(np.identity(dim, dtype=complex))


def test_energy_resolution():
    """Check that energy resolution works and gives the same output as
    setting the equivalent number of moments.
    """
    ham = kwant.rmt.gaussian(dim)
    rng = 1
    energy_resolution = 0.05

    sp1 = SpectralDensity(ham, rng=rng, energy_resolution=energy_resolution)
    # number of moments for this energy resolution
    num_moments = sp1.num_moments
    # use the same number of moments but not passing energy resolution
    sp2 = SpectralDensity(ham, rng=rng, num_moments=num_moments)

    # Check bit for bit equality
    assert np.all(sp1.densities == sp2.densities)


def test_operator_none():
    """Check operator=None gives the same results as operator=np.identity(),
    with the same random vectors.
    """
    ham = kwant.rmt.gaussian(dim)
    identity = np.identity(dim)

    sp1 = SpectralDensity(ham, operator=None, rng=1)
    sp2 = SpectralDensity(ham, operator=identity, rng=1)

    # different algorithms are used so these arrays are equal up to TOL_SP
    assert_allclose_sp(sp1.densities, sp2.densities)


def test_bounds():
    """Check operator=None gives the same results as operator=np.identity(),
    with the same random vectors.
    """
    ham = kwant.rmt.gaussian(dim)
    epsilon = 0.05
    tol = epsilon*0.5
    rng = ensure_rng(1)
    sp1 = SpectralDensity(ham, bounds=None, eps=epsilon, rng=rng)
    # re initialize to obtain the same vector v0
    rng = ensure_rng(1)
    v0 = np.exp(2j * np.pi * rng.random_sample(dim))
    lmax = float(sla.eigsh(
        ham, k=1, which='LA', return_eigenvectors=False, tol=tol, v0=v0))
    lmin = float(sla.eigsh(
        ham, k=1, which='SA', return_eigenvectors=False, tol=tol, v0=v0))
    sp2 = SpectralDensity(ham, bounds=(lmin, lmax), eps=epsilon, rng=1)

    # different algorithms are used so these arrays are equal up to TOL_SP
    assert_allclose_sp(sp1.densities, sp2.densities)


def test_operator_user():
    """Check operator=None gives the same results as operator=np.identity(),
    with the same random vectors.
    """
    ham = kwant.rmt.gaussian(dim)
    identity = np.identity(dim)
    user_op = lambda bra, ket: np.vdot(bra, np.dot(identity, ket))

    sp1 = SpectralDensity(ham, operator=user_op, rng=1)
    sp2 = SpectralDensity(ham, operator=identity, rng=1)

    # different algorithms are used so these arrays are equal up to TOL_SP
    assert_allclose_sp(sp1.densities, sp2.densities)


def test_kwant_syst():
    """Check that when a kwant system is passed, the results are the same as
    for the Hamiltonian (in both sparse and dense formats) of the same kwant
    system.
    """
    syst = make_chain()
    spectrum_syst = make_spectrum(syst, p, rng=1)
    ham_sparse = syst.hamiltonian_submatrix(sparse=True)
    ham_dense = syst.hamiltonian_submatrix(sparse=False)

    spectrum_sparse = make_spectrum(ham_sparse, p, rng=1)
    spectrum_dense = make_spectrum(ham_dense, p, rng=1)

    # same algorithms are used so these arrays are equal up to TOL
    assert_allclose(spectrum_syst.densities, spectrum_sparse.densities)
    assert_allclose(spectrum_syst.densities, spectrum_dense.densities)


def test_kwant_op():
    """Check that the kwant.operator.Density gives the same result as the
    identity operator.
    """
    syst = make_chain()
    kwant_op = kwant.operator.Density(syst)
    spectrum_syst = make_spectrum(syst, p, operator=kwant_op, rng=1)

    ham = syst.hamiltonian_submatrix()
    identity = np.identity(dim)
    spectrum = make_spectrum(ham, p, operator=identity, rng=1)

    assert spectrum_syst.densities.shape[1] == ham.shape[0]
    # same algorithms are used so these arrays are equal up to TOL
    assert_allclose(np.sum(spectrum_syst.densities, axis=1),
                    spectrum.densities)
    # test that the densities are arrays when calling the instance
    e = spectrum_syst.energies
    assert_allclose_sp(spectrum_syst(e), spectrum_syst.densities)


def test_kwant_op_current():
    """Check that the kwant.operator.Density gives the same result as the
    identity operator when the system has ``params``.
    """
    params = {'r':dim, 'pot':1, 't':-2}

    # build the system using parameters default values
    # to be later rewritten
    syst = make_chain_with_params()
    current = kwant.operator.Current(syst)
    # pass parameters to kpm to evaluate hamiltonian and operator
    spectrum_syst = make_spectrum(syst, p, operator=current, rng=1,
                                  params=params)

    # compare with explicit hamiltonian and operator
    ham = syst.hamiltonian_submatrix(params=params)
    def my_current(bra, ket):
        return current(bra, ket, params=params)
    # do not pass parameters to kpm
    spectrum = make_spectrum(ham, p, operator=my_current, rng=1)

    # same algorithms are used so these arrays are equal up to TOL
    assert_allclose(spectrum_syst.densities, spectrum.densities)


def test_kwant_op_integrate():
    """Check that the kwant.operator.Density gives the same result as the
    identity operator.
    """
    syst = make_chain()
    kwant_op = kwant.operator.Density(syst)
    spectrum_syst = make_spectrum(syst, p, operator=kwant_op, rng=1)

    ham = syst.hamiltonian_submatrix()
    identity = np.identity(dim)
    spectrum = make_spectrum(ham, p, operator=identity, rng=1)
    ones = lambda x: np.ones_like(x)

    assert spectrum_syst.densities.shape[1] == ham.shape[0]
    # same algorithms are used so these arrays are equal up to TOL
    assert_allclose(np.sum(spectrum_syst.integrate(distribution_function=ones)),
                    spectrum.integrate())


# ## test for methods to work as expected

# ### increase num_moments


def test_increase_num_moments():
    spectrum = make_spectrum(ham, p, rng=1)
    precise = copy(p)
    precise.num_moments = 2 * p.num_moments

    spectrum_raise = make_spectrum(ham, precise, rng=1)
    spectrum.add_moments(p.num_moments)

    # Check bit for bit equality
    assert np.all(np.array(spectrum_raise._moments_list) ==
                  np.array(spectrum._moments_list))

    # test case when there are an odd number of moments
    # (this code path is treated differently internally)
    odd = copy(p)
    odd.num_moments = p.num_moments + 1
    spectrum_odd = make_spectrum(ham, odd, rng=1)

    assert not np.all(np.asarray(spectrum_odd._moments_list)[:, -1] == 0)

    # test when increasing num_moments by 1 from an even to an odd number
    spectrum_even = make_spectrum(ham, p, rng=1)
    spectrum_even.add_moments(num_moments=1)
    assert np.all(np.array(spectrum_even._moments_list) ==
                  np.array(spectrum_odd._moments_list))


# ### increase num_moments with an operator (different algorithm)

def test_increase_num_moments_op():
    ham = kwant.rmt.gaussian(dim)
    identity = np.identity(dim)

    sp1 = make_spectrum(ham, p, operator=None, rng=1)
    sp2 = make_spectrum(ham, p, operator=identity, rng=1)

    sp1.add_moments(num_moments=sp1.num_moments)
    sp2.add_moments(num_moments=sp2.num_moments)

    # different algorithms are used so these arrays are equal up to TOL_SP
    assert_allclose_sp(sp1.densities, sp2.densities)

# ### increase num_random_vecs


def test_increase_num_vectors():
    precise = copy(p)
    precise.num_vectors = 2 * p.num_vectors

    spectrum_raise = make_spectrum(ham, precise, rng=1)
    spectrum = make_spectrum(ham, p, rng=1)
    spectrum.add_vectors(num_vectors=p.num_vectors)
    # Check bit for bit equality

    assert np.all(spectrum_raise.densities == spectrum.densities)


def test_invalid_input():

    with pytest.raises(TypeError):
        SpectralDensity(ham, num_moments=10, energy_resolution=0.1)

    s = SpectralDensity(ham)
    with pytest.raises(TypeError):
        s.add_moments(num_moments=20, energy_resolution=0.1)

    for v in (-20, 0, 10.5):
        for p in ('num_vectors', 'num_moments'):
            with pytest.raises(ValueError):
                SpectralDensity(ham, **{p: v})
        s = SpectralDensity(ham)
        with pytest.raises(ValueError):
            s.add_moments(num_moments=v)
        with pytest.raises(ValueError):
            s.add_vectors(num_vectors=v)

    for v in (-1, -0.5, 0):
        with pytest.raises(ValueError):
            SpectralDensity(ham, eps=v)
        s = SpectralDensity(ham)
        with pytest.raises(ValueError):
            s.add_moments(energy_resolution=v)

    with pytest.raises(ValueError):
        s.add_moments(energy_resolution=10)


# ### Convergence for higher number of moments ###


def test_check_convergence_decreasing_values():
    difference_list = []
    depth = 2

    for i in range(depth):
        precise = SimpleNamespace(
            num_moments=10*dim + 100*i*dim,
            num_vectors=dim//2,
            )
        results = []
        np.random.seed(1)
        iterations = 3

        for ii in range(iterations):
            results.append(deviation_from_eigenvalues(dim, precise))

        difference_list.append(results)

    assert(np.all(np.diff(np.sum(difference_list, axis=1) / iterations) < 0.))
    assert(np.sum(difference_list, axis=1)[-1] / iterations < TOL_WEAK)

# ### Convergence for a custom vector_factory


def test_convergence_custom_vector_factory():
    rng = ensure_rng(1)

    def random_binary_vectors():
        while True:
            yield np.rint(rng.random_sample(dim)) * 2 - 1

    # extracted from `deviation_from_eigenvalues
    def deviation(ham, spectrum):
        filter_index = find_peaks_in_an_array(spectrum.densities)
        try:
            return 1./(2 * spectrum._a) * np.max(np.abs(
                spectrum.energies[filter_index] - np.linalg.eigh(ham)[0]))
        except ValueError:
            """This means that two eigenvalues are nearly degenerate, therefore
            the maximum deviation is the minimum difference between a peak and
            all the eigenvalues."""

            return np.max([1./(2 * spectrum._a) *
                           np.min(np.abs(spectrum.energies[filter_index] -
                                         np.linalg.eigh(ham)[0][i]))
                           for i in range(len(ham))])

    difference_list = []
    depth = 2

    for i in range(depth):
        precise = SimpleNamespace(
            num_moments=10*dim + 100*i*dim,
            num_vectors=dim//2,
            )
        results = []
        iterations = 3

        for ii in range(iterations):
            ham = kwant.rmt.gaussian(dim)
            spectrum = make_spectrum(ham, precise,
                                     vector_factory=random_binary_vectors())
            results.append(deviation(ham, spectrum))

        difference_list.append(results)

    assert np.all(np.diff(np.sum(difference_list, axis=1) / iterations) < 0.)
    assert np.sum(difference_list, axis=1)[-1]/iterations < TOL_WEAK

# ## Consistency checks for internal functions

# ### check call


def test_call_no_argument():
    ham = kwant.rmt.gaussian(dim)
    spectrum = make_spectrum(ham, p)
    energies, densities = spectrum()

    # different algorithms are used so these arrays are equal up to TOL_SP
    assert_allclose_sp(energies, spectrum.energies)
    assert_allclose_sp(densities, spectrum.densities)


def test_call():
    ham = kwant.rmt.gaussian(dim)
    spectrum = make_spectrum(ham, p)
    densities_array = np.array([spectrum(e) for e in spectrum.energies])

    # different algorithms are used so these arrays are equal up to TOL_SP
    assert_allclose_sp(densities_array, spectrum.densities)

# ### check integrate


def test_integrate():
    ham = kwant.rmt.gaussian(dim)
    spectrum = make_spectrum(ham, p)
    ones = lambda x: np.ones_like(x)
    assert np.abs(
        (spectrum.integrate() - simps(spectrum.densities, x=spectrum.energies))
        / spectrum.integrate()) < TOL_SP
    assert np.abs(spectrum.integrate() - spectrum.integrate(
        distribution_function=ones)) < TOL

# ### check increase_energy_resolution


def test_increase_energy_resolution():
    spectrum, filter_index = make_spectrum_and_peaks(ham, p)

    old_sampling_points = 2 * spectrum.num_moments
    tol = np.max(np.abs(np.diff(spectrum.energies))) / 2

    spectrum.add_moments(energy_resolution=tol)
    new_sampling_points = 2 * spectrum.num_moments

    assert old_sampling_points < new_sampling_points
    assert np.max(np.abs(np.diff(spectrum.energies))) < tol


# ### check _rescale


def test_rescale():
    ham = kwant.rmt.gaussian(dim)
    spectrum, filter_index = make_spectrum_and_peaks(ham, p)

    ham_operator, (a, b) = _rescale(ham, eps=0.05, v0=None, bounds=None)
    rescaled_eigvalues, rescaled_eigvectors = np.linalg.eigh((
        ham - b * np.identity(len(ham))) / a)

    assert np.all(-1 < rescaled_eigvalues) and np.all(rescaled_eigvalues < 1)

    """The test
    `assert np.max(np.abs(eigvectors - rescaled_eigvectors)) < TOL`
    is not good because eigenvectors could be defined up to a phase. It should
    compare that the eigenvectors are proportional to eachother. One way to
    test this is that the product gives a complex number in the unit circle."""
    eigvalues, eigvectors = np.linalg.eigh(ham)
    assert np.all(1 - np.abs(np.vdot(eigvectors, rescaled_eigvectors)) < TOL)
