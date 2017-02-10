# -*- coding: utf-8 -*-
# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['SpectralDensity']

import warnings

import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.fftpack as fft

from . import system
from ._common import ensure_isinstance, ensure_rng


class SpectralDensity:
    """Calculate the spectral density of an operator.

    The **SpectralDensity** class makes use of the kernel polynomial
    method (KPM), presented in [1]_, to obtain the spectral density
    :math:`ρ_A(e)`, as a function of the energy :math:`e`, of some
    operator :math:`A` that acts on a kwant system or a Hamiltonian.
    In general

    .. math::
       ρ_A(e) = ρ(e) A(e),

    where :math:`ρ(e)` is the density of states, and :math:`A(e)` is
    the expectation value of :math:`A` for all the eigenstates with
    energy :math:`e`.

    The **SpectralDensity** class can be called as

    >>> rho = kwant.kpm.SpectralDensity(fsyst, operator=A),

    with optional
    arguments that specify the accuracy of the approximation. The
    parameters ``num_moments``, ``num_rand_vecs``, and
    ``num_sampling_points`` can be specified when creating an instance
    or by calling

    >>> rho.increase_accuracy(num_moments=200,
    ...                       num_rand_vecs=10,
    ...                       num_sampling_points=400)

    Parameters
    ----------
    syst_or_ham : `~kwant.system.FiniteSystem` or matrix Hamiltonian
        If a system is passed, it should contain no leads.
    args : tuple, optional
        Positional arguments to pass to the system.
    operator : operator, dense matrix, or sparse matrix, optional
        Operator for which the spectral density will be evaluated. If
        it is callable, the ``densities`` at each energy will have the
        dimension of the result of `operator(bra, ket)`. If it has a
        ``dot`` method, such as ``numpy.ndarray`` and ``scipy.sparse.
        matrices``, the densities will be scalars.
        If no operator is provided, the density of states is calculated
        with a faster algorithm.
    num_rand_vecs : integer, default: 10
        Number of random vectors for the KPM method.
    num_moments : integer, default: 100
        Number of moments, order of the KPM expansion.
    num_sampling_points : integer, optional
        Number of points where the spectral density will be evaluated.
        If not provided, ``2\*num_moments`` will be used.
    vector_factory : function, optional
        The user defined function ``f(n)`` generates random vectors of
        length ``n`` that will be used in the algorithm.
        If not provided, random phase vectors are used.
        The default random vectors are optimal for most cases, see the
        discussions in [1]_ and [2]_.
    bounds : pair of floats, optional
        Lower and upper bounds for the eigenvalue spectrum of the system.
        If not provided, they are computed.
    epsilon : float, default: 0.05
        Parameter to ensure that the rescaled spectrum lies in the
        interval [-1, 1]; required for stability.
    rng : seed, or random number generator, optional
        Random number generator used by ``vector_factory``.
        If not provided, numpy's rng will be used; if it is an Integer,
        it will be used to seed numpy's rng, and if it is a random
        number generator, this is the one used.

    Notes
    -----
    When passing an ``operator`` defined in `~kwant.operator`, the
    result of ``operator(bra, ket)`` depends on the attribute ``sum``
    of such operator. If ``sum=True``, densities will be scalars, that
    is, total densities of the system. If ``sum=False`` the densities
    will be arrays of the length of the system, that is, local
    densities.

    .. [1] `Rev. Mod. Phys., Vol. 78, No. 1 (2006)
       <https://arxiv.org/abs/cond-mat/0504627>`_.
    .. [2] `Phys. Rev. E 69, 057701 (2004)
       <https://arxiv.org/abs/cond-mat/0401202>`_

    Examples
    --------
    In the following example, we will obtain the density of states of a
    graphene sheet, defined as a honeycomb lattice with first nearest
    neighbors coupling.

    We start by importing kwant and defining a
    `~kwant.system.FiniteSystem`,

    >>> import kwant
    ...
    >>> def circle(pos):
    ...     x, y = pos
    ...     return x**2 + y**2 < 100
    ...
    >>> lat = kwant.lattice.honeycomb()
    >>> syst = kwant.Builder()
    >>> syst[lat.shape(circle, (0, 0))] = 0
    >>> syst[lat.neighbors()] = 1

    and after finalizing the system, create an instance of
    **SpectralDensity**

    >>> fsyst = syst.finalized()
    >>> rho = kwant.kpm.SpectralDensity(fsyst)

    The ``energies`` and ``densities`` can be accesed with

    >>> energies, densities = rho()

    or

    >>> energies, densities = rho.energies, rho.densities

    Attributes
    ----------
    energies : array of floats
        Array of sampling points with length ``num_sampling_points`` in
        the range of the spectrum.
    densities : array of floats
        Spectral density of the ``operator`` evaluated at the energies.

    """

    def __init__(self, syst_or_ham, args=(), operator=None, num_rand_vecs=10,
                 num_moments=100, num_sampling_points=None,
                 vector_factory=None, bounds=None, epsilon=0.05, rng=None):
        rng = ensure_rng(rng)
        # self.epsilon ensures that the rescaled Hamiltonian has a
        # spectrum strictly in the interval (-1,1).
        self.epsilon = epsilon
        # Check if syst_or_ham is a finalized System or a Hamiltonian
        try:
            ham = scipy.sparse.csr_matrix(syst_or_ham)
        except:
            try:
                ensure_isinstance(syst_or_ham, system.System)
                ham = scipy.sparse.csr_matrix(
                    syst_or_ham.hamiltonian_submatrix(args=args, sparse=True))
            except TypeError:
                raise ValueError('Parameter `syst_or_ham` is not a '
                                 'Hamiltonian neither a (finalized) '
                                 '`kwant.system`.')
        # Check if operator is a sparse matrix or None.
        if operator is None:
            self.operator = None
        else:
            if callable(operator):
                self.operator = operator
            elif hasattr(operator, 'dot') and not hasattr(operator, 'act'):
                operator = scipy.sparse.csr_matrix(operator)
                self.operator = (lambda bra, ket:
                                 np.vdot(bra, operator.dot(ket)))
            else:
                raise ValueError('Parameter `operator` has no `.dot` '
                                 'attribute and is not callable.')
        self.num_rand_vecs = num_rand_vecs
        self.num_moments = num_moments
        # Default number of sampling points
        if num_sampling_points is None:
            self.num_sampling_points = 2 * self.num_moments
        elif num_sampling_points < self.num_moments:
            raise ValueError('The number of sampling points should be larger '
                             'than the number of moments.')
        else:
            self.num_sampling_points = num_sampling_points

        # calculate raw moments
        self._vector_factory = vector_factory or \
            (lambda n: np.exp(2j * np.pi * rng.random_sample(n)))
        # store this vector for reproducibility
        self._v0 = self._vector_factory(ham.shape[0])
        self._rand_vect_list = []
        self._last_two_alphas = []
        self._moments_list = [0.] * self.num_rand_vecs
        # Hamiltonian rescaled as in Eq. (24)
        self.ham, (self._a, self._b) = _rescale(ham, epsilon=self.epsilon,
                                                v0=self._v0, bounds=bounds)
        self.bounds = (self._b - self._a, self._b + self._a)
        self._update_moments_list(self.num_moments, self.num_rand_vecs)
        # sum moments of all random vectors
        moments = np.sum(np.asarray(self._moments_list), axis=0)
        # divide by the number of random vectors
        moments = moments / self.num_rand_vecs
        # divide by the length of the vectors to normalize
        moments = moments / self.ham.shape[0]
        # divide by scale factor to reflect the integral rescaling
        moments = moments / self._a

        # obtain energies, densities, and gammas of Eq. 83
        xk_rescaled, rho, self._gammas = _calc_fft_moments(
            moments, self.num_sampling_points)
        # energies to normal scale
        self.energies = xk_rescaled*self._a + self._b
        self.densities = rho

    def __call__(self, energy=None):
        """Return the spectral density evaluated at ``energy``.

        Parameters
        ----------
        energy : float or sequence of float, optional

        Returns
        -------
        float, if ``energy`` is float, array of float if ``energy``
        is a sequence, a tuple (energies, densities) if
        ``energy`` was not provided.

        Notes
        -----
        If ``energy`` is not provided, then the densities are obtained
        by Fast Fourier Transform of the Chebyshev moments.
        """
        if energy is None:
            return self.energies, self.densities
        else:
            energy = np.asarray(energy, dtype=complex)
            rescaled_energy = (energy - self._b) / self._a
            g_e = np.pi * np.sqrt(1 - rescaled_energy) *\
                np.sqrt(1 + rescaled_energy)

            # calculate the coefficients for Chebyshev expansion
            # sum moments of all random vectors
            moments = np.sum(np.asarray(self._moments_list), axis=0)
            # divide by the number of random vectors
            moments = moments / self.num_rand_vecs
            # divide by the length of the vectors to normalize
            moments = moments / self.ham.shape[0]
            # divide by scale factor to reflect the integral rescaling
            moments = moments / self._a

            m = np.arange(self.num_moments)

            kernel = ((self.num_moments - m + 1) *
                      np.cos(np.pi * m/(self.num_moments + 1)) +
                      np.sin(np.pi * m/(self.num_moments + 1)) /
                      np.tan(np.pi/(self.num_moments + 1)))
            kernel = kernel / (self.num_moments + 1)

            coef_cheb = np.zeros_like(moments)
            coef_cheb[0] = moments[0]
            coef_cheb[1:] = 2 * moments[1:] * kernel[1:]

            return (np.polynomial.chebyshev.chebval(
                    rescaled_energy, coef_cheb) / g_e).real

    def average(self, distribution_function=None):
        """Returns the total spectral density.

        Returns the integral over the whole spectrum with an optional
        distribution function. ``distribution_function`` should be able
        to take arrays as input. Defined using Gauss-Chebyshev
        integration.
        """
        # This factor divides the sum to normalize the Gauss integral
        # and rescales the integral back with ``self._a`` to normal
        # scale.
        factor = self._a/self.num_sampling_points
        if distribution_function is None:
            rho = self._gammas
        else:
            # The evaluation of the distribution function should be at
            # the energies without rescaling.
            distribution_array = distribution_function(self.energies)
            rho = np.transpose(self._gammas.transpose() * distribution_array)
        return factor * np.sum(rho, axis=0)

    def increase_energy_resolution(self, tol=0.05, increase_num_moments=True):
        """Set the minimum energy resolution to ``tol``.

        The energy resolution is increased by increasing the number of
        sampling points. By default the number of moments is also
        increased.
        """
        resolution = 2 * self._a / (self.num_sampling_points / 1.6)
        if tol > resolution:
            warnings.warn('Energy resolution is already smaller than tol.')
        else:
            num_sampling_points = np.ceil((1.6 * 2*self._a) / tol)
            num_sampling_points = num_sampling_points.astype(int)
            if increase_num_moments:
                num_moments = num_sampling_points // 2
                self.increase_accuracy(num_moments=num_moments,
                                       num_sampling_points=num_sampling_points)
            else:
                self.increase_accuracy(num_sampling_points=num_sampling_points)

    def increase_accuracy(self, num_moments=None, num_rand_vecs=None,
                          num_sampling_points=None):
        """Increase the number of moments, random vectors, or sampling
        points.

        Parameters
        ----------
        num_moments, num_rand_vecs, num_sampling_points : integer
            Number of Chebyshev moments, random vectors used for
            sampling, and sampling points for the calculation of
            densities.
        """
        # Calls to update_moments_list should be in this order:
        # 1st. update random vectors and recalculate current moments,
        # 2nd. update moments with the updated random vectors,
        # 3rd. update sampling points and FFT densities.
        num_rand_vecs = num_rand_vecs or self.num_rand_vecs
        self._update_moments_list(self.num_moments, num_rand_vecs)
        self.num_rand_vecs = num_rand_vecs

        num_moments = num_moments or self.num_moments
        self._update_moments_list(num_moments, self.num_rand_vecs)
        self.num_moments = num_moments

        num_sampling_points = num_sampling_points or self.num_sampling_points
        if num_sampling_points < self.num_moments:
            raise ValueError(
                'The number of sampling points should be larger than the '
                'number of moments.')
        self.num_sampling_points = num_sampling_points

        # sum moments of all random vectors
        moments = np.sum(np.asarray(self._moments_list), axis=0)
        # divide by the number of random vectors
        moments = moments / self.num_rand_vecs
        # divide by the length of the vectors to normalize
        moments = moments / self.ham.shape[0]
        # divide by scale factor to reflect the integral rescaling
        moments = moments / self._a

        xk_rescaled, rho, self._gammas = _calc_fft_moments(
            moments, self.num_sampling_points)

        # Sampling points to normal scale
        self.energies = xk_rescaled * self._a + self._b
        self.densities = rho

    def _update_moments_list(self, n_moments, n_rand):
        """Calculate the Chebyshev moments of an operator's spectral
        density.

        The algorithm is based on the KPM method as depicted in `Rev.
        Mod. Phys., Vol. 78, No. 1 (2006)
        <https://arxiv.org/abs/cond-mat/0504627>`_.

        Parameters
        ----------
        n_moments : integer
            Number of Chebyshev moments.
        n_rand : integer
            Number of random vectors used for sampling.
        """
        dim = self.ham.shape[0]

        if len(self._rand_vect_list) == 0:
            for r in range(n_rand):
                self._rand_vect_list.append(self._vector_factory(dim))
            r_start = 0
            r_stop = self.num_rand_vecs
        else:
            if self.num_rand_vecs == n_rand:
                r_start = 0
                r_stop = self.num_rand_vecs
            elif self.num_rand_vecs < n_rand:
                new_rand_vect = n_rand - self.num_rand_vecs
                for r in range(new_rand_vect):
                    self._rand_vect_list.append(self._vector_factory(dim))
                self._moments_list.extend([0.] * new_rand_vect)
                r_start = self.num_rand_vecs
                r_stop = n_rand
            else:
                warnings.warn('Decreasing number of random vectors')
                r_start = 0
                r_stop = n_rand

        if n_moments == self.num_moments:
            start = 2
            new_moments = 0
        else:
            start = self.num_moments
            new_moments = n_moments - self.num_moments
        stop = n_moments

        for r in range(r_start, r_stop):
            alpha_zero = self._rand_vect_list[r]

            one_moment = [0.] * n_moments
            if new_moments == 0:
                alpha = alpha_zero
                alpha_next = self.ham.matvec(alpha)
                if self.operator is None:
                    one_moment[0] = np.vdot(alpha_zero, alpha_zero)
                    one_moment[1] = np.vdot(alpha_zero, alpha_next)
                else:
                    one_moment[0] = self.operator(alpha_zero, alpha_zero)
                    one_moment[1] = self.operator(alpha_zero, alpha_next)

            elif new_moments > 0:
                (alpha, alpha_next) = self._last_two_alphas[r]
                one_moment[0:self.num_moments] = self._moments_list[r]
            else:
                raise ValueError('Attempt to decrease the number of moments.')
            # Iteration over the moments
            # Two cases can occur, depicted in Eq. (28) and in Eq. (29),
            # respectively.
            # ----
            # In the first case, self.operator is None and we can use
            # Eqs. (34) and (35) to obtain the density of states, with
            # two moments ``one_moment`` for every new alpha.
            # ----
            # In the second case, the operator is not None and a matrix
            # multiplication should be used.
            if self.operator is None:
                for n in range(start//2, stop//2):
                    alpha_save = alpha_next
                    alpha_next = 2 * self.ham.matvec(alpha_next) - alpha
                    alpha = alpha_save
                    # Following Eqs. (34) and (35)
                    one_moment[2*n] = 2 * np.vdot(alpha, alpha) -\
                        one_moment[0]
                    one_moment[2*n+1] = 2 * np.vdot(alpha_next, alpha) -\
                        one_moment[1]
            else:
                for n in range(start, stop):
                    alpha_save = alpha_next
                    alpha_next = 2 * self.ham.matvec(alpha_next) - alpha
                    alpha = alpha_save
                    one_moment[n] = self.operator(alpha_zero, alpha_next)

            self._last_two_alphas.append((alpha, alpha_next))
            self._moments_list[r] = np.asarray(one_moment).real


# ### Auxiliary functions


def _rescale(ham, epsilon, v0, bounds):
    """Rescale a Hamiltonian and return a LinearOperator

    Parameters
    ----------
    ham : 2D array
        Hamiltonian of the system.
    epsilon : scalar
        Ensures that the bounds 'a' and 'b' are strict.
    v0 : random vector, or None
        Used as the initial residual vector for the algorithm that
        finds the lowest and highest eigenvalues.
    bounds : tuple, or None
        Boundaries of the spectrum. If not provided the maximum and
        minimum eigenvalues are calculated.
    """
    # Relative tolerance to which to calculate eigenvalues.  Because after
    # rescaling we will add epsilon / 2 to the spectral bounds, we don't need
    # to know the bounds more accurately than epsilon / 2.
    tol = epsilon / 2

    if bounds:
        lmin, lmax = bounds
    else:
        lmax = float(sla.eigsh(ham, k=1, which='LA', return_eigenvectors=False,
                               tol=tol, v0=v0))
        lmin = float(sla.eigsh(ham, k=1, which='SA', return_eigenvectors=False,
                               tol=tol, v0=v0))

    a = np.abs(lmax-lmin) / (2. - epsilon)
    b = (lmax+lmin) / 2.

    if lmax - lmin <= abs(lmax + lmin) * tol / 2:
        raise ValueError(
            'The Hamiltonian has a single eigenvalue, it is not possible to '
            'obtain a spectral density.')

    def rescaled(v):
        return (ham.dot(v) - b * v) / a

    rescaled_ham = sla.LinearOperator(shape=ham.shape, matvec=rescaled)

    return rescaled_ham, (a, b)


def _calc_fft_moments(moments, n_sampling):
    """This function takes the normalised moments and returns an array
    of points and an array of the evaluated function at those points.
    """
    moments = np.asarray(moments)
    # extra_shape handles the case where operators have vector outputs
    n_moments, *extra_shape = moments.shape
    moments_ext = np.zeros([n_sampling] + extra_shape)

    # Jackson kernel, as in Eq. (71), and kernel improved moments,
    # as in Eq. (81).
    m = np.arange(n_moments)
    kernel = ((n_moments - m + 1) * np.cos(np.pi * m / (n_moments + 1)) +
              np.sin(np.pi * m / (n_moments + 1)) /
              np.tan(np.pi / (n_moments + 1))) / (n_moments + 1)

    # special points at the abscissas of Chebyshev integration
    k = np.arange(0, n_sampling)
    xk_rescaled = np.cos(np.pi * (k + 0.5) / n_sampling)
    # prefactor in Eq. (80)
    gk = np.pi * np.sqrt(1 - xk_rescaled ** 2)

    # transposes handle the case where operators have vector outputs
    moments_ext[:n_moments] = np.transpose(moments.transpose() * kernel)
    # The function evaluated in these special data points is the FFT of
    # the moments as in Eq. (83).
    gammas = np.transpose(fft.dct(moments_ext.transpose(), type=3))

    # Element-wise division of moments_FFT over gk, as in Eq. (83).
    rho = np.transpose(np.divide(gammas.transpose(), gk))

    # Reverse energies and densities to set ascending order.
    return xk_rescaled[::-1], rho[::-1], gammas[::-1]
