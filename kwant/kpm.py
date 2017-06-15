# -*- coding: utf-8 -*-
# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import math
import numpy as np
import scipy
import scipy.sparse.linalg as sla
import scipy.fftpack as fft

from . import system
from ._common import ensure_rng
from .operator import _LocalOperator

__all__ = ['SpectralDensity']


class SpectralDensity:
    """Calculate the spectral density of an operator.

    This class makes use of the kernel polynomial
    method (KPM), presented in [1]_, to obtain the spectral density
    :math:`ρ_A(e)`, as a function of the energy :math:`e`, of some
    operator :math:`A` that acts on a kwant system or a Hamiltonian.
    In general

    .. math::
       ρ_A(e) = ρ(e) A(e),

    where :math:`ρ(e) = \sum_{k=0}^{D-1} δ(E-E_k)` is the density of
    states, and :math:`A(e)` is the expectation value of :math:`A` for
    all the eigenstates with energy :math:`e`.

    Parameters
    ----------
    hamiltonian : `~kwant.system.FiniteSystem` or matrix Hamiltonian
        If a system is passed, it should contain no leads.
    params : dict, optional
        Additional parameters to pass to the Hamiltonian and operator.
    operator : operator, dense matrix, or sparse matrix, optional
        Operator for which the spectral density will be evaluated. If
        it is callable, the ``densities`` at each energy will have the
        dimension of the result of `operator(bra, ket)`. If it has a
        ``dot`` method, such as ``numpy.ndarray`` and
        ``scipy.sparse.matrices``, the densities will be scalars.
        If no operator is provided, the density of states is calculated
        with a faster algorithm.
    num_vectors : positive int, default: 10
        Number of random vectors for the KPM method.
    num_moments : positive int, default: 100
        Number of moments, order of the KPM expansion. Mutually exclusive
        with 'energy_resolution'.
    energy_resolution : positive float, optional
        The resolution in energy of the KPM approximation to the spectral
        density. Mutually exclusive with 'num_moments'.
    vector_factory : function, optional
        The user defined function ``f(n)`` generates random vectors of
        length ``n`` that will be used in the algorithm.
        If not provided, random phase vectors are used.
        The default random vectors are optimal for most cases, see the
        discussions in [1]_ and [2]_.
    bounds : pair of floats, optional
        Lower and upper bounds for the eigenvalue spectrum of the system.
        If not provided, they are computed.
    eps : positive float, default: 0.05
        Parameter to ensure that the rescaled spectrum lies in the
        interval ``(-1, 1)``; required for stability.
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
    `~kwant.kpm.SpectralDensity`

    >>> fsyst = syst.finalized()
    >>> rho = kwant.kpm.SpectralDensity(fsyst)

    The ``energies`` and ``densities`` can be accessed with

    >>> energies, densities = rho()

    or

    >>> energies, densities = rho.energies, rho.densities

    Attributes
    ----------
    energies : array of floats
        Array of sampling points with length ``2 * num_moments`` in
        the range of the spectrum.
    densities : array of floats
        Spectral density of the ``operator`` evaluated at the energies.
    """

    def __init__(self, hamiltonian, params=None, operator=None,
                 num_vectors=10, num_moments=None, energy_resolution=None,
                 vector_factory=None, bounds=None, eps=0.05, rng=None):

        if num_moments and energy_resolution:
            raise TypeError("either 'num_moments' or 'energy_resolution' "
                            "must be provided.")

        rng = ensure_rng(rng)
        # self.eps ensures that the rescaled Hamiltonian has a
        # spectrum strictly in the interval (-1,1).
        self.eps = eps

        # Normalize the format of 'ham'
        if isinstance(hamiltonian, system.System):
            hamiltonian = hamiltonian.hamiltonian_submatrix(params=params,
                                                            sparse=True)
        try:
            hamiltonian = scipy.sparse.csr_matrix(hamiltonian)
        except Exception:
            raise ValueError("'hamiltonian' is neither a matrix "
                             "nor a Kwant system.")

        # Normalize 'operator' to a common format.
        if operator is None:
            self.operator = None
        elif isinstance(operator, _LocalOperator):
            self.operator = operator.bind(params=params)
        elif callable(operator):
            self.operator = operator
        elif hasattr(operator, 'dot'):
            operator = scipy.sparse.csr_matrix(operator)
            self.operator = lambda bra, ket: np.vdot(bra, operator.dot(ket))
        else:
            raise ValueError('Parameter `operator` has no `.dot` '
                             'attribute and is not callable.')

        self._vector_factory = vector_factory or \
            (lambda n: np.exp(2j * np.pi * rng.random_sample(n)))
        # store this vector for reproducibility
        self._v0 = self._vector_factory(hamiltonian.shape[0])
        self._rand_vect_list = []
        # Hamiltonian rescaled as in Eq. (24)
        self.hamiltonian, (self._a, self._b) = _rescale(hamiltonian,
                                                        eps=self.eps,
                                                        v0=self._v0,
                                                        bounds=bounds)
        self.bounds = (self._b - self._a, self._b + self._a)

        if energy_resolution:
            num_moments = math.ceil((1.6 * self._a) / energy_resolution)
        elif num_moments is None:
            num_moments = 100

        must_be_positive_int = ['num_vectors', 'num_moments']
        for var in must_be_positive_int:
            val = locals()[var]
            if val <= 0 or val != int(val):
                raise ValueError('{} must be a positive integer'.format(var))
        if eps <= 0:
            raise ValueError('eps must be positive')

        for r in range(num_vectors):
            self._rand_vect_list.append(
                self._vector_factory(self.hamiltonian.shape[0]))
        self._last_two_alphas = [0.] * num_vectors
        self._moments_list = [0.] * num_vectors

        self.num_moments = num_moments
        self.num_vectors = 0  # new random vectors will be used
        self._update_moments_list(self.num_moments, num_vectors)
        self.num_vectors = num_vectors

        moments = self._moments()
        xk_rescaled, rho, self._gammas = _calc_fft_moments(
            moments, 2 * self.num_moments)
        self.energies = xk_rescaled * self._a + self._b
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
            g_e = (np.pi * np.sqrt(1 - rescaled_energy)
                   * np.sqrt(1 + rescaled_energy))

            moments = self._moments()

            m = np.arange(self.num_moments)

            kernel = ((self.num_moments - m + 1) *
                      np.cos(np.pi * m/(self.num_moments + 1)) +
                      np.sin(np.pi * m/(self.num_moments + 1)) /
                      np.tan(np.pi/(self.num_moments + 1)))
            kernel = kernel / (self.num_moments + 1)

            # transposes handle the case where operators have vector outputs
            coef_cheb = np.transpose(moments.transpose() * kernel)
            coef_cheb[1:] = 2 * coef_cheb[1:]

            return np.transpose(np.polynomial.chebyshev.chebval(
                    rescaled_energy, coef_cheb) / g_e).real

    def integrate(self, distribution_function=None):
        """Returns the total spectral density.

        Returns the integral over the whole spectrum with an optional
        distribution function. ``distribution_function`` should be able
        to take arrays as input. Defined using Gauss-Chebyshev
        integration.
        """
        # This factor divides the sum to normalize the Gauss integral
        # and rescales the integral back with ``self._a`` to normal
        # scale.
        factor = self._a / (2 * self.num_moments)
        if distribution_function is None:
            rho = self._gammas
        else:
            # The evaluation of the distribution function should be at
            # the energies without rescaling.
            distribution_array = distribution_function(self.energies)
            rho = np.transpose(self._gammas.transpose() * distribution_array)
        return factor * np.sum(rho, axis=0)

    def add_moments(self, num_moments=None, *, energy_resolution=None):
        """Increase the number of Chebyshev moments.

        Parameters
        ----------
        num_moments: positive int
            The number of Chebyshev moments to add. Mutually
            exclusive with 'energy_resolution'.
        energy_resolution: positive float, optional
            Features wider than this resolution are visible
            in the spectral density. Mutually exclusive with
            'num_moments'.
        """
        if not ((num_moments is None) ^ (energy_resolution is None)):
            raise TypeError("either 'num_moments' or 'energy_resolution' "
                            "must be provided.")

        if energy_resolution:
            if energy_resolution <= 0:
                raise ValueError("'energy_resolution' must be positive"
                                 .format(energy_resolution))
            # factor of 1.6 comes from the fact that we use the
            # Jackson kernel when calculating the FFT, which has
            # maximal slope π/2. Rounding to 1.6 ensures that the
            # energy resolution is sufficient.
            present_resolution = self._a * 1.6 / self.num_moments
            if present_resolution < energy_resolution:
                raise ValueError('Energy resolution is already smaller '
                                 'than the requested resolution')
            num_moments = math.ceil((1.6 * self._a) / energy_resolution)

        if (num_moments is None or num_moments <= 0
            or num_moments != int(num_moments)):
            raise ValueError("'num_moments' must be a positive integer")

        self._update_moments_list(self.num_moments + num_moments,
                                  self.num_vectors)
        self.num_moments += num_moments

        # recalculate quantities derived from the moments
        moments = self._moments()
        xk_rescaled, rho, self._gammas = _calc_fft_moments(
            moments, 2 * self.num_moments)
        self.energies = xk_rescaled * self._a + self._b
        self.densities = rho

    def add_vectors(self, num_vectors):
        """Increase the number of random vectors.

        Parameters
        ----------
        num_vectors: positive int
            The number of random vectors to add.
        """
        if num_vectors <= 0 or num_vectors != int(num_vectors):
            raise ValueError("'num_vectors' must be a positive integer")
        for r in range(num_vectors):
            self._rand_vect_list.append(
                self._vector_factory(self.hamiltonian.shape[0]))
        self._moments_list.extend([0.] * num_vectors)
        self._last_two_alphas.extend([0.] * num_vectors)
        self._update_moments_list(self.num_moments,
                                  self.num_vectors + num_vectors)
        self.num_vectors += num_vectors

        # recalculate quantities derived from the moments
        moments = self._moments()
        xk_rescaled, rho, self._gammas = _calc_fft_moments(
            moments, 2 * self.num_moments)
        self.energies = xk_rescaled * self._a + self._b
        self.densities = rho

    def _moments(self):
        # sum moments of all random vectors
        moments = np.sum(np.asarray(self._moments_list).real, axis=0)
        # divide by the number of random vectors
        moments /= self.num_vectors
        # divide by scale factor to reflect the integral rescaling
        return moments / self._a

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

        if self.num_vectors == n_rand:
            r_start = 0
            new_rand_vect = 0
        elif self.num_vectors < n_rand:
            r_start = self.num_vectors
            new_rand_vect = n_rand - self.num_vectors
        else:
            raise ValueError('Cannot decrease number of random vectors')

        if n_moments == self.num_moments:
            m_start = 2
            new_moments = 0
            if new_rand_vect == 0:
                # nothing new to calculate
                return
        else:
            new_moments = n_moments - self.num_moments
            m_start = self.num_moments
            if new_moments < 0:
                raise ValueError('Cannot decrease number of moments')

            if new_rand_vect != 0:
                raise ValueError("Only 'num_moments' *or* 'num_vectors' "
                                 "may be updated at a time.")

        for r in range(r_start, n_rand):
            alpha_zero = self._rand_vect_list[r]

            one_moment = [0.] * n_moments
            if new_rand_vect > 0:
                alpha = alpha_zero
                alpha_next = self.hamiltonian.matvec(alpha)
                if self.operator is None:
                    one_moment[0] = np.vdot(alpha_zero, alpha_zero)
                    one_moment[1] = np.vdot(alpha_zero, alpha_next)
                else:
                    one_moment[0] = self.operator(alpha_zero, alpha_zero)
                    one_moment[1] = self.operator(alpha_zero, alpha_next)

            if new_moments > 0:
                (alpha, alpha_next) = self._last_two_alphas[r]
                one_moment[0:self.num_moments] = self._moments_list[r]
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
                for n in range(m_start//2, n_moments//2):
                    alpha_save = alpha_next
                    alpha_next = (2 * self.hamiltonian.matvec(alpha_next)
                                  - alpha)
                    alpha = alpha_save
                    # Following Eqs. (34) and (35)
                    one_moment[2*n] = (2 * np.vdot(alpha, alpha)
                                       - one_moment[0])
                    one_moment[2*n+1] = (2 * np.vdot(alpha_next, alpha)
                                         - one_moment[1])
                if n_moments % 2:
                    # odd moment
                    one_moment[n_moments - 1] = (
                        2 * np.vdot(alpha_next, alpha_next) - one_moment[0])
            else:
                for n in range(m_start, n_moments):
                    alpha_save = alpha_next
                    alpha_next = (2 * self.hamiltonian.matvec(alpha_next)
                                  - alpha)
                    alpha = alpha_save
                    one_moment[n] = self.operator(alpha_zero, alpha_next)

            self._last_two_alphas[r] = (alpha, alpha_next)
            self._moments_list[r] = one_moment[:]


# ### Auxiliary functions


def _rescale(hamiltonian, eps, v0, bounds):
    """Rescale a Hamiltonian and return a LinearOperator

    Parameters
    ----------
    hamiltonian : 2D array
        Hamiltonian of the system.
    eps : scalar
        Ensures that the bounds 'a' and 'b' are strict.
    v0 : random vector, or None
        Used as the initial residual vector for the algorithm that
        finds the lowest and highest eigenvalues.
    bounds : tuple, or None
        Boundaries of the spectrum. If not provided the maximum and
        minimum eigenvalues are calculated.
    """
    # Relative tolerance to which to calculate eigenvalues.  Because after
    # rescaling we will add eps / 2 to the spectral bounds, we don't need
    # to know the bounds more accurately than eps / 2.
    tol = eps / 2

    if bounds:
        lmin, lmax = bounds
    else:
        lmax = float(sla.eigsh(hamiltonian, k=1, which='LA',
                               return_eigenvectors=False, tol=tol, v0=v0))
        lmin = float(sla.eigsh(hamiltonian, k=1, which='SA',
                               return_eigenvectors=False, tol=tol, v0=v0))

    a = np.abs(lmax-lmin) / (2. - eps)
    b = (lmax+lmin) / 2.

    if lmax - lmin <= abs(lmax + lmin) * tol / 2:
        raise ValueError(
            'The Hamiltonian has a single eigenvalue, it is not possible to '
            'obtain a spectral density.')

    def rescaled(v):
        return (hamiltonian.dot(v) - b * v) / a

    rescaled_ham = sla.LinearOperator(shape=hamiltonian.shape, matvec=rescaled)

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
