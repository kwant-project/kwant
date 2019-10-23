# -*- coding: utf-8 -*-
# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.
import math
from operator import add
from collections.abc import Iterable
from functools import reduce
import numpy as np
from numpy.polynomial.chebyshev import chebval
from scipy.sparse import coo_matrix, csr_matrix
from scipy.integrate import simps
from scipy.sparse.linalg import eigsh, LinearOperator
import scipy.fftpack as fft

from . import system
from ._common import ensure_rng
from .operator import (_LocalOperator, _get_tot_norbs, _get_all_orbs,
                       _normalize_site_where)
from .graph.defs import gint_dtype

__all__ = ['SpectralDensity', 'Correlator', 'conductivity',
           'RandomVectors', 'LocalVectors', 'jackson_kernel', 'lorentz_kernel',
           'fermi_distribution']

SAMPLING = 2 # number of sampling points to number of moments ratio

class SpectralDensity:
    r"""Calculate the spectral density of an operator.

    This class makes use of the kernel polynomial
    method (KPM), presented in [1]_, to obtain the spectral density
    :math:`ρ_A(e)`, as a function of the energy :math:`e`, of some
    operator :math:`A` that acts on a kwant system or a Hamiltonian.
    In general

    .. math::
       ρ_A(E) = ρ(E) A(E),

    where :math:`ρ(E) = \sum_{k=0}^{D-1} δ(E-E_k)` is the density of
    states, and :math:`A(E)` is the expectation value of :math:`A` for
    all the eigenstates with energy :math:`E`.

    Parameters
    ----------
    hamiltonian : `~kwant.system.FiniteSystem` or matrix Hamiltonian
        If a system is passed, it should contain no leads.
    params : dict, optional
        Additional parameters to pass to the Hamiltonian and operator.
    operator : operator, dense matrix, or sparse matrix, optional
        Operator for which the spectral density will be evaluated. If
        it is callable, the ``densities`` at each energy will have the
        dimension of the result of ``operator(bra, ket)``. If it has a
        ``dot`` method, such as ``numpy.ndarray`` and
        ``scipy.sparse.matrices``, the densities will be scalars.
    num_vectors : positive int, or None, default: 10
        Number of vectors used in the KPM expansion. If ``None``, the
        number of vectors used equals the length of the 'vector_factory'.
    num_moments : positive int, default: 100
        Number of moments, order of the KPM expansion. Mutually exclusive
        with ``energy_resolution``.
    energy_resolution : positive float, optional
        The resolution in energy of the KPM approximation to the spectral
        density. Mutually exclusive with ``num_moments``.
    vector_factory : iterable, optional
        If provided, it should contain (or yield) vectors of the size of
        the system. If not provided, random phase vectors are used.
        The default random vectors are optimal for most cases, see the
        discussions in [1]_ and [2]_.
    bounds : pair of floats, optional
        Lower and upper bounds for the eigenvalue spectrum of the system.
        If not provided, they are computed.
    eps : positive float, default: 0.05
        Parameter to ensure that the rescaled spectrum lies in the
        interval ``(-1, 1)``; required for stability.
    rng : seed, or random number generator, optional
        Random number generator used for the calculation of the spectral
        bounds, and to generate random vectors (if ``vector_factory`` is
        not provided). If not provided, numpy's rng will be used; if it
        is an Integer, it will be used to seed numpy's rng, and if it is
        a random number generator, this is the one used.
    kernel : callable, optional
        Callable that takes moments and returns stabilized moments.
        By default, the `~kwant.kpm.jackson_kernel` is used.
        The Lorentz kernel is also accesible by passing
        `~kwant.kpm.lorentz_kernel`.
    mean : bool, default: ``True``
        If ``True``, return the mean spectral density for the vectors
        used, otherwise return an array of densities for each vector.
    accumulate_vectors : bool, default: ``True``
        Whether to save or discard each vector produced by the vector
        factory. If it is set to ``False``, it is not possible to
        increase the number of moments, but less memory is used.

    Notes
    -----
    When passing an operator defined in `~kwant.operator`, the
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
    >>> syst[lat.neighbors()] = -1

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
                 vector_factory=None, bounds=None, eps=0.05, rng=None,
                 kernel=None, mean=True, accumulate_vectors=True):

        if num_moments and energy_resolution:
            raise TypeError("either 'num_moments' or 'energy_resolution' "
                            "must be provided.")

        # self.eps ensures that the rescaled Hamiltonian has a
        # spectrum strictly in the interval (-1,1).
        self.eps = eps

        # Normalize the format of 'ham'
        if isinstance(hamiltonian, system.System):
            hamiltonian = hamiltonian.hamiltonian_submatrix(params=params,
                                                            sparse=True)
        try:
            hamiltonian = csr_matrix(hamiltonian)
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
            operator = csr_matrix(operator)
            self.operator = lambda bra, ket: np.vdot(bra, operator.dot(ket))
        else:
            raise ValueError("Parameter 'operator' has no '.dot' "
                             "attribute and is not callable.")

        self.mean = mean
        rng = ensure_rng(rng)
        # store this vector for reproducibility
        self._v0 = np.exp(2j * np.pi * rng.random_sample(hamiltonian.shape[0]))

        if eps <= 0:
            raise ValueError("'eps' must be positive")

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

        if num_moments <= 0 or num_moments != int(num_moments):
                raise ValueError("'num_moments' must be a positive integer")

        if vector_factory is None:
            self._vector_factory = _VectorFactory(
                RandomVectors(hamiltonian, rng=rng),
                num_vectors=num_vectors,
                accumulate=accumulate_vectors)
        else:
            if not isinstance(vector_factory, Iterable):
                raise TypeError('vector_factory must be iterable')
            try:
                len(vector_factory)
            except TypeError:
                if num_vectors is None:
                    raise ValueError('num_vectors must be provided if'
                                     'vector_factory has no length.')
            self._vector_factory = _VectorFactory(
                vector_factory,
                num_vectors=num_vectors,
                accumulate=accumulate_vectors)
        num_vectors = self._vector_factory.num_vectors

        self._last_two_alphas = []
        self._moments_list = []

        self.num_moments = num_moments
        self._update_moments_list(self.num_moments, num_vectors)

        # set kernel before calling moments
        self.kernel = kernel if kernel is not None else jackson_kernel
        moments = self._moments()
        self.densities, self._gammas = _calc_fft_moments(moments)

    @property
    def energies(self):
        return (self._a * _chebyshev_nodes(SAMPLING * self.num_moments)
                + self._b)
    @property
    def num_vectors(self):
        return len(self._moments_list)

    def __call__(self, energy=None):
        """Return the spectral density evaluated at ``energy``.

        Parameters
        ----------
        energy : float or sequence of floats, optional

        Returns
        -------
        energies : array of floats
            Drawn from the nodes of the highest Chebyshev polynomial.
            Not returned if 'energy' was not provided
        densities : float or array of floats
            single ``float`` if the ``energy`` parameter is a single
            ``float``, else an array of ``float``.

        Notes
        -----
        If ``energy`` is not provided, then the densities are obtained
        by Fast Fourier Transform of the Chebyshev moments.
        """
        if energy is None:
            return self.energies, self.densities
        else:
            energy = np.asarray(energy)
            e = (energy - self._b) / self._a
            g_e = (np.pi * np.sqrt(1 - e) * np.sqrt(1 + e))

            moments = self._moments()
            # factor 2 comes from the norm of the Chebyshev polynomials
            moments[1:] = 2 * moments[1:]

            return np.transpose(chebval(e, moments) / g_e)

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
            exclusive with ``energy_resolution``.
        energy_resolution: positive float, optional
            Features wider than this resolution are visible
            in the spectral density. Mutually exclusive with
            ``num_moments``.
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
        self.densities, self._gammas = _calc_fft_moments(moments)

    def add_vectors(self, num_vectors=None):
        """Increase the number of vectors

        Parameters
        ----------
        num_vectors: positive int, optional
            The number of vectors to add.
        """
        self._vector_factory.add_vectors(num_vectors)
        num_vectors = self._vector_factory.num_vectors - self.num_vectors

        self._update_moments_list(self.num_moments,
                                  self.num_vectors + num_vectors)

        # recalculate quantities derived from the moments
        moments = self._moments()
        self.densities, self._gammas = _calc_fft_moments(moments)

    def _moments(self):
        moments = np.real_if_close(self._moments_list)
        # put moments in the first axis, to return an array of densities
        moments = np.swapaxes(moments, 0, 1)
        if self.mean:
            moments = np.mean(moments, axis=1)
        # divide by scale factor to reflect the integral rescaling
        moments /= self._a
        # stabilized moments with a kernel
        moments = self.kernel(moments)
        return moments

    def _update_moments_list(self, n_moments, num_vectors):
        """Calculate the Chebyshev moments of an operator's spectral
        density.

        The algorithm is based on the KPM method as depicted in `Rev.
        Mod. Phys., Vol. 78, No. 1 (2006)
        <https://arxiv.org/abs/cond-mat/0504627>`_.

        Parameters
        ----------
        n_moments : integer
            Number of Chebyshev moments.
        num_vectors : integer
            Number of vectors used for sampling.
        """

        if self.num_vectors == num_vectors:
            r_start = 0
            new_vectors = 0
        elif self.num_vectors < num_vectors:
            r_start = self.num_vectors
            new_vectors = num_vectors - self.num_vectors
        else:
            raise ValueError('Cannot decrease number of vectors')
        self._moments_list.extend([0.] * new_vectors)
        self._last_two_alphas.extend([0.] * new_vectors)

        if n_moments == self.num_moments:
            m_start = 2
            new_moments = 0
            if new_vectors == 0:
                # nothing new to calculate
                return
        else:
            if not self._vector_factory.accumulate:
                raise ValueError("Cannot increase the number of moments if "
                                 "'accumulate_vectors' is 'False'.")
            new_moments = n_moments - self.num_moments
            m_start = self.num_moments
            if new_moments < 0:
                raise ValueError('Cannot decrease number of moments')

            if new_vectors != 0:
                raise ValueError("Only 'num_moments' *or* 'num_vectors' "
                                 "may be updated at a time.")

        for r in range(r_start, num_vectors):
            alpha_zero = self._vector_factory[r]

            one_moment = [0.] * n_moments
            if new_vectors > 0:
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

            if self._vector_factory.accumulate:
                self._last_two_alphas[r] = (alpha, alpha_next)
                self._moments_list[r] = one_moment[:]
            else:
                self._moments_list[r] = one_moment


class Correlator:
    """Calculates the response of the correlation between two operators.

    The response tensor :math:`χ_{α β}` of an operator :math:`O_α`
    to a perturbation in an operator :math:`O_β`, is defined here based
    on [3]_, and [4]_, and takes the form

    .. math::
       χ_{α β}(µ, T) =
       \\int_{-\\infty}^{\\infty}{\\mathrm{d}E} f(µ-E, T)
       \\left({O_α ρ(E) O_β \\frac{\\mathrm{d}G^{+}}{\\mathrm{d}E}} -
       {O_α \\frac{\\mathrm{d}G^{-}}{\\mathrm{d}E} O_β ρ(E)}\\right)

    .. [3] `Phys. Rev. Lett. 114, 116602 (2015)
       <https://arxiv.org/abs/1410.8140>`_.
    .. [4] `Phys. Rev. B 92, 184415 (2015)
       <https://doi.org/10.1103/PhysRevB.92.184415>`_

    Internally, the correlation is approximated with a
    two dimensional KPM expansion,

    .. math::

       χ_{α β}(µ, T) =
       \\int_{-1}^1{\\mathrm{d}E} \\frac{f(µ-E,T)}{(1-E^2)^2}
       \\sum_{m,n}Γ_{n m}(E)µ_{n m}^{α β},

    with coefficients

    .. math::

       Γ_{m n}(E) =
       (E - i n \\sqrt{1 - E^2}) e^{i n \\arccos(E)} T_m(E)

       + (E + i m \\sqrt{1 - E^2}) e^{-i m \\arccos(E)} T_n(E),

    and moments matrix
    :math:`µ_{n m}^{α β} = \\mathrm{Tr}(O_α T_m(H) O_β T_n(H))`.

    The trace is calculated creating two instances of
    `~kwant.kpm.SpectralDensity`, and saving the vectors
    :math:`Ψ_{n r} = O_β T_n(H)\\rvert r\\rangle`,
    and :math:`Ω_{m r} = T_m(H) O_α\\rvert r\\rangle` , where
    :math:`\\rvert r\\rangle` is a vector provided by the
    ``vector_factory``.
    The moments matrix is formed with the product
    :math:`µ_{m n} = \\langle Ω_{m r} \\rvert Ψ_{n r}\\rangle` for
    every :math:`\\rvert r\\rangle`.

    Parameters
    ----------
    hamiltonian : `~kwant.system.FiniteSystem` or matrix Hamiltonian
        If a system is passed, it should contain no leads.
    operator1, operator2 : operators, dense matrix, or sparse matrix, optional
        Operators to be passed to two different instances of
        `~kwant.kpm.SpectralDensity`.
    **kwargs : dict
        Keyword arguments to pass to `~kwant.kpm.SpectralDensity`.

    Notes
    -----
    The ``operator1`` must act to the right as :math:`O_α\\rvert r\\rangle`.
    """

    def __init__(self, hamiltonian, operator1=None, operator2=None, **kwargs):

        # Normalize 'operator1' and 'operator2' to functions that take
        # and return a vector.
        params = kwargs.get('params')
        self.mean = kwargs.get('mean', True)
        accumulate_vectors = kwargs.get('accumulate_vectors', False)
        kwargs['accumulate_vectors'] = True
        kwargs.pop('operator', None)
        self.operator1 = _normalize_operator(operator1, params)
        self.operator2 = _normalize_operator(operator2, params)

        # initialize `SpectralDensity` to get `T_n(H)|r>` with a fake operator
        def fake_op(bra, ket): return ket

        # The vector factory used is the one passed by the user (or rng)
        # to save the vectors, accumulate_vectors must be 'True'
        self._spectrum_R = SpectralDensity(hamiltonian, operator=fake_op,
                                           **kwargs)
        self._a = self._spectrum_R._a
        self._b = self._spectrum_R._b
        _a = self._a * (1 - self._spectrum_R.eps / 2)
        bounds = (self._b - _a, self._b + _a)
        self.num_vectors = self._spectrum_R.num_vectors
        self.num_moments = self._spectrum_R.num_moments
        # apply operator2 to obtain `Psi_{n,r} = op2 T_n(H)|r>`
        self._update_psi()

        # instantiate the second `SpectralDensity`
        # `accumulate_vectors` is set to the user defined option
        # rewrite the bounds to match the rescaled bounds in the next call
        kwargs['accumulate_vectors'] = accumulate_vectors
        kwargs['num_vectors'] = self.num_vectors
        kwargs['num_moments'] = self.num_moments
        kwargs['energy_resolution'] = None
        # Now we must take operator1 applied to the initial
        # vectors to get `op1|r>`
        # The vector factory used is defined below to ensure applying the
        # same initial vectors stored in `self._vector_factory.saved_vectors`
        kwargs['vector_factory'] = self._op_factory()
        kwargs['bounds'] = bounds
        self._spectrum_L = SpectralDensity(hamiltonian, operator=fake_op,
                                           **kwargs)
        # and now self._moments_list is `Omega_{m,r} = T_m(H) op1|r>`
        # The shape of '_omega' is '(num_vecs, num_moments, dim_output)',
        # where 'dim_output' is the dimension of the output of 'operator1'
        self._omega = np.array(self._spectrum_L._moments_list)

        self._calculate_moments_matrix()
        self._build_integral_factor()

    def __call__(self, mu=0, temperature=0):
        """Returns the linear response :math:`χ_{α β}(µ, T)`

        Parameters
        ----------
        mu : float
            Chemical potential defined in the same units of energy as
            the Hamiltonian.
        temperature : float
            Temperature in units of energy, the same as defined in the
            Hamiltonian.
        """
        e = self.energies
        e_rescaled = (e - self._b) / self._a

        # rescale the energy to compare with the chemical potential
        distribution_array = fermi_distribution(e, mu, temperature)
        integrand = np.divide(distribution_array, (1 - e_rescaled ** 2) ** 2)
        integrand = np.multiply(integrand, self._integral_factor)
        integral = simps(integrand, x=e_rescaled)
        # gives the linear response in units of volume * e^2/h
        prefactor = 2 * 4**2 / ((2 * self._a) ** 2)
        return prefactor * integral

    @property
    def energies(self):
        return self._spectrum_R.energies

    def add_moments(self, num_moments=None, *, energy_resolution=None):
        """Increase the number of Chebyshev moments

        Parameters
        ----------
        num_moments: positive int, optional
            The number of Chebyshev moments to add. Mutually
            exclusive with 'energy_resolution'.
        energy_resolution: positive float, optional
            Features wider than this resolution are visible
            in the spectral density. Mutually exclusive with
            ``num_moments``.
        """

        self._spectrum_R.add_moments(num_moments=num_moments,
                                     energy_resolution=energy_resolution)
        self.num_moments = self._spectrum_R.num_moments
        # apply operator2 to obtain `Psi_{n,r} = op2
        self._update_psi()

        self._spectrum_L.add_moments(num_moments=num_moments,
                                     energy_resolution=energy_resolution)
        self._omega = np.array(self._spectrum_L._moments_list)

        self._calculate_moments_matrix()
        self._build_integral_factor()

    def add_vectors(self, num_vectors=None):
        """Increase the number of vectors

        Parameters
        ----------
        num_vectors: positive int, optional
            The number of vectors to add.
        """
        # get `T_n(H)|r>` with a fake operator
        self._spectrum_R.add_vectors(num_vectors)
        # apply operator2 to obtain `Psi_{n,r} = op2 T_n(H)|r>`
        self._update_psi()

        # _spectrum_L vector_factory is linked to _spectrum_R vector_factory
        self._spectrum_L.add_vectors(num_vectors)
        self.num_vectors = self._spectrum_L.num_vectors
        # and now self._moments_list is `Omega_{m,r} = T_m(H) op1|r>`
        self._omega = np.array(self._spectrum_L._moments_list)

        self._calculate_moments_matrix()
        self._build_integral_factor()

    def _calculate_moments_matrix(self):
        """Return the moments matrix, averaged over the vectors used """
        # The final matrix is ready to be computed as
        # `µ_{m,n} = <Omega_{m,r} | Psi_{n,r}>`
        # for every `r` in `num_vectors`.
        # 'moments_matrix' will be an array of moments matrix for each vector
        # the shape of `moments_matrix` is
        # `(num_vecs, num_moments, num_moments)`
        self.moments_matrix = self._omega.conjugate() @ self._psi
        if self.mean:
            self.moments_matrix = np.mean(self.moments_matrix, axis=0)

    def _op_factory(self):
        """Factory of vectors ``operator1(vec[idx])``.

        This factory will get updated with more vectors when
        ``_spectrum_R._vector_factory`` gets updated to include more
        vectors.
        """
        for vector in self._spectrum_R._vector_factory:
            yield self.operator1(vector)
        return

    def _update_psi(self):
        """Axes are swapped in the end the get the shape
        '(num_vecs, dim_output, num_moments)', where 'dim_output'
        is the dimension of the output of 'operator2'."""
        self._psi = np.array([
            [
                self.operator2(self._spectrum_R._moments_list[r][n])
                for n in range(self._spectrum_R.num_moments)
            ]
            for r in range(self._spectrum_R.num_vectors)
        ]).swapaxes(1, 2)

    def _build_integral_factor(self):
        """ Build the integral factor

        .. math::
           Γ_{m n}(E)
           = (E - i n \\sqrt{1 - E^2}) e^{i n \\arccos(E)} T_m(E)

           + (E + i m \\sqrt{1 - E^2}) e^{-i m \\arccos(E)} T_n(E),

        times the moments matrix :math:`µ_{m n}` and sum over :math:`m`
        and :math:`n`. :math:`E` is the array of the sampling points
        selected as the Chebyshev nodes.
        """

        n_moments = self.num_moments

        # get kernel array
        g_kernel = self._spectrum_R.kernel(np.ones(n_moments))
        g_kernel[0] /= 2
        mu_kernel = np.outer(g_kernel, g_kernel) * self.moments_matrix

        e = (self.energies - self._b) / self._a

        # arrays for faster calculation
        sqrt_e = np.sqrt(1 - e ** 2)
        arccos_e = np.arccos(e)

        exp_n = np.exp(1j * np.outer(arccos_e, np.arange(n_moments)))
        t_n = np.real(exp_n)

        e_plus = (np.outer(e, np.ones(n_moments)) -
                  1j * np.outer(sqrt_e, np.arange(n_moments)))
        e_plus = e_plus * exp_n

        big_gamma = e_plus[:, None, :] * t_n[:, :, None]
        big_gamma += big_gamma.conj().swapaxes(1, 2)

        self._integral_factor = np.tensordot(mu_kernel, big_gamma.T)


def conductivity(hamiltonian, alpha='x', beta='x', positions=None, **kwargs):
    """Returns a callable object to obtain the elements of the
    conductivity tensor using the Kubo-Bastin approach.

    A `~kwant.kpm.Correlator` instance is created to obtain the
    correlation between two components of the current operator

    .. math::

       σ_{α β}(µ, T) =
       \\frac{1}{V} \\int_{-\\infty}^{\\infty}{\\mathrm{d}E} f(µ-E, T)
       \\left({j_α ρ(E) j_β \\frac{\\mathrm{d}G^{+}}{\\mathrm{d}E}} -
       {j_α \\frac{\\mathrm{d}G^{-}}{\\mathrm{d}E} j_β ρ(E)}\\right),

    where :math:`V` is the volume where the conductivity is sampled.
    In this implementation it is assumed that the vectors are normalized
    and :math:`V=1`, otherwise the result of this calculation must be
    normalized with the corresponding volume.

    The equations used here are based on [3]_ and [4]_

    .. [3] `Phys. Rev. Lett. 114, 116602 (2015)
       <https://arxiv.org/abs/1410.8140>`_.
    .. [4] `Phys. Rev. B 92, 184415 (2015)
       <https://doi.org/10.1103/PhysRevB.92.184415>`_

    Parameters
    ----------
    hamiltonian : `~kwant.system.FiniteSystem` or matrix Hamiltonian
        If a system is passed, it should contain no leads.
    alpha, beta : str, or operators
        If ``hamiltonian`` is a kwant system, or if the ``positions``
        are provided, ``alpha`` and ``beta`` can be the directions of the
        velocities as strings {'x', 'y', 'z'}.
        Otherwise ``alpha`` and ``beta`` should be the proper velocity
        operators, which can be members of `~kwant.operator` or matrices.
    positions : array of float, optioinal
        If ``hamiltonian`` is a matrix, the velocities can be calculated
        internally by passing the positions of each orbital in the system
        when ``alpha`` or ``beta`` are one of the directions {'x', 'y', 'z'}.
    **kwargs : dict
        Keyword arguments to pass to `~kwant.kpm.Correlator`.

    Examples
    --------
    We will obtain the conductivity of the Haldane model, defined as a
    honeycomb lattice with first nearest neighbors coupling, and
    imaginary second nearest neighbors coupling.

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
    >>> syst[lat.neighbors()] = -1
    >>> syst[lat.a.neighbors()] = -0.5j
    >>> syst[lat.b.neighbors()] = 0.5j
    >>> fsyst = syst.finalized()

    Now we can call `~kwant.kpm.conductivity` to calculate the transverse
    conductivity at chemical potential 0 and temperature 0.01.

    >>> cond = kwant.kpm.conductivity(fsyst, alpha='x', beta='y')
    >>> cond(mu=0, temperature=0.01)
    """

    if positions is None and not isinstance(hamiltonian, system.System):
        raise ValueError("If 'hamiltonian' is a matrix, positions "
                         "must be provided")

    params = kwargs.get('params')
    alpha = _velocity(hamiltonian, params, alpha, positions)
    beta = _velocity(hamiltonian, params, beta, positions)

    correlator = Correlator(
        hamiltonian, operator1=alpha, operator2=beta, **kwargs)

    return correlator


class _VectorFactory:
    """Factory for Hilbert space vectors.

    Parameters
    ----------
    vectors : iterable
        Iterable of Hilbert space vectors.
    num_vectors : int, optional
        Total number of vectors. If not specified, will be set to the
        length of 'vectors'.
    accumulate : bool, default: True
        If True, the attribute 'saved_vectors' will store the vectors
        generated.
    """

    def __init__(self, vectors=None, num_vectors=None, accumulate=True):
        assert isinstance(vectors, Iterable)
        try:
            _len = len(vectors)
            if num_vectors is None:
                num_vectors = _len
        except TypeError:
            _len = np.inf
            if num_vectors is None:
                raise ValueError("'num_vectors' must be specified when "
                                 "'vectors' has no len() method.")
        self._max_vectors = _len
        self._iterator = iter(vectors)

        self.accumulate = accumulate
        self.saved_vectors = []

        self.num_vectors = 0
        self.add_vectors(num_vectors=num_vectors)

        self._last_idx = -np.inf
        self._last_vector = None

    def _fill_in_saved_vectors(self, index):
        if index < self._last_idx and not self.accumulate:
            raise ValueError("Cannot get previous values if 'accumulate' "
                             "is False")

        if index >= self.num_vectors:
            raise IndexError('Requested more vectors than available')

        self._last_idx = index
        if self.accumulate:
            if self.saved_vectors[index] is None:
                self.saved_vectors[index] = next(self._iterator)
        else:
            self._last_vector = next(self._iterator)

    def __getitem__(self, index):
        self._fill_in_saved_vectors(index)
        if self.accumulate:
            return self.saved_vectors[index]
        return self._last_vector

    def add_vectors(self, num_vectors=None):
        """Increase the number of vectors

        Parameters
        ----------
        num_vectors: positive int, optional
            The number of vectors to add.
        """
        if num_vectors is None:
            raise ValueError("'num_vectors' must be specified.")
        else:
            if num_vectors <= 0 or num_vectors != int(num_vectors):
                raise ValueError("'num_vectors' must be a positive integer")
            elif self.num_vectors + num_vectors > self._max_vectors:
                raise ValueError("'num_vectors' is larger than available "
                                 "vectors")

        self.num_vectors += num_vectors

        if self.accumulate:
            self.saved_vectors.extend([None] * num_vectors)


def RandomVectors(syst, where=None, rng=None):
    """Returns a random phase vector iterator for the sites in 'where'.

    Parameters
    ----------
    syst : `~kwant.system.FiniteSystem` or matrix Hamiltonian
        If a system is passed, it should contain no leads.
    where : sequence of `int` or `~kwant.builder.Site`, or callable, optional
        Spatial range of the vectors produced. If ``syst`` is a
        `~kwant.system.FiniteSystem`, where behaves as in
        `~kwant.operator.Density`. If ``syst`` is a matrix, ``where``
        must be a list of integers with the indices where column vectors
        are nonzero.
    """
    rng = ensure_rng(rng)
    tot_norbs, orbs = _normalize_orbs_where(syst, where)
    while True:
        vector = np.zeros(tot_norbs, dtype=complex)
        vector[orbs] = np.exp(2j * np.pi * rng.random_sample(len(orbs)))
        yield vector


class LocalVectors:
    """Generates a local vector iterator for the sites in 'where'.

    Parameters
    ----------
    syst : `~kwant.system.FiniteSystem` or matrix Hamiltonian
        If a system is passed, it should contain no leads.
    where : sequence of `int` or `~kwant.builder.Site`, or callable, optional
        Spatial range of the vectors produced. If ``syst`` is a
        `~kwant.system.FiniteSystem`, where behaves as in
        `~kwant.operator.Density`. If ``syst`` is a matrix, ``where``
        must be a list of integers with the indices where column vectors
        are nonzero.
    """
    def __init__(self, syst, where=None, *args):
        self.tot_norbs, self.orbs = _normalize_orbs_where(syst, where)
        self._idx = 0

    def __len__(self,):
        return len(self.orbs)

    def __iter__(self,):
        return self

    def __next__(self,):
        if self._idx < len(self):
            vector = np.zeros(self.tot_norbs)
            vector[self.orbs[self._idx]] = 1
            self._idx = self._idx + 1
            return vector
        raise StopIteration('Too many vectors requested from this generator')

# ### Auxiliary functions

def fermi_distribution(energy, mu, temperature):
    """Returns the Fermi distribution f(e, µ, T) evaluated at 'e'.

    Parameters
    ----------
    energy : float or sequence of floats
        Energy array where the Fermi distribution is evaluated.
    mu : float
        Chemical potential defined in the same units of energy as
        the Hamiltonian.
    temperature : float
        Temperature in units of energy, the same as defined in the
        Hamiltonian.
    """
    if temperature < 0:
        raise ValueError("temperature must be non-negative")
    elif temperature == 0:
        return np.array(np.less(energy - mu, 0), dtype=float)
    else:
        return 1 / (1 + np.exp((energy - mu) / temperature))

def _from_where_to_orbs(syst, where):
    """Returns a list of slices of the orbitals in 'where'"""
    assert isinstance(syst, system.System)
    where = _normalize_site_where(syst, where)
    _site_ranges = np.asarray(syst.site_ranges, dtype=gint_dtype)
    offsets, norbs = _get_all_orbs(where, _site_ranges)
    # concatenate all the orbitals
    orbs = [list(range(start, start+orbs))
            for start, orbs in zip(offsets[:, 0], norbs[:, 0])]
    orbs = reduce(add, orbs)
    return orbs


def _normalize_orbs_where(syst, where):
    """Return total number of orbitals and a list of slices of
    orbitals in 'where'"""
    if isinstance(syst, system.System):
        tot_norbs = _get_tot_norbs(syst)
        orbs = _from_where_to_orbs(syst, where)
    else:
        try:
            tot_norbs = csr_matrix(syst).shape[0]
        except TypeError:
            raise TypeError("'syst' is neither a matrix "
                             "nor a Kwant system.")
        orbs = (range(tot_norbs) if where is None
                else np.asarray(where, int))
    return tot_norbs, orbs


def _velocity(hamiltonian, params, op_type, positions):
    """Construct the velocity operator

    Parameters
    ----------
    hamiltonian : ndarray or a Kwant system
        System for which the velocity operator is calculated.
    params : dict
        Parametres of the system
    op_type : str, matrix or operator
        If ``op_type`` is a 'str' in {'x', 'y', 'z'}, the velocity operator
        is calculated using the ``hamiltonian`` and ``positions``, else
        if ``op_type`` is an operator or a matrix, this is the velocity
        operator.
    positions : ndarray of shape ``(N, dim)``
        Positions of each orbital. This parameter is not used if
        ``hamiltonian`` is a Kwant system.
    """
    directions = dict(x=0, y=1, z=2)

    if isinstance(op_type, _LocalOperator):
        operator = op_type
    elif isinstance(op_type, str):
        direction = directions[op_type]
        if isinstance(hamiltonian, system.System):
            operator, norbs, norbs = hamiltonian.hamiltonian_submatrix(
                params=params, sparse=True, return_norb=True
            )
            positions = np.vstack([[hamiltonian.pos(i)] * norb
                                   for i, norb in enumerate(norbs)])
        elif positions is not None:
            operator = coo_matrix(hamiltonian, copy=True)
        displacements = positions[operator.col] - positions[operator.row]
        if direction > displacements.shape[1]:
            raise ValueError("{} is not an allowed direction.".format(op_type))
        operator.data *= 1j * displacements[:, direction]
        operator = operator.tocsr()
    else:
        try:
            operator = csr_matrix(op_type)
        except Exception:
            raise ValueError("Velocity operator must be provided as a matrix, "
                             "a kwant operator, or a direction.")
    return operator


def _normalize_operator(op, params):
    """Normalize 'op' to a function that takes and returns a vector."""
    if op is None:
        def r_op(v): return v
    elif isinstance(op, _LocalOperator):
        op = op.bind(params=params)
        r_op = op.act
    elif callable(op):
        r_op = op
    elif hasattr(op, 'dot'):
        op = csr_matrix(op)
        r_op = op.dot
    else:
        raise TypeError("The operators must have a '.dot' "
                        "attribute or must be callable.")
    return r_op


def jackson_kernel(moments):
    """Convolutes ``moments`` with the Jackson kernel.

    Taken from Eq. (71) of `Rev. Mod. Phys., Vol. 78, No. 1 (2006)
    <https://arxiv.org/abs/cond-mat/0504627>`_.
    """

    n_moments, *extra_shape = moments.shape
    m = np.arange(n_moments)
    kernel_array = ((n_moments - m + 1) *
                    np.cos(np.pi * m/(n_moments + 1)) +
                    np.sin(np.pi * m/(n_moments + 1)) /
                    np.tan(np.pi/(n_moments + 1)))
    kernel_array /= n_moments + 1

    # transposes handle the case where operators have vector outputs
    conv_moments = np.transpose(moments.transpose() * kernel_array)
    return conv_moments


def lorentz_kernel(moments, l=4):
    """Convolutes ``moments`` with the Lorentz kernel.

    Taken from Eq. (71) of `Rev. Mod. Phys., Vol. 78, No. 1 (2006)
    <https://arxiv.org/abs/cond-mat/0504627>`_.

    The additional parameter ``l`` controls the decay of the kernel.
    """

    n_moments, *extra_shape = moments.shape

    m = np.arange(n_moments)
    kernel_array = np.sinh(l * (1 - m / n_moments)) / np.sinh(l)

    # transposes handle the case where operators have vector outputs
    conv_moments = np.transpose(moments.transpose() * kernel_array)
    return conv_moments


def _rescale(hamiltonian, eps, v0, bounds):
    """Rescale a Hamiltonian and return a LinearOperator

    Parameters
    ----------
    hamiltonian : 2D array
        Hamiltonian of the system.
    eps : scalar
        Ensures that the bounds are strict.
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
        lmax = float(eigsh(hamiltonian, k=1, which='LA',
                           return_eigenvectors=False, tol=tol, v0=v0))
        lmin = float(eigsh(hamiltonian, k=1, which='SA',
                           return_eigenvectors=False, tol=tol, v0=v0))

    a = np.abs(lmax-lmin) / (2. - eps)
    b = (lmax+lmin) / 2.

    if lmax - lmin <= abs(lmax + lmin) * tol / 2:
        raise ValueError(
            'The Hamiltonian has a single eigenvalue, it is not possible to '
            'obtain a spectral density.')

    def rescaled(v):
        return (hamiltonian.dot(v) - b * v) / a

    rescaled_ham = LinearOperator(shape=hamiltonian.shape, matvec=rescaled)

    return rescaled_ham, (a, b)

def _chebyshev_nodes(n_sampling):
    """Return an array of 'n_sampling' points in the interval (-1,1)"""
    raw, step = np.linspace(np.pi, 0, n_sampling,
                            endpoint=False, retstep=True)
    return np.cos(raw + step / 2)


def _calc_fft_moments(moments):
    """This function takes the stabilized moments and returns an array
    of points and an array of the evaluated function at those points.
    """
    moments = np.asarray(moments)
    # extra_shape handles the case where operators have vector outputs
    n_moments, *extra_shape = moments.shape
    n_sampling = SAMPLING * n_moments
    moments_ext = np.zeros([n_sampling] + extra_shape, dtype=moments.dtype)

    # special points at the abscissas of Chebyshev integration
    e_rescaled = _chebyshev_nodes(n_sampling)

    # transposes handle the case where operators have vector outputs
    moments_ext[:n_moments] = moments
    # The function evaluated in these special data points is the FFT of
    # the moments as in Eq. (83).
    # The order of gammas must be reversed to match the energies in
    # ascending order
    gammas = np.transpose(fft.dct(moments_ext.transpose(), type=3))[::-1]

    # Element-wise division of moments_FFT over gk, as in Eq. (83).
    gk = np.pi * np.sqrt(1 - e_rescaled ** 2)
    rho = np.transpose(np.divide(gammas.transpose(), gk))

    return rho, gammas
