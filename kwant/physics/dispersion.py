# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import math
import numpy as np
from .. import system
from .._common import ensure_isinstance

__all__ = ['Bands']


class Bands:
    """
    Class of callable objects for the computation of energy bands.

    Parameters
    ----------
    sys : `kwant.system.InfiniteSystem`
        The low level infinite system for which the energies are to be
        calculated.
    args : tuple, defaults to empty
        Positional arguments to pass to the ``hamiltonian`` method.
        Mutually exclusive with 'params'.
    params : dict, optional
        Dictionary of parameter names and their values. Mutually exclusive
        with 'args'.

    Notes
    -----
    An instance of this class can be called like a function.  Given a momentum
    (currently this must be a scalar as all infinite systems are quasi-1-d), it
    returns a NumPy array containing the eigenenergies of all modes at this
    momentum. Velocities and velocity derivatives are calculated if the flag
    ``deriv`` is set.

    Examples
    --------
    >>> bands = kwant.physics.Bands(some_syst)
    >>> momenta = numpy.linspace(-numpy.pi, numpy.pi, 101)
    >>> energies = [bands(k) for k in momenta]
    >>> pyplot.plot(momenta, energies)
    >>> pyplot.show()
    """

    def __init__(self, sys, args=(), *, params=None):
        syst = sys
        ensure_isinstance(syst, system.InfiniteSystem)
        self.ham = syst.cell_hamiltonian(args, params=params)
        if not np.allclose(self.ham, self.ham.T.conj()):
            raise ValueError('The cell Hamiltonian is not Hermitian.')
        hop = syst.inter_cell_hopping(args, params=params)
        self.hop = np.empty(self.ham.shape, dtype=complex)
        self.hop[:, : hop.shape[1]] = hop
        self.hop[:, hop.shape[1]:] = 0

    def __call__(self, k, deriv=0):
        """Calculate all energies :math:`E`, velocities :math:`v`
        and velocity derivatives `v'` for a given momentum :math:`k`

        :math:`E_n, \quad v_n = dE_n / dk, \quad v'_n = d^2E_n / dk^2,
        \quad n \in \{0, nbands - 1\}`

        :math:`nbands` is the number of open modes

        Parameters
        ----------
        k : float
            momentum

        deriv : {0, 1, 2}, optional
            Maximal derivative order to calculate. Default is zero


        Returns
        ----------
        ener : numpy float array
            energies (and optionally also higher momentum derivatives)

            if deriv = 0
                numpy float array of the energies :math:`E`, shape (nbands,)
            if deriv > 0
                numpy float array, shape (deriv + 1, nbands) of
                energies and derivatives :math:`(E, E', E'')`

        Notes
        -----
        *   The ordering of the energies and velocities is the same and
            according to the magnitude of the energy eigenvalues.
            Therefore, the bands are not continously ordered.
        *   The curvature `E''` can be only calculated
            for non-degenerate bands.
        """
        # Equation to solve is
        # (V^\dagger e^{ik} + H + V e^{-ik}) \psi = E \psi
        # we obtain the derivatives by perturbing around momentum k

        mat = self.hop * complex(math.cos(k), -math.sin(k))
        ham = mat + mat.conjugate().transpose() + self.ham
        if deriv == 0:
            return np.sort(np.linalg.eigvalsh(ham).real)

        ener, psis = np.linalg.eigh(ham)
        h1 = 1j*(- mat + mat.conjugate().transpose())
        ph1p = np.dot(psis.conjugate().transpose(), np.dot(h1, psis))
        velo = np.real(np.diag(ph1p))
        if deriv == 1:
            return np.array([ener, velo])

        ediff = ener.reshape(-1, 1) - ener.reshape(1, -1)
        ediff = np.divide(1, ediff, out=np.zeros_like(ediff), where=ediff != 0)
        h2 = - (mat + mat.conjugate().transpose())
        curv = (np.real(np.diag(
                np.dot(psis.conjugate().transpose(), np.dot(h2, psis)))) +
                2 * np.sum(ediff * np.abs(ph1p)**2, axis=1))
        if deriv == 2:
            return np.array([ener, velo, curv])
        raise NotImplementedError('deriv= {} not implemented'.format(deriv))
