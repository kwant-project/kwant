# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Band structure calculation for the leads"""

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
    momentum. If `derivative_order > 0` or `return_eigenvectors = True`,
    additional arrays are returned returned.

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

        # Switch computation from diag(a^\dagger @ b) to sum(a.T * b, axis=0),
        # which is computationally more efficient for larger matrices.
        # a and b are NxN matrices, and we switch for N > _crossover_size.
        # Both ways are numerically (but not necessarily bitwise) equivalent.
        self._crossover_size = 8

    def __call__(self, k, derivative_order=0, return_eigenvectors=False):
        """Calculate all energies :math:`E` (optionally higher derivatives
        :math:`E', E''` and eigenvectors) for a given momentum :math:`k`

        :math:`E_n, \quad E'_n = dE_n / dk, \quad E''_n = d^2E_n / dk^2,
        \quad n \in \{0, nbands - 1\}`

        :math:`nbands` is the number of open modes

        Eigenvectors are orthonormal.


        Parameters
        ----------
        k : float
            momentum

        derivative_order : {0, 1, 2}, optional
            Maximal derivative order to calculate. Default is zero

        return_eigenvectors : bool, optional
            if set to `True` return the eigenvectors as last tuple element


        Returns
        ----------
        energies : numpy float array, size `nbands`
            energies :math:`E`
        velocities : numpy float array, size `nbands`
            velocities (first order energy derivatives :math:`E'`)
        curvatures : numpy float array, size `nbands`
            curvatures (second energy derivatives :math:`E''`)
        eigenvectors : numpy float array, size `nbands`
            eigenvectors

        The number of retured elements varies. If `derivative_order > 0`
        we return all derivatives up to the order `derivative_order`,
        and total the number of returned elements is `derivative_order + 1`
        If 'return_eigenvectors=True', in addition the eigenvectors are
        returned as the last element. In that case, the total number of
        returned elements is `derivative_order + 2`.

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
        # The perturbed Hamiltonian H(k) is
        # H(k+dk) = H_0(k) + H_1(k) dk + H_2(k) dk^2 + O(dk^3)
        # and we use h0, h1 and h2 in the code to refer to above terms

        mat = self.hop * complex(math.cos(k), -math.sin(k))
        h0 = mat + mat.conjugate().transpose() + self.ham
        if derivative_order == 0:
            if return_eigenvectors:
                energies, eigenvectors = np.linalg.eigh(h0)
                return energies, eigenvectors
            return np.sort(np.linalg.eigvalsh(h0).real)

        energies, eigenvectors = np.linalg.eigh(h0)
        h1 = 1j*(- mat + mat.conjugate().transpose())
        if derivative_order == 1 and len(energies) > self._crossover_size:
            velocities = np.sum(eigenvectors.conjugate() * (h1 @ eigenvectors),
                                axis=0).real
        else:
            ph1p = eigenvectors.conjugate().transpose() @ h1 @ eigenvectors
            velocities = np.diag(ph1p).real
        if derivative_order == 1:
            if return_eigenvectors:
                return energies, velocities, eigenvectors
            return energies, velocities

        # ediff_{i,j} =  1 / (E_i - E_j) if i != j, else 0
        ediff = energies.reshape(-1, 1) - energies.reshape(1, -1)
        ediff = np.divide(1, ediff, out=np.zeros_like(ediff), where=ediff != 0)
        h2 = - (mat + mat.conjugate().transpose())
        curvatures = np.sum(eigenvectors.conjugate() * (h2 @ eigenvectors)
                            - 2 * ediff * np.abs(ph1p)**2, axis=0).real
        # above expression is similar to: curvatures =
        # np.diag(eigenvectors.conjugate().transpose() @ h2 @ eigenvectors
        #         + 2 * ediff @ np.abs(ph1p)**2).real
        if derivative_order == 2:
            if return_eigenvectors:
                return energies, velocities, curvatures, eigenvectors
            return energies, velocities, curvatures
        raise NotImplementedError('derivative_order= {} not implemented'
                                  .format(derivative_order))
