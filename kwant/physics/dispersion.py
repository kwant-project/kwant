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

    Notes
    -----
    An instance of this class can be called like a function.  Given a momentum
    (currently this must be a scalar as all infinite systems are quasi-1-d), it
    returns a NumPy array containing the eigenenergies of all modes at this
    momentum

    Examples
    --------
    >>> bands = kwant.physics.Bands(some_syst)
    >>> momenta = numpy.linspace(-numpy.pi, numpy.pi, 101)
    >>> energies = [bands(k) for k in momenta]
    >>> pyplot.plot(momenta, energies)
    >>> pyplot.show()
    """

    def __init__(self, sys, args=()):
        syst = sys
        ensure_isinstance(syst, system.InfiniteSystem)
        self.ham = syst.cell_hamiltonian(args)
        if not np.allclose(self.ham, self.ham.T.conj()):
            raise ValueError('The cell Hamiltonian is not Hermitian.')
        hop = syst.inter_cell_hopping(args)
        self.hop = np.empty(self.ham.shape, dtype=complex)
        self.hop[:, : hop.shape[1]] = hop
        self.hop[:, hop.shape[1]:] = 0

    def __call__(self, k):
        # Note: Equation to solve is
        #       (V^\dagger e^{ik} + H + V e^{-ik}) \psi = E \psi
        mat = self.hop * complex(math.cos(k), -math.sin(k))
        mat += mat.conjugate().transpose() + self.ham
        return np.sort(np.linalg.eigvalsh(mat).real)
