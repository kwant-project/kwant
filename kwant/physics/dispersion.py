# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import math
import numpy as np

__all__ = ['Bands']

class Bands(object):
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
    >>> bands = kwant.physics.Bands(some_sys)
    >>> momenta = numpy.linspace(-numpy.pi, numpy.pi, 101)
    >>> energies = [bands(k) for k in momenta]
    >>> pyplot.plot(momenta, energies)
    >>> pyplot.show()
    """

    def __init__(self, sys, args=()):
        self.ham = sys.cell_hamiltonian(args=args)
        if not np.allclose(self.ham, self.ham.T.conj()):
            raise ValueError('The cell Hamiltonian is not Hermitian.')
        hop = sys.inter_cell_hopping(args=args)
        self.hop = np.empty(self.ham.shape, dtype=complex)
        self.hop[:, : hop.shape[1]] = hop
        self.hop[:, hop.shape[1]:] = 0

    def __call__(self, k):
        mat = self.hop * complex(math.cos(k), math.sin(k))
        mat += mat.conjugate().transpose() + self.ham
        return np.sort(np.linalg.eigvalsh(mat).real)
