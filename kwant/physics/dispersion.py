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

    Notes
    -----
    An instance of this class can be called like a function.  Given a momentum
    (currently this must be a scalar as all infinite systems are quasi-1-d), it
    returns a NumPy array containing the eigenenergies of all modes at this
    momentum
    """

    def __init__(self, sys):
        self.ham = sys.slice_hamiltonian()
        if not np.allclose(self.ham, self.ham.T.conj()):
            raise ValueError('The slice Hamiltonian is not Hermitian.')
        hop = sys.inter_slice_hopping()
        self.hop = np.empty(self.ham.shape, dtype=complex)
        self.hop[:, : hop.shape[1]] = hop
        self.hop[:, hop.shape[1]:] = 0

    def __call__(self, k):
        mat = self.hop * complex(math.cos(k), math.sin(k))
        mat += mat.conjugate().transpose() + self.ham
        return np.sort(np.linalg.eigvalsh(mat).real)
