# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['DiscreteSymmetry']

import numpy as np
from scipy.sparse import identity, csr_matrix, hstack


def almost_identity(mat):
    return np.all(abs(mat - identity(mat.shape[0])).data < 1e-10)


def _column_sum(matrix):
    """Sum the columns of a sparse matrix.

    This is fully analogous to ``matrix.sum(0)``, and uses an implementation
    similar to that in scipy v1.1.0, however it avoids using ``numpy.matrix``
    interface and therefore does not raise a ``PendingDeprecationWarning``.
    This should be removed once we depend on scipy v1.2.0, where the warning is
    silenced.
    """
    return matrix.T @ np.ones(matrix.shape[0])


def cond_conj(op, conj):
    return op.conj() if conj else op


_conj = [True, True, False]
_names = ['Time reversal', 'Particle-hole', 'Chiral']
_signs = [1, -1, -1]

class DiscreteSymmetry:
    r"""A collection of discrete symmetries and conservation laws.

    Parameters
    ----------
    projectors : iterable of rectangular or square sparse matrices
        Projectors that block-diagonalize the Hamiltonian.
    time_reversal : square sparse matrix
        The unitary part of the time-reversal symmetry operator.
    particle_hole : square sparse matrix
        The unitary part of the particle-hole symmetry operator.
    chiral : square sparse matrix
        The chiral symmetry operator.

    Notes
    -----

    When computing scattering modes, the representation of the
    modes is chosen to reflect declared discrete symmetries and
    conservation laws.

    `projectors` block diagonalize the Hamiltonian, and modes are computed
    separately in each block. The ordering of blocks is the same as of
    `projectors`. If `conservation_law` is declared in
    `~kwant.builder.Builder`, `projectors` is computed as the projectors
    onto its orthogonal eigensubspaces. The projectors are stored in the
    order of ascending eigenvalues of `conservation_law`.

    Symmetrization using discrete symmetries varies depending on whether
    a conservation law/projectors are declared. Consider the case with no
    conservation law declared. With `time_reversal` declared, the outgoing
    modes are chosen as the time-reversed partners of the incoming modes,
    i.e. :math:`\psi_{out}(-k) = T \psi_{in}(k)` with k the momentum.
    `chiral` also relates incoming and outgoing modes, such that
    :math:`\psi_{out}(k) = C \psi_{in}(k)`. `particle_hole` gives symmetric
    incoming and outgoing modes separately, such that
    :math:`\psi_{in/out}(-k) = P \psi_{in/out}(k)`, except when k=-k, at
    k = 0 or :math:`\pi`. In this case, each mode is chosen as an eigenstate
    :math:`P \psi = \psi` if :math:`P^2=1`. If :math:`P^2=-1`, we
    symmetrize the modes by generating pairs of orthogonal modes
    :math:`\psi` and :math:`P\psi`. Because `chiral` and `particle_hole`
    flip the sign of energy, they only apply at zero energy.

    Discrete symmetries can be combined with a conservation law if they
    leave each block invariant or transform it to another block. With S
    a discrete symmetry and :math:`P_i` and :math:`P_j` projectors onto
    blocks i and j of the Hamiltonian, :math:`S_{ji} = P_j^+ S P_i` is the
    symmetry projection that maps from block i to block j.
    :math:`S_{ji} = P_j^+ S P_i` must for each j be nonzero for exactly
    one i. If S leaves block i invariant, the modes within block i are
    symmetrized using the nonzero projection :math:`S_{ii}`, like in the
    case without a conservation law. If S transforms between blocks i and
    j, the modes of the block with the larger index are obtained by
    transforming the modes of the block with the lower index. Thus, with
    :math:`\psi_i` and :math:`\psi_j` the modes of blocks i and j, we have
    :math:`\psi_j = S_{ji} \psi_i`.
    """
    def __init__(self, projectors=None, time_reversal=None, particle_hole=None,
                 chiral=None):
        # Normalize format
        if projectors is not None:
            try:
                projectors = [proj.tocsr() for proj in projectors]
            except AttributeError:
                raise TypeError("projectors must be a sequence of "
                                "sparse matrices.")

        symms = (time_reversal, particle_hole, chiral)
        try:
            symms = [symm.tocsr() if symm is not None else None
                     for symm in symms]
        except AttributeError:
            raise TypeError("Symmetries must be sparse matrices.")

        if None not in (time_reversal, particle_hole, chiral):
            product = chiral.dot(particle_hole).dot(time_reversal.conj())
            if not almost_identity(product):
                raise ValueError("Product of all symmetries "
                                 "should be equal to identity.")

        # Auto-compute a missing symmetry.
        if None not in (time_reversal, particle_hole) and chiral is None:
            chiral = particle_hole.dot(time_reversal.conj())
        elif None not in (particle_hole, chiral) and time_reversal is None:
            # Product of particle hole with itself to get the sign right.
            time_reversal = (particle_hole.dot(particle_hole.conj()).dot(
                             particle_hole).dot(chiral.conj()))
        elif None not in (chiral, time_reversal) and particle_hole is None:
            particle_hole = (time_reversal.dot(time_reversal.conj()).dot(
                             chiral).dot(time_reversal))

        symms = [time_reversal, particle_hole, chiral]
        for i, symm, conj, name in zip(range(3), symms, _conj, _names):
            if symm is None:
                continue
            if not almost_identity(symm.T.conj().dot(symm)):
                raise ValueError("{} symmetry should be "
                                 "{}unitary.".format(name, "anti" * conj))
            symm_squared = symm.dot(cond_conj(symm, conj))
            if not (almost_identity(symm_squared) or
                    (conj and almost_identity(-symm_squared))):
                raise ValueError("{} symmetry should square to "
                                 "{}1.".format(name, "±" * conj))
            symms[i] = symm

        if projectors is not None:
            projectors = [
                p[:, _column_sum(abs(p)) > 1e-10] for p in projectors
            ]
            if not almost_identity(sum(projector.dot(projector.conj().T) for
                                       projector in projectors)):
                raise ValueError("Projectors should stack to a unitary matrix.")

            # Check that the symmetries and conservation law are in canonical
            # form.  In the projector basis, each discrete symmetry should have
            # at most one nonzero block per row.
            for (symm, name, conj) in zip(symms, _names, _conj):
                if symm is None:
                    continue
                for p_i in projectors:
                    p_hc = p_i.T.conj()
                    nonzero_blocks = sum(
                            np.any(abs(p_hc.dot(symm).dot(
                                (cond_conj(p_j, conj))).data > 1e-7))
                                         for p_j in projectors)
                    if nonzero_blocks > 1:
                        raise ValueError(name + ' symmetry is not in canonical '
                                         'form: see DiscreteSymmetry '
                                         'documentation.')
        self.projectors = projectors
        self.time_reversal, self.particle_hole, self.chiral = symms

    def validate(self, matrix):
        """Check if a matrix satisfies the discrete symmetries.

        Parameters
        ----------
        matrix : sparse or dense matrix
            If rectangular, it is padded by zeros to be square.

        Returns
        -------
        broken_symmetries : list
            List of strings, the names of symmetries broken by the
            matrix: any combination of "Conservation law", "Time reversal",
            "Particle-hole", "Chiral". If no symmetries are broken, returns
            an empty list.
        """
        # Extra transposes are to enforse sparse dot product in case matrix is
        # dense.
        n, m = matrix.shape
        if n != m:
            if isinstance(matrix, np.ndarray):
                new_matrix = np.empty((n, n), dtype=matrix.dtype)
                new_matrix[:, :m] = matrix
                new_matrix[:, m:] = 0
                matrix = new_matrix
            else:
                matrix = hstack([matrix, csr_matrix((n, n-m))])
        broken_symmetries = []
        if self.projectors is not None:
            for proj in self.projectors:
                full = proj.dot(proj.T.conj())
                commutator = full.dot(matrix) - (full.T.dot(matrix.T)).T
                if np.linalg.norm(commutator.data) > 1e-8:
                    broken_symmetries.append('Conservation law')
                    break
        for symm, conj, sign, name in zip(self[1:], _conj, _signs, _names):
            if symm is None:
                continue
            commutator = symm.T.conj().dot((symm.T.dot(matrix.T)).T)
            commutator = commutator - sign * cond_conj(matrix, conj)
            if np.linalg.norm(commutator.data) > 1e-8:
                broken_symmetries.append(name)
        return broken_symmetries

    def __getitem__(self, item):
        return (self.projectors, self.time_reversal,
                self.particle_hole, self.chiral)[item]
