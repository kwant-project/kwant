# Copyright 2011-2016 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.


import numpy as np
import scipy.linalg as la
from scipy import sparse
from scipy.sparse import csr_matrix as csr
from pytest import raises
from kwant.physics import DiscreteSymmetry
import kwant
from kwant._common import ensure_rng

def test_projectors():
    """Test setting projectors"""
    cons_vals = [-1, -1, 1, 1, 1, 2]
    U = kwant.rmt.circular(6, 'A', rng=8)
    cons = U.T.conj().dot(np.diag(cons_vals)).dot(U)
    # Get the projectors.
    evals, evecs = np.linalg.eigh(cons)
    # Make sure the ordering is correct.
    assert np.allclose(evals-np.array(cons_vals), 0)
    projectors = [np.reshape(evecs[:, :2], (6, 2)),
                  np.reshape(evecs[:, 2:5], (6, 3)),
                  np.reshape(evecs[:, 5], (6, 1))]
    # Ensure that the projectors sum to a unitary and set them.
    assert np.allclose(sum(projector.dot(projector.conj().T) for projector in
                           projectors), np.eye(6))
    # set_symmetry only accepts sparse matrices, so setting np.array
    # projectors should throw an error.
    raises(TypeError, DiscreteSymmetry, projectors=projectors)
    # Make the projectors sparse. Can now set them without trouble.
    projectors = [sparse.coo_matrix(p) for p in projectors]
    DiscreteSymmetry(projectors=projectors)
    # Break one projector - setting them should now throw an error.
    projectors = [p.toarray() for p in projectors]
    projectors[1][2, :] = 0
    assert not np.allclose(sum(projector.dot(projector.conj().T) for projector
                               in projectors), np.eye(6))
    projectors = [sparse.csr_matrix(p) for p in projectors]
    raises(ValueError, DiscreteSymmetry, projectors=projectors)
    # Test setting square projectors.
    p0 = np.eye(20)
    p0[:, ::2] = 0
    p1 = np.eye(20)
    p1[:, 1::2] = 0
    projectors = [sparse.coo_matrix(p0), sparse.coo_matrix(p1)]
    symmetry = DiscreteSymmetry(projectors=projectors)
    # Check that set_symmetry removes zero columns properly.
    p0 = p0[:, 1::2]
    p1 = p1[:, ::2]
    assert np.allclose(symmetry.projectors[0].toarray(), p0)
    assert np.allclose(symmetry.projectors[1].toarray(), p1)

def test_set_discrete_symm():
    n = 20
    # All discrete symmetries
    sym = 'CII'
    t_mat = np.array(kwant.rmt.h_t_matrix[sym])
    t_mat = np.kron(np.identity(n // len(t_mat)), t_mat)
    p_mat = np.array(kwant.rmt.h_p_matrix[sym])
    p_mat = np.kron(np.identity(n // len(p_mat)), p_mat)
    c_mat = p_mat.dot(t_mat.conj())

    # Set one symmetry at a time.
    symms = [(t_mat, None, None), (None, p_mat, None), (None, None, c_mat)]
    for symm in symms:
        T, P, C = symm
        # Try to set the symmetries as numpy arrays. An error should be thrown.
        raises(TypeError, DiscreteSymmetry, time_reversal=T, particle_hole=P,
               chiral=C)

        s_symm = (sparse.coo_matrix(op) if op is not None else None
                  for op in symm)
        T, P, C = s_symm
        DiscreteSymmetry(time_reversal=T, particle_hole=P, chiral=C)
        # Break the symmetry operator - an error should be thrown.
        b_symm = (sparse.csr_matrix(kwant.rmt.circular(op.shape[0], 'A',
                                                       rng=2))
                  if op is not None else None for op in symm)
        T, P, C = b_symm
        raises(ValueError, DiscreteSymmetry, time_reversal=T, particle_hole=P,
               chiral=C)

    # Set all three symmetries
    t_mat = sparse.csr_matrix(t_mat)
    p_mat = sparse.csr_matrix(p_mat)
    c_mat = sparse.csr_matrix(c_mat)
    DiscreteSymmetry(time_reversal=t_mat, particle_hole=p_mat, chiral=c_mat)

    # Break chiral symmetry - replace it with identity,
    # such that the product of symmetries is incorrect,
    # and an error should the thrown.
    raises(ValueError, DiscreteSymmetry, time_reversal=t_mat,
           particle_hole=p_mat, chiral=sparse.eye(n))
    raises(ValueError, DiscreteSymmetry, time_reversal=2*t_mat)
    raises(ValueError, DiscreteSymmetry, chiral=2*t_mat)

    # Check that with two symmetries specified, the third one is computed.
    symm = DiscreteSymmetry(time_reversal=t_mat, particle_hole=p_mat)
    assert np.allclose(symm.chiral.toarray(), c_mat.toarray())
    DiscreteSymmetry(time_reversal=t_mat, chiral=c_mat)
    assert np.allclose(symm.particle_hole.toarray(), p_mat.toarray())
    DiscreteSymmetry(particle_hole=p_mat, chiral=c_mat)
    assert np.allclose(symm.time_reversal.toarray(), t_mat.toarray())


def test_projectors_and_symmetry():
    # Consider a Hamiltonian with three blocks: the first one has a discrete
    # symmetry and the other two are related by the same discrete symmetry (but
    # do not possess it themselves).  Further, the first block is twice the
    # size of the latter two. For this situation and if the symmetry is chiral
    # or time reversal symmetry, the full symmetry operators are:
    n = 20
    c_mat1 = np.kron(np.identity(n), np.diag([1, -1]))
    c_mat2 = np.kron(np.identity(n // 2), np.diag([1, -1]))
    sx = np.array([[0,1],[1,0]])
    C_mat = la.block_diag(c_mat1, np.kron(sx, c_mat2))
    assert np.allclose(C_mat.dot(C_mat), np.eye(C_mat.shape[0]))
    t_mat = np.array(kwant.rmt.h_t_matrix['CII'])
    t_mat1 = np.kron(np.identity(2*n // len(t_mat)), t_mat)
    t_mat2 = np.kron(np.identity(n // len(t_mat)), t_mat)
    T_mat = la.block_diag(t_mat1, np.kron(sx, t_mat2))
    assert np.allclose(T_mat.dot(T_mat.conj()), -np.eye(T_mat.shape[0]))
    # The projectors that block diagonalize the Hamiltonian are
    I = np.eye(4*n)
    projectors = [I[:, :2*n], I[:, 2*n:3*n], I[:, 3*n:]]
    projectors = [sparse.csr_matrix(p) for p in projectors]
    # The symmetry and projectors are in canonical form and they commute,
    # so declaring should not throw an error or raise a warning.
    U = sparse.csr_matrix(kwant.rmt.circular(4*n, 'A', rng=234))

    for (C, T) in [(sparse.csr_matrix(C_mat), None),
                   (None, sparse.csr_matrix(T_mat))]:
        DiscreteSymmetry(projectors=projectors, chiral=C, time_reversal=T)
        # Transform symmetries with a random unitary, such that they are
        # no longer in canonical form, but leave projectors as is. An error
        # should be thrown.
        if C is not None: C = U.T.conj().dot(C).dot(U)
        if T is not None: T = U.T.conj().dot(T).dot(U.conj())
        raises(ValueError, DiscreteSymmetry, projectors=projectors, chiral=C,
               time_reversal=T)


def test_validate():
    csr = sparse.csr_matrix
    sym = DiscreteSymmetry(projectors=[csr(np.array([[1], [0]])),
                                       csr(np.array([[0], [1]]))])
    assert sym.validate(csr(np.array([[0], [1]]))) == ['Conservation law']
    assert sym.validate(np.array([[1], [0]])) == []
    assert sym.validate(np.eye(2)) == []
    assert sym.validate(1 - np.eye(2)) == ['Conservation law']

    sym = DiscreteSymmetry(particle_hole=sparse.identity(2))
    assert sym.validate(1j * sparse.identity(2)) == []
    assert sym.validate(sparse.identity(2)) == ['Particle-hole']

    sym = DiscreteSymmetry(time_reversal=sparse.identity(2))
    assert sym.validate(sparse.identity(2)) == []
    assert sym.validate(1j * sparse.identity(2)) == ['Time reversal']

    sym = DiscreteSymmetry(chiral=csr(np.diag((1, -1))))
    assert sym.validate(np.eye(2)) == ['Chiral']
    assert sym.validate(1 - np.eye(2)) == []


def random_onsite_hop(n, rng=0):
    rng = ensure_rng(rng)
    onsite = rng.randn(n, n) + 1j * rng.randn(n, n)
    onsite = onsite + onsite.T.conj()
    hop = rng.rand(n, n) + 1j * rng.rand(n, n)
    return onsite, hop


def test_validate_commutator():
    symm_class = ['AI', 'AII', 'D', 'C', 'AIII', 'BDI']
    sym_dict = {'AI': ['Time reversal'],
                'AII': ['Time reversal'],
                'D': ['Particle-hole'],
                'C': ['Particle-hole'],
                'AIII': ['Chiral'],
                'BDI': ['Time reversal', 'Particle-hole', 'Chiral']}
    n = 10
    rng = 10
    for sym in symm_class:
        # Random matrix in symmetry class
        h = kwant.rmt.gaussian(n, sym, rng=rng)
        if kwant.rmt.p(sym):
            p_mat = np.array(kwant.rmt.h_p_matrix[sym])
            p_mat = csr(np.kron(np.identity(n // len(p_mat)), p_mat))
        else:
            p_mat = None
        if kwant.rmt.t(sym):
            t_mat = np.array(kwant.rmt.h_t_matrix[sym])
            t_mat = csr(np.kron(np.identity(n // len(t_mat)), t_mat))
        else:
            t_mat = None
        if kwant.rmt.c(sym):
            c_mat = csr(np.kron(np.identity(n // 2), np.diag([1, -1])))
        else:
            c_mat = None
        disc_symm = DiscreteSymmetry(particle_hole=p_mat,
                                     time_reversal=t_mat,
                                     chiral=c_mat)
        assert disc_symm.validate(h) == []
        a = random_onsite_hop(n, rng=rng)[1]
        for symmetry in disc_symm.validate(a):
            assert symmetry in sym_dict[sym]
