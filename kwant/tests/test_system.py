import numpy as np
from scipy import sparse
from nose.tools import assert_raises, assert_almost_equal
import kwant

def test_hamiltonian_submatrix():
    sys = kwant.Builder()
    sys.default_site_group = kwant.lattice.Chain()
    for i in xrange(3):
        sys[(i,)] = 0.5 * i
    for i in xrange(2):
        sys[(i,), (i + 1,)] = 1j * (i + 1)

    sys2 = sys.finalized()
    mat = sys2.hamiltonian_submatrix()[0]
    assert mat.shape == (3, 3)
    # Sorting is required due to unknown compression order of builder.
    perm = np.argsort(sys2.onsite_hamiltonians)
    mat_should_be = np.mat('0 1j 0; -1j 0.5 2j; 0 -2j 1')

    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)

    mat = sys2.hamiltonian_submatrix(sparse=True)[0]
    assert sparse.isspmatrix_coo(mat)
    mat = mat.todense()
    mat = mat[perm, :]
    mat = mat[:, perm]
    np.testing.assert_array_equal(mat, mat_should_be)

    mat = sys2.hamiltonian_submatrix(perm[[0, 1]], perm[[2]])[0]
    np.testing.assert_array_equal(mat, mat_should_be[: 2, 2])

    mat = sys2.hamiltonian_submatrix(perm[[0, 1]], perm[[2]], sparse=True)[0]
    mat = mat.todense()
    np.testing.assert_array_equal(mat, mat_should_be[: 2, 2])

    # Test for correct treatment of matrix input.
    sys = kwant.Builder()
    sys.default_site_group = kwant.lattice.Chain()
    sys[(0,)] = np.mat('0 1j; -1j 0')
    sys[(1,)] = np.mat('1')
    sys[(2,)] = np.mat('2')
    sys[(1,), (0,)] = np.mat('1 2j')
    sys[(2,), (1,)] = np.mat('3j')
    sys2 = sys.finalized()
    mat_dense = sys2.hamiltonian_submatrix()[0]
    mat_sp = sys2.hamiltonian_submatrix(sparse=True)[0].todense()
    np.testing.assert_array_equal(mat_sp, mat_dense)

    # Test for shape errors.
    sys[(0,), (2,)] = np.mat('1 2')
    sys2 = sys.finalized()
    assert_raises(ValueError, sys2.hamiltonian_submatrix)
    assert_raises(ValueError, sys2.hamiltonian_submatrix, None, None, True)

def test_energies():
    sys = kwant.Builder(kwant.TranslationalSymmetry([(-1, 0)]))
    sys.default_site_group = kwant.lattice.Square()
    sys[[(0, 0), (0, 1)]] = complex(0)
    sys[[((0, 0), (0, 1)),
         ((0, 0), (1, 0))]] = complex(0, 1)
    for e in sys.finalized().energies(0):
        assert_almost_equal(abs(e), 1)
