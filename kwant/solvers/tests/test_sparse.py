from nose.plugins.skip import Skip, SkipTest
from  kwant.solvers.sparse import solve, ldos, wave_func
import kwant.solvers.sparse
import _test_sparse

def test_output():
    _test_sparse.test_output(solve)


def test_one_lead():
    _test_sparse.test_one_lead(solve)


def test_smatrix_shape():
    _test_sparse.test_smatrix_shape(solve)


def test_two_equal_leads():
    _test_sparse.test_two_equal_leads(solve)


def test_graph_system():
    _test_sparse.test_graph_system(solve)


def test_singular_graph_system():
    _test_sparse.test_singular_graph_system(solve)


def test_tricky_singular_hopping():
    _test_sparse.test_tricky_singular_hopping(solve)


def test_self_energy():
    _test_sparse.test_self_energy(solve)


def test_self_energy_reflection():
    _test_sparse.test_self_energy_reflection(solve)


def test_very_singular_leads():
    _test_sparse.test_very_singular_leads(solve)


def test_umfpack_del():
    if kwant.solvers.sparse.uses_umfpack:
        assert hasattr(kwant.solvers.sparse.umfpack.UmfpackContext,
                       '__del__')
    else:
        raise SkipTest


def test_ldos():
    _test_sparse.test_ldos(ldos)


def test_wavefunc_ldos_consistency():
    _test_sparse.test_wavefunc_ldos_consistency(wave_func, ldos)
