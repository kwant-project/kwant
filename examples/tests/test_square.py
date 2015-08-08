import square
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_almost_equal

def test_nodeid_to_from_pos():
    s = square.System((3, 4), 1)
    assert_raises(Exception, s.nodeid_from_pos, (0, -2))
    assert_raises(Exception, s.nodeid_from_pos, (-1, 3))
    assert_raises(Exception, s.nodeid_from_pos, (3, 1))
    assert_raises(Exception, s.pos_from_nodeid, -1)
    assert_raises(Exception, s.pos_from_nodeid, 12)
    assert_equal(s.nodeid_from_pos((0, 0)), 0)
    assert_equal(s.nodeid_from_pos(s.pos_from_nodeid(7)), 7)
    assert_equal(s.pos_from_nodeid(s.nodeid_from_pos((2, 3))), (2, 3))

def test_hamiltonian():
    sys = square.System((4, 5), 1)
    for i in range(sys.graph.num_nodes):
        shape = sys.hamiltonian(i, i).shape
        assert_equal(len(shape), 2)
        assert_equal(shape[0], 1)
        for j in sys.graph.out_neighbors(i):
            m = sys.hamiltonian(i, j)
            shape = m.shape
            m_herm = m.T.conj()
            assert_almost_equal(m, m_herm)
            assert_almost_equal(m_herm, sys.hamiltonian(j, i))

def test_selfenergy():
    sys = square.System((2, 4), 1)
    for lead in range(len(sys.lead_interfaces)):
        se = sys.leads[lead].selfenergy(0)
        assert_equal(len(se.shape), 2)
        assert_equal(se.shape[0], se.shape[1])
        assert_equal(se.shape[0], len(sys.lead_interfaces[lead]))
