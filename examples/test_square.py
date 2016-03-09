import square
from pytest import raises
from numpy.testing import assert_almost_equal

def test_nodeid_to_from_pos():
    s = square.System((3, 4), 1)
    raises(Exception, s.nodeid_from_pos, (0, -2))
    raises(Exception, s.nodeid_from_pos, (-1, 3))
    raises(Exception, s.nodeid_from_pos, (3, 1))
    raises(Exception, s.pos_from_nodeid, -1)
    raises(Exception, s.pos_from_nodeid, 12)
    assert s.nodeid_from_pos((0, 0)) == 0
    assert s.nodeid_from_pos(s.pos_from_nodeid(7)) == 7
    assert s.pos_from_nodeid(s.nodeid_from_pos((2, 3))) == (2, 3)

def test_hamiltonian():
    sys = square.System((4, 5), 1)
    for i in range(sys.graph.num_nodes):
        shape = sys.hamiltonian(i, i).shape
        assert len(shape) == 2
        assert shape[0] == 1
        for j in sys.graph.out_neighbors(i):
            m = sys.hamiltonian(i, j)
            shape = m.shape
            m_herm = m.T.conj()
            assert_almost_equal(m, m_herm)
            assert_almost_equal(m_herm, sys.hamiltonian(j, i))

def test_selfenergy():
    syst = square.System((2, 4), 1)
    for lead in range(len(syst.lead_interfaces)):
        se = syst.leads[lead].selfenergy(0)
        assert len(se.shape) == 2
        assert se.shape[0] == se.shape[1]
        assert se.shape[0] == len(syst.lead_interfaces[lead])
