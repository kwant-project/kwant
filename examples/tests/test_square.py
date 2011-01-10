from kwant import square
from nose.tools import assert_equal, assert_raises
from numpy.testing import assert_almost_equal

def test_nodeid_to_from_pos():
    s = square.System((3, 4), 1)
    assert_raises(StandardError, s.nodeid_from_pos, (0, -2))
    assert_raises(StandardError, s.nodeid_from_pos, (-1, 3))
    assert_raises(StandardError, s.nodeid_from_pos, (3, 1))
    assert_raises(StandardError, s.pos_from_nodeid, -1)
    assert_raises(StandardError, s.pos_from_nodeid, 12)
    assert_equal(s.nodeid_from_pos((0, 0)), 0)
    assert_equal(s.nodeid_from_pos(s.pos_from_nodeid(7)), 7)
    assert_equal(s.pos_from_nodeid(s.nodeid_from_pos((2, 3))), (2, 3))

def test_hamiltonian():
    sys = square.System((4, 5), 1)
    for i in xrange(sys.graph.num_nodes):
        shape = sys.hamiltonian(i, i).shape
        assert_equal(len(shape), 2)
        assert_equal(shape[0], sys.num_orbitals(i))
        for j in sys.graph.out_neighbors(i):
            m = sys.hamiltonian(i, j)
            shape = m.shape
            m_herm = m.T.conj()
            assert_almost_equal(m, m_herm)
            assert_almost_equal(m_herm, sys.hamiltonian(j, i))

def test_self_energy():
    sys = square.System((2, 4), 1)
    for lead in xrange(len(sys.lead_neighbor_seqs)):
        n_orb = sum(
            sys.num_orbitals(site) for site in sys.lead_neighbor_seqs[lead])
        se = sys.self_energy(lead, 0)
        assert_equal(len(se.shape), 2)
        assert_equal(se.shape[0], se.shape[1])
        assert_equal(se.shape[0], n_orb)
