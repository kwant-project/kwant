"""An example of how to directly implement a system without using
kwant.Builder.
"""

import numpy as np
from matplotlib import pyplot
import kwant
from kwant.physics.leads import square_selfenergy

__all__ = ['System']


class Lead:
    def __init__(self, width, t, potential):
        self.width = width
        self.t = t
        self.potential = potential

    def selfenergy(self, fermi_energy, args=(), params=None):
        assert not args
        assert params is None
        return square_selfenergy(self.width, self.t,
                                 self.potential + fermi_energy)


class System(kwant.system.FiniteSystem):
    def __init__(self, shape, hopping,
                 potential=0, lead_potentials=(0, 0),
                 return_scalars_as_matrix=True):
        """`potential` can be a container (indexed by a pair of integers) or a
        function (taking a pair of integers as its parameter) or a number.
        Checked in this order.
        """
        assert len(shape) == 2
        for s in shape:
            assert int(s) == s
            assert s >= 1

        self.as_matrix = return_scalars_as_matrix
        self.shape = shape
        if hasattr(potential, '__getitem__'):
            self.pot = potential.__getitem__
        elif hasattr(potential, '__call__'):
            self.pot = potential
        else:
            self.pot = lambda xy: potential
        self.t = hopping

        # Build rectangular mesh graph
        g = kwant.graph.Graph()
        increment = [1, shape[0]]
        for along, across in [(0, 1), (1, 0)]:
            # Add edges in direction "along".
            if shape[along] < 2: continue
            edges = np.empty((2 * shape[across], 2), dtype=int)
            edges[:shape[across], 0] = np.arange(
                0, shape[across] * increment[across], increment[across])
            edges[:shape[across], 1] = edges[:shape[across], 0]
            edges[:shape[across], 1] += increment[along]
            edges[shape[across]:, (0, 1)] = edges[:shape[across], (1, 0)]
            g.add_edges(edges)
            for i in range(shape[along] - 2):
                edges += increment[along]
                g.add_edges(edges)
        self.graph = g.compressed()

        self.lead_interfaces = []
        for x in [0, shape[0] - 1]:
            # We have to use list here, as numpy.array does not understand
            # generators.
            interface = list(self.nodeid_from_pos((x, y))
                             for y in range(shape[1]))
            self.lead_interfaces.append(np.array(interface))

        self.leads = [Lead(shape[1], hopping, lead_potentials[i])
                      for i in range(2)]

    def hamiltonian(self, i, j, *args, params=None):
        """Return an submatrix of the tight-binding Hamiltonian."""
        assert not args
        assert params is None
        if i == j:
            # An on-site Hamiltonian has been requested.
            result = 4 * self.t + self.pot(self.pos_from_nodeid(i))
        else:
            # A hopping element has been requested.
            result = -self.t
        if self.as_matrix:
            result = np.array([[result]], dtype=complex)
        return result

    def nodeid_from_pos(self, pos):
        for i in range(2):
            assert int(pos[i]) == pos[i]
            assert pos[i] >= 0 and pos[i] < self.shape[i]
        return pos[0] + pos[1] * self.shape[0]

    def pos_from_nodeid(self, nodeid):
        result = (nodeid % self.shape[0]), (nodeid // self.shape[0])
        assert result[1] >= 0 and result[1] < self.shape[1]
        return result


def main():
    syst = System((10, 5), 1)
    energies = [0.04 * i for i in range(100)]
    data = [kwant.greens_function(syst, energy).transmission(1, 0)
            for energy in energies]

    pyplot.plot(energies, data)
    pyplot.xlabel("energy [in units of t]")
    pyplot.ylabel("conductance [in units of e^2/h]")
    pyplot.show()


if __name__ == '__main__':
    main()
