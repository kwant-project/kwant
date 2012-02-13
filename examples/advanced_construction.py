"""An example of advanced system creation."""

from __future__ import division
import kwant
import numpy
from math import tanh, sqrt


def make_system(R=50):
    sigma_0 = numpy.identity(2)
    sigma_x = numpy.array([[0, 1], [1, 0]])
    sigma_y = numpy.array([[0, -1j], [1j, 0]])
    sigma_z = numpy.array([[1, 0], [0, -1]])

    def in_ring(pos):
        return R**2 / 4 < pos[0]**2 + pos[1]**2 < R**2

    def pot(site):
        x, y = site.pos
        return (0.1 * tanh(x / R) + tanh(2 * y / R)) * sigma_z

    def in_lead(pos):
        return -1 < pos[0] < 1.3 and - R/4 < pos[1] < R/4

    lat = kwant.lattice.Honeycomb()

    sys = kwant.Builder()
    sys[lat.shape(in_ring, (3 * R / 4, 0))] = pot
    for hopping in lat.nearest:
        sys[sys.possible_hoppings(*hopping)] = sigma_y

    lead = kwant.Builder(kwant.TranslationalSymmetry([lat.vec((-1, 0))]))
    lead[lat.shape(in_lead, (0,0))] = sigma_0
    for hopping in lat.nearest:
        lead[lead.possible_hoppings(*hopping)] = sigma_x
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())

    return sys.finalized()


def main():
    fsys = make_system(100)
    print kwant.solve(fsys, 0.1).transmission(0, 1)


if __name__ == '__main__':
    main()
