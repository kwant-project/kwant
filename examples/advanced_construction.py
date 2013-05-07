"""An example of advanced system creation."""

from __future__ import division
from math import tanh
import tinyarray as ta
import kwant


sigma_0 = ta.identity(2)
sigma_x = ta.array([[0, 1], [1, 0]])
sigma_y = ta.array([[0, -1j], [1j, 0]])
sigma_z = ta.array([[1, 0], [0, -1]])


def make_system(R):
    def in_ring(pos):
        x, y = pos
        return R**2 / 4 < x**2 + y**2 < R**2

    def in_lead(pos):
        x, y = pos
        return -R / 4 < y < R / 4

    def pot(site):
        x, y = site.pos
        return (0.1 * tanh(x / R) + tanh(2 * y / R)) * sigma_z

    lat = kwant.lattice.honeycomb()

    sys = kwant.Builder()
    sys[lat.shape(in_ring, (3 * R / 4, 0))] = pot
    sys[lat.neighbors()] = sigma_y

    lead = kwant.Builder(kwant.TranslationalSymmetry(lat.vec((-1, 0))))
    lead[lat.shape(in_lead, (0, 0))] = sigma_0
    lead[lat.neighbors()] = sigma_x
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())

    return sys.finalized()


def main():
    sys = make_system(100)
    print kwant.solve(sys, 1.1).transmission(0, 1)


if __name__ == '__main__':
    main()
