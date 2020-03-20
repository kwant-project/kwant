"""An example of advanced system creation."""

from math import tanh
from cmath import exp
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

    def pot(site, B):
        x, y = site.pos
        return (0.1 * tanh(x / R) + tanh(2 * y / R)) * sigma_z

    def hop(site1, site2, B):
        x1, y1 = site1.pos
        x2, y2 = site2.pos
        return - exp(.5j * B * (x1 - x2) * (y1 + y2)) * sigma_0

    lat = kwant.lattice.honeycomb()

    syst = kwant.Builder()
    syst[lat.shape(in_ring, (3 * R / 4, 0))] = pot
    syst[lat.neighbors()] = hop

    lead = kwant.Builder(kwant.TranslationalSymmetry(lat.vec((-1, 0))))
    lead[lat.shape(in_lead, (0, 0))] = sigma_0
    lead[lat.neighbors()] = sigma_x
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst.finalized()


def main():
    syst = make_system(100)
    print(kwant.smatrix(syst, 1.1, params=dict(B=0.1)).transmission(0, 1))


if __name__ == '__main__':
    main()
