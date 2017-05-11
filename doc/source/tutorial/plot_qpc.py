import math
import cmath

import matplotlib.pyplot as plt

import kwant


#HIDDEN_BEGIN_syst
def hopping(sitei, sitej, phi):
    xi, yi = sitei.pos
    xj, yj = sitej.pos
    return -cmath.exp(-0.5j * phi * (xi - xj) * (yi + yj))


def onsite(site, salt):
    return 0.05 * kwant.digest.gauss(site.tag, salt) + 4


def make_system(L=50):
    def central_region(pos):
        x, y = pos
        return -2*L < x < 2*L and \
            abs(y) < L - 37.5 * math.exp(-x**2 / 12**2)

    lat = kwant.lattice.square(norbs=1)
    syst = kwant.Builder()

    syst[lat.shape(central_region, (0, 0))] = onsite
    syst[lat.neighbors()] = hopping

    sym = kwant.TranslationalSymmetry((-1, 0))
    lead = kwant.Builder(sym)
    lead[(lat(0, y) for y in range(-L + 1, L))] = 4
    lead[lat.neighbors()] = hopping

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst.finalized()
#HIDDEN_END_syst


def main():
    syst = make_system()

    kwant.plot(syst)

#HIDDEN_BEGIN_wf
    params = dict(phi=1/40, salt="")
    psi = kwant.wave_function(syst, energy=0.15, params=params)(0)
    J = kwant.operator.Current(syst).bind(params=params)
    D = kwant.operator.Density(syst).bind(params=params)
    # Calculate density and current due to modes from the left lead
    density = sum(D(p) for p in psi)
    current = sum(J(p) for p in psi)
#HIDDEN_END_wf

#HIDDEN_BEGIN_density
    kwant.plotter.map(syst, density, method='linear')
#HIDDEN_END_density

#HIDDEN_BEGIN_current
    kwant.plotter.current(syst, current)
#HIDDEN_END_current


if __name__ == '__main__':
    main()
