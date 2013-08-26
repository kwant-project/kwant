import math
from cmath import exp
import kwant
from kwant.contrib.digest import gauss


def hopping(sitei, sitej, phi, salt):
    xi, yi = sitei.pos
    xj, yj = sitej.pos
    return -exp(-0.5j * phi * (xi - xj) * (yi + yj))


def onsite(site, phi, salt):
    return 0.3 * gauss(repr(site), salt) + 4


def test_qhe(W=16, L=8):
    def central_region(pos):
        x, y = pos
        return -L < x < L and \
            abs(y) < W - 5.5 * math.exp(-x**2 / 5**2)

    lat = kwant.lattice.square()
    sys = kwant.Builder()

    sys[lat.shape(central_region, (0, 0))] = onsite
    sys[lat.neighbors()] = hopping

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[(lat(0, y) for y in range(-W + 1, W))] = 4
    lead[lat.neighbors()] = hopping

    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())
    sys = sys.finalized()

    #### The following chunk of code can be uncommented to visualize the
    #### conductance plateaus.
    # from matplotlib import pyplot
    # import numpy
    # reciprocal_phis = numpy.linspace(0.1, 7, 200)
    # conductances = []
    # for phi in 1 / reciprocal_phis:
    #     smatrix = kwant.smatrix(sys, 1.0, [phi, ""])
    #     conductances.append(smatrix.transmission(1, 0))
    # pyplot.plot(reciprocal_phis, conductances)
    # pyplot.show()

    for r_phis, T_nominal, max_err in [((1.3, 2.1), 1, 1e-7),
                                       ((3.2, 3.7), 2, 1e-3),
                                       ((5.2, 5.5), 3, 1e-1)]:
        for r_phi in r_phis:
            args = (1.0 / r_phi, "")
            T_actual = kwant.smatrix(sys, 1.0, args).transmission(1, 0)
            assert abs(T_nominal - T_actual) < max_err
