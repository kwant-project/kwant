# Physics background
# ------------------
# - conductance of a NS-junction (Andreev reflection, superconducting gap)
#
# Kwant features highlighted
# --------------------------
# - Implementing electron and hole ("orbital") degrees of freedom
#   using different lattices

import kwant

# For plotting
import pylab

# For matrix support
import numpy

def make_system(a=1, W=10, L=10, barrier=1.5, barrierpos=(3, 4),
                mu=0.4, Delta=0.1, Deltapos=4, t=1.0):
    # Start with an empty tight-binding system and two square lattices,
    # corresponding to electron and hole degree of freedom
    lat_e = kwant.lattice.Square(a)
    lat_h = kwant.lattice.Square(a)

    sys = kwant.Builder()

    #### Define the scattering region. ####
    sys[(lat_e(x, y) for x in range(L) for y in range(W))] = 4 * t - mu
    sys[(lat_h(x, y) for x in range(L) for y in range(W))] = mu - 4 * t

    # the tunnel barrier
    sys[(lat_e(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = 4 * t + barrier - mu
    sys[(lat_h(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = mu - 4 * t - barrier

    # hoppings in x and y-directions, for both electrons and holes
    sys[sys.possible_hoppings((1, 0), lat_e, lat_e)] = - t
    sys[sys.possible_hoppings((0, 1), lat_e, lat_e)] = - t
    sys[sys.possible_hoppings((1, 0), lat_h, lat_h)] = t
    sys[sys.possible_hoppings((0, 1), lat_h, lat_h)] = t

    # Superconducting order parameter enters as hopping between
    # electrons and holes
    sys[((lat_e(x,y), lat_h(x, y)) for x in range(Deltapos, L)
         for y in range(W))] = Delta

    #### Define the leads. ####
    # left electron lead
    sym_lead0 = kwant.TranslationalSymmetry([lat_e.vec((-1, 0))])
    lead0 = kwant.Builder(sym_lead0)

    lead0[(lat_e(0, j) for j in xrange(W))] = 4 * t - mu
    # hoppings in x and y-direction
    lead0[lead0.possible_hoppings((1, 0), lat_e, lat_e)] = - t
    lead0[lead0.possible_hoppings((0, 1), lat_e, lat_e)] = - t

    # left hole lead
    sym_lead1 = kwant.TranslationalSymmetry([lat_h.vec((-1, 0))])
    lead1 = kwant.Builder(sym_lead1)

    lead1[(lat_h(0, j) for j in xrange(W))] = mu - 4 * t
    # hoppings in x and y-direction
    lead1[lead1.possible_hoppings((1, 0), lat_h, lat_h)] = t
    lead1[lead1.possible_hoppings((0, 1), lat_h, lat_h)] = t

    # Then the lead to the right
    # this one is superconducting and thus is comprised of electrons
    # AND holes
    sym_lead2 = kwant.TranslationalSymmetry([lat_e.vec((1, 0))])
    lead2 = kwant.Builder(sym_lead2)

    lead2[(lat_e(0, j) for j in xrange(W))] = 4 * t - mu
    lead2[(lat_h(0, j) for j in xrange(W))] = mu - 4 * t
    # hoppings in x and y-direction
    lead2[lead2.possible_hoppings((1, 0), lat_e, lat_e)] = - t
    lead2[lead2.possible_hoppings((0, 1), lat_e, lat_e)] = - t
    lead2[lead2.possible_hoppings((1, 0), lat_h, lat_h)] = t
    lead2[lead2.possible_hoppings((0, 1), lat_h, lat_h)] = t
    lead2[((lat_e(0, j), lat_h(0, j)) for j in xrange(W))] = Delta

    #### Attach the leads and return the finalized system. ####
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)
    sys.attach_lead(lead2)

    return sys.finalized()

def plot_conductance(fsys, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.solve(fsys, energy)
        # Conductance is N - R_ee + R_he
        data.append(smatrix.submatrix(0, 0).shape[0] -
                    smatrix.transmission(0, 0) +
                    smatrix.transmission(1, 0))

    pylab.plot(energies, data)
    pylab.xlabel("energy [in units of t]")
    pylab.ylabel("conductance [in units of e^2/h]")
    pylab.show()


def main():
    fsys = make_system()

    # Check that the system looks as intended.
    kwant.plot(fsys)

    plot_conductance(fsys, energies=[0.002 * i for i in xrange(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
