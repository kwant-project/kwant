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
from matplotlib import pyplot


#HIDDEN_BEGIN_zuuw
def make_system(a=1, W=10, L=10, barrier=1.5, barrierpos=(3, 4),
                mu=0.4, Delta=0.1, Deltapos=4, t=1.0):
    # Start with an empty tight-binding system and two square lattices,
    # corresponding to electron and hole degree of freedom
    lat_e = kwant.lattice.square(a, name='e')
    lat_h = kwant.lattice.square(a, name='h')
#HIDDEN_END_zuuw

#HIDDEN_BEGIN_pqmp
    sys = kwant.Builder()

    #### Define the scattering region. ####
    sys[(lat_e(x, y) for x in range(L) for y in range(W))] = 4 * t - mu
    sys[(lat_h(x, y) for x in range(L) for y in range(W))] = mu - 4 * t

    # the tunnel barrier
    sys[(lat_e(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = 4 * t + barrier - mu
    sys[(lat_h(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = mu - 4 * t - barrier

    # hoppings for both electrons and holes
    sys[lat_e.nearest] = -t
    sys[lat_h.nearest] = t

    # Superconducting order parameter enters as hopping between
    # electrons and holes
    sys[((lat_e(x, y), lat_h(x, y)) for x in range(Deltapos, L)
         for y in range(W))] = Delta
#HIDDEN_END_pqmp

    #### Define the leads. ####
#HIDDEN_BEGIN_ttth
    # Symmetry for the left leads.
    sym_left = kwant.TranslationalSymmetry((-a, 0))

    # left electron lead
    lead0 = kwant.Builder(sym_left)
    lead0[(lat_e(0, j) for j in xrange(W))] = 4 * t - mu
    lead0[lat_e.nearest] = -t

    # left hole lead
    lead1 = kwant.Builder(sym_left)
    lead1[(lat_h(0, j) for j in xrange(W))] = mu - 4 * t
    lead1[lat_h.nearest] = t
#HIDDEN_END_ttth

    # Then the lead to the right
    # this one is superconducting and thus is comprised of electrons
    # AND holes
#HIDDEN_BEGIN_mhiw
    sym_right = kwant.TranslationalSymmetry((a, 0))
    lead2 = kwant.Builder(sym_right)
    lead2 += lead0
    lead2 += lead1
    lead2[((lat_e(0, j), lat_h(0, j)) for j in xrange(W))] = Delta
#HIDDEN_END_mhiw

    #### Attach the leads and return the system. ####
#HIDDEN_BEGIN_ozsr
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)
    sys.attach_lead(lead2)

    return sys
#HIDDEN_END_ozsr


#HIDDEN_BEGIN_jbjt
def plot_conductance(sys, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.solve(sys, energy)
        # Conductance is N - R_ee + R_he
        data.append(smatrix.submatrix(0, 0).shape[0] -
                    smatrix.transmission(0, 0) +
                    smatrix.transmission(1, 0))
#HIDDEN_END_jbjt

    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [e^2/h]")
    pyplot.show()


def main():
    sys = make_system()

    # Check that the system looks as intended.
    kwant.plot(sys)

    # Finalize the system.
    sys = sys.finalized()

    plot_conductance(sys, energies=[0.002 * i for i in xrange(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
