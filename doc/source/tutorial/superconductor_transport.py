# Tutorial 2.6.2. "Lattice description": using different lattices
# ===============================================================
#
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
    syst = kwant.Builder()

    #### Define the scattering region. ####
    syst[(lat_e(x, y) for x in range(L) for y in range(W))] = 4 * t - mu
    syst[(lat_h(x, y) for x in range(L) for y in range(W))] = mu - 4 * t

    # the tunnel barrier
    syst[(lat_e(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = 4 * t + barrier - mu
    syst[(lat_h(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = mu - 4 * t - barrier

    # hoppings for both electrons and holes
    syst[lat_e.neighbors()] = -t
    syst[lat_h.neighbors()] = t

    # Superconducting order parameter enters as hopping between
    # electrons and holes
    syst[((lat_e(x, y), lat_h(x, y)) for x in range(Deltapos, L)
         for y in range(W))] = Delta
#HIDDEN_END_pqmp

    #### Define the leads. ####
#HIDDEN_BEGIN_ttth
    # Symmetry for the left leads.
    sym_left = kwant.TranslationalSymmetry((-a, 0))

    # left electron lead
    lead0 = kwant.Builder(sym_left)
    lead0[(lat_e(0, j) for j in range(W))] = 4 * t - mu
    lead0[lat_e.neighbors()] = -t

    # left hole lead
    lead1 = kwant.Builder(sym_left)
    lead1[(lat_h(0, j) for j in range(W))] = mu - 4 * t
    lead1[lat_h.neighbors()] = t
#HIDDEN_END_ttth

    # Then the lead to the right
    # this one is superconducting and thus is comprised of electrons
    # AND holes
#HIDDEN_BEGIN_mhiw
    sym_right = kwant.TranslationalSymmetry((a, 0))
    lead2 = kwant.Builder(sym_right)
    lead2 += lead0
    lead2 += lead1
    lead2[((lat_e(0, j), lat_h(0, j)) for j in range(W))] = Delta
#HIDDEN_END_mhiw

    #### Attach the leads and return the system. ####
#HIDDEN_BEGIN_ozsr
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    syst.attach_lead(lead2)

    return syst
#HIDDEN_END_ozsr


#HIDDEN_BEGIN_jbjt
def plot_conductance(syst, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
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
    syst = make_system()

    # Check that the system looks as intended.
    kwant.plot(syst)

    # Finalize the system.
    syst = syst.finalized()

    plot_conductance(syst, energies=[0.002 * i for i in range(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
