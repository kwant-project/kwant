# Tutorial 2.3.2. Spatially dependent values through functions
# ============================================================
#
# Physics background
# ------------------
#  transmission through a quantum well
#
# Kwant features highlighted
# --------------------------
#  - Functions as values in Builder

import kwant

# For plotting
from matplotlib import pyplot


#HIDDEN_BEGIN_ehso
def make_system(a=1, t=1.0, W=10, L=30, L_well=10):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.square(a)

    sys = kwant.Builder()

    #### Define the scattering region. ####
    # Potential profile
    def potential(site, pot):
        (x, y) = site.pos
        if (L - L_well) / 2 < x < (L + L_well) / 2:
            return pot
        else:
            return 0
#HIDDEN_END_ehso

#HIDDEN_BEGIN_coid
    def onsite(site, pot=0):
        return 4 * t + potential(site, pot)

    sys[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    sys[lat.neighbors()] = -t
#HIDDEN_END_coid

    #### Define and attach the leads. ####
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0, j) for j in range(W))] = 4 * t
    lead[lat.neighbors()] = -t
    sys.attach_lead(lead)
    sys.attach_lead(lead.reversed())

    return sys


def plot_conductance(sys, energy, welldepths):
#HIDDEN_BEGIN_sqvr

    # Compute conductance
    data = []
    for welldepth in welldepths:
        smatrix = kwant.smatrix(sys, energy, args=[-welldepth])
        data.append(smatrix.transmission(1, 0))

    pyplot.figure()
    pyplot.plot(welldepths, data)
    pyplot.xlabel("well depth [t]")
    pyplot.ylabel("conductance [e^2/h]")
    pyplot.show()
#HIDDEN_END_sqvr


def main():
    sys = make_system()

    # Check that the system looks as intended.
    kwant.plot(sys)

    # Finalize the system.
    sys = sys.finalized()

    # We should see conductance steps.
    plot_conductance(sys, energy=0.2,
                     welldepths=[0.01 * i for i in range(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
