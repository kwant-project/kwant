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

# global variable governing the behavior of potential() in
# make_system()
#HIDDEN The following code line is included verbatim in the tutorial text
#HIDDEN because nested code examples are not supported.  Remember to update
#HIDDEN the tutorial text when you modify this line.
#HIDDEN_BEGIN_ehso
pot = 0

def make_system(a=1, t=1.0, W=10, L=30, L_well=10):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.Square(a)

    sys = kwant.Builder()

    #### Define the scattering region. ####
    # Potential profile
    def potential(site):
        (x, y) = site.pos
        if (L - L_well) / 2 < x < (L + L_well) / 2:
            # The potential value is provided using a global variable
            return pot
        else:
            return 0
#HIDDEN_END_ehso

#HIDDEN_BEGIN_coid
    def onsite(site):
        return 4 * t + potential(site)

    sys[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    for hopping in lat.nearest:
        sys[sys.possible_hoppings(*hopping)] = -t
#HIDDEN_END_coid

    #### Define the leads. ####
    # First the lead to the left, ...
    # (Note: in the current version, TranslationalSymmetry takes a
    # realspace vector)
    sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
    lead0 = kwant.Builder(sym_lead0)

    lead0[(lat(0, j) for j in xrange(W))] = 4 * t
    for hopping in lat.nearest:
        lead0[lead0.possible_hoppings(*hopping)] = -t

    # ... then the lead to the right.  We use a method that returns a copy of
    # `lead0` with its direction reversed.
    lead1 = lead0.reversed()

    #### Attach the leads and return the finalized system. ####
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)

    return sys


def plot_conductance(sys, energy, welldepths):
    # We specify that we want to not only read, but also write to a
    # global variable.
#HIDDEN_BEGIN_sqvr
    global pot

    # Compute conductance
    data = []
    for welldepth in welldepths:
        # Set the global variable that defines the potential well depth
        pot = -welldepth

        smatrix = kwant.solve(sys, energy)
        data.append(smatrix.transmission(1, 0))

    pyplot.figure()
    pyplot.plot(welldepths, data)
    pyplot.xlabel("well depth [in units of t]")
    pyplot.ylabel("conductance [in units of e^2/h]")
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
                     welldepths=[0.01 * i for i in xrange(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
