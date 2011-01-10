# Physics background
# ------------------
#  transmission through a quantum well
#
# Kwant features highlighted
# --------------------------
#  - Functions as values in Builder

import kwant

# For plotting
import pylab

# global variable governing the behavior of potential() in
# make_system()
pot = 0

def make_system(a=1, t=1.0, W=10, L=30, L_well=10):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.Square(a)

    sys = kwant.Builder()
    sys.default_site_group = lat

    #### Define the scattering region. ####
    # Potential profile
    def potential(site):
        (x, y) = site.pos
        if (L - L_well) / 2 < x < (L + L_well) / 2:
            # The potential value is provided using a global variable
            return pot
        else:
            return 0

    def onsite(site):
        return 4 * t + potential(site)

    sys[((x, y) for x in range(L) for y in range(W))] = onsite
    for hopping in lat.nearest:
        sys[sys.possible_hoppings(*hopping)] = -t

    #### Define the leads. ####
    # First the lead to the left, ...
    # (Note: in the current version, TranslationalSymmetry takes a
    # realspace vector)
    sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
    lead0 = kwant.Builder(sym_lead0)
    lead0.default_site_group = lat

    lead0[((0, j) for j in xrange(W))] = 4 * t
    for hopping in lat.nearest:
        lead0[lead0.possible_hoppings(*hopping)] = - t

    # ... then the lead to the right.  We use a method that returns a copy of
    # `lead0` with its direction reversed.
    lead1 = lead0.reversed()

    #### Attach the leads and return the finalized system. ####
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)

    return sys.finalized()

def plot_conductance(fsys, energy, welldepths):
    # We specify that we want to not only read, but also write to a
    # global variable.
    global pot

    # Compute conductance
    data = []
    for welldepth in welldepths:
        # Set the global variable that defines the potential well depth
        pot = -welldepth

        smatrix = kwant.solve(fsys, energy)
        data.append(smatrix.transmission(1, 0))

    pylab.plot(welldepths, data)
    pylab.xlabel("well depth [in units of t]")
    pylab.ylabel("conductance [in units of e^2/h]")
    pylab.show()


def main():
    fsys = make_system()

    # Check that the system looks as intended.
    kwant.plot(fsys)

    # We should see conductance steps.
    plot_conductance(fsys, energy=0.2,
                     welldepths=[0.01 * i for i in xrange(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
