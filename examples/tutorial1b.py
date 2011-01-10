# Physics background
# ------------------
#  Conductance of a quantum wire; subbands
#
# Kwant features highlighted
# --------------------------
#  - Using iterables and possible_hoppings() for making systems
#  - introducing `reversed()` for the leads
#
# Note: Does the same as tutorial1a.py, but using other features of kwant
#

import kwant

# For plotting
import pylab

def make_system(a=1, t=1.0, W=10, L=30):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.Square(a)

    sys = kwant.Builder()
    sys.default_site_group = lat

    #### Define the scattering region. ####
    sys[((x, y) for x in range(L) for y in range(W))] = 4 * t
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

def plot_conductance(fsys, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.solvers.sparse.solve(fsys, energy)
        data.append(smatrix.transmission(1, 0))

    pylab.plot(energies, data)
    pylab.xlabel("energy [in units of t]")
    pylab.ylabel("conductance [in units of e^2/h]")
    pylab.show()


def main():
    fsys = make_system()

    # Check that the system looks as intended.
    kwant.plot(fsys)

    # We should see conductance steps.
    plot_conductance(fsys, energies=[0.01 * i for i in xrange(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
