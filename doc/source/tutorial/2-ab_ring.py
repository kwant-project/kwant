# Physics background
# ------------------
#  Flux-dependent transmission through a quantum ring
#
# Kwant features highlighted
# --------------------------
#  - More complex shapes with lattices
#  - Allows for discussion of subtleties of `attach_lead` (not in the
#    example, but in the tutorial main text)
#  - Modifcations of hoppings/sites after they have been added

from cmath import exp
from math import pi

import kwant

# For plotting
from matplotlib import pyplot


#HIDDEN_BEGIN_eusz
def make_system(a=1, t=1.0, W=10, r1=10, r2=20):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity).

    lat = kwant.lattice.square(a)

    sys = kwant.Builder()

    #### Define the scattering region. ####
    # Now, we aim for a more complex shape, namely a ring (or annulus)
    def ring(pos):
        (x, y) = pos
        rsq = x ** 2 + y ** 2
        return (r1 ** 2 < rsq < r2 ** 2)
#HIDDEN_END_eusz

    # and add the corresponding lattice points using the `shape`-function
#HIDDEN_BEGIN_lcak
    sys[lat.shape(ring, (0, r1 + 1))] = 4 * t
    for hopping in lat.nearest:
        sys[sys.possible_hoppings(*hopping)] = -t
#HIDDEN_END_lcak

    # In order to introduce a flux through the ring, we introduce a phase
    # on the hoppings on the line cut through one of the arms

    # since we want to change the flux without modifying Builder repeatedly,
    # we define the modified hoppings as a function that takes the flux
    # through the global variable phi.
#HIDDEN_BEGIN_lvkt
    def fluxphase(site1, site2):
        return exp(1j * phi)

    def crosses_branchcut(hop):
        ix0, iy0 = hop[0].tag

        # possible_hoppings with the argument (1, 0) below
        # returns hoppings ordered as ((i+1, j), (i, j))
        return iy0 < 0 and ix0 == 1  # ix1 == 0 then implied

    # Modify only those hopings in x-direction that cross the branch cut
    sys[(hop for hop in sys.possible_hoppings((1, 0), lat, lat)
         if crosses_branchcut(hop))] = fluxphase
#HIDDEN_END_lvkt

    #### Define the leads. ####
    # left lead
#HIDDEN_BEGIN_qwgr
    sym_lead0 = kwant.TranslationalSymmetry((-a, 0))
    lead0 = kwant.Builder(sym_lead0)

    def lead_shape(pos):
        (x, y) = pos
        return (-1 < x < 1) and (-W / 2 < y < W / 2)

    lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
    for hopping in lat.nearest:
        lead0[lead0.possible_hoppings(*hopping)] = -t
#HIDDEN_END_qwgr

    # Then the lead to the right
    # [again, obtained using reversed()]
    lead1 = lead0.reversed()

    #### Attach the leads and return the system. ####
#HIDDEN_BEGIN_skbz
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)
#HIDDEN_END_skbz

    return sys


def plot_conductance(sys, energy, fluxes):
    # compute conductance
    # global variable phi controls the flux
    global phi

    normalized_fluxes = [flux / (2 * pi) for flux in fluxes]
    data = []
    for flux in fluxes:
        phi = flux

        smatrix = kwant.solve(sys, energy)
        data.append(smatrix.transmission(1, 0))

    pyplot.figure()
    pyplot.plot(normalized_fluxes, data)
    pyplot.xlabel("flux [in units of the flux quantum]")
    pyplot.ylabel("conductance [in units of e^2/h]")
    pyplot.show()


def main():
    sys = make_system()

    # Check that the system looks as intended.
    kwant.plot(sys)

    # Finalize the system.
    sys = sys.finalized()

    # We should see a conductance that is periodic with the flux quantum
    plot_conductance(sys, energy=0.15, fluxes=[0.01 * i * 3 * 2 * pi
                                                for i in xrange(100)])

# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
