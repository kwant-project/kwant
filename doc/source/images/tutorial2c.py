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
import pylab

import latex, html

def make_system(a=1, t=1.0, W=10, r1=10, r2=20):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity).

    lat = kwant.lattice.Square(a)

    sys = kwant.Builder()

    #### Define the scattering region. ####
    # Now, we aim for a more complex shape, namely a ring (or annulus)
    def ring(pos):
        (x, y) = pos
        rsq = x**2 + y**2
        return ( r1**2 < rsq < r2**2)

    # and add the corresponding lattice points using the `shape`-function
    sys[lat.shape(ring, (0, 11))] = 4 * t
    for hopping in lat.nearest:
        sys[sys.possible_hoppings(*hopping)] = - t

    # In order to introduce a flux through the ring, we introduce a phase
    # on the hoppings on the line cut through one of the arms

    # since we want to change the flux without modifying Builder repeatedly,
    # we define the modified hoppings as a function that takes the flux
    # through the global variable phi.
    def fluxphase(site1, site2):
        return exp(1j * phi)

    def crosses_branchcut(hop):
        ix0, iy0 = hop[0].tag

        # possible_hoppings with the argument (1, 0) below
        # returns hoppings ordered as ((i+1, j), (i, j))
        return iy0 < 0 and ix0 == 1 # ix1 == 0 then implied

    # Modify only those hopings in x-direction that cross the branch cut
    sys[(hop for hop in sys.possible_hoppings((1,0), lat, lat)
         if crosses_branchcut(hop))] = fluxphase

    #### Define the leads. ####
    # left lead
    sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
    lead0 = kwant.Builder(sym_lead0)

    def lead_shape(pos):
        (x, y) = pos
        return (-1 < x < 1) and ( -W/2 < y < W/2  )

    lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
    for hopping in lat.nearest:
        lead0[lead0.possible_hoppings(*hopping)] = - t

    # Then the lead to the right
    # (again, obtained using reverse()
    lead1 = lead0.reversed()

    #### Attach the leads and return the finalized system. ####
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)

    return sys


def make_system_note1(a=1, t=1.0, W=10, r1=10, r2=20):
    lat = kwant.lattice.Square(a)
    sys = kwant.Builder()
    def ring(pos):
        (x, y) = pos
        rsq = x**2 + y**2
        return ( r1**2 < rsq < r2**2)
    sys[lat.shape(ring, (0, 11))] = 4 * t
    for hopping in lat.nearest:
        sys[sys.possible_hoppings(*hopping)] = - t
    sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
    lead0 = kwant.Builder(sym_lead0)
    def lead_shape(pos):
        (x, y) = pos
        return (-1 < x < 1) and ( 0.5 * W < y < 1.5 * W )
    lead0[lat.shape(lead_shape, (0, W))] = 4 * t
    for hopping in lat.nearest:
        lead0[lead0.possible_hoppings(*hopping)] = - t
    lead1 = lead0.reversed()
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)
    return sys


def make_system_note2(a=1, t=1.0, W=10, r1=10, r2=20):
    lat = kwant.lattice.Square(a)
    sys = kwant.Builder()
    def ring(pos):
        (x, y) = pos
        rsq = x**2 + y**2
        return ( r1**2 < rsq < r2**2)
    sys[lat.shape(ring, (0, 11))] = 4 * t
    for hopping in lat.nearest:
        sys[sys.possible_hoppings(*hopping)] = - t
    sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
    lead0 = kwant.Builder(sym_lead0)
    def lead_shape(pos):
        (x, y) = pos
        return (-1 < x < 1) and ( -W/2 < y < W/2  )
    lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
    for hopping in lat.nearest:
        lead0[lead0.possible_hoppings(*hopping)] = - t
    lead1 = lead0.reversed()
    sys.attach_lead(lead0)
    sys.attach_lead(lead1, lat(0, 0))
    return sys


def plot_conductance(fsys, energy, fluxes):
    # compute conductance
    # global variable phi controls the flux
    global phi

    normalized_fluxes = [flux/(2 * pi) for flux in fluxes]
    data = []
    for flux in fluxes:
        phi = flux

        smatrix = kwant.solve(fsys, energy)
        data.append(smatrix.transmission(1, 0))

    pylab.plot(normalized_fluxes, data)
    pylab.xlabel("flux [in units of the flux quantum]",
                 fontsize=latex.mpl_label_size)
    pylab.ylabel("conductance [in units of e^2/h]",
                 fontsize=latex.mpl_label_size)
    fig = pylab.gcf()
    pylab.setp(fig.get_axes()[0].get_xticklabels(),
               fontsize=latex.mpl_tick_size)
    pylab.setp(fig.get_axes()[0].get_yticklabels(),
               fontsize=latex.mpl_tick_size)
    fig.set_size_inches(latex.mpl_width_in, latex.mpl_width_in*3./4.)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    fig.savefig("tutorial2c_result.pdf")
    fig.savefig("tutorial2c_result.png",
                dpi=(html.figwidth_px/latex.mpl_width_in))


def main():
    sys = make_system()

    # Check that the system looks as intended.
    kwant.plot(sys, "tutorial2c_sys.pdf", width=latex.figwidth_pt)
    kwant.plot(sys, "tutorial2c_sys.png", width=html.figwidth_px)

    # Finalize the system.
    fsys = sys.finalized()

    # We should see a conductance that is periodic with the flux quantum
    plot_conductance(fsys, energy=0.15, fluxes=[0.01 * i * 3 * 2 * pi
                                                for i in xrange(100)])

    # Finally, some plots needed for the notes
    sys = make_system_note1()
    kwant.plot(sys, "tutorial2c_note1.pdf", width=latex.figwidth_small_pt)
    kwant.plot(sys, "tutorial2c_note1.png", width=html.figwidth_small_px)
    sys = make_system_note2()
    kwant.plot(sys, "tutorial2c_note2.pdf", width=latex.figwidth_small_pt)
    kwant.plot(sys, "tutorial2c_note2.png", width=html.figwidth_small_px)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
