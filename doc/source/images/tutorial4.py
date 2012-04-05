# Physics background
# ------------------
#  Transport through a graphene quantum dot with a pn-junction
#
# Kwant features highlighted
# --------------------------
#  - Application of all the aspects of tutorials 1-3 to a more complicated
#    lattice, namely graphene

from __future__ import division # so that 1/2 == 0.5, and not 0
from math import pi, sqrt
import numpy as np

import kwant
import latex, html

# For computing eigenvalues
import scipy.sparse.linalg as sla

# For plotting
import pylab


# Define the graphene lattice
sin_30, cos_30 = (1/2, np.sqrt(3)/2)
graphene = kwant.make_lattice([(1, 0), (sin_30, cos_30)],
                              [(0, 0), (0, 1/np.sqrt(3))])
a, b = graphene.sublattices


def make_system(r=10, w=2.0, pot=0.1):

    #### Define the scattering region. ####
    # circular scattering region
    def circle(pos):
        x, y = pos
        return x**2 + y**2 < r**2

    sys= kwant.Builder()

    # w: width and pot: potential maximum of the p-n junction
    def potential(site):
        (x, y) = site.pos
        d = y * cos_30 + x * sin_30
        return pot * np.tanh(d / w)

    sys[graphene.shape(circle, (0,0))] = potential

    # specify the hoppings of the graphene lattice in the
    # format expected by possibe_hoppings()
    hoppings = (((0, 0), b, a), ((0, 1), b, a), ((-1, 1), b, a))
    for hopping in hoppings:
        sys[sys.possible_hoppings(*hopping)] = - 1

    # Modify the scattering region
    del sys[a(0,0)]
    sys[a(-2,1), b(2, 2)] = -1

    #### Define the leads. ####
    # left lead
    sym0 = kwant.TranslationalSymmetry([graphene.vec((-1, 0))])

    def lead0_shape(pos):
        x, y = pos
        return (-1 < x < 1) and (-0.4 * r < y < 0.4 * r)

    lead0 = kwant.Builder(sym0)
    lead0[graphene.shape(lead0_shape, (0,0))] = - pot
    for hopping in hoppings:
        lead0[lead0.possible_hoppings(*hopping)] = - 1

    # The second lead, going to the top right
    sym1 = kwant.TranslationalSymmetry([graphene.vec((0, 1))])

    def lead1_shape(pos):
        x, y = pos
        u = x * sin_30 + y * cos_30
        v = y * sin_30 - x * cos_30
        return (-1 < u < 1) and (-0.4 * r < v < 0.4 * r)

    lead1 = kwant.Builder(sym1)
    lead1[graphene.shape(lead1_shape, (0,0))] = pot
    for hopping in hoppings:
        lead1[lead1.possible_hoppings(*hopping)] = - 1

    return sys, [lead0, lead1]


def compute_evs(sys):
    # Compute some eigenvalues of the closed system
    sparse_mat = sys.hamiltonian_submatrix(sparse=True)

    try:
        # This requires scipy version >= 0.9.0
        # Failure (i.e. insufficient scipy version) is not critical
        # for the remainder of the tutorial, hence the try-block
        evs = scipy.sparse.linalg.eigs(sparse_mat, 2)[0]
        print evs
    except:
        pass


def plot_conductance(fsys, energies):
    # Compute transmission as a function of energy
    data = []
    for energy in energies:
        smatrix = kwant.solve(fsys, energy)
        data.append(smatrix.transmission(0, 1))

    pylab.clf()
    pylab.plot(energies, data)
    pylab.xlabel("energy [in units of t]",
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
    fig.savefig("tutorial4_result.pdf")
    fig.savefig("tutorial4_result.png",
                dpi=(html.figwidth_px/latex.mpl_width_in))


def plot_bandstructure(flead, momenta):
    # Use the method ``energies`` of the finalized lead to compute
    # the bandstructure
    energy_list = [flead.energies(k) for k in momenta]

    pylab.clf()
    pylab.plot(momenta, energy_list)
    pylab.xlabel("momentum [in untis of (lattice constant)^-1]",
                 fontsize=latex.mpl_label_size)
    pylab.ylabel("energy [in units of t]",
                 fontsize=latex.mpl_label_size)
    fig = pylab.gcf()
    pylab.setp(fig.get_axes()[0].get_xticklabels(),
               fontsize=latex.mpl_tick_size)
    pylab.setp(fig.get_axes()[0].get_yticklabels(),
               fontsize=latex.mpl_tick_size)
    fig.set_size_inches(latex.mpl_width_in, latex.mpl_width_in*3./4.)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    fig.savefig("tutorial4_bs.pdf")
    fig.savefig("tutorial4_bs.png",
                dpi=(html.figwidth_px/latex.mpl_width_in))


def main():
    pot = 0.1
    sys, leads = make_system(pot=pot)

    # To highlight the two sublattices of graphene, we plot one with
    # a filled, and the other one with an open circle:
    plotter_symbols = {a: kwant.plotter.Circle(r=0.3),
                       b: kwant.plotter.Circle(r=0.3,
                                               fcol=kwant.plotter.white,
                                               lcol=kwant.plotter.black)}

    # Plot the closed system without leads.
    kwant.plot(sys, symbols=plotter_symbols,
               filename="tutorial4_sys1.pdf", width=latex.figwidth_pt)
    kwant.plot(sys, symbols=plotter_symbols,
               filename="tutorial4_sys1.png", width=html.figwidth_px)

    # Compute some eigenvalues.
    compute_evs(sys.finalized())

    # Attach the leads to the system.
    for lead in leads:
        sys.attach_lead(lead)

    # Then, plot the system with leads.
    kwant.plot(sys, symbols=plotter_symbols,
               filename="tutorial4_sys2.pdf", width=latex.figwidth_pt)
    kwant.plot(sys, symbols=plotter_symbols,
               filename="tutorial4_sys2.png", width=html.figwidth_px)

    # Finalize the system.
    fsys = sys.finalized()

    # Compute the band structure of lead 0.
    momenta = np.arange(-pi, pi + .01, 0.1 * pi)
    plot_bandstructure(fsys.leads[0], momenta)

    # Plot conductance.
    energies = np.arange(-2 * pot, 2 * pot, pot / 10.5)
    plot_conductance(fsys, energies)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
