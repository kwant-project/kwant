# Physics background
# ------------------
#  band structure of a simple quantum wire in tight-binding approximation
#
# Kwant features highlighted
# --------------------------
#  - Computing the band structure of a finalized lead.

import kwant

from math import pi

# For plotting
from matplotlib import pyplot

import latex, html


def make_lead(a=1, t=1.0, W=10):
    # Start with an empty lead with a single square lattice
    lat = kwant.lattice.Square(a)

    sym_lead = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
    lead = kwant.Builder(sym_lead)

    # build up one unit cell of the lead, and add the hoppings
    # to the next unit cell
    for j in xrange(W):
        lead[lat(0, j)] = 4 * t

        if j > 0:
            lead[lat(0, j), lat(0, j - 1)] = - t

        lead[lat(1, j), lat(0, j)] = - t

    return lead


def plot_bandstructure(lead, momenta):
    # Use the method ``energies`` of the finalized lead to compute
    # the bandstructure
    energy_list = [lead.energies(k) for k in momenta]

    fig = pyplot.figure()
    pyplot.plot(momenta, energy_list)
    pyplot.xlabel("momentum [in units of (lattice constant)^-1]",
                 fontsize=latex.mpl_label_size)
    pyplot.ylabel("energy [in units of t]",
                 fontsize=latex.mpl_label_size)
    pyplot.setp(fig.get_axes()[0].get_xticklabels(),
               fontsize=latex.mpl_tick_size)
    pyplot.setp(fig.get_axes()[0].get_yticklabels(),
               fontsize=latex.mpl_tick_size)
    fig.set_size_inches(latex.mpl_width_in, latex.mpl_width_in*3./4.)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    fig.savefig("3-band_structure_result.pdf")
    fig.savefig("3-band_structure_result.png",
                dpi=(html.figwidth_px/latex.mpl_width_in))


def main():
    lead = make_lead().finalized()

    # list of momenta at which the bands should be computed
    momenta = [-pi + 0.02 * pi * i for i in xrange(101)]

    plot_bandstructure(lead, momenta)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
