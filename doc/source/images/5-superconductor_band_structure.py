# Physics background
# ------------------
#  band structure of a superconducting quantum wire in tight-binding
#  approximation
#
# Kwant features highlighted
# --------------------------
#  - Repetition of previously used concepts (band structure calculations,
#    matrices as values in Builder).
#  - Main motivation is to contrast to the implementation of superconductivity
#    in tutorial5b.py

import kwant

import numpy as np
from math import pi

# For plotting
from matplotlib import pyplot

import latex, html

tau_x = np.array([[0, 1], [1, 0]])
tau_z = np.array([[1, 0], [0, -1]])


def make_lead(a=1, t=1.0, mu=0.7, Delta=0.1, W=10):
    # Start with an empty lead with a single square lattice
    lat = kwant.lattice.Square(a)

    sym_lead = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
    lead = kwant.Builder(sym_lead)

    # build up one unit cell of the lead, and add the hoppings
    # to the next unit cell
    for j in xrange(W):
        lead[lat(0, j)] = (4 * t - mu) * tau_z + Delta * tau_x

        if j > 0:
            lead[lat(0, j), lat(0, j - 1)] = - t * tau_z

        lead[lat(1, j), lat(0, j)] = - t * tau_z

    return lead


def plot_bandstructure(lead, momenta):
    # Use the method ``energies`` of the finalized lead to compute
    # the bandstructure
    energy_list = [lead.energies(k) for k in momenta]

    fig = pyplot.figure()
    pyplot.plot(momenta, energy_list)
    pyplot.xlabel("momentum [in untis of (lattice constant)^-1]")
    pyplot.ylabel("energy [in units of t]")
    pyplot.ylim([-0.8, 0.8])
    pyplot.setp(fig.get_axes()[0].get_xticklabels(),
               fontsize=latex.mpl_tick_size)
    pyplot.setp(fig.get_axes()[0].get_yticklabels(),
               fontsize=latex.mpl_tick_size)
    fig.set_size_inches(latex.mpl_width_in, latex.mpl_width_in*3./4.)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    fig.savefig("5-superconductor_band_structure_result.pdf")
    fig.savefig("5-superconductor_band_structure_result.png",
                dpi=(html.figwidth_px/latex.mpl_width_in))


def main():
    # Make system and finalize it right away.
    lead = make_lead().finalized()

    # list of momenta at which the bands should be computed
    momenta = np.linspace(-1.5, 1.5, 201)

    plot_bandstructure(lead, momenta)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
