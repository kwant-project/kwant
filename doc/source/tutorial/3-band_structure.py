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


#HIDDEN_BEGIN_zxip
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
            lead[lat(0, j), lat(0, j - 1)] = -t

        lead[lat(1, j), lat(0, j)] = -t

    return lead
#HIDDEN_END_zxip


#HIDDEN_BEGIN_pejz
def plot_bandstructure(lead, momenta):
    # Use the method ``energies`` of the finalized lead to compute
    # the bandstructure
    energy_list = [lead.energies(k) for k in momenta]

    pyplot.figure()
    pyplot.plot(momenta, energy_list)
    pyplot.xlabel("momentum [in units of (lattice constant)^-1]")
    pyplot.ylabel("energy [in units of t]")
    pyplot.show()


def main():
    lead = make_lead().finalized()

    # list of momenta at which the bands should be computed
    momenta = [-pi + 0.02 * pi * i for i in xrange(101)]

    plot_bandstructure(lead, momenta)
#HIDDEN_END_pejz


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
