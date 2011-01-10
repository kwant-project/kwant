# Physics background
# ------------------
#  band structure of a simple quantum wire in tight-binding approximation
#
# Kwant features highlighted
# --------------------------
#  - Computing the band structure of a finalized lead.

import kwant
import numpy as np
from math import pi

import latex, html

a = 1
lat = kwant.lattice.Square(a)

t = 1.0
W = 10

# Define a lead:

sym_lead = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
lead = kwant.Builder(sym_lead)
lead.default_site_group = lat

for j in xrange(W):
    lead[(0, j)] = 4 * t

    if j > 0:
        lead[(0, j), (0, j-1)] = - t

    lead[(1, j), (0, j)] = - t

# Now compute the band structure

# Only a finalized lead has the information about bandstructure
flead = lead.finalized()

momenta = np.arange(-pi, pi + .01, 0.02 * pi)
energy_list = [flead.energies(k) for k in momenta]

import pylab
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
fig.savefig("tutorial3a_result.pdf")
fig.savefig("tutorial3a_result.png",
            dpi=(html.figwidth_px/latex.mpl_width_in))
