# Physics background
# ------------------
#  Conductance of a quantum wire; subbands
#
# Kwant features highlighted
# --------------------------
#  - Builder for setting up transport systems easily
#  - Making scattering region and leads
#  - Using the simple sparse solver for computing Landauer conductance

import kwant
import latex, html

# First, define the tight-binding system

sys = kwant.Builder()

# Here, we are only working with square lattices
a = 1
lat = kwant.lattice.Square(a)

t = 1.0
W = 10
L = 30

# Define the scattering region

for i in xrange(L):
    for j in xrange(W):
        sys[lat(i, j)] = 4 * t

        # hoppig in y-direction
        if j > 0:
            sys[lat(i, j), lat(i, j - 1)] = - t

        #hopping in x-direction
        if i > 0:
            sys[lat(i, j), lat(i - 1, j)] = -t

# Then, define the leads:

# First the lead to the left

# (Note: in the current version, TranslationalSymmetry takes a
# realspace vector)
sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
lead0 = kwant.Builder(sym_lead0)

for j in xrange(W):
    lead0[lat(0, j)] = 4 * t

    if j > 0:
        lead0[lat(0, j), lat(0, j - 1)] = - t

    lead0[lat(1, j), lat(0, j)] = - t

# Then the lead to the right

sym_lead1 = kwant.TranslationalSymmetry([lat.vec((1, 0))])
lead1 = kwant.Builder(sym_lead1)

for j in xrange(W):
    lead1[lat(0, j)] = 4 * t

    if j > 0:
        lead1[lat(0, j), lat(0, j - 1)] = - t

    lead1[lat(1, j), lat(0, j)] = - t

# Then attach the leads to the system

sys.attach_lead(lead0)
sys.attach_lead(lead1)

# Plot it, to make sure it's OK

kwant.plot(sys, "1-quantum_wire_sys.pdf", width=latex.figwidth_pt)
kwant.plot(sys, "1-quantum_wire_sys.png", width=html.figwidth_px)

# Finalize the system

sys = sys.finalized()

# Now that we have the system, we can compute conductance

energies = []
data = []
for ie in xrange(100):
    energy = ie * 0.01

    # compute the scattering matrix at energy energy
    smatrix = kwant.solve(sys, energy)

    # compute the transmission probability from lead 0 to
    # lead 1
    energies.append(energy)
    data.append(smatrix.transmission(1, 0))

# Use matplotlib to write output
# We should see conductance steps
from matplotlib import pyplot

fig = pyplot.figure()
pyplot.plot(energies, data)
pyplot.xlabel("energy [in units of t]",
                 fontsize=latex.mpl_label_size)
pyplot.ylabel("conductance [in units of e^2/h]",
                 fontsize=latex.mpl_label_size)
pyplot.setp(fig.get_axes()[0].get_xticklabels(),
           fontsize=latex.mpl_tick_size)
pyplot.setp(fig.get_axes()[0].get_yticklabels(),
           fontsize=latex.mpl_tick_size)
fig.set_size_inches(latex.mpl_width_in, latex.mpl_width_in*3./4.)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
fig.savefig("1-quantum_wire_result.pdf")
fig.savefig("1-quantum_wire_result.png", dpi=(html.figwidth_px/latex.mpl_width_in))
