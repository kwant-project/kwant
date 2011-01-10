# Physics background
# ------------------
#  transmission through a quantum well
#
# Kwant features highlighted
# --------------------------
#  - Functions as values in Builder

import kwant

import latex, html

# First, define the tight-binding system

sys = kwant.Builder()

# Here, we are only working with square lattices

# for simplicity, take lattice constant = 1
a = 1
lat = kwant.lattice.Square(a)

t = 1.0
alpha = 0.5
e_z = 0.08
W = 10
L = 30

# Define the scattering region

def rectangle(pos):
    (x, y) = pos
    return ( -0.5 < x < L - 0.5 ) and ( -0.5 < y < W - 0.5 )

def potential(site):
    (x, y) = site.pos
    if 10 < x < 20:
        return pot
    else:
        return 0

def onsite(site):
    return 4 * t + potential(site)

sys[lat.shape(rectangle, (0, 0))] = onsite
for hopping in lat.nearest:
    sys[sys.possible_hoppings(*hopping)] = - t

# Then, define the leads:

# First the lead to the left

# (Note: in the current version, TranslationalSymmetry takes a
# realspace vector)
sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
lead0 = kwant.Builder(sym_lead0)
lead0.default_site_group = lat

def lead_shape(pos):
    (x, y) = pos
    return (-1 < x < 1) and ( -0.5 < y < W - 0.5 )

lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
for hopping in lat.nearest:
    lead0[lead0.possible_hoppings(*hopping)] = - t

# Then the lead to the right
# there we can use a special function that simply reverses the direction

lead1 = lead0.reversed()

# Then attach the leads to the system

sys.attach_lead(lead0)
sys.attach_lead(lead1)

# finalize the system

fsys = sys.finalized()

# Now that we have the system, we can compute conductance

energy = 0.2
wellpot = []
data = []
for ipot in xrange(100):
    pot = - ipot * 0.01

    # compute the scattering matrix at energy energy
    smatrix = kwant.solvers.sparse.solve(fsys, energy)

    # compute the transmission probability from lead 0 to
    # lead 1
    wellpot.append(-pot)
    data.append(smatrix.transmission(1, 0))

# Use matplotlib to write output
# We should see conductance steps
import pylab

pylab.plot(wellpot, data)
pylab.xlabel("well depth [in units of t]",
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
fig.savefig("tutorial2b_result.pdf")
fig.savefig("tutorial2b_result.png",
            dpi=(html.figwidth_px/latex.mpl_width_in))
