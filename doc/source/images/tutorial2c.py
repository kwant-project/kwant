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

import latex, html

# First, define the tight-binding system

sys = kwant.Builder()

# Here, we are only working with square lattices

# for simplicity, take lattice constant = 1
a = 1
lat = kwant.lattice.Square(a)

t = 1.0
W = 10
r1 = 10
r2 = 20

# Define the scattering region
# Now, we aim for a more compelx shape, namely a ring (or annulus)

def ring(pos):
    (x, y) = pos
    rsq = x**2 + y**2
    return ( r1**2 < rsq < r2**2)

sys[lat.shape(ring, (0, 11))] = 4 * t
for hopping in lat.nearest:
    sys[sys.possible_hoppings(*hopping)] = - t

# In order to introduce a flux through the ring, we introduce a phase
# on the hoppings on the line cut through one of the arms

# since we want to change the flux without modifying Builder repeatedly,
# we define the modified hoppings as a function that takes the flux
# as a global variable.

def fluxphase(site1, site2):
    return exp(1j * phi)

# Now go through all the hoppings and modify those in the lower
# arm of the ring that go from x=0 to x=1

for (site1, site2) in sys.hoppings():
    ix1, iy1 = site1.tag
    ix2, iy2 = site2.tag

    hopx = tuple(sorted((ix1, ix2)))

    if hopx == (0, 1) and iy1 == iy2 and iy1 < 0:
        sys[lat(hopx[1], iy1), lat(hopx[0], iy1)] = fluxphase

# Then, define the leads:

# First the lead to the left

# (Note: in the current version, TranslationalSymmetry takes a
# realspace vector)
sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
lead0 = kwant.Builder(sym_lead0)
lead0.default_site_group = lat

def lead_shape(pos):
    (x, y) = pos
    return (-1 < x < 1) and ( -W/2 < y < W/2  )

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

# and plot it, to make sure it's proper

kwant.plot(fsys, "tutorial2c_sys.pdf", width=latex.figwidth_pt)
kwant.plot(fsys, "tutorial2c_sys.png", width=html.figwidth_px)

# Now that we have the system, we can compute conductance

energy = 0.15
phases = []
data = []
for iphi in xrange(100):
    phi = iphi * 0.01 * 3 * 2 * pi

    # compute the scattering matrix at energy energy
    smatrix = kwant.solve(fsys, energy)

    # compute the transmission probability from lead 0 to
    # lead 1
    phases.append(phi / (2 * pi))
    data.append(smatrix.transmission(1, 0))

# Use matplotlib to write output
# We should see conductance steps
import pylab

pylab.plot(phases, data)
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

# Finally, some plots needed for the notes

sys = kwant.Builder()

sys[lat.shape(ring, (0, 11))] = 4 * t
for hopping in lat.nearest:
    sys[sys.possible_hoppings(*hopping)] = - t

sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
lead0 = kwant.Builder(sym_lead0)
lead0.default_site_group = lat

def lead_shape(pos):
    (x, y) = pos
    return (-1 < x < 1) and ( 0.5 * W < y < 1.5 * W )

lead0[lat.shape(lead_shape, (0, W))] = 4 * t
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

# and plot it, to make sure it's proper

kwant.plot(fsys, "tutorial2c_note1.pdf", width=latex.figwidth_small_pt)
kwant.plot(fsys, "tutorial2c_note1.png", width=html.figwidth_small_px)

sys = kwant.Builder()

sys[lat.shape(ring, (0, 11))] = 4 * t
for hopping in lat.nearest:
    sys[sys.possible_hoppings(*hopping)] = - t

sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
lead0 = kwant.Builder(sym_lead0)
lead0.default_site_group = lat

def lead_shape(pos):
    (x, y) = pos
    return (-1 < x < 1) and ( -W/2 < y < W/2  )

lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
for hopping in lat.nearest:
    lead0[lead0.possible_hoppings(*hopping)] = - t

# Then the lead to the right
# there we can use a special function that simply reverses the direction

lead1 = lead0.reversed()

# Then attach the leads to the system

sys.attach_lead(lead0)
sys.attach_lead(lead1, lat(0, 0))

# finalize the system

fsys = sys.finalized()

# and plot it, to make sure it's proper

kwant.plot(fsys, "tutorial2c_note2.pdf", width=latex.figwidth_small_pt)
kwant.plot(fsys, "tutorial2c_note2.png", width=html.figwidth_small_px)
