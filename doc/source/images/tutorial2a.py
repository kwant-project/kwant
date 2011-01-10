# Physics background
# ------------------
#  Gaps in quantum wires with spin-orbit coupling and Zeeman splititng,
#  as theoretically predicted in
#   http://prl.aps.org/abstract/PRL/v90/i25/e256601
#  and (supposedly) experimentally oberved in
#   http://www.nature.com/nphys/journal/v6/n5/abs/nphys1626.html
#
# Kwant features highlighted
# --------------------------
#  - Numpy matrices as values in Builder

import kwant
import numpy

import latex, html

# define sigma-matrices for convenience
sigma_0 = numpy.eye(2)
sigma_x = numpy.array([[0, 1], [1, 0]])
sigma_y = numpy.array([[0, -1j], [1j, 0]])
sigma_z = numpy.array([[1, 0], [0, -1]])

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

sys[lat.shape(rectangle, (0, 0))] = 4 * t * sigma_0 + e_z * sigma_z
# hoppings in x-direction
sys[sys.possible_hoppings((1, 0), lat, lat)] = - t * sigma_0 - \
    1j * alpha * sigma_y
# hoppings in y-directions
sys[sys.possible_hoppings((0, 1), lat, lat)] = - t * sigma_0 + \
    1j * alpha * sigma_x

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

lead0[lat.shape(lead_shape, (0, 0))] = 4 * t * sigma_0 + e_z * sigma_z
# hoppings in x-direction
lead0[lead0.possible_hoppings((1, 0), lat, lat)] = - t * sigma_0 - \
    1j * alpha * sigma_y
# hoppings in y-directions
lead0[lead0.possible_hoppings((0, 1), lat, lat)] = - t * sigma_0 + \
    1j * alpha * sigma_x

# Then the lead to the right
# there we can use a special function that simply reverses the direction

lead1 = lead0.reversed()

# Then attach the leads to the system

sys.attach_lead(lead0)
sys.attach_lead(lead1)

# finalize the system

fsys = sys.finalized()

# Now that we have the system, we can compute conductance

energies = []
data = []
for ie in xrange(100):
    energy = ie * 0.01 - 0.3

    # compute the scattering matrix at energy energy
    smatrix = kwant.solvers.sparse.solve(fsys, energy)

    # compute the transmission probability from lead 0 to
    # lead 1
    energies.append(energy)
    data.append(smatrix.transmission(1, 0))

# Use matplotlib to write output
# We should see conductance steps
import pylab

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
fig.savefig("tutorial2a_result.pdf")
fig.savefig("tutorial2a_result.png",
            dpi=(html.figwidth_px/latex.mpl_width_in))
