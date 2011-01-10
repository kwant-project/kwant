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

# First, define the tight-binding system

sys = kwant.Builder()

# Here, we are only working with square lattices
a = 1
lat = kwant.lattice.Square(a)
sys.default_site_group = lat

t = 1.0
W = 10
L = 30

# Define the scattering region

for i in xrange(L):
    for j in xrange(W):
        sys[(i, j)] = 4 * t

        # hoppig in y-direction
        if j > 0 :
            sys[(i, j), (i, j-1)] = - t

        #hopping in x-direction
        if i > 0:
            sys[(i, j), (i-1, j)] = -t

# Then, define the leads:

# First the lead to the left

# (Note: in the current version, TranslationalSymmetry takes a
# realspace vector)
sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
lead0 = kwant.Builder(sym_lead0)
lead0.default_site_group = lat

for j in xrange(W):
    lead0[(0, j)] = 4 * t

    if j > 0:
        lead0[(0, j), (0, j-1)] = - t

    lead0[(1, j), (0, j)] = - t

# Then the lead to the right

sym_lead1 = kwant.TranslationalSymmetry([lat.vec((1, 0))])
lead1 = kwant.Builder(sym_lead1)
lead1.default_site_group = lat

for j in xrange(W):
    lead1[(0, j)] = 4 * t

    if j > 0:
        lead1[(0, j), (0, j-1)] = - t

    lead1[(1, j), (0, j)] = - t

# Then attach the leads to the system

sys.attach_lead(lead0)
sys.attach_lead(lead1)

# finalize the system

fsys = sys.finalized()

# and plot it, to make sure it's proper

kwant.plot(fsys)

# Now that we have the system, we can compute conductance

energies = []
data = []
for ie in xrange(100):
    energy = ie * 0.01

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
pylab.xlabel("energy [in units of t]")
pylab.ylabel("conductance [in units of e^2/h]")
pylab.show()
