# Physics background
# ------------------
#  Conductance of a quantum wire; subbands
#
# Kwant features highlighted
# --------------------------
#  - Builder for setting up transport systems easily
#  - Making scattering region and leads
#  - Using the simple sparse solver for computing Landauer conductance

from matplotlib import pyplot
#HIDDEN_BEGIN_dwhx
import kwant
#HIDDEN_END_dwhx

# First, define the tight-binding system

#HIDDEN_BEGIN_goiq
sys = kwant.Builder()
#HIDDEN_END_goiq

# Here, we are only working with square lattices
#HIDDEN_BEGIN_suwo
a = 1
lat = kwant.lattice.square(a)
#HIDDEN_END_suwo

#HIDDEN_BEGIN_zfvr
t = 1.0
W = 10
L = 30

# Define the scattering region

for i in xrange(L):
    for j in xrange(W):
        sys[lat(i, j)] = 4 * t

        # hoppig in y-direction
        if j > 0:
            sys[lat(i, j), lat(i, j - 1)] = -t

        #hopping in x-direction
        if i > 0:
            sys[lat(i, j), lat(i - 1, j)] = -t
#HIDDEN_END_zfvr

# Then, define the leads:

# First the lead to the left

# (Note: TranslationalSymmetry takes a real-space vector)
#HIDDEN_BEGIN_xcmc
sym_lead0 = kwant.TranslationalSymmetry((-a, 0))
lead0 = kwant.Builder(sym_lead0)
#HIDDEN_END_xcmc

#HIDDEN_BEGIN_ndez
for j in xrange(W):
    lead0[lat(0, j)] = 4 * t

    if j > 0:
        lead0[lat(0, j), lat(0, j - 1)] = -t

    lead0[lat(1, j), lat(0, j)] = -t
#HIDDEN_END_ndez

# Then the lead to the right
#HIDDEN_BEGIN_xhqc

sym_lead1 = kwant.TranslationalSymmetry((a, 0))
lead1 = kwant.Builder(sym_lead1)

for j in xrange(W):
    lead1[lat(0, j)] = 4 * t

    if j > 0:
        lead1[lat(0, j), lat(0, j - 1)] = -t

    lead1[lat(1, j), lat(0, j)] = -t
#HIDDEN_END_xhqc

# Then attach the leads to the system

#HIDDEN_BEGIN_fskr
sys.attach_lead(lead0)
sys.attach_lead(lead1)
#HIDDEN_END_fskr

# Plot it, to make sure it's OK

#HIDDEN_BEGIN_wsgh
kwant.plot(sys)
#HIDDEN_END_wsgh

# Finalize the system

#HIDDEN_BEGIN_dngj
sys = sys.finalized()
#HIDDEN_END_dngj

# Now that we have the system, we can compute conductance

#HIDDEN_BEGIN_buzn
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
#HIDDEN_END_buzn

# Use matplotlib to write output
# We should see conductance steps
#HIDDEN_BEGIN_lliv

pyplot.figure()
pyplot.plot(energies, data)
pyplot.xlabel("energy [t]")
pyplot.ylabel("conductance [e^2/h]")
pyplot.show()
#HIDDEN_END_lliv
