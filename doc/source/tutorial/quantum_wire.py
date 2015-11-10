# Tutorial 2.2.2. Transport through a quantum wire
# ================================================
#
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

for i in range(L):
    for j in range(W):
        # On-site Hamiltonian
        sys[lat(i, j)] = 4 * t

        # Hopping in y-direction
        if j > 0:
            sys[lat(i, j), lat(i, j - 1)] = -t

        # Hopping in x-direction
        if i > 0:
            sys[lat(i, j), lat(i - 1, j)] = -t
#HIDDEN_END_zfvr

# Then, define and attach the leads:

# First the lead to the left
# (Note: TranslationalSymmetry takes a real-space vector)
#HIDDEN_BEGIN_xcmc
sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
left_lead = kwant.Builder(sym_left_lead)
#HIDDEN_END_xcmc

#HIDDEN_BEGIN_ndez
for j in range(W):
    left_lead[lat(0, j)] = 4 * t
    if j > 0:
        left_lead[lat(0, j), lat(0, j - 1)] = -t
    left_lead[lat(1, j), lat(0, j)] = -t
#HIDDEN_END_ndez

#HIDDEN_BEGIN_fskr
sys.attach_lead(left_lead)
#HIDDEN_END_fskr

# Then the lead to the right
#HIDDEN_BEGIN_xhqc
sym_right_lead = kwant.TranslationalSymmetry((a, 0))
right_lead = kwant.Builder(sym_right_lead)

for j in range(W):
    right_lead[lat(0, j)] = 4 * t
    if j > 0:
        right_lead[lat(0, j), lat(0, j - 1)] = -t
    right_lead[lat(1, j), lat(0, j)] = -t

sys.attach_lead(right_lead)
#HIDDEN_END_xhqc

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
for ie in range(100):
    energy = ie * 0.01

    # compute the scattering matrix at a given energy
    smatrix = kwant.smatrix(sys, energy)

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
