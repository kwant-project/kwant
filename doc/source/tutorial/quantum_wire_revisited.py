# Tutorial 2.2.3. Building the same system with less code
# =======================================================
#
# Physics background
# ------------------
#  Conductance of a quantum wire; subbands
#
# Kwant features highlighted
# --------------------------
#  - Using iterables and builder.HoppingKind for making systems
#  - introducing `reversed()` for the leads
#
# Note: Does the same as tutorial1a.py, but using other features of Kwant.

#HIDDEN_BEGIN_xkzy
import kwant

# For plotting
from matplotlib import pyplot


def make_system(a=1, t=1.0, W=10, L=30):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity.
    lat = kwant.lattice.square(a)

    syst = kwant.Builder()
#HIDDEN_END_xkzy

    #### Define the scattering region. ####
#HIDDEN_BEGIN_vvjt
    syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * t
#HIDDEN_END_vvjt
#HIDDEN_BEGIN_nooi
    syst[lat.neighbors()] = -t
#HIDDEN_END_nooi

    #### Define and attach the leads. ####
    # Construct the left lead.
#HIDDEN_BEGIN_iepx
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0, j) for j in range(W))] = 4 * t
    lead[lat.neighbors()] = -t
#HIDDEN_END_iepx

    # Attach the left lead and its reversed copy.
#HIDDEN_BEGIN_yxot
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst
#HIDDEN_END_yxot


#HIDDEN_BEGIN_ayuk
def plot_conductance(syst, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix.transmission(1, 0))

    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [e^2/h]")
    pyplot.show()
#HIDDEN_END_ayuk


#HIDDEN_BEGIN_cjel
def main():
    syst = make_system()

    # Check that the system looks as intended.
    kwant.plot(syst)

    # Finalize the system.
    syst = syst.finalized()

    # We should see conductance steps.
    plot_conductance(syst, energies=[0.01 * i for i in range(100)])
#HIDDEN_END_cjel


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
#HIDDEN_BEGIN_ypbj
if __name__ == '__main__':
    main()
#HIDDEN_END_ypbj
