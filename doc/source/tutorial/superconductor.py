# Tutorial 2.6. "Superconductors": orbitals, conservation laws and symmetries
# ===========================================================================
#
# Physics background
# ------------------
# - conductance of a NS-junction (Andreev reflection, superconducting gap)
#
# Kwant features highlighted
# --------------------------
# - Implementing electron and hole ("orbital") degrees of freedom
#   using conservation laws.
# - Use of discrete symmetries to relate scattering states.

import kwant

import tinyarray
import numpy as np

# For plotting
from matplotlib import pyplot

tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])

#HIDDEN_BEGIN_nbvn
def make_system(a=1, W=10, L=10, barrier=1.5, barrierpos=(3, 4),
                mu=0.4, Delta=0.1, Deltapos=4, t=1.0, phs=True):
    # Start with an empty tight-binding system. On each site, there
    # are now electron and hole orbitals, so we must specify the
    # number of orbitals per site. The orbital structure is the same
    # as in the Hamiltonian.
    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()

    #### Define the scattering region. ####
    # The superconducting order parameter couples electron and hole orbitals
    # on each site, and hence enters as an onsite potential.
    # The pairing is only included beyond the point 'Deltapos' in the scattering region.
    syst[(lat(x, y) for x in range(Deltapos) for y in range(W))] = (4 * t - mu) * tau_z
    syst[(lat(x, y) for x in range(Deltapos, L) for y in range(W))] = (4 * t - mu) * tau_z + Delta * tau_x

    # The tunnel barrier
    syst[(lat(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = (4 * t + barrier - mu) * tau_z

    # Hoppings
    syst[lat.neighbors()] = -t * tau_z
#HIDDEN_END_nbvn
#HIDDEN_BEGIN_ttth
    #### Define the leads. ####
    # Left lead - normal, so the order parameter is zero.
    sym_left = kwant.TranslationalSymmetry((-a, 0))
    # Specify the conservation law used to treat electrons and holes separately.
    # We only do this in the left lead, where the pairing is zero.
    lead0 = kwant.Builder(sym_left, conservation_law=-tau_z, particle_hole=tau_y)
    lead0[(lat(0, j) for j in range(W))] = (4 * t - mu) * tau_z
    lead0[lat.neighbors()] = -t * tau_z
#HIDDEN_END_ttth
#HIDDEN_BEGIN_zuuw
    # Right lead - superconducting, so the order parameter is included.
    sym_right = kwant.TranslationalSymmetry((a, 0))
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(0, j) for j in range(W))] = (4 * t - mu) * tau_z + Delta * tau_x
    lead1[lat.neighbors()] = -t * tau_z

    #### Attach the leads and return the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)

    return syst
#HIDDEN_END_zuuw

#HIDDEN_BEGIN_jbjt
def plot_conductance(syst, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
        # Conductance is N - R_ee + R_he
        data.append(smatrix.submatrix((0, 0), (0, 0)).shape[0] -
                    smatrix.transmission((0, 0), (0, 0)) +
                    smatrix.transmission((0, 1), (0, 0)))
#HIDDEN_END_jbjt
    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [e^2/h]")
    pyplot.show()

#HIDDEN_BEGIN_pqmp
def check_PHS(syst):
    # Scattering matrix
    s = kwant.smatrix(syst, energy=0)
    # Electron to electron block
    s_ee = s.submatrix((0,0), (0,0))
    # Hole to hole block
    s_hh = s.submatrix((0,1), (0,1))
    print('s_ee: \n', np.round(s_ee, 3))
    print('s_hh: \n', np.round(s_hh[::-1, ::-1], 3))
    print('s_ee - s_hh^*: \n',
          np.round(s_ee - s_hh[::-1, ::-1].conj(), 3), '\n')
    # Electron to hole block
    s_he = s.submatrix((0,1), (0,0))
    # Hole to electron block
    s_eh = s.submatrix((0,0), (0,1))
    print('s_he: \n', np.round(s_he, 3))
    print('s_eh: \n', np.round(s_eh[::-1, ::-1], 3))
    print('s_he + s_eh^*: \n',
          np.round(s_he + s_eh[::-1, ::-1].conj(), 3))
#HIDDEN_END_pqmp

def main():
    syst = make_system(W=10)

    # Check that the system looks as intended.
    kwant.plot(syst)

    # Finalize the system.
    syst = syst.finalized()

    # Check particle-hole symmetry of the scattering matrix
    check_PHS(syst)

    # Compute and plot the conductance
    plot_conductance(syst, energies=[0.002 * i for i in range(-10, 100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
