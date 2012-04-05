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

# For plotting
import pylab

# For matrix support
import numpy

# define Pauli-matrices for convenience
sigma_0 = numpy.eye(2)
sigma_x = numpy.array([[0, 1], [1, 0]])
sigma_y = numpy.array([[0, -1j], [1j, 0]])
sigma_z = numpy.array([[1, 0], [0, -1]])


def make_system(a=1, t=1.0, alpha=0.5, e_z=0.08, W=10, L=30):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity).
    lat = kwant.lattice.Square(a)

    sys = kwant.Builder()
    sys.default_site_group = lat

    #### Define the scattering region. ####
    sys[((x, y) for x in range(L) for y in range(W))] = 4 * t * sigma_0 + \
        e_z * sigma_z
    # hoppings in x-direction
    sys[sys.possible_hoppings((1, 0), lat, lat)] = - t * sigma_0 - \
        1j * alpha * sigma_y
    # hoppings in y-directions
    sys[sys.possible_hoppings((0, 1), lat, lat)] = - t * sigma_0 + \
        1j * alpha * sigma_x

    #### Define the leads. ####
    # left lead
    sym_lead0 = kwant.TranslationalSymmetry([lat.vec((-1, 0))])
    lead0 = kwant.Builder(sym_lead0)
    lead0.default_site_group = lat

    lead0[((0, j) for j in xrange(W))] = 4 * t * sigma_0 + e_z * sigma_z
    # hoppings in x-direction
    lead0[lead0.possible_hoppings((1, 0), lat, lat)] = - t * sigma_0 - \
        1j * alpha * sigma_y
    # hoppings in y-directions
    lead0[lead0.possible_hoppings((0, 1), lat, lat)] = - t * sigma_0 + \
        1j * alpha * sigma_x

    # Then the lead to the right
    # (again, obtained using reverse()
    lead1 = lead0.reversed()

    #### Attach the leads and return the finalized system. ####
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)

    return sys

def plot_conductance(fsys, energies):
    # Compute conductance
    data = []
    for energy in energies:
        smatrix = kwant.solve(fsys, energy)
        data.append(smatrix.transmission(1, 0))

    pylab.plot(energies, data)
    pylab.xlabel("energy [in units of t]")
    pylab.ylabel("conductance [in units of e^2/h]")
    pylab.show()


def main():
    sys = make_system()

    # Check that the system looks as intended.
    kwant.plot(sys)

    # Finalize the system.
    fsys = sys.finalized()

    # We should see non-monotonic conductance steps.
    plot_conductance(fsys, energies=[0.01 * i - 0.3 for i in xrange(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
