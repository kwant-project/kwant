# Physics background
# ------------------
#  Fock-darwin spectrum of a quantum dot (energy spectrum in
#  as a function of a magnetic field)
#
# Kwant features highlighted
# --------------------------
#  - Use of `hamiltonian_submatrix` in order to obtain a Hamiltonian
#    matrix.

from cmath import exp
import numpy as np
import kwant

# For eigenvalue computation
#HIDDEN_BEGIN_tibv
import scipy.sparse.linalg as sla
#HIDDEN_END_tibv

# For plotting
from matplotlib import pyplot


def make_system(a=1, t=1.0, r=10):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity).

#HIDDEN_BEGIN_qlyd
    lat = kwant.lattice.square(a)

    sys = kwant.Builder()

    # Define the quantum dot
    def circle(pos):
        (x, y) = pos
        rsq = x ** 2 + y ** 2
        return rsq < r ** 2

    def hopx(site1, site2, B=0):
        # The magnetic field is controlled by the parameter B
        y = site1.pos[1]
        return -t * exp(-1j * B * y)

    sys[lat.shape(circle, (0, 0))] = 4 * t
    # hoppings in x-direction
    sys[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
    # hoppings in y-directions
    sys[kwant.builder.HoppingKind((0, 1), lat, lat)] = -t

    # It's a closed system for a change, so no leads
    return sys
#HIDDEN_END_qlyd


#HIDDEN_BEGIN_yvri
def plot_spectrum(sys, Bfields):

    # In the following, we compute the spectrum of the quantum dot
    # using dense matrix methods. This works in this toy example, as
    # the system is tiny. In a real example, one would want to use
    # sparse matrix methods

    energies = []
    for B in Bfields:
        # Obtain the Hamiltonian as a dense matrix
        ham_mat = sys.hamiltonian_submatrix(args=[B], sparse=True)

        # we only calculate the 15 lowest eigenvalues
        ev = sla.eigsh(ham_mat, k=15, which='SM', return_eigenvectors=False)

        energies.append(ev)

    pyplot.figure()
    pyplot.plot(Bfields, energies)
    pyplot.xlabel("magnetic field [arbitrary units]")
    pyplot.ylabel("energy [t]")
    pyplot.show()
#HIDDEN_END_yvri


#HIDDEN_BEGIN_wave
def plot_wave_function(sys):
    # Calculate the wave functions in the system.
    ham_mat = sys.hamiltonian_submatrix(sparse=True)
    evecs = sla.eigsh(ham_mat, k=20, which='SM')[1]

    # Plot the probability density of the 10th eigenmode.
    kwant.plotter.map(sys, np.abs(evecs[:, 9])**2,
                      colorbar=False, oversampling=1)
#HIDDEN_END_wave


def main():
    sys = make_system()

    # Check that the system looks as intended.
    kwant.plot(sys)

    # Finalize the system.
    sys = sys.finalized()

    # The following try-clause can be removed once SciPy 0.9 becomes uncommon.
    try:
        # We should observe energy levels that flow towards Landau
        # level energies with increasing magnetic field.
        plot_spectrum(sys, [iB * 0.002 for iB in xrange(100)])

        # Plot an eigenmode of a circular dot. Here we create a larger system for
        # better spatial resolution.
        sys = make_system(r=30).finalized()
        plot_wave_function(sys)
    except ValueError as e:
        if e.message == "Input matrix is not real-valued.":
            print "The calculation of eigenvalues failed because of a bug in SciPy 0.9."
            print "Please upgrade to a newer version of SciPy."
        else:
            raise


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
