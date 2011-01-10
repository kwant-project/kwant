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
import kwant

# For eigenvalue computation
import scipy.linalg as la

# For plotting
import pylab

def make_system(a=1, t=1.0, r=10):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity).

    lat = kwant.lattice.Square(a)

    sys = kwant.Builder()

    # Define the quantum dot
    def circle(pos):
        (x, y) = pos
        rsq = x**2 + y**2
        return rsq < r**2

    def hopx(site1, site2):
        # The magnetic field is controlled by the global variable B
        y = site1.pos[1]
        return - t * exp(-1j * B * y)

    sys[lat.shape(circle, (0, 0))] = 4 * t
    # hoppings in x-direction
    sys[sys.possible_hoppings((1, 0), lat, lat)] = hopx
    # hoppings in y-directions
    sys[sys.possible_hoppings((0, 1), lat, lat)] = - t

    # It's a closed system for a change, so no leads
    return sys.finalized()


def plot_spectrum(fsys, Bfields):
    # global variable B controls the magnetic field
    global B

    # In the following, we compute the spectrum of the quantum dot
    # using dense matrix methods. This works in this toy example, as
    # the system is tiny. In a real example, one would want to use
    # sparse matrix methods

    energies = []
    for Bfield in Bfields:
        B = Bfield

        # Obtain the Hamiltonian as a dense matrix
        ham_mat = fsys.hamiltonian_submatrix()

        ev = la.eigh(ham_mat, eigvals_only=True)

        # we only plot the 15 lowest eigenvalues
        energies.append(ev[:15])

    pylab.plot(Bfields, energies)
    pylab.xlabel("magnetic field [some arbitrary units]")
    pylab.ylabel("energy [in units of t]")
    pylab.show()


def main():
    fsys = make_system()

    # Check that the system looks as intended.
    kwant.plot(fsys)

    # We should observe energy levels that flow towards Landau
    # level energies with increasing magnetic field
    plot_spectrum(fsys, [iB * 0.002 for iB in xrange(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
