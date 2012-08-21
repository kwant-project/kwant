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
from matplotlib import pyplot

import latex, html


def make_system(a=1, t=1.0, r=10):
    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity).

    lat = kwant.lattice.Square(a)

    sys = kwant.Builder()

    # Define the quantum dot
    def circle(pos):
        (x, y) = pos
        rsq = x ** 2 + y ** 2
        return rsq < r ** 2

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
    return sys


def plot_spectrum(sys, Bfields):
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
        ham_mat = sys.hamiltonian_submatrix()[0]

        ev = la.eigh(ham_mat, eigvals_only=True)

        # we only plot the 15 lowest eigenvalues
        energies.append(ev[:15])

    fig = pyplot.figure()
    pyplot.plot(Bfields, energies)
    pyplot.xlabel("magnetic field [some arbitrary units]",
                 fontsize=latex.mpl_label_size)
    pyplot.ylabel("energy [in units of t]",
                 fontsize=latex.mpl_label_size)
    pyplot.setp(fig.get_axes()[0].get_xticklabels(),
               fontsize=latex.mpl_tick_size)
    pyplot.setp(fig.get_axes()[0].get_yticklabels(),
               fontsize=latex.mpl_tick_size)
    fig.set_size_inches(latex.mpl_width_in, latex.mpl_width_in*3./4.)
    fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
    fig.savefig("3-closed_system_result.pdf")
    fig.savefig("3-closed_system_result.png",
                dpi=(html.figwidth_px/latex.mpl_width_in))


def main():
    sys = make_system()

    # Check that the system looks as intended.
    kwant.plot(sys, filename="3-closed_system_sys.pdf",
               width=latex.figwidth_pt)
    kwant.plot(sys, filename="3-closed_system_sys.png",
               width=html.figwidth_px)

    # Finalize the system.
    sys = sys.finalized()

    # We should observe energy levels that flow towards Landau
    # level energies with increasing magnetic field
    plot_spectrum(sys, [iB * 0.002 for iB in xrange(100)])


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
