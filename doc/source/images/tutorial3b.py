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

import latex, html

# First, define the tight-binding system

sys = kwant.Builder()

# Here, we are only working with square lattices

# for simplicity, take lattice constant = 1
a = 1
lat = kwant.lattice.Square(a)

t = 1.0
r = 10

# Define the quantum dot

def circle(pos):
    (x, y) = pos
    rsq = x**2 + y**2
    return rsq < r**2

def hopx(site1, site2):
    y = site1.pos[1]
    return - t * exp(-1j * B * y)

sys[lat.shape(circle, (0, 0))] = 4 * t
# hoppings in x-direction
sys[sys.possible_hoppings((1, 0), lat, lat)] = hopx
# hoppings in y-directions
sys[sys.possible_hoppings((0, 1), lat, lat)] = - t

# It's a closed system for a change, so no leads

# finalize the system

fsys = sys.finalized()

# and plot it, to make sure it's proper

kwant.plot(fsys, "tutorial3b_sys.pdf")
kwant.plot(fsys, "tutorial3b_sys.png")

# In the following, we compute the spectrum of the quantum dot
# using dense matrix methods. This works in this toy example,
# as the system is tiny. In a real example, one would want to use
# sparse matrix methods

import scipy.linalg as la

Bs = []
energies = []
for iB in xrange(100):
    B = iB * 0.002

# Obtain the Hamiltonian as a dense matrix
    ham_mat = fsys.hamiltonian_submatrix()

    ev = la.eigh(ham_mat, eigvals_only=True)

    Bs.append(B)
    energies.append(ev[:15])

import pylab

pylab.plot(Bs, energies)
pylab.xlabel("magnetic field [some arbitrary units]",
             fontsize=latex.mpl_label_size)
pylab.ylabel("energy [in units of t]",
             fontsize=latex.mpl_label_size)
fig = pylab.gcf()
pylab.setp(fig.get_axes()[0].get_xticklabels(),
           fontsize=latex.mpl_tick_size)
pylab.setp(fig.get_axes()[0].get_yticklabels(),
           fontsize=latex.mpl_tick_size)
fig.set_size_inches(latex.mpl_width_in, latex.mpl_width_in*3./4.)
fig.subplots_adjust(left=0.15, right=0.95, top=0.95, bottom=0.15)
fig.savefig("tutorial3b_result.pdf")
fig.savefig("tutorial3b_result.png",
            dpi=(html.figwidth_px/latex.mpl_width_in))
