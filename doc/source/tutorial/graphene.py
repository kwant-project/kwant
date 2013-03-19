# Physics background
# ------------------
#  Transport through a graphene quantum dot with a pn-junction
#
# Kwant features highlighted
# --------------------------
#  - Application of all the aspects of tutorials 1-3 to a more complicated
#    lattice, namely graphene

from __future__ import division  # so that 1/2 == 0.5, and not 0
from math import pi, sqrt, tanh

import kwant

# For computing eigenvalues
import scipy.sparse.linalg as sla

# For plotting
from matplotlib import pyplot


# Define the graphene lattice
sin_30, cos_30 = (1 / 2, sqrt(3) / 2)
#HIDDEN_BEGIN_hnla
graphene = kwant.lattice.general([(1, 0), (sin_30, cos_30)],
                                 [(0, 0), (0, 1 / sqrt(3))])
a, b = graphene.sublattices
#HIDDEN_END_hnla


#HIDDEN_BEGIN_shzy
def make_system(r=10, w=2.0, pot=0.1):

    #### Define the scattering region. ####
    # circular scattering region
    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    sys = kwant.Builder()

    # w: width and pot: potential maximum of the p-n junction
    def potential(site):
        (x, y) = site.pos
        d = y * cos_30 + x * sin_30
        return pot * tanh(d / w)

    sys[graphene.shape(circle, (0, 0))] = potential
#HIDDEN_END_shzy

    # specify the hoppings of the graphene lattice in the
    # format expected by builder.HoppingKind
#HIDDEN_BEGIN_hsmc
    hoppings = (((0, 0), a, b), ((0, 1), a, b), ((-1, 1), a, b))
#HIDDEN_END_hsmc
#HIDDEN_BEGIN_bfwb
    sys[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -1
#HIDDEN_END_bfwb

    # Modify the scattering region
#HIDDEN_BEGIN_efut
    del sys[a(0, 0)]
    sys[a(-2, 1), b(2, 2)] = -1
#HIDDEN_END_efut

    #### Define the leads. ####
#HIDDEN_BEGIN_aakh
    # left lead
    sym0 = kwant.TranslationalSymmetry(graphene.vec((-1, 0)))

    def lead0_shape(pos):
        x, y = pos
        return (-1 < x < 1) and (-0.4 * r < y < 0.4 * r)

    lead0 = kwant.Builder(sym0)
    lead0[graphene.shape(lead0_shape, (0, 0))] = -pot
    lead0[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -1

    # The second lead, going to the top right
    sym1 = kwant.TranslationalSymmetry(graphene.vec((0, 1)))

    def lead1_shape(pos):
        x, y = pos
        u = x * sin_30 + y * cos_30
        v = y * sin_30 - x * cos_30
        return (-1 < u < 1) and (-0.4 * r < v < 0.4 * r)

    lead1 = kwant.Builder(sym1)
    lead1[graphene.shape(lead1_shape, (0, 0))] = pot
    lead1[[kwant.builder.HoppingKind(*hopping) for hopping in hoppings]] = -1
#HIDDEN_END_aakh

#HIDDEN_BEGIN_kmmw
    return sys, [lead0, lead1]
#HIDDEN_END_kmmw


#HIDDEN_BEGIN_zydk
def compute_evs(sys):
    # Compute some eigenvalues of the closed system
    sparse_mat = sys.hamiltonian_submatrix(sparse=True)

    try:
        # This requires SciPy version >= 0.9.0
        # Failure (i.e. insufficient SciPy version) is not critical
        # for the remainder of the tutorial, hence the try-block
        evs = sla.eigs(sparse_mat, 2)[0]
        print evs
    except:
        pass
#HIDDEN_END_zydk


def plot_conductance(sys, energies):
    # Compute transmission as a function of energy
    data = []
    for energy in energies:
        smatrix = kwant.solve(sys, energy)
        data.append(smatrix.transmission(0, 1))

    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [in units of t]")
    pyplot.ylabel("conductance [in units of e^2/h]")
    pyplot.show()


def plot_bandstructure(flead, momenta):
    bands = kwant.physics.Bands(flead)
    energies = [bands(k) for k in momenta]

    pyplot.figure()
    pyplot.plot(momenta, energies)
    pyplot.xlabel("momentum [in units of (lattice constant)^-1]")
    pyplot.ylabel("energy [in units of t]")
    pyplot.show()


#HIDDEN The part of the following code block which begins with group_colors
#HIDDEN is included verbatim in the tutorial text because nested code examples
#HIDDEN are not supported.  Remember to update the tutorial text when you
#HIDDEN modify this block.
#HIDDEN_BEGIN_itkk
def main():
    pot = 0.1
    sys, leads = make_system(pot=pot)

    # To highlight the two sublattices of graphene, we plot one with
    # a filled, and the other one with an open circle:
    def group_colors(site):
        return 0 if site.group == a else 1

    # Plot the closed system without leads.
    kwant.plot(sys, site_color=group_colors, colorbar=False)
#HIDDEN_END_itkk

    # Compute some eigenvalues.
#HIDDEN_BEGIN_jmbi
    compute_evs(sys.finalized())
#HIDDEN_END_jmbi

    # Attach the leads to the system.
    for lead in leads:
        sys.attach_lead(lead)

    # Then, plot the system with leads.
    kwant.plot(sys, site_color=group_colors, colorbar=False)

    # Finalize the system.
    sys = sys.finalized()

    # Compute the band structure of lead 0.
    momenta = [-pi + 0.02 * pi * i for i in xrange(101)]
    plot_bandstructure(sys.leads[0], momenta)

    # Plot conductance.
    energies = [-2 * pot + 4. / 50. * pot * i for i in xrange(51)]
    plot_conductance(sys, energies)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
