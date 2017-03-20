# Tutorial 2.7.1. 2D example: graphene quantum dot
# ================================================
#
# Physics background
# ------------------
#  - graphene edge states
#
# Kwant features highlighted
# --------------------------
#  - demonstrate different ways of plotting

import kwant
from matplotlib import pyplot

#HIDDEN_BEGIN_makesyst
lat = kwant.lattice.honeycomb()
a, b = lat.sublattices

def make_system(r=8, t=-1, tp=-0.1):

    def circle(pos):
        x, y = pos
        return x**2 + y**2 < r**2

    syst = kwant.Builder()
    syst[lat.shape(circle, (0, 0))] = 0
    syst[lat.neighbors()] = t
    syst.eradicate_dangling()
    if tp:
        syst[lat.neighbors(2)] = tp

    return syst
#HIDDEN_END_makesyst


#HIDDEN_BEGIN_plotsyst1
def plot_system(syst):
    kwant.plot(syst)
#HIDDEN_END_plotsyst1
    # the standard plot is ok, but not very intelligible. One can do
    # better by playing wioth colors and linewidths

    # use color and linewidths to get a better plot
#HIDDEN_BEGIN_plotsyst2
    def family_color(site):
        return 'black' if site.family == a else 'white'

    def hopping_lw(site1, site2):
        return 0.04 if site1.family == site2.family else 0.1

    kwant.plot(syst, site_lw=0.1, site_color=family_color, hop_lw=hopping_lw)
#HIDDEN_END_plotsyst2


#HIDDEN_BEGIN_plotdata1
def plot_data(syst, n):
    import scipy.linalg as la

    syst = syst.finalized()
    ham = syst.hamiltonian_submatrix()
    evecs = la.eigh(ham)[1]

    wf = abs(evecs[:, n])**2
#HIDDEN_END_plotdata1

    # the usual - works great in general, looks just a bit crufty for
    # small systems

#HIDDEN_BEGIN_plotdata2
    kwant.plotter.map(syst, wf, oversampling=10, cmap='gist_heat_r')
#HIDDEN_END_plotdata2

    # use two different sort of triangles to cleanly fill the space
#HIDDEN_BEGIN_plotdata3
    def family_shape(i):
        site = syst.sites[i]
        return ('p', 3, 180) if site.family == a else ('p', 3, 0)

    def family_color(i):
        return 'black' if syst.sites[i].family == a else 'white'

    kwant.plot(syst, site_color=wf, site_symbol=family_shape,
               site_size=0.5, hop_lw=0, cmap='gist_heat_r')
#HIDDEN_END_plotdata3

    # plot by changing the symbols itself
#HIDDEN_BEGIN_plotdata4
    def site_size(i):
        return 3 * wf[i] / wf.max()

    kwant.plot(syst, site_size=site_size, site_color=(0, 0, 1, 0.3),
               hop_lw=0.1)
#HIDDEN_END_plotdata4


def main():
    # plot the graphene system in different styles
    syst = make_system()

    plot_system(syst)

    # compute a wavefunction (number 225) and plot it in different
    # styles
    syst = make_system(tp=0)

    plot_data(syst, 225)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
