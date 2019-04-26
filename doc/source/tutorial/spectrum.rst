Beyond transport: Band structure and closed systems
---------------------------------------------------

Band structure calculations
...........................

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download:script:`band_structure`

.. jupyter-kernel::
    :id: band_structure

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.4.1. Band structure calculations
    # ===========================================
    #
    # Physics background
    # ------------------
    #  band structure of a simple quantum wire in tight-binding approximation
    #
    # Kwant features highlighted
    # --------------------------
    #  - Computing the band structure of a finalized lead.

    import kwant

    # For plotting
    from matplotlib import pyplot

When doing transport simulations, one also often needs to know the band
structure of the leads, i.e. the energies of the propagating plane waves in the
leads as a function of momentum. This band structure contains information about
the number of modes, their momenta and velocities.

In this example, we aim to compute the band structure of a simple
tight-binding wire.

Computing band structures in Kwant is easy. Just define a lead in the
usual way:

.. jupyter-execute::

    def make_lead(a=1, t=1.0, W=10):
        # Start with an empty lead with a single square lattice
        lat = kwant.lattice.square(a)

        sym_lead = kwant.TranslationalSymmetry((-a, 0))
        lead = kwant.Builder(sym_lead)

        # build up one unit cell of the lead, and add the hoppings
        # to the next unit cell
        for j in range(W):
            lead[lat(0, j)] = 4 * t

            if j > 0:
                lead[lat(0, j), lat(0, j - 1)] = -t

            lead[lat(1, j), lat(0, j)] = -t

        return lead

"Usual way" means defining a translational symmetry vector, as well
as one unit cell of the lead, and the hoppings to neighboring
unit cells. This information is enough to make the infinite, translationally
invariant system needed for band structure calculations.

In the previous examples `~kwant.builder.Builder` instances like the one
created above were attached as leads to the ``Builder`` instance of the
scattering region and the latter was finalized.  The thus created system
contained implicitly finalized versions of the attached leads. However, now
we are working with a single lead and there is no scattering region. Hence, we
have to finalize the ``Builder`` of our sole lead explicitly.

That finalized lead is then passed to `~kwant.plotter.bands`. This function
calculates energies of various bands at a range of momenta and plots the
calculated energies. It is really a convenience function, and if one needs to
do something more profound with the dispersion relation these energies may be
calculated directly using `~kwant.physics.Bands`. For now we just plot the
bandstructure:

.. jupyter-execute::

    def main():
        lead = make_lead().finalized()
        kwant.plotter.bands(lead, show=False)
        pyplot.xlabel("momentum [(lattice constant)^-1]")
        pyplot.ylabel("energy [t]")
        pyplot.show()

This gives the result:

.. jupyter-execute::
    :hide-code:

    # Call the main function if the script gets executed (as opposed to imported).
    # See <http://docs.python.org/library/__main__.html>.
    if __name__ == '__main__':
        main()

where we observe the cosine-like dispersion of the square lattice. Close
to ``k=0`` this agrees well with the quadratic dispersion this tight-binding
Hamiltonian is approximating.

.. _closed-systems:

Closed systems
..............

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download:script:`closed_system`

.. jupyter-kernel::
    :id: closed_system

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.4.2. Closed systems
    # ==============================
    #
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
    from matplotlib import pyplot
    import kwant

Although Kwant is (currently) mainly aimed towards transport problems, it
can also easily be used to compute properties of closed systems -- after
all, a closed system is nothing more than a scattering region without leads!

In this example, we compute the wave functions of a closed circular quantum dot
and its spectrum as a function of magnetic field (Fock-Darwin spectrum).

To compute the eigenenergies and eigenstates, we will make use of the sparse
linear algebra functionality of `SciPy <https://www.scipy.org>`_, which
interfaces the ARPACK package:


.. jupyter-execute::

    # For eigenvalue computation
    import scipy.sparse.linalg as sla

We set up the system using the `shape`-function as in
:ref:`tutorial-abring`, but do not add any leads:

.. jupyter-execute::
    :hide-code:

    a = 1
    t = 1.0
    r = 10

.. jupyter-execute::

    def make_system(a=1, t=1.0, r=10):

        lat = kwant.lattice.square(a, norbs=1)

        syst = kwant.Builder()

        # Define the quantum dot
        def circle(pos):
            (x, y) = pos
            rsq = x ** 2 + y ** 2
            return rsq < r ** 2

        def hopx(site1, site2, B):
            # The magnetic field is controlled by the parameter B
            y = site1.pos[1]
            return -t * exp(-1j * B * y)

        syst[lat.shape(circle, (0, 0))] = 4 * t
        # hoppings in x-direction
        syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = hopx
        # hoppings in y-directions
        syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = -t

        # It's a closed system for a change, so no leads
        return syst

We add the magnetic field using a function and a global variable as we
did in the two previous tutorial. (Here, the gauge is chosen such that
:math:`A_x(y) = - B y` and :math:`A_y=0`.)

The spectrum can be obtained by diagonalizing the Hamiltonian of the
system, which in turn can be obtained from the finalized
system using `~kwant.system.System.hamiltonian_submatrix`:

.. jupyter-execute::

    def plot_spectrum(syst, Bfields):

        energies = []
        for B in Bfields:
            # Obtain the Hamiltonian as a sparse matrix
            ham_mat = syst.hamiltonian_submatrix(params=dict(B=B), sparse=True)

            # we only calculate the 15 lowest eigenvalues
            ev = sla.eigsh(ham_mat.tocsc(), k=15, sigma=0,
                           return_eigenvectors=False)

            energies.append(ev)

        pyplot.figure()
        pyplot.plot(Bfields, energies)
        pyplot.xlabel("magnetic field [arbitrary units]")
        pyplot.ylabel("energy [t]")
        pyplot.show()

Note that we use sparse linear algebra to efficiently calculate only a
few lowest eigenvalues. Finally, we obtain the result:

.. jupyter-execute::
    :hide-code:

    syst = make_system()

    syst = syst.finalized()

    # We should observe energy levels that flow towards Landau
    # level energies with increasing magnetic field.
    plot_spectrum(syst, [iB * 0.002 for iB in range(100)])

At zero magnetic field several energy levels are degenerate (since our
quantum dot is rather symmetric). These degeneracies are split
by the magnetic field, and the eigenenergies flow towards the
Landau level energies at higher magnetic fields [#]_.

The eigenvectors are obtained very similarly, and can be plotted directly
using `~kwant.plotter.map`:

.. jupyter-execute::
    :hide-code:

    def sorted_eigs(ev):
        evals, evecs = ev
        evals, evecs = map(np.array, zip(*sorted(zip(evals, evecs.transpose()))))
        return evals, evecs.transpose()

.. jupyter-execute::

    def plot_wave_function(syst, B=0.001):
        # Calculate the wave functions in the system.
        ham_mat = syst.hamiltonian_submatrix(sparse=True, params=dict(B=B))
        evals, evecs = sorted_eigs(sla.eigsh(ham_mat.tocsc(), k=20, sigma=0))

        # Plot the probability density of the 10th eigenmode.
        kwant.plotter.map(syst, np.abs(evecs[:, 9])**2,
                          colorbar=False, oversampling=1)

.. jupyter-execute::
    :hide-code:

    syst = make_system(r=30)

    # Plot an eigenmode of a circular dot. Here we create a larger system for
    # better spatial resolution.
    syst = make_system(r=30).finalized()
    plot_wave_function(syst);

The last two arguments to `~kwant.plotter.map` are optional.  The first prevents
a colorbar from appearing.  The second, ``oversampling=1``, makes the image look
better for the special case of a square lattice.


As our model breaks time reversal symmetry (because of the applied magnetic
field) we can also see an interesting property of the eigenstates, namely
that they can have *non-zero* local current. We can calculate the local
current due to a state by using `kwant.operator.Current` and plotting
it using `kwant.plotter.current`:

.. jupyter-execute::

    def plot_current(syst, B=0.001):
        # Calculate the wave functions in the system.
        ham_mat = syst.hamiltonian_submatrix(sparse=True, params=dict(B=B))
        evals, evecs = sorted_eigs(sla.eigsh(ham_mat.tocsc(), k=20, sigma=0))

        # Calculate and plot the local current of the 10th eigenmode.
        J = kwant.operator.Current(syst)
        current = J(evecs[:, 9], params=dict(B=B))
        kwant.plotter.current(syst, current, colorbar=False)

.. jupyter-execute::
    :hide-code:

    plot_current(syst);

.. specialnote:: Technical details

  - `~kwant.system.System.hamiltonian_submatrix` can also return a sparse
    matrix, if the optional argument ``sparse=True``. The sparse matrix is in
    SciPy's ``scipy.sparse.coo_matrix`` format, which can be easily be converted
    to various other sparse matrix formats (see `SciPy's documentation
    <https://docs.scipy.org/doc/scipy/reference/>`_).

.. rubric:: Footnotes

.. [#] Again, in this tutorial example no care was taken into choosing
       appropriate material parameters or units. For this reason, magnetic
       field is given only in "arbitrary units".
