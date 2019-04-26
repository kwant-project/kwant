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
    :download:`closed_system.py </code/download/closed_system.py>`

Although Kwant is (currently) mainly aimed towards transport problems, it
can also easily be used to compute properties of closed systems -- after
all, a closed system is nothing more than a scattering region without leads!

In this example, we compute the wave functions of a closed circular quantum dot
and its spectrum as a function of magnetic field (Fock-Darwin spectrum).

To compute the eigenenergies and eigenstates, we will make use of the sparse
linear algebra functionality of `SciPy <https://www.scipy.org>`_, which
interfaces the ARPACK package:

.. literalinclude:: /code/include/closed_system.py
    :start-after: #HIDDEN_BEGIN_tibv
    :end-before: #HIDDEN_END_tibv

We set up the system using the `shape`-function as in
:ref:`tutorial-abring`, but do not add any leads:

.. literalinclude:: /code/include/closed_system.py
    :start-after: #HIDDEN_BEGIN_qlyd
    :end-before: #HIDDEN_END_qlyd

We add the magnetic field using a function and a global variable as we
did in the two previous tutorial. (Here, the gauge is chosen such that
:math:`A_x(y) = - B y` and :math:`A_y=0`.)

The spectrum can be obtained by diagonalizing the Hamiltonian of the
system, which in turn can be obtained from the finalized
system using `~kwant.system.System.hamiltonian_submatrix`:

.. literalinclude:: /code/include/closed_system.py
    :start-after: #HIDDEN_BEGIN_yvri
    :end-before: #HIDDEN_END_yvri

Note that we use sparse linear algebra to efficiently calculate only a
few lowest eigenvalues. Finally, we obtain the result:

.. image:: /code/figure/closed_system_result.*

At zero magnetic field several energy levels are degenerate (since our
quantum dot is rather symmetric). These degeneracies are split
by the magnetic field, and the eigenenergies flow towards the
Landau level energies at higher magnetic fields [#]_.

The eigenvectors are obtained very similarly, and can be plotted directly
using `~kwant.plotter.map`:

.. literalinclude:: /code/include/closed_system.py
    :start-after: #HIDDEN_BEGIN_wave
    :end-before: #HIDDEN_END_wave

.. image:: /code/figure/closed_system_eigenvector.*

The last two arguments to `~kwant.plotter.map` are optional.  The first prevents
a colorbar from appearing.  The second, ``oversampling=1``, makes the image look
better for the special case of a square lattice.


As our model breaks time reversal symmetry (because of the applied magnetic
field) we can also see an interesting property of the eigenstates, namely
that they can have *non-zero* local current. We can calculate the local
current due to a state by using `kwant.operator.Current` and plotting
it using `kwant.plotter.current`:

.. literalinclude:: /code/include/closed_system.py
    :start-after: #HIDDEN_BEGIN_current
    :end-before: #HIDDEN_END_current

.. image:: /code/figure/closed_system_current.*

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
