Beyond transport: Band structures and closed systems
----------------------------------------------------

Band structure calculations
...........................

When doing transport simulations, one also often needs to know the
band structure of the leads, i.e. the energies of the propagating
plane waves in the leads as a function of momentum. This band structure
contains information about the number of modes, the velocities, ...

In this example, we aim to compute the bandstructure of a simple
tight-binding wire.

Computing band structures in kwant is easy. Just define a lead in the
usual way:

.. literalinclude:: 3-band_structure.py
    :start-after: #HIDDEN_BEGIN_zxip
    :end-before: #HIDDEN_END_zxip

"Usual way" means defining a translational symmetry vector, as well
as one unit cell of the lead, and the hoppings to neighboring
unit cells. This information is enough to make the infinite, translationally
invariant system needed for band structure calculations.

In the previous examples `~kwant.builder.Builder` instances like the one
created above were attached as leads to the ``Builder`` instance of the
scattering region and the latter was finalized.  The thus created system
contained implicitly finalized versions of the attached leads.  But now we are
working with a single lead and there is no scattering region.  So we have to
finalized the ``Builder`` of our sole lead explicitly.

That finalized lead is then passed as to `kwant.physics.Bands`.  This
creates an object that behaves just like a function: when called with a
momentum ``k`` as parameter it returns the eigenenergies of the translational
invariant system for that momentum.  Computing these eigenenergies for a range
of momenta then yields the bandstructure:

.. literalinclude:: 3-band_structure.py
    :start-after: #HIDDEN_BEGIN_pejz
    :end-before: #HIDDEN_END_pejz

This gives the result:

.. image:: ../images/3-band_structure_result.*

where we observe the cosine-like dispersion of the square lattice. Close
to ``k=0`` this agrees well with the quadratic dispersion this tight-binding
Hamiltonian is approximating.

.. seealso::
     The full source code can be found in
     :download:`tutorial/3-band_structure.py <../../../tutorial/3-band_structure.py>`

Closed systems
..............

Although kwant is (currently) mainly aimed towards transport problem, it
can also easily be used to compute properties of closed systems -- after
all, a closed system is nothing more than a scattering region without leads!

In this example, we compute the spectrum of a closed, (approximately)
circular quantum dot as a function of magnetic field
(Fock-Darwin spectrum).

To compute the eigenenergies, we will make use of the linear algebra
functionality of `scipy <www.scipy.org>`_:

.. literalinclude:: 3-closed_system.py
    :start-after: #HIDDEN_BEGIN_tibv
    :end-before: #HIDDEN_END_tibv

We set up the system using the `shape`-function as in
:ref:`tutorial-abring`, but do not add any leads:

.. literalinclude:: 3-closed_system.py
    :start-after: #HIDDEN_BEGIN_qlyd
    :end-before: #HIDDEN_END_qlyd

We add the magnetic field using a function and a global variable as we
did in the two previous tutorial. (Here, the gauge is chosen such that
:math:`A_x(y) = - B y` and :math:`A_y=0`.)

The spectrum can be obtained by diagonalizing the Hamiltonian of the
system, which in turn can be obtained from the finalized
system using `~kwant.system.System.hamiltonian_submatrix`:

.. literalinclude:: 3-closed_system.py
    :start-after: #HIDDEN_BEGIN_yvri
    :end-before: #HIDDEN_END_yvri

In this toy model we use dense matrices and dense matrix algebra since
the system is very small. (In a real application one would probably
want to use sparse matrix methods.) Finally, we obtain the result:

.. image:: ../images/3-closed_system_result.*

At zero magnetic field several energy levels are degenerate (since our
quantum dot is rather symmetric). These degeneracies are split
by the magnetic field, and the eigenenergies flow towards the
Landau level energies at higher magnetic fields [#]

.. seealso::
    The full source code can be found in
    :download:`tutorial/3-closed_system.py <../../../tutorial/3-closed_system.py>`

.. specialnote:: Technical details

  - `~kwant.system.System.hamiltonian_submatrix` can also return a sparse
    matrix, if the optional argument ``sparse=True``. The sparse matrix is in
    SciPy's `scipy.sparse.coo_matrix` format, which can be easily be converted
    to various other sparse matrix formats (see `SciPy's documentation
    <http://docs.scipy.org/doc/scipy/reference/>`_).

.. rubric:: Footnotes

.. [#] Again, in this tutorial example no care was taken into choosing
       appropriate material parameters or units. For this reason, magnetic
       field is given only in "arbitrary units"
