First steps: setting up a simple system and computing conductance
-----------------------------------------------------------------

.. _tutorial_discretization_schrodinger:

Discretization of a Schrödinger Hamiltonian
...........................................

As first example, we compute the transmission probability through a
two-dimensional quantum wire.  The wire is described by the two-dimensional
Schrödinger equation

.. math::
    H = \frac{-\hbar^2}{2m}(\partial_x^2 + \partial_y^2) + V(y)

with a hard-wall confinement :math:`V(y)` in the y-direction.

To be able to implement the quantum wire with Kwant, the continuous Hamiltonian
:math:`H` has to be discretized thus turning it into a tight-binding
model.  For simplicity, we discretize :math:`H` on the sites of a square
lattice with lattice constant :math:`a`.  Each site with the integer
lattice coordinates :math:`(i, j)` has the real-space coordinates :math:`(x, y)
= (ai, aj)`.

Introducing the discretized positional states

.. math::
    \ket{i, j} \equiv \ket{ai, aj} = \ket{x, y}

the second-order differential operators can be expressed in the limit :math:`a
\to 0` as

.. math::
    \partial_x^2 = \frac{1}{a^2} \sum_{i, j} \left(\ket{i+1, j}\bra{i, j} +
    \ket{i, j}\bra{i+1, j} -2 \ket{i, j}\bra{i, j} \right),

and an equivalent expression for :math:`\partial_y^2`.  Subsitituting them in
the Hamiltonian gives us

.. math::
    H = \sum_{i,j} \big[ \left(V(ai, aj) + 4t\right)\ket{i,j}\bra{i,j}
    - t \big( \ket{i+1,j}\bra{i,j} + \ket{i,j}\bra{i+1,j}
    + \ket{i,j+1}\bra{i,j} + \ket{i,j}\bra{i,j+1} \big) \big]

with

.. math::
    t = \frac{\hbar^2}{2ma^2}.

For finite :math:`a`, this discretized Hamiltonian approximates the continuous
one to any required accuracy.  The approximation is good for all quantum states
with a wave length considerably larger than :math:`a`.

The remainder of this section demonstrates how to realize the discretized
Hamiltonian in Kwant and how to perform transmission calculations.  For
simplicity, we choose to work in such units that :math:`t = a = 1`.

.. _tutorial_quantum_wire:

Transport through a quantum wire
................................

.. seealso::
    You can execute the code examples live in your browser by
    activating thebelab:

    .. thebe-button:: Activate Thebelab

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download-script:`quantum_wire`

In order to use Kwant, we need to import it:

.. jupyter-kernel::
    :id: quantum_wire

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.2.2. Transport through a quantum wire
    # ================================================
    #
    # Physics background
    # ------------------
    #  Conductance of a quantum wire; subbands
    #
    # Kwant features highlighted
    # --------------------------
    #  - Builder for setting up transport systems easily
    #  - Making scattering region and leads
    #  - Using the simple sparse solver for computing Landauer conductance

    # For plotting
    from matplotlib import pyplot

.. jupyter-execute:: boilerplate.py
    :hide-code:

.. jupyter-execute::

    import kwant

Enabling Kwant is as easy as this [#]_ !

The first step is now to define the system with scattering region and
leads. For this we make use of the `~kwant.builder.Builder` type that allows to
define a system in a convenient way. We need to create an instance of it:

.. jupyter-execute::

    # First define the tight-binding system

    syst = kwant.Builder()

Observe that we just accessed `~kwant.builder.Builder` by the name
``kwant.Builder``.  We could have just as well written
``kwant.builder.Builder`` instead.  Kwant consists of a number of sub-packages
that are all covered in the :doc:`reference documentation
<../reference/index>`.  For convenience, some of the most widely-used members
of the sub-packages are also accessible directly through the top-level `kwant`
package.

Apart from `~kwant.builder.Builder` we also need to specify
what kind of sites we want to add to the system. Here we work with
a square lattice. For simplicity, we set the lattice constant to unity:

.. jupyter-execute::

    a = 1
    lat = kwant.lattice.square(a, norbs=1)

Since we work with a square lattice, we label the points with two
integer coordinates `(i, j)`. `~kwant.builder.Builder` then
allows us to add matrix elements corresponding to lattice points:
``syst[lat(i, j)] = ...`` sets the on-site energy for the point `(i, j)`,
and ``syst[lat(i1, j1), lat(i2, j2)] = ...`` the hopping matrix element
**from** point `(i2, j2)` **to** point `(i1, j1)`. In specifying ``norbs=1``
in the definition of the lattice we tell Kwant that there is 1 degree
of freedom per lattice site.

Note that we need to specify sites for `~kwant.builder.Builder`
in the form ``lat(i, j)``. The lattice object `lat` does the
translation from integer coordinates to proper site format
needed in Builder (more about that in the technical details below).

We now build a rectangular scattering region that is `W`
lattice points wide and `L` lattice points long:

.. jupyter-execute::

    t = 1.0
    W, L = 10, 30

    # Define the scattering region

    for i in range(L):
        for j in range(W):
            # On-site Hamiltonian
            syst[lat(i, j)] = 4 * t

            # Hopping in y-direction
            if j > 0:
                syst[lat(i, j), lat(i, j - 1)] = -t

            # Hopping in x-direction
            if i > 0:
                syst[lat(i, j), lat(i - 1, j)] = -t

Observe how the above code corresponds directly to the terms of the discretized
Hamiltonian:
"On-site Hamiltonian" implements

.. math::
    \sum_{i,j} \left(V(ai, aj) + 4t\right)\ket{i,j}\bra{i,j}

(with zero potential).  "Hopping in x-direction" implements

.. math::
    \sum_{i,j} -t \big( \ket{i+1,j}\bra{i,j} + \ket{i,j}\bra{i+1,j} \big),

and "Hopping in y-direction" implements

.. math::
    \sum_{i,j} -t \big( \ket{i,j+1}\bra{i,j} + \ket{i,j}\bra{i,j+1} \big).

The hard-wall confinement is realized by not having hoppings (and sites) beyond
a certain region of space.


Next, we define the leads. Leads are also constructed using
`~kwant.builder.Builder`, but in this case, the
system must have a translational symmetry:

.. jupyter-execute::

     # Then, define and attach the leads:

     # First the lead to the left
     # (Note: TranslationalSymmetry takes a real-space vector)
     sym_left_lead = kwant.TranslationalSymmetry((-a, 0))
     left_lead = kwant.Builder(sym_left_lead)

Here, the `~kwant.builder.Builder` takes a
`~kwant.lattice.TranslationalSymmetry` as the optional parameter. Note that the
(real-space) vector ``(-a, 0)`` defining the translational symmetry must point
in a direction *away* from the scattering region, *into* the lead -- hence, lead
0 [#]_ will be the left lead, extending to infinity to the left.

For the lead itself it is enough to add the points of one unit cell as well
as the hoppings inside one unit cell and to the next unit cell of the lead.
For a square lattice, and a lead in y-direction the unit cell is
simply a vertical line of points:

.. jupyter-execute::

    for j in range(W):
        left_lead[lat(0, j)] = 4 * t
        if j > 0:
            left_lead[lat(0, j), lat(0, j - 1)] = -t
        left_lead[lat(1, j), lat(0, j)] = -t

Note that here it doesn't matter if you add the hoppings to the next or the
previous unit cell -- the translational symmetry takes care of that.  The
isolated, infinite is attached at the correct position using

.. jupyter-execute::
    :hide-output:

     syst.attach_lead(left_lead)

This call returns the lead number which will be used to refer to the lead when
computing transmissions (further down in this tutorial). More details about
attaching leads can be found in the tutorial :ref:`tutorial-abring`.

We also want to add a lead on the right side. The only difference to
the left lead is that the vector of the translational
symmetry must point to the right, the remaining code is the same:

.. jupyter-execute::
    :hide-output:

    # Then the lead to the right

    sym_right_lead = kwant.TranslationalSymmetry((a, 0))
    right_lead = kwant.Builder(sym_right_lead)

    for j in range(W):
        right_lead[lat(0, j)] = 4 * t
        if j > 0:
            right_lead[lat(0, j), lat(0, j - 1)] = -t
        right_lead[lat(1, j), lat(0, j)] = -t

    syst.attach_lead(right_lead)

Note that here we added points with x-coordinate 0, just as for the left lead.
You might object that the right lead should be placed `L`
(or `L+1`?) points to the right with respect to the left lead. In fact,
you do not need to worry about that.

Now we have finished building our system! We plot it, to make sure we didn't
make any mistakes:

.. jupyter-execute::

    kwant.plot(syst);

The system is represented in the usual way for tight-binding systems:
dots represent the lattice points `(i, j)`, and for every
nonzero hopping element between points there is a line connecting these
points. From the leads, only a few (default 2) unit cells are shown, with
fading color.

In order to use our system for a transport calculation, we need to finalize it

.. jupyter-execute::

     # Finalize the system
     syst = syst.finalized()

Having successfully created a system, we now can immediately start to compute
its conductance as a function of energy:

.. jupyter-execute::

    # Now that we have the system, we can compute conductance
    energies = []
    data = []
    for ie in range(100):
        energy = ie * 0.01

        # compute the scattering matrix at a given energy
        smatrix = kwant.smatrix(syst, energy)

        # compute the transmission probability from lead 0 to
        # lead 1
        energies.append(energy)
        data.append(smatrix.transmission(1, 0))

We use ``kwant.smatrix`` which is a short name for
`kwant.solvers.default.smatrix` of the default solver module
`kwant.solvers.default`.  ``kwant.smatrix`` computes the scattering matrix
``smatrix`` solving a sparse linear system.  ``smatrix`` itself allows to
directly compute the total transmission probability from lead 0 to lead 1 as
``smatrix.transmission(1, 0)``. The numbering used to refer to the leads here
is the same as the numbering assigned by the call to
`~kwant.builder.Builder.attach_lead` earlier in the tutorial.

Finally we can use ``matplotlib`` to make a plot of the computed data
(although writing to file and using an external viewer such as
gnuplot or xmgrace is just as viable)

.. jupyter-execute::

    # Use matplotlib to write output
    # We should see conductance steps
    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [e^2/h]")
    pyplot.show()

We see a conductance quantized in units of :math:`e^2/h`,
increasing in steps as the energy is increased. The
value of the conductance is determined by the number of occupied
subbands that increases with energy.


.. admonition:: Technical details
    :class: dropdown note

    - In the example above, when building the system, only one direction
      of hopping is given, i.e. ``syst[lat(i, j), lat(i, j-1)] = ...`` and
      not also ``syst[lat(i, j-1), lat(i, j)] = ...``. The reason is that
      `~kwant.builder.Builder` automatically adds the other
      direction of the hopping such that the resulting system is Hermitian.

      However, it does not hurt to define the opposite direction of hopping as
      well::

        syst[lat(1, 0), lat(0, 0)] = -t
        syst[lat(0, 0), lat(1, 0)] = -t.conj()

      (assuming that `t` is complex) is perfectly fine. However,
      be aware that also

      ::

          syst[lat(1, 0), lat(0, 0)] = -1
          syst[lat(0, 0), lat(1, 0)] = -2

      is valid code. In the latter case, the hopping ``syst[lat(1, 0),
      lat(0, 0)]`` is overwritten by the last line and also equals to -2.

    - Some more details the relation between `~kwant.builder.Builder`
      and the square lattice `lat` in the example:

      Technically, `~kwant.builder.Builder` expects
      **sites** as indices. Sites themselves have a certain type, and
      belong to a **site family**. A site family is also used to convert
      something that represents a site (like a tuple) into a
      proper `~kwant.builder.Site` object that can be used with
      `~kwant.builder.Builder`.

      In the above example, `lat` is the site family. ``lat(i, j)``
      then translates the description of a lattice site in terms of two
      integer indices (which is the natural way to do here) into
      a proper `~kwant.builder.Site` object.

      The concept of site families and sites allows `~kwant.builder.Builder`
      to mix arbitrary lattices and site families

    - In the example, we wrote

      ::

        syst = syst.finalized()

      In doing so, we transform the `~kwant.builder.Builder` object (with which
      we built up the system step by step) into a `~kwant.system.System`
      that has a fixed structure (which we cannot change any more).

      Note that this means that we cannot access the `~kwant.builder.Builder`
      object any more. This is not necesarry any more, as the computational
      routines all expect finalized systems. It even has the advantage
      that python is now free to release the memory occupied by the
      `~kwant.builder.Builder` which, for large systems, can be considerable.
      Roughly speaking, the above code corresponds to

      ::

          fsyst = syst.finalized()
          del syst
          syst = fsyst

    - Even though the vector passed to the
      `~kwant.lattice.TranslationalSymmetry` is specified in real space, it must
      be compatible with the lattice symmetries.  A single lead can consists of
      sites belonging to more than one lattice, but of course the translational
      symmetry of the lead has to be shared by all of them.

    - Instead of plotting to the screen (which is standard)
      `~kwant.plotter.plot` can also write to a file specified by the argument
      ``file``.


.. rubric:: Footnotes

.. [#] https://xkcd.com/353/
.. [#] Leads are numbered in the python convention, starting from 0.

Building the same system with less code
.......................................

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download-script:`quantum_wire_revisited`


.. jupyter-kernel::
    :id: quantum_wire_revisited


Kwant allows for more than one way to build a system. The reason is that
`~kwant.builder.Builder` is essentially just a container that can be filled in
different ways. Here we present a more compact rewrite of the previous example
(still with the same results).

Also, the previous example was written in the form of a Python script with
little structure, and with everything governed by global variables.  This is OK
for such a simple example, but for larger projects it makes sense to partition
the code into separate entities. In this example we therefore also aim at more
structure.

We begin the program collecting all imports in the beginning of the
file and defining the a square lattice and empty scattering region.

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.2.3. Building the same system with less code
    # =======================================================
    #
    # Physics background
    # ------------------
    #  Conductance of a quantum wire; subbands
    #
    # Kwant features highlighted
    # --------------------------
    #  - Using iterables and builder.HoppingKind for making systems
    #  - introducing `reversed()` for the leads
    #
    # Note: Does the same as tutorial1a.py, but using other features of Kwant.

.. jupyter-execute::

    import kwant

    # For plotting
    from matplotlib import pyplot

.. jupyter-execute:: boilerplate.py
    :hide-code:

.. jupyter-execute::

    a = 1
    t = 1.0
    W, L = 10, 30

    # Start with an empty tight-binding system and a single square lattice.
    # `a` is the lattice constant (by default set to 1 for simplicity).
    # Each lattice site has 1 degree of freedom, hence norbs=1.
    lat = kwant.lattice.square(a, norbs=1)

    syst = kwant.Builder()


Previously, the scattering region was build using two ``for``-loops.
Instead, we now write:


.. jupyter-execute::

    syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * t


Here, all lattice points are added at once in the first line.  The
construct ``((i, j) for i in range(L) for j in range(W))`` is a
generator that iterates over all points in the rectangle as did the
two ``for``-loops in the previous example. In fact, a
`~kwant.builder.Builder` can not only be indexed by a single
lattice point -- it also allows for lists of points, or, as in this
example, a generator (as is also used in list comprehensions in
python).

Having added all lattice points in one line, we now turn to the
hoppings. In this case, an iterable like for the lattice
points becomes a bit cumbersome, and we use instead another
feature of Kwant:

.. jupyter-execute::

    syst[lat.neighbors()] = -t

In regular lattices, hoppings form large groups such that hoppings within a
group can be transformed into one another by lattice translations. In order to
allow to easily manipulate such hoppings, an object
`~kwant.builder.HoppingKind` is provided. When given a `~kwant.builder.Builder`
as an argument, `~kwant.builder.HoppingKind` yields all the hoppings of a
certain kind that can be added to this builder without adding new sites. When
`~kwant.builder.HoppingKind` is given to `~kwant.builder.Builder` as a key, it
means that something is done to all the possible hoppings of this kind. A list
of `~kwant.builder.HoppingKind` objects corresponding to nearest neighbors in
lattices in Kwant is obtained using ``lat.neighbors()``. ``syst[lat.neighbors()]
= -t`` then sets all of those hopping matrix elements at once. In order to set
values for all the nth-nearest neighbors at once, one can similarly use
``syst[lat.neighbors(n)] = -t``. More detailed example of using
`~kwant.builder.HoppingKind` directly will be provided in
:ref:`tutorial_spinorbit`.

The left lead is constructed in an analogous way:

.. jupyter-execute::

    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0, j) for j in range(W))] = 4 * t
    lead[lat.neighbors()] = -t

The previous example duplicated almost identical code for the left and
the right lead.  The only difference was the direction of the translational
symmetry vector.  Here, we only construct the left lead, and use the method
`~kwant.builder.Builder.reversed` of `~kwant.builder.Builder` to obtain a copy
of a lead pointing in the opposite direction.  Both leads are attached as
before:

.. jupyter-execute::
    :hide-output:

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

The remainder of the script proceeds identically. We first finalize the system:

.. jupyter-execute::

    syst = syst.finalized()

and then calculate the transmission and plot:

.. jupyter-execute::

    energies = []
    data = []
    for ie in range(100):
        energy = ie * 0.01
        smatrix = kwant.smatrix(syst, energy)
        energies.append(energy)
        data.append(smatrix.transmission(1, 0))

    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [e^2/h]")
    pyplot.show()

.. admonition:: Technical details
    :class: dropdown note

    - We have seen different ways to add lattice points to a
      `~kwant.builder.Builder`. It allows to

      * add single points, specified as sites
      * add several points at once using a generator (as in this example)
      * add several points at once using a list (typically less
        effective compared to a generator)

      For technical reasons it is not possible to add several points
      using a tuple of sites. Hence it is worth noting
      a subtle detail in:

      >>> syst[(lat(x, y) for x in range(L) for y in range(W))] = 4 * t

      Note that ``(lat(x, y) for x in range(L) for y in range(W))`` is not
      a tuple, but a generator.

      Let us elaborate a bit more on this using a simpler example:

      >>> a = (0, 1, 2, 3)
      >>> b = (i for i in range(4))

      Here, `a` is a tuple, whereas `b` is a generator. One difference
      is that one can subscript tuples, but not generators:

      >>> a[0]
      0
      >>> b[0]
      Traceback (most recent call last):
        File "<stdin>", line 1, in <module>
      TypeError: 'generator' object is unsubscriptable

      However, both can be used in ``for``-loops, for example.

    - In the example, we have added all the hoppings using
      `~kwant.builder.HoppingKind`. In fact,
      hoppings can be added in the same fashion as sites, namely specifying

      * a single hopping
      * several hoppings via a generator
      * several hoppings via a list

      A hopping is defined using two sites. If several hoppings are
      added at once, these two sites should be encapsulated in a tuple.
      In particular, one must write::

          syst[((lat(0,j+1), lat(0, j)) for j in range(W-1)] = ...

      or::

          syst[[(site1, site2), (site3, site4), ...]] = ...

      You might wonder, why it is then possible to write for a single hopping::

         syst[site1, site2] = ...

      instead of ::

         syst[(site1, site2)] = ...

      In fact, due to the way python handles subscripting, ``syst[site1, site2]``
      is the same as ``syst[(site1, site2)]``.

      (This is the deeper reason why several sites cannot be added as a tuple --
      it would be impossible to distinguish whether one would like to add two
      separate sites, or one hopping.


Tips for organizing your simulation scripts
...........................................

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download-script:`quantum_wire_organized`


.. jupyter-kernel::
    :id: quantum_wire_organized

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.2.4. Organizing a simulation script
    # ==============================================
    #
    # Physics background
    # ------------------
    #  Conductance of a quantum wire; subbands
    #
    # Note: Does the same as quantum_write_revisited.py, but features
    #       better code organization


The above two examples illustrate some of the core features of Kwant, however
the code was presented in a style which is good for exposition, but which is
bad for making your code understandable and reusable. In this example we will
lay out some best practices for writing your own simulation scripts.

In the above examples we constructed a single Kwant system, using global variables
for parameters such as the lattice constant and the length and width of the system.
Instead, it is preferable to create a *function* that you can call, and which will
return a Kwant ``Builder``:

.. jupyter-execute::

    from matplotlib import pyplot
    import kwant

.. jupyter-execute:: boilerplate.py
    :hide-code:

.. jupyter-execute::

    def make_system(L, W, a=1, t=1.0):
        lat = kwant.lattice.square(a, norbs=1)

        syst = kwant.Builder()
        syst[(lat(i, j) for i in range(L) for j in range(W))] = 4 * t
        syst[lat.neighbors()] = -t

        lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
        lead[(lat(0, j) for j in range(W))] = 4 * t
        lead[lat.neighbors()] = -t

        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

        return syst


By encapsulating system creation within ``make_system`` we *document* our code
by telling readers that *this* is how we create a system, and that creating a system
depends on *these* parameters (the length and width of the system, in this case, as well
as the lattice constant and the value for the hopping parameter). By defining a function
we also ensure that we can consistently create different systems (e.g. of different sizes)
of the same type (rectangular slab).

We similarly encapsulate the part of the script that does computation and plotting into
a function ``plot_conductance``:

.. jupyter-execute::

    def plot_conductance(syst, energies):
        # Compute conductance
        data = []
        for energy in energies:
            smatrix = kwant.smatrix(syst, energy)
            data.append(smatrix.transmission(1, 0))

        pyplot.figure()
        pyplot.plot(energies, data)
        pyplot.xlabel("energy [t]")
        pyplot.ylabel("conductance [e^2/h]")
        pyplot.show()

And the ``main`` function that glues together the components that we previously defined:

.. jupyter-execute::

    def main():
        syst = make_system(W=10, L=30)

        # Check that the system looks as intended.
        kwant.plot(syst)

        # Finalize the system.
        fsyst = syst.finalized()

        # We should see conductance steps.
        plot_conductance(fsyst, energies=[0.01 * i for i in range(100)])


Finally, we use the following standard Python construct [#]_ to execute
``main`` if the program is used as a script (i.e. executed as
``python quantum_wire_organized.py``):


.. jupyter-execute::

    # Call the main function if the script gets executed (as opposed to imported).
    # See <https://docs.python.org/library/__main__.html>.
    if __name__ == '__main__':
        main()

If the example, however, is imported inside Python using ``import
quantum_wire_organized as qw``, ``main`` is not executed automatically.
Instead, you can execute it manually using ``qw.main()``.  On the other
hand, you also have access to the other functions, ``make_system`` and
``plot_conductance``, and can thus play with the parameters.

The result of this example should be identical to the previous one.


.. rubric:: Footnotes

.. [#] https://docs.python.org/3/library/__main__.html
