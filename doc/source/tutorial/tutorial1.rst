First steps in kwant: Setting up a simple system and computing transport
------------------------------------------------------------------------

Transport through a quantum wire
................................

As first example, we compute the transmission probability
through a two-dimensional quantum wire. For this we use a tight-binding
model representing the two-dimensional Schroedinger equation

.. math::

    H = \frac{\hbar^2}{2 m} (\partial_x^2+\partial_y^2) + V(y)

with a hard wall confinement :math:`V(y)` in y-direction.

In order to use kwant, we need to import it:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_dwhx
    :end-before: #HIDDEN_END_dwhx

Enabling kwant is as easy as this [#]_ !

The first step is now the definition of the system with scattering region
and leads. For this we make use of the `~kwant.builder.Builder` class
that allows for a convenient way to define the system. For this we need to
create an instance of the `~kwant.builder.Builder` class:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_goiq
    :end-before: #HIDDEN_END_goiq

Apart from `~kwant.builder.Builder` we also need to specify
what kind of sites we want to add to the system. Here we work with
a square lattice. For simplicity, we set the lattice constant to unity:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_suwo
    :end-before: #HIDDEN_END_suwo

Since we work with a square lattice, we label the points with two
integer coordinates `(i, j)`. `~kwant.builder.Builder` then
allows us to add matrix elements corresponding to lattice points:
``sys[lat(i, j)] = ...`` sets the on-site energy for the point `(i, j)`,
and ``sys[lat(i1, j1), lat(i2, j2)] = ...`` the hopping matrix element
**from** point `(i2, j2)` **to** point `(i1, j1)`.

Note that we need to specify sites for `~kwant.builder.Builder`
in the form ``lat(i, j)``. The lattice object `lat` does the
translation from integer coordinates to proper site format
needed in Builder (more about that in the technical details below).

We now build a rectangular scattering region that is `W`
lattice points wide and `L` lattice points long:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_zfvr
    :end-before: #HIDDEN_END_zfvr

Next, we define the leads. Leads are also constructed using
`~kwant.builder.Builder`, but in this case, the
system must have a translational symmetry:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_xcmc
    :end-before: #HIDDEN_END_xcmc

Here, the `~kwant.builder.Builder` takes the translational symmetry
as an optional parameter. Note that the (real space)
vector ``lat.vec((-1, 0))`` defining the translational symmetry
must point in a direction *away* from the scattering region, *into*
the lead -- hence, lead 0 [#]_ will be the left lead, extending to
infinity to the left.

For the lead itself it is enough to add the points of one unit cell as well
as the hoppings inside one unit cell and to the next unit cell of the lead.
For a square lattice, and a lead in y-direction the unit cell is
simply a vertical line of points:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_ndez
    :end-before: #HIDDEN_END_ndez

Note that here it doesn't matter if you add the hoppings to the next or the
previous unit cell -- the translational symmetry takes care of that.

We also want to add a lead on the right side. The only difference to
the left lead is that the vector of the translational
symmetry must point to the right, the remaining code is the same:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_xhqc
    :end-before: #HIDDEN_END_xhqc

Note that here we added points with x-coordinate 0, just as for the left lead.
You might object that the right lead should be placed `L`
(or `L+1`?) points to the right with respect to the left lead. In fact,
you do not need to worry about that. The `~kwant.builder.Builder` with
`~kwant.lattice.TranslationalSymmetry` represents a lead which is
infinitely extended. These isolated, infinite leads can then be simply
attached at the right position using:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_fskr
    :end-before: #HIDDEN_END_fskr

More details about attaching leads can be found in the tutorial
:ref:`tutorial-abring`.

Now we have finished building our system! We plot it, to make sure we didn't
make any mistakes:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_wsgh
    :end-before: #HIDDEN_END_wsgh

This should bring up this picture:

.. image:: /images/1-quantum_wire_sys.*

The system is represented in the usual way for tight-binding systems:
dots represent the lattice points `(i, j)`, and for every
nonzero hopping element between points there is a line connecting these
points. From the leads, only a few (default 2) unit cells are shown, with
fading color.

In order to use our system for a transport calculation, we need to finalize it

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_dngj
    :end-before: #HIDDEN_END_dngj

Having successfully created a system, we now can immediately start to compute
its conductance as a function of energy:

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_buzn
    :end-before: #HIDDEN_END_buzn

Currently **CHANGE**, there is only one algorithm implemented to compute the
conductance: :func:`kwant.solve <kwant.solvers.common.SparseSolver.solve>`
which computes the scattering matrix `smatrix` solving a sparse linear system.
`smatrix` itself allows you to directly compute the total
transmission probability from lead 0 to lead 1 as
``smatrix.transmission(1, 0)``.

Finally we can use `matplotlib` to make a plot of the computed data
(although writing to file and using an external viewer such as
gnuplot or xmgrace is just as viable)

.. literalinclude:: 1-quantum_wire.py
    :start-after: #HIDDEN_BEGIN_lliv
    :end-before: #HIDDEN_END_lliv

This should yield the result

.. image:: /images/1-quantum_wire_result.*

We see a conductance quantized in units of :math:`e^2/h`,
increasing in steps as the energy is increased. The
value of the conductance is determined by the number of occupied
subbands that increases with energy.


.. seealso::
     The full source code can be found in
     :download:`tutorial/1-quantum_wire.py <../../../tutorial/1-quantum_wire.py>`

.. specialnote:: Technical details

   - In the example above, when building the system, only one direction
     of hopping is given, i.e. ``sys[lat(i, j), lat(i, j-1)] = ...`` and
     not also ``sys[lat(i, j-1), lat(i, j)] = ...``. The reason is that
     `~kwant.builder.Builder` automatically adds the other
     direction of the hopping such that the resulting system is Hermitian.

     However, it does not hurt to define the opposite direction of hopping as
     well::

         sys[lat(1, 0), lat(0, 0)] = - t
         sys[lat(0, 0), lat(1, 0)] = - t.conj()

     (assuming that `t` is complex) is perfectly fine. However,
     be aware that also

     ::

         sys[lat(1, 0), lat(0, 0)] = - 1
         sys[lat(0, 0), lat(1, 0)] = - 2

     is valid code. In the latter case, the hopping ``sys[lat(1, 0),
     lat(0, 0)]`` is overwritten by the last line and also equals to -2.

   - Some more details the relation between `~kwant.builder.Builder`
     and the square lattice `lat` in the example:

     Technically, `~kwant.builder.Builder` expects
     **sites** as indices. Sites themselves have a certain type, and
     belong to a **site group**. A site group is also used to convert
     something that represents a site (like a tuple) into a
     proper `~kwant.builder.Site` object that can be used with
     `~kwant.builder.Builder`.

     In the above example, `lat` is the site group. ``lat(i, j)``
     then translates the description of a lattice site in terms of two
     integer indices (which is the natural way to do here) into
     a proper `~kwant.builder.Site` object.

     The concept of site groups and sites allows `~kwant.builder.Builder`
     to mix arbitrary lattices and site groups

   - In the example, we wrote

     ::

         sys = sys.finalized()

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

         fsys = sys.finalized()
         del sys
         sys = fsys

   - Note that the vector passed to the `~kwant.lattice.TranslationalSymmetry`
     (in fact, what is passed is a list of vectors -- there could be more than
     on periodic direction. However, for a lead there is only one.) is
     a realspace vector: ``lat.vec((1,0))``. Here, ``lat.vec``
     converts the integer indices `(1,0)` into a realspace vector. In
     this particular example, this is trivial (even more as ``a=1``),
     but it is not so any more for more complicated lattices.

     Even though the translational symmetry vector is specified in
     realspace, it must be compatible with the lattice symmetries
     (in principle, there could be more than one lattice). Hence,
     it will be typically specified using ``lat.vec``, as this
     is guaranteed to be a proper lattice vector, compatible
     with the lattice symmetry.

   - Instead of plotting to the screen (which is standard, if the
     Python Image Library PIL **CHANGE [if plotter is changed]** is installed),
     :func:`plot <kwant.plotter.plot>`
     can also write to the file specified by the argument `filename`.
     (for details, see the documentation of :func:`plot <kwant.plotter.plot>`.)


.. rubric:: Footnotes

.. [#] http://xkcd.com/353/
.. [#] Leads are numbered in the python convention, starting from 0.

The same but different: Alternative system building
...................................................

kwant is very flexible, and often allows you more than one way to
build up your system. The reason is that `~kwant.builder.Builder`
is essentially just a container, and allows for different
ways to be filled. Here we present a more compact rewrite of
the previous example (still with the same results).

Also, the previous example was written in the form of a pythons script
with little structure, and everything governed by global variables.
This is OK for such a simple example, but for larger projects it makes
sense to structure different functionality into different functional
entities. In this example we therefore also aim at more structure.

We begin the program collecting all imports in the beginning of the
file and put the build-up of the system into a separate function
`make_system`:

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_xkzy
    :end-before: #HIDDEN_END_xkzy

Previously, the scattering region was build using two ``for``-loops.
Instead, we now write:

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_vvjt
    :end-before: #HIDDEN_END_vvjt

Here, all lattice points are added at once in the first line.  The
construct ``((i, j) for i in xrange(L) for j in xrange(W))`` is a
generator that iterates over all points in the rectangle as did the
two ``for``-loops in the previous example. In fact, a
`~kwant.builder.Builder` can not only be indexed by a single
lattice point -- it also allows for lists of points, or, as in this
example, a generator (as is also used in list comprehensions in
python).

Having added all lattice points in one line, we now turn to the
hoppings. In this case, an iterable like for the lattice
points becomes a bit cumbersome, and we use instead another
feature of kwant:

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_nooi
    :end-before: #HIDDEN_END_nooi

In regular lattices, one has only very few types of different hoppings
(by one lattice point in x or y-direction in the case of a square
lattice considered here). For the square lattice, these types of
hoppings are stored as a list in ``lat.nearest``, and the ``for``-loop
runs over all of them.
`~kwant.builder.Builder.possible_hoppings` takes as an argument
one type of hopping (more about that in the notes below;
details on the hopping definition will be discussed in
:ref:`tutorial_spinorbit`), and generates all
hoppings of this type that are possible with all the lattice points
that were added before.  ``sys[sys.possible_hoppings(*hopping)] = -t``
then sets all of those hopping matrix elements at once.

The leads can be constructed in an analogous way:

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_iepx
    :end-before: #HIDDEN_END_iepx

Note that in the previous example, we essentially used the same code
for the right and the left lead, the only difference was the direction
of the translational symmetry vector. The
`~kwant.builder.Builder` used for the lead provides a method
`~kwant.builder.Builder.reversed` that returns a copy of the
lead, but with it's translational vector reversed.  This can thus be
used to obtain a lead pointing in the opposite direction, i.e. makes a
right lead from a left lead:

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_xkdo
    :end-before: #HIDDEN_END_xkdo

The remainder of the code is identical to the previous example
(except for a bit of reorganization into functions):

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_yxot
    :end-before: #HIDDEN_END_yxot

and

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_ayuk
    :end-before: #HIDDEN_END_ayuk

Finally, we use a python trick to make our example usable both
as a script, as well as allowing it to be imported as a module.
We collect all statements that should be executed in the script
in a ``main()``-function:

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_cjel
    :end-before: #HIDDEN_END_cjel

Finally, we use the following python construct [#]_ that executes
``main()`` if the program is used as a script (i.e. executed as
``python tutorial1b.py``):

.. literalinclude:: 1-quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_ypbj
    :end-before: #HIDDEN_END_ypbj

If the example however is imported using ``import tutorial1b``,
``main()`` is not executed automatically. Instead, you can execute it
manually using ``tutorial1b.main()``.  On the other hand, you also
have access to the other functions, ``make_system()`` and
``plot_conductance()``, and can thus play with the parameters.

The result of the example should be identical to the previous one.

.. seealso::
    The full source code can be found in
    :download:`tutorial/1-quantum_wire_revisited.py <../../../tutorial/1-quantum_wire_revisited.py>`

.. specialnote:: Technical details

   - In

     .. literalinclude:: 1-quantum_wire_revisited.py
       :start-after: #HIDDEN_BEGIN_nooi
       :end-before: #HIDDEN_END_nooi

     we write ``*hopping`` instead of ``hopping``. The reason is as follows:
     `~kwant.builder.Builder.possible_hoppings` expects the hopping to
     be defined using three parameters (in particular, a tuple
     containing a relative lattice vector, and two (sub)lattice objects that
     indicate the start and end lattice, more about that in
     a :ref:`later tutorial <tutorial_spinorbit>`). ``lat.nearest``
     is a list of tuples, with every tuple containing the three
     parameters expected by `~kwant.builder.Builder.possible_hoppings`.

     Hence, ``hopping`` is a tuple. But passing it to
     `~kwant.builder.Builder.possible_hoppings` would fail,
     as three parameters are expected (not a single tuple). ``*hopping``
     unpacks the tuple into these three separate parameters (see
     <http://docs.python.org/tutorial/controlflow.html#unpacking-argument-lists>)

   - We have seen different ways to add lattice points to a
     `~kwant.builder.Builder`. It allows to

     * add single points, specified as sites
     * add several points at once using a generator (as in this example)
     * add several points at once using a list (typically less
       effective compared to a generator)

     For technical reasons it is not possible to add several points
     using a tuple of sites. Hence it is worth noting
     a subtle detail in

     .. literalinclude:: 1-quantum_wire_revisited.py
         :start-after: #HIDDEN_BEGIN_vvjt
         :end-before: #HIDDEN_END_vvjt

     Note that ``(lat(x, y) for x in range(L) for y in range(W))`` is not
     a tuple, but a generator.

     Let us elaborate a bit more on this using a simpler example:

     >>> a = (0, 1, 2, 3)
     >>> b = (i for i in xrange(4))

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
     `~kwant.builder.Builder.possible_hoppings`. In fact,
     hoppings can be added in the same fashion as sites, namely specifying

     * a single hopping
     * several hoppings via a generator
     * several hoppings via a list

     A hopping is defined using two sites. If several hoppings are
     added at once, these two sites should be encapsulated in a tuple.
     In particular, one must write::

         sys[((lat(0,j+1), lat(0, j)) for j in xrange(W-1)] = ...

     or::

         sys[[(site1, site2), (site3, site4), ...]] = ...

     You might wonder, why it is then possible to write for a single hopping::

        sys[site1, site2] = ...

     instead of ::

        sys[(site1, site2)] = ...

     In fact, due to the way python handles subscripting, ``sys[site1, site2]``
     is the same as ``sys[(site1, site2)]``.

     (This is the deeper reason why several sites cannot be added as a tuple --
     it would be impossible to distinguish whether one would like to add two
     separate sites, or one hopping.

.. rubric:: Footnotes

.. [#] http://docs.python.org/library/__main__.html
