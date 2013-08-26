First steps in Kwant: Setting up a simple system and computing transport
------------------------------------------------------------------------

Transport through a quantum wire
................................

As first example, we compute the transmission probability
through a two-dimensional quantum wire. For this we use a tight-binding
model representing the two-dimensional Schroedinger equation

.. math::

    H = \frac{-\hbar^2}{2 m} (\partial_x^2+\partial_y^2) + V(y)

with a hard wall confinement :math:`V(y)` in y-direction.

In order to use Kwant, we need to import it:

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_dwhx
    :end-before: #HIDDEN_END_dwhx

Enabling Kwant is as easy as this [#]_ !

The first step is now the definition of the system with scattering region and
leads. For this we make use of the `~kwant.builder.Builder` type that allows to
define a system in a convenient way. We need to create an instance of it:

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_goiq
    :end-before: #HIDDEN_END_goiq

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

.. literalinclude:: quantum_wire.py
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

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_zfvr
    :end-before: #HIDDEN_END_zfvr

Next, we define the leads. Leads are also constructed using
`~kwant.builder.Builder`, but in this case, the
system must have a translational symmetry:

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_xcmc
    :end-before: #HIDDEN_END_xcmc

Here, the `~kwant.builder.Builder` takes a
`~kwant.lattice.TranslationalSymmetry` as the optional parameter. Note that the
(real-space) vector ``(-a, 0)`` defining the translational symmetry must point
in a direction *away* from the scattering region, *into* the lead -- hence, lead
0 [#]_ will be the left lead, extending to infinity to the left.

For the lead itself it is enough to add the points of one unit cell as well
as the hoppings inside one unit cell and to the next unit cell of the lead.
For a square lattice, and a lead in y-direction the unit cell is
simply a vertical line of points:

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_ndez
    :end-before: #HIDDEN_END_ndez

Note that here it doesn't matter if you add the hoppings to the next or the
previous unit cell -- the translational symmetry takes care of that.  The
isolated, infinite is attached at the correct position using

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_fskr
    :end-before: #HIDDEN_END_fskr

More details about attaching leads can be found in the tutorial
:ref:`tutorial-abring`.

We also want to add a lead on the right side. The only difference to
the left lead is that the vector of the translational
symmetry must point to the right, the remaining code is the same:

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_xhqc
    :end-before: #HIDDEN_END_xhqc

Note that here we added points with x-coordinate 0, just as for the left lead.
You might object that the right lead should be placed `L`
(or `L+1`?) points to the right with respect to the left lead. In fact,
you do not need to worry about that.

Now we have finished building our system! We plot it, to make sure we didn't
make any mistakes:

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_wsgh
    :end-before: #HIDDEN_END_wsgh

This should bring up this picture:

.. image:: /images/quantum_wire_sys.*

The system is represented in the usual way for tight-binding systems:
dots represent the lattice points `(i, j)`, and for every
nonzero hopping element between points there is a line connecting these
points. From the leads, only a few (default 2) unit cells are shown, with
fading color.

In order to use our system for a transport calculation, we need to finalize it

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_dngj
    :end-before: #HIDDEN_END_dngj

Having successfully created a system, we now can immediately start to compute
its conductance as a function of energy:

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_buzn
    :end-before: #HIDDEN_END_buzn

We use ``kwant.smatrix`` which is a short name for
`kwant.solvers.default.smatrix` of the default solver module
`kwant.solvers.default`.  ``kwant.smatrix`` computes the scattering matrix
``smatrix`` solving a sparse linear system.  ``smatrix`` itself allows to
directly compute the total transmission probability from lead 0 to lead 1 as
``smatrix.transmission(1, 0)``.

Finally we can use `matplotlib` to make a plot of the computed data
(although writing to file and using an external viewer such as
gnuplot or xmgrace is just as viable)

.. literalinclude:: quantum_wire.py
    :start-after: #HIDDEN_BEGIN_lliv
    :end-before: #HIDDEN_END_lliv

This should yield the result

.. image:: /images/quantum_wire_result.*

We see a conductance quantized in units of :math:`e^2/h`,
increasing in steps as the energy is increased. The
value of the conductance is determined by the number of occupied
subbands that increases with energy.


.. seealso::
     The full source code can be found in
     :download:`tutorial/quantum_wire.py <../../../tutorial/quantum_wire.py>`

.. specialnote:: Technical details

   - In the example above, when building the system, only one direction
     of hopping is given, i.e. ``sys[lat(i, j), lat(i, j-1)] = ...`` and
     not also ``sys[lat(i, j-1), lat(i, j)] = ...``. The reason is that
     `~kwant.builder.Builder` automatically adds the other
     direction of the hopping such that the resulting system is Hermitian.

     However, it does not hurt to define the opposite direction of hopping as
     well::

         sys[lat(1, 0), lat(0, 0)] = -t
         sys[lat(0, 0), lat(1, 0)] = -t.conj()

     (assuming that `t` is complex) is perfectly fine. However,
     be aware that also

     ::

         sys[lat(1, 0), lat(0, 0)] = -1
         sys[lat(0, 0), lat(1, 0)] = -2

     is valid code. In the latter case, the hopping ``sys[lat(1, 0),
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

   - Even though the vector passed to the
     `~kwant.lattice.TranslationalSymmetry` is specified in real space, it must
     be compatible with the lattice symmetries.  A single lead can consists of
     sites belonging to more than one lattice, but of course the translational
     symmetry of the lead has to be shared by all of them.

   - Instead of plotting to the screen (which is standard)
     `~kwant.plotter.plot` can also write to a file specified by the argument
     ``file``.  For the plotting to the screen to work the module
     ``matplotlib.pyplot`` has to be imported.  (An informative error message
     will remind you if you forget.)  The reason for this is pretty technical:
     matplotlib's "backend" can only be chosen before ``matplotlib.pyplot`` has
     been imported.  Would Kwant import that module by itself, it would deprive
     you of the possibility to choose a non-default backend later.


.. rubric:: Footnotes

.. [#] http://xkcd.com/353/
.. [#] Leads are numbered in the python convention, starting from 0.

The same but different: Alternative system building
...................................................

Kwant is very flexible, and often allows you more than one way to
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

.. literalinclude:: quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_xkzy
    :end-before: #HIDDEN_END_xkzy

Previously, the scattering region was build using two ``for``-loops.
Instead, we now write:

.. literalinclude:: quantum_wire_revisited.py
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
feature of Kwant:

.. literalinclude:: quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_nooi
    :end-before: #HIDDEN_END_nooi

In regular lattices, hoppings form large groups such that hoppings within a
group can be transformed into one another by lattice translations. In order to
allow to easily manipulate such hoppings, an object
`~kwant.builder.HoppingKind` is provided. When given a `~kwant.builder.Builder`
as an argument, `~kwant.builder.HoppingKind` yields all the hoppings of a
certain kind that can be added to this builder without adding new sites. When
`~kwant.builder.HoppingKind` is given to `~kwant.builder.Builder` as a key, it
means that something is done to all the possible hoppings of this kind. A list
of `~kwant.builder.HoppingKind` objects corresponding to nearest neighbors in
lattices in Kwant is obtained using ``lat.neighbors()``. ``sys[lat.neighbors()]
= -t`` then sets all of those hopping matrix elements at once. In order to set
values for all the nth-nearest neighbors at once, one can similarly use
``sys[lat.neighbors(n)] = -t``. More detailed example of using
`~kwant.builder.HoppingKind` directly will be provided in
:ref:`tutorial_spinorbit`.

The left lead is constructed in an analogous way:

.. literalinclude:: quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_iepx
    :end-before: #HIDDEN_END_iepx

The previous example duplicated almost identical code for the left and
the right lead.  The only difference was the direction of the translational
symmetry vector.  Here, we only construct the left lead, and use the method
`~kwant.builder.Builder.reversed` of `~kwant.builder.Builder` to obtain a copy
of a lead pointing in the opposite direction.  Both leads are attached as
before and the finished system returned:

.. literalinclude:: quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_yxot
    :end-before: #HIDDEN_END_yxot

The remainder of the script has been organized into two functions.  One for the
plotting of the conductance.

.. literalinclude:: quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_ayuk
    :end-before: #HIDDEN_END_ayuk

And one ``main`` function.

.. literalinclude:: quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_cjel
    :end-before: #HIDDEN_END_cjel

Finally, we use the following standard Python construct [#]_ to execute
``main`` if the program is used as a script (i.e. executed as
``python tutorial1b.py``):

.. literalinclude:: quantum_wire_revisited.py
    :start-after: #HIDDEN_BEGIN_ypbj
    :end-before: #HIDDEN_END_ypbj

If the example, however, is imported inside Python using ``import tutorial1b``,
``main`` is not executed automatically.  Instead, you can execute it
manually using ``tutorial1b.main()``.  On the other hand, you also
have access to the other functions, ``make_system`` and
``plot_conductance``, and can thus play with the parameters.

The result of the example should be identical to the previous one.

.. seealso::
    The full source code can be found in
    :download:`tutorial/quantum_wire_revisited.py <../../../tutorial/quantum_wire_revisited.py>`

.. specialnote:: Technical details

   - We have seen different ways to add lattice points to a
     `~kwant.builder.Builder`. It allows to

     * add single points, specified as sites
     * add several points at once using a generator (as in this example)
     * add several points at once using a list (typically less
       effective compared to a generator)

     For technical reasons it is not possible to add several points
     using a tuple of sites. Hence it is worth noting
     a subtle detail in

     .. literalinclude:: quantum_wire_revisited.py
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
     `~kwant.builder.HoppingKind`. In fact,
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
