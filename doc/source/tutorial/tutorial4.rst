.. _tutorial-graphene:

Using a more complicated lattice (graphene)
-------------------------------------------

In the following example, we are going to calculate the
conductance through a graphene quantum dot with a p-n junction
and two non-collinear leads. In the process, we will touch
all of the topics that we have seen in the previous tutorials,
but now for the honeycomb lattice. As you will see, everything
carries over nicely.
spectrum of a quantum dot.

We begin from defining the honeycomb lattice of graphene. This is
in principle already done in `kwant.lattice.Honeycomb`, but we do it
explicitly here to show how to define a new lattice:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 24-27

The first argument to the `make_lattice` function is the list of primitive
vectors of the lattice; the second one is the coordinates of basis atoms.
The honeycomb lattice has two basis atoms. Each type of basis atom by itself
forms a regular lattice of the same type as well, and those *sublattices*
are referenced as `a` and `b` above.

In the next step we define the shape of the scattering region (circle again)
and add all lattice points using the ``shape()``-functionality:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 30-31, 34-39, 41-46

As you can see, this works exactly the same for any kind of lattice.
We add the onsite energies using a function describing the p-n junction;
in contrast to the previous examples, the potential value is this time taken
from the scope of `make_system()`, since we keep the potential fixed
in this example.

As a next step we add the hoppings, making use of
`~kwant.builder.Builder.possible_hoppings`. Since we use our home-made
lattice (instead of `kwant.lattice.Honeycomb`), we have to define
the hoppings ourselves:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 50

The nearest-neighbor model for graphene contains only
hoppings between different basis atoms. For these type of
hoppings, it is not enough to specify relative lattice indices,
but we also need to specify the proper target and source
sublattices. Remember that the format of the hopping specification
is ``(i,j), target, source``. In the previous examples (i.e.
:ref:`tutorial_spinorbit`) ``target=source=lat``, whereas here
we have to specify different sublattices. Furthermore,
note that the directions given by the lattice indices
`(1, 0)` and `(0, 1)` are not orthogonal any more, as they are given with
respect to the two primitive vectors ``[(1, 0), (sin_30, cos_30)]``.

Adding the hoppings however still works the same way:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 51-52

Modifying the scattering region is also possible as before. Let's
do something crazy, and remove an atom in sublattice A
(which removes also the hoppings from/to this site) as well
as add an additional link:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 55-56

Note that the conversion from a tuple `(i,j)` to site
is done be the sublattices `a` and `b`.

Later, we will compute some eigenvalues of the closed
scattering region without leads. For that, obtain a finalized
snapshot of the system:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 60

Adding leads to the scattering region is done as before:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 64-93

Note here that the translational vectors ``graphene.vec((-1, 0))`` and
``graphene.vec((0, 1))`` are *not* orthogonal any more as they would
have been in a square lattice-- they follow the non-orthogonal
primitive vectors defined in the beginning.

In the end, we return not only the finalized system with leads, but
also a finalized copy of the closed system (for eigenvalues)
as well as a finalized lead (for band structure calculations).

The computation of some eigenvalues of the closed system is done
in the following piece of code:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 96-101, 104-107

Here we use in contrast to the previous example a sparse matrix and
the sparse linear algebra functionality of scipy (this requires
scipy version >= 0.9.0; since the remaining part of the example does not
depend on this eigenenergy calculation, a ``try``-block simply skips this
calculation if a lower scipy version is installed.)

The code for computing the band structure and the conductance is identical
to the previous examples, and needs not be further explained here.

Finally, in the `main()` function we make and
plot the system:

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 135-137, 142-147

Here we customize the plotting: `plotter_symbols` is a dictionary
which the sublattice objects `a` and `b` as keys, and the
`~kwant.plotter.Circle` objects specify that the sublattice `a` should
be drawn using a filled black circle, and `b` using a white circle
with a black outline. The radius of the circle is given in relative
units: :func:`plot <kwant.plotter.plot>` uses a typical length
scale as a reference length. By default, the typical length scale is
the smallest distance between lattice points.  :func:`plot
<kwant.plotter.plot>` can find this length by itself, but must then go
through all hoppings. Alternatively, one can specify the typical
length scale using the argument `a` as in the example (not to be
confused with the sublattice `a`) which is here set to the distance
between carbon atoms in the graphene lattice. Specifying ``r=0.3`` in
`~kwant.plotter.Circle` hence means that the radius of the circle is
30% of the carbon-carbon distance. Using this relative units it is
easy to make good-looking plots where the symbols cover a well-defined
part of the plot.

Plotting the closed system gives this result:

.. image:: ../images/tutorial4_sys1.*

and computing the eigenvalues of largest magnitude,

.. literalinclude:: ../../../examples/tutorial4.py
    :lines: 148

should yield two eigenvalues similar to `[ 3.07869311
+1.02714523e-17j, -3.06233144 -6.68085759e-18j]` (round-off might
change the imaginary part which should in exact arithmetics be equal
to zero).

The remaining code of `main()` plots the system with leads:

.. image:: ../images/tutorial4_sys2.*

It computes the band structure of one of the leads:

.. image:: ../images/tutorial4_bs.*

showing all the features of a zigzag lead, including the flat
edge state bands (note that the band structure is not symmetric around
zero energy, as we have a potential in the leads).

Finally the transmission through the system is computed,

.. image:: ../images/tutorial4_result.*

showing the typical resonance-like transmission probability through
an open quantum dot

.. seealso::
    The full source code can be found in
    :download:`examples/tutorial4.py <../../../examples/tutorial4.py>`

.. specialnote:: Technical details

 - Apart from circles, the `kwant.plotter` module also has regular
   `~kwant.plotter.Polygon`'s as predefined symbols. It is also
   easy to define any custom symbol. Furthermore, plotting offers
   many more options to customize plots. See the documentation of
   :func:`plot <kwant.plotter.plot>` for more details.

 - In a lattice with more than one basis atom, you can always act either
   on all sublattice at the same time, or on a single sublattice only.

   For example, you can add lattice points for all sublattices in the
   current example using::

       sys[graphene.shape(...)] = ...

   or just for a single sublattice::

       sys[a.shape(...)] = ...

   and the same of course with `b`. Also, you can selectively remove points::

       del sys[graphene.shape(...)]
       del sys[a.shape(...)]

   where the first line removes points in both sublattices, whereas the
   second line removes only points in one sublattice.
