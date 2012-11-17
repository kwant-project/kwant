.. _tutorial-graphene:

Using a more complicated lattice (graphene)
-------------------------------------------

In the following example, we are going to calculate the
conductance through a graphene quantum dot with a p-n junction
and two non-collinear leads. In the process, we will touch
all of the topics that we have seen in the previous tutorials,
but now for the honeycomb lattice. As you will see, everything
carries over nicely.

We begin by defining the honeycomb lattice of graphene. This is
in principle already done in `kwant.lattice.Honeycomb`, but we do it
explicitly here to show how to define a new lattice:

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_hnla
    :end-before: #HIDDEN_END_hnla

The first argument to the `~kwant.lattice.make_lattice` function is the list of
primitive vectors of the lattice; the second one is the coordinates of basis
atoms.  The honeycomb lattice has two basis atoms. Each type of basis atom by
itself forms a regular lattice of the same type as well, and those
*sublattices* are referenced as `a` and `b` above.

In the next step we define the shape of the scattering region (circle again)
and add all lattice points using the ``shape()``-functionality:

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_shzy
    :end-before: #HIDDEN_END_shzy

As you can see, this works exactly the same for any kind of lattice.
We add the onsite energies using a function describing the p-n junction;
in contrast to the previous tutorial, the potential value is this time taken
from the scope of `make_system()`, since we keep the potential fixed
in this example.

As a next step we add the hoppings, making use of
`~kwant.builder.Builder.possible_hoppings`. Since we use our home-made
lattice (instead of `kwant.lattice.Honeycomb`), we have to define
the hoppings ourselves:

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_hsmc
    :end-before: #HIDDEN_END_hsmc

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

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_bfwb
    :end-before: #HIDDEN_END_bfwb

Modifying the scattering region is also possible as before. Let's
do something crazy, and remove an atom in sublattice A
(which removes also the hoppings from/to this site) as well
as add an additional link:

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_efut
    :end-before: #HIDDEN_END_efut

Note again that the conversion from a tuple `(i,j)` to site
is done by the sublattices `a` and `b`.

The leads are defined almost as before:

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_aakh
    :end-before: #HIDDEN_END_aakh

Note the method `~kwant.lattice.PolyatomicLattice.vec` used in calculating the
parameter for `~kwant.lattice.TranslationalSymmetry`.  The latter expects a
real space symmetry vector, but for many lattices symmetry vectors are more
easily expressed in the natural coordinate system of the lattice.  The ``vec``
method of lattices maps a lattice vector to a real space vector.

Observe also that the translational vectors ``graphene.vec((-1, 0))`` and
``graphene.vec((0, 1))`` are *not* orthogonal any more as they would have been
in a square lattice -- they follow the non-orthogonal primitive vectors defined
in the beginning.

Later, we will compute some eigenvalues of the closed scattering region without
leads. This is why we postpone attaching the leads to the system. Instead,
we return the scattering region and the leads separately.

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_kmmw
    :end-before: #HIDDEN_END_kmmw

The computation of some eigenvalues of the closed system is done
in the following piece of code:

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_zydk
    :end-before: #HIDDEN_END_zydk

Here we use in contrast to the previous example a sparse matrix and
the sparse linear algebra functionality of scipy (this requires
scipy version >= 0.9.0; since the remaining part of the example does not
depend on this eigenenergy calculation, a ``try``-block simply skips this
calculation if a lower scipy version is installed.)

The code for computing the band structure and the conductance is identical
to the previous examples, and needs not be further explained here.

Finally, in the `main()` function we make and
plot the system:

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_itkk
    :end-before: #HIDDEN_END_itkk

We customize the plotting: we set the `site_colors` argument of
`~kwant.plotter.plot` to a function which returns 0 for
sublattice `a` and 1 for sublattice `b`::

    def group_colors(site):
        return 0 if site.group == a else 1

The function `~kwant.plotter.plot` shows these values using a color scale
(grayscale by default). The symbol `size` is specified in points, and is
independent on the overall figure size.

Plotting the closed system gives this result:

.. image:: ../images/4-graphene_sys1.*

Computing the eigenvalues of largest magnitude,

.. literalinclude:: 4-graphene.py
    :start-after: #HIDDEN_BEGIN_jmbi
    :end-before: #HIDDEN_END_jmbi

should yield two eigenvalues similar to `[ 3.07869311 +1.02714523e-17j,
-3.06233144 -6.68085759e-18j]` (round-off might change the imaginary part which
would be equal to zero for exact arithmetics).

The remaining code of `main()` attaches the leads to the system and plots it
again:

.. image:: ../images/4-graphene_sys2.*

It computes the band structure of one of lead 0:

.. image:: ../images/4-graphene_bs.*

showing all the features of a zigzag lead, including the flat
edge state bands (note that the band structure is not symmetric around
zero energy, as we have a potential in the leads).

Finally the transmission through the system is computed,

.. image:: ../images/4-graphene_result.*

showing the typical resonance-like transmission probability through
an open quantum dot

.. seealso::
    The full source code can be found in
    :download:`tutorial/4-graphene.py <../../../tutorial/4-graphene.py>`

.. specialnote:: Technical details

 - In a lattice with more than one basis atom, you can always act either
   on all sublattices at the same time, or on a single sublattice only.

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
