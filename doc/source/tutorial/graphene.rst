.. _tutorial-graphene:

Beyond square lattices: graphene
--------------------------------

.. seealso::
    The complete source code of this example can be found in
    :download:`graphene.py </code/download/graphene.py>`

In the following example, we are going to calculate the
conductance through a graphene quantum dot with a p-n junction
and two non-collinear leads. In the process, we will touch
all of the topics that we have seen in the previous tutorials,
but now for the honeycomb lattice. As you will see, everything
carries over nicely.

We begin by defining the honeycomb lattice of graphene. This is
in principle already done in `kwant.lattice.honeycomb`, but we do it
explicitly here to show how to define a new lattice:

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_hnla
    :end-before: #HIDDEN_END_hnla

The first argument to the `~kwant.lattice.general` function is the list of
primitive vectors of the lattice; the second one is the coordinates of basis
atoms.  The honeycomb lattice has two basis atoms. Each type of basis atom by
itself forms a regular lattice of the same type as well, and those
*sublattices* are referenced as `a` and `b` above.

In the next step we define the shape of the scattering region (circle again)
and add all lattice points using the ``shape``-functionality:

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_shzy
    :end-before: #HIDDEN_END_shzy

As you can see, this works exactly the same for any kind of lattice.
We add the onsite energies using a function describing the p-n junction;
in contrast to the previous tutorial, the potential value is this time taken
from the scope of `make_system`, since we keep the potential fixed
in this example.

As a next step we add the hoppings, making use of
`~kwant.builder.HoppingKind`. For illustration purposes we define
the hoppings ourselves instead of using ``graphene.neighbors()``:

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_hsmc
    :end-before: #HIDDEN_END_hsmc

The nearest-neighbor model for graphene contains only
hoppings between different basis atoms. For this type of
hoppings, it is not enough to specify relative lattice indices,
but we also need to specify the proper target and source
sublattices. Remember that the format of the hopping specification
is ``(i,j), target, source``. In the previous examples (i.e.
:ref:`tutorial_spinorbit`) ``target=source=lat``, whereas here
we have to specify different sublattices. Furthermore,
note that the directions given by the lattice indices
`(1, 0)` and `(0, 1)` are not orthogonal anymore, since they are given with
respect to the two primitive vectors ``[(1, 0), (sin_30, cos_30)]``.

Adding the hoppings however still works the same way:

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_bfwb
    :end-before: #HIDDEN_END_bfwb

Modifying the scattering region is also possible as before. Let's
do something crazy, and remove an atom in sublattice A
(which removes also the hoppings from/to this site) as well
as add an additional link:

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_efut
    :end-before: #HIDDEN_END_efut

Note again that the conversion from a tuple `(i,j)` to site
is done by the sublattices `a` and `b`.

The leads are defined almost as before:

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_aakh
    :end-before: #HIDDEN_END_aakh

Note the method `~kwant.lattice.Polyatomic.vec` used in calculating the
parameter for `~kwant.lattice.TranslationalSymmetry`.  The latter expects a
real-space symmetry vector, but for many lattices symmetry vectors are more
easily expressed in the natural coordinate system of the lattice. The
`~kwant.attices.Polyatomic.vec`-method is thus used to map a lattice vector
to a real-space vector.

Observe also that the translational vectors ``graphene.vec((-1, 0))`` and
``graphene.vec((0, 1))`` are *not* orthogonal any more as they would have been
in a square lattice -- they follow the non-orthogonal primitive vectors defined
in the beginning.

Later, we will compute some eigenvalues of the closed scattering region without
leads. This is why we postpone attaching the leads to the system. Instead,
we return the scattering region and the leads separately.

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_kmmw
    :end-before: #HIDDEN_END_kmmw

The computation of some eigenvalues of the closed system is done
in the following piece of code:

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_zydk
    :end-before: #HIDDEN_END_zydk

The code for computing the band structure and the conductance is identical
to the previous examples, and needs not be further explained here.

Finally, in the `main` function we make and plot the system:

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_itkk
    :end-before: #HIDDEN_END_itkk

We customize the plotting: we set the `site_colors` argument of
`~kwant.plotter.plot` to a function which returns 0 for
sublattice `a` and 1 for sublattice `b`::

    def family_colors(site):
        return 0 if site.family == a else 1

The function `~kwant.plotter.plot` shows these values using a color scale
(grayscale by default). The symbol `size` is specified in points, and is
independent on the overall figure size.

Plotting the closed system gives this result:

.. image:: /code/figure/graphene_syst1.*

Computing the eigenvalues of largest magnitude,

.. literalinclude:: /code/include/graphene.py
    :start-after: #HIDDEN_BEGIN_jmbi
    :end-before: #HIDDEN_END_jmbi

should yield two eigenvalues equal to ``[ 3.07869311,
-3.06233144]``.

The remaining code of `main` attaches the leads to the system and plots it
again:

.. image:: /code/figure/graphene_syst2.*

It computes the band structure of one of lead 0:

.. image:: /code/figure/graphene_bs.*

showing all the features of a zigzag lead, including the flat
edge state bands (note that the band structure is not symmetric around
zero energy, due to a potential in the leads).

Finally the transmission through the system is computed,

.. image:: /code/figure/graphene_result.*

showing the typical resonance-like transmission probability through
an open quantum dot

.. specialnote:: Technical details

 - In a lattice with more than one basis atom, you can always act either
   on all sublattices at the same time, or on a single sublattice only.

   For example, you can add lattice points for all sublattices in the
   current example using::

       syst[graphene.shape(...)] = ...

   or just for a single sublattice::

       syst[a.shape(...)] = ...

   and the same of course with `b`. Also, you can selectively remove points::

       del syst[graphene.shape(...)]
       del syst[a.shape(...)]

   where the first line removes points in both sublattices, whereas the
   second line removes only points in one sublattice.
