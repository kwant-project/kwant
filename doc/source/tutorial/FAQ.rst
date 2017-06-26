--------------------------
Frequently Asked Questions
--------------------------
It is important to read the tutorials before looking at the questions. This FAQ
is aimed to add complementary explainations that are not in the tutorials. The `Kwant paper <https://downloads.kwant-project.org/doc/kwant-paper.pdf>`_ also digs deeper into Kwant's structure.


What is a system, and what is a builder?
========================================
A Kwant system represents a tight-binding model. It contains a graph of edges
and vertices that are assigned values, and which is used to construct
the Hamiltonian for the model being simulated.

In Kwant the specification of the tight-binding model is separated from the use
of the model in numerical calculations. The `~kwant.builder.Builder` is used
when specifying/constructing the model, then the
`~kwant.builder.Builder.finalize` method is called, which produces a so-called
low-level `~kwant.system.System` that can be used by Kwant's solvers.

In the documentation and in mailing list discussions, the term general term
"system" can be used to refer either to a ``Builder`` or to a low-level
``System``, and the context will determine which specific class is being
referred to. The terms "builder" and "low-level system" (or "finalized system")
refer respectively to ``Builder`` and ``System``.


What is a site?
===============
Kwant is a tool for working with tight-binding models, which can be viewed as a
graph composed of edges and vertices.  Site objects are Kwant’s abstraction for
the vertices.  Sites have two attributes: a **family** and a **tag** .  The
combination of family and tag uniquely define a site.

For example let us create an empty tight binding system and add two sites:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ122
    :end-before: #HIDDEN_END_FAQ122

.. image:: ../images/FAQ122.*

In the above snippet we added 2 sites: ``lat(1 ,0)`` and ``lat(0 , 1)``. Both
of these sites belong to the same family, ``lat``, but have different tags:
``(1, 0)`` and ``(0, 1)`` respectively.


What is a site family, and what is a tag?
=========================================
a site family "groups" related sites together, and a tag serves as a unique
identifier for a site within a given family.

In the previous example we saw a family that was suggestively called ``lat``,
which had sites whose tags were pairs of integers. In this specific example
the site family also happens to be a regular Bravais lattice, and the tags take
on the meaning of lattice coordinates for a site on this lattice.

The concept of families and tags is, however, more general. For example, one
could implement a mesh that can be locally refined in certain areas, by having
a family where sites belong to a `quadtree
<https://en.wikipedia.org/wiki/Quadtree>`_, or an amorphous blob where sites
are tagged by letters of the alphabet.


What is a lattice?
==================
Kwant allows users to define and use Bravais lattices for dealing with
collections of regularly placed sites. They know about things like which sites
on the lattice are neighbors, and how to fill a region of realspace with sites.
``Monatomic`` lattices have a single site in their basis, while ``Polyatomic``
have more than one site in their basis.

``Monatomic`` lattices in Kwant *are also site families*, with sites that are
tagged with tuples of integers: the site's coordinates in the basis of
primitive vectors of the lattice. ``Polyatomic`` lattices, however, are *not*
site families, as lattice coordinates are not enough information to uniquely
identify a site if there is more than one site in the basis. ``Polyatomic``
lattices do, however, have an attribute ``sublattices`` that is a list of
``Monatomic`` lattices that together make up the whole ``Polyatomic`` lattice.

For example:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ123
    :end-before: #HIDDEN_END_FAQ123

.. image:: ../images/FAQ123.*

Above, we created 2 ``Monatomic`` lattices (``lat1`` and ``lat2``).  ``(1, 0)``
and ``(0, 1)`` are the primitive vectors and ``(0, 0)`` and ``(0.5, 0.5)`` are
the origins of the two lattices. Next we create a ``Polyatomic`` lattice with the
same primitive vectors and 2 sites in the basis.

The two sublattices are equivalent to the two monatomic lattices that we
created previously. Because a ``Polyatomic`` lattice knows about its
sublattices, we can do things like calculate neighboring sites, even between
sublattices, which would not be possible with the two separate ``Monatomic``
lattices.

the `kwant.lattice` module also defines several functions, such as
`~kwant.lattice.square` and `~kwant.lattice.honeycomb`, which serve as a
convenience for creating lattices of common types, without having to
explicitly specify all of the lattice vectors and basis vectors.


How do I plot a polyatomic lattice with different colors?
=========================================================
In the following example we shall use a kagome lattice, which has 3 sublattices.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ8
    :end-before: #HIDDEN_END_FAQ8

As we can see below, we create a new plotting function that assigns a color for each family, and a different size for the hoppings depending on the family of the two sites. Finally we add sites and hoppings to our system and plot it with the new function.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ9
    :end-before: #HIDDEN_END_FAQ9

.. image:: ../images/FAQ6B.*


What is a hopping?
==================
A hopping is simply a pair of sites, which defines an edge of the graph
that makes up our tight-binding model.

Starting from the example code from `What is a site?`_, we can add a hopping
to our system in the following way:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ124
    :end-before: #HIDDEN_END_FAQ124

.. image:: ../images/FAQ124.*

Visually, we represent a hopping as a line that joins two sites.

Whenever we add a hopping ``(site_a, site_b)`` to a system and assign it a
value ``v``, implicitly the hopping ``(site_b, site_a)`` is also added, with
value the Hermitian conjugate of ``v``.


How do I create all hoppings in a given direction?
==================================================
This can be obtained using a `~kwant.builder.HoppingKind`. A ``HoppingKind`` is
a way of specifying all hoppings of a particular "kind", between two site
families. For example ``HoppingKind((1, 0), lat_a, lat_b)`` represents all
hoppings of the form ``(lat_a(x + (1, 0)), lat_b(x))``, where ``x`` is a tag
(here, a pair of integers).

The following example shows how this can be used:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ4
    :end-before: #HIDDEN_END_FAQ4

.. image:: ../images/FAQ3.*

Note that ``HoppingKind`` only works with site families so you cannot use
them directly with ``Polyatomic`` lattices; you have to explicitly specify
the sublattices when creating a ``HoppingKind``:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ13
    :end-before: #HIDDEN_END_FAQ13

Here, we want the hoppings between the sites from sublattice b with a direction of (0,1) in the lattice coordinates.

.. image:: ../images/FAQ10.*

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ14
    :end-before: #HIDDEN_END_FAQ14

Here, we create hoppings between the sites of the same lattice coordinates but from different families.

.. image:: ../images/FAQ11.*


How do I create the hoppings between adjacent sites?
====================================================
``Polyatomic`` and ``Monatomic`` lattices have a method `~kwant.lattice.Polyatomic.neighbors`
that returns a (or several) ``HoppingKind`` that connects sites with their
(n-nearest) neighors:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ5
    :end-before: #HIDDEN_END_FAQ5

.. image:: ../images/FAQ4.*
.. image:: ../images/FAQ5.*

As we can see in the figure above, ``lat.neighbors()`` (on the left) returns the hoppings between the first nearest neighbors and ``lat.neighbors(2)`` (on the right) returns the hoppings between the second nearest neighbors.

When using a ``Polyatomic`` lattice ``neighbors()`` knows about the different
sublattices:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ10
    :end-before: #HIDDEN_END_FAQ10


.. image:: ../images/FAQ7.*

However, if we use the ``neighbors()`` method of a single sublattice, we will
only get the neighbors *on that sublattice*:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ11
    :end-before: #HIDDEN_END_FAQ11

.. image:: ../images/FAQ8.*

Note in the above that there are *only* hoppings between the blue sites. This
is an artifact of the visualisation: the red and green sites just happen to lie
in the path of the hoppings, but are not connected by them.


How do I make a hole in a system?
=================================
To make a hole in the system, we use ``del syst[site]``. In the following
example we remove all sites inside some "hole" region:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ2
    :end-before: #HIDDEN_END_FAQ2

.. image:: ../images/FAQ1.*
.. image:: ../images/FAQ2.*

``del syst[site]`` also works after hoppings have been added to the system.
If a site is deleted, then all the hoppings to/from that site are also deleted.


How can I get access to a system's sites?
=========================================
The ways of accessing system sites is slightly different depending on whether
we are talking about a ``Builder`` or ``System`` (see `What is a system, and
what is a builder?`_ if you do not know the difference).

We can access the sites of a ``Builder`` by using its `~kwant.builder.Builder.sites` method:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ3
    :end-before: #HIDDEN_END_FAQ3

The ``sites()`` method returns an *iterator* over the system sites, and in the
above example we create a list from the contents of this iterator, which
contains all the sites. At this stage the ordering of sites is not fixed, so if
you add more sites to the ``Builder`` and call ``sites()`` again, the sites may
well be returned in a different order.

After finalization, when we are dealing with a ``System``, the sites themselves
are stored in a list, which can be accessed via the ``sites`` attribute:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ7
    :end-before: #HIDDEN_END_FAQ7

The order of sites in a ``System`` is fixed, and also defines the ordering of
the system Hamiltonian, system wavefunctions etc. (see `How does Kwant order components of an individual wavefunction?`_ for details).

``System`` also contains the inverse mapping, ``id_by_site`` which gives us
the index of a given site within the system:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ72
    :end-before: #HIDDEN_END_FAQ72


How do I create a lead with a lattice different from the scattering region?
===========================================================================
Let us take the example of a system containing sites from a honeycomb lattice,
which we want to connect to leads that contain sites from a square lattice.

First we construct the central system:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAA
    :end-before: #HIDDEN_END_FAQAA

.. image:: ../images/FAQAA.*

and the lead:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAB
    :end-before: #HIDDEN_END_FAQAB

.. image:: ../images/FAQAB.*

We cannot simply use `~kwant.builder.Builder.attach_lead` to attach this lead to the
system with the honeycomb lattice because Kwant does not know how sites from
these two lattices should be connected.

We must first add a layer of sites from the square lattice to the system and manually
add the hoppings from these sites to the sites from the honeycomb lattice:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAC
    :end-before: #HIDDEN_END_FAQAC

.. image:: ../images/FAQAC.*

``attach_lead()`` will now be able to attach the lead:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAD
    :end-before: #HIDDEN_END_FAQAD

.. image:: ../images/FAQAD.*


How do I cut a finite system out of a system with translational symmetries?
===========================================================================
This can be achieved using the `~kwant.builder.Builder.fill` method to fill a
``Builder`` with a ``Builder`` with higher symmetry.

When using the ``fill()`` method, we need two systems: the template and the
target. The template is a ``Builder`` with some translational symmetry that
will be repeated in the desired shape to create the final system.

For example, say we want to create a simple model on a cubic lattice:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQccc
    :end-before: #HIDDEN_END_FAQccc

We have now created our "template" ``Builder`` which has 3 translational
symmetries. Next we will fill a system with no translational symmetries with
sites and hoppings from the template inside a cuboid:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQddd
    :end-before: #HIDDEN_END_FAQddd

.. image:: ../images/FAQaaa.*

We can then use the original template to create a lead, which has 1 translational
symmetry. We can then use this lead as a template to fill another section of
the system with a cylinder of sites and hoppings:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQeee
    :end-before: #HIDDEN_END_FAQeee

.. image:: ../images/FAQbbb.*


How does Kwant order the propagating modes of a lead?
=====================================================
A very useful feature of kwant is to calculate the transverse wavefunctions of
propagating modes in a system with 1 translational symmetry.  This can be
achieved with the `~kwant.system.InfiniteSystem.modes` method, which returns a
pair of objects, the first of which contains the propagating modes of the
system in a `~kwant.physics.PropagatingModes` object:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_PM
    :end-before: #HIDDEN_END_PM

``PropagatingModes`` contains the wavefunctions, velocities and momenta of the
modes at the requested energy (2.5 in this example).  In order to understand
the order in which these quantities are returned it is often useful to look at
the a section of the band structure for the system in question:

.. image:: ../images/FAQPM.*

On the above band structure we have labelled the 4 modes in the order
that they appear in the output of ``modes()`` at energy 2.5. Note that
the modes are sorted in the following way:

    + First all the modes with negative velocity, then all the modes with
      positive velocity
    + Negative velocity modes are ordered by *increasing* momentum
    + Positive velocity modes are ordered by *decreasing* momentum

For more complicated systems and band structures this can lead to some
possibly unintuitive orderings:

.. image:: ../images/FAQPMC.*


How does Kwant order scattering states?
=======================================
Scattering states calculated using `~kwant.solvers.default.wave_function` are returned in the
same order as the "incoming" modes of `~kwant.system.InfiniteSystem.modes`.
Kwant considers that the translational symmetry of a lead points "towards
infinity" (*not* towards the system) which means that the incoming modes are
those that have *negative* velocities:


How does Kwant order components of an individual wavefunction?
==============================================================
In `How can I get access to a system's sites?`_ we saw that the sites of a finalized system
are available as a list through the ``sites`` attribute, and that one can look up the index
of a site with the ``id_by_site`` attribute.

When all the site families present in a system have only 1 degree of freedom
per site (i.e.  the all the onsites are scalars) then the index into a
wavefunction defined over the system is exactly the site index:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_ORD1
    :end-before: #HIDDEN_END_ORD1
.. literalinclude:: ../images/FAQORD1.txt

We see that the wavefunction on a single site is a single complex number, as
expected.

If a site family have more than 1 degree of freedom per site (e.g. spin or
particle-hole) then Kwant places degrees of freedom on the same site adjacent
to one another.  In the case where all site families in the system have the
*same* number of degrees of freedom, we can then simply *reshape* the
wavefunction into a matrix, where the row number indexes the site, and the
column number indexes the degree of freedom on that site:

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_ORD2
    :end-before: #HIDDEN_END_ORD2
.. literalinclude:: ../images/FAQORD2.txt

We see that the wavefunction on a single site is a *vector* of 2 complex numbers,
as we expect.

If there are different site families present in the system that have *different*
numbers of orbitals per site, then the situation becomes much more involved,
because we cannot simply "reshape" the wavefunction like we did in the
preceding example.
