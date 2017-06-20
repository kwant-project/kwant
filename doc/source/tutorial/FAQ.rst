---------------------------------------------------------------
Frequently Asked Questions :
---------------------------------------------------------------

It is important to read the tutorials before looking at the questions. This FAQ is made to add complementary explanations that are not in the tutorials.


What is a site?
=================

Kwant is based on the tight-binding model which is composed of edges and vertices.
Site objects are Kwant’s abstraction for the vertices. The sites have two attributes: the **family** and the **tag** .

For example :

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ122
    :end-before: #HIDDEN_END_FAQ122

.. image:: ../images/FAQ122.*

As we can see in the figure, we added 2 sites to the system. Each site has a different tag ( ``(1,0)`` and ``(1,1)`` ) but they have the same family ``lat``

What is a lattice?
====================================

The lattice contains the spatial repartition of the sites in the system. There are two kind of lattices in Kwant :
	- The monatomic lattices
	- The polyatomic lattices


The monatomic class represents a bravais lattice with a single site in the basis, it contains an origin and multiple vectors which define a frame of reference. The polyatomic lattice has N sites per basis which correspond to N monatomic sublattices.


.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ123
    :end-before: #HIDDEN_END_FAQ123

.. image:: ../images/FAQ123.*

As we can see above, we created 2 monatomic lattices,  ``(1,0)`` and ``(0,1)`` are the primitive vectors and ``(0,0)`` and ``(0.5,0.5)`` are the origins of the two lattices. We can create a polyatomic lattice with the same primitive vectors and 2 sites in the basis. It leads to the same result. The two sublattices are equivalent to the two monatomic lattices.

What is a family ? a tag?
============================

The concept of lattices and families are very interrelated. The family represents a sublattice and defines the tag.
In the monatomic case, the family has the same name as the the lattice.

In the polyatomic case, there are multiple families which represent the different monatomic sublattices.

If we take the example from above, there are 4 families: ``subA , subB , lat1, lat2``. However, ``lat3`` doesn't have a family because it is polyatomic.

The tag represents the coordinates of a site in his family. In the common case where the site family is a regular lattice, the site tag is simply given by its integer lattice coordinates.

There are multiple :doc:`predefined lattices <../reference/kwant.lattice>` implemented in Kwant that we can use, it is simple to see how much sublattices there are by using ``lat.sublattices`` and look at what it returns.


What is a hopping?
====================

If we take the example of `What is a site?`_ we can simply add a hopping with :

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ124
    :end-before: #HIDDEN_END_FAQ124

.. image:: ../images/FAQ124.*

As we can see, a hopping is a tupple of site, it represents an edge of the tight-binding system.

How to make a hole in a system?
=================================
To make a hole in the system, we use ``del syst[key]`` , we can either use the real-space coordinates with ``.pos`` or use the lattice coordinates ``tag`` . Here, we use the real-space position to delete sites in the middle.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ2
    :end-before: #HIDDEN_END_FAQ2

.. image:: ../images/FAQ1.*
.. image:: ../images/FAQ2.*


Note that we can make the hole after creating the hoppings, then the hoppings to the sites that are deleted are removed as well.

How can we get access to the sites of our system?
===================================================

In kwant, the tight-binding model is split in two steps. First, we have a Builder where it is easy to add, remove or modify sites and hoppings. Then, we have a System which is optimized for computation. The separation is made with the finalization. The way we can have access to the list of sites is different before and after the finalization.


.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ3
    :end-before: #HIDDEN_END_FAQ3

Before finalizing, we get access to the sites with : ``syst.sites()`` .

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ7
    :end-before: #HIDDEN_END_FAQ7

After finalizing the system, the order of the sites changes, so we need ``id_by_site`` to find the site's id in the finalized system. We can also use ``lat.closest(pos)`` to get the tag of a site at a given position.

How to plot a polyatomic lattice with different colors?
===============================================================================
We take a kagome lattice as example here, it has 3 sublattices.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ8
    :end-before: #HIDDEN_END_FAQ8

As we can see below, we create a new plotting function that assigns a color for each family, and a different size for the hoppings depending on the family of the two sites. Finally we add sites and hoppings to our system and plot it with the new function.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ9
    :end-before: #HIDDEN_END_FAQ9

.. image:: ../images/FAQ6B.*

How to create every hoppings in a given direction using ``Hoppingkind``?
==========================================================================


If the lattice is monatomic:

``HoppingKind`` can be used to easily create a hopping in a specific direction.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ4
    :end-before: #HIDDEN_END_FAQ4

.. image:: ../images/FAQ3.*

We create hoppings between the sites of the lattice ``lat`` . ``(1,0)`` represents the direction of the hoppings we want to assign.

If the lattice is polyatomic:

``Hoppingkind`` can only be used in the monatomic sublattices. As we can see below, it can create hoppings between different sublattices with a given direction.

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



How to create the hoppings between adjacent sites?
====================================================

``lat.neighbors(n)`` returns ``HoppingKind`` in the directions of the n-nearest neighbors.

In the monatomic case:

``lat.neighbors(n)`` creates the hoppings between the n-nearest neighbors.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ5
    :end-before: #HIDDEN_END_FAQ5

.. image:: ../images/FAQ4.*
.. image:: ../images/FAQ5.*

As we can see in the figure above, ``lat.neighbors()`` (on the left) returns the hoppings between the first nearest neighbors and ``lat.neighbors(2)`` (on the right) returns the hoppings between the second nearest neighbors.

In the polyatomic case:

It is possible to use ``.neighbors()`` with the lattice or the sublattices as explained below.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ10
    :end-before: #HIDDEN_END_FAQ10

It creates hoppings between the first nearest neighbors of every sites of the system.

.. image:: ../images/FAQ7.*

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ11
    :end-before: #HIDDEN_END_FAQ11

It creates hoppings between the first nearest neighbors of the sites of the sublattice a (in red).

.. image:: ../images/FAQ8.*

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQ12A
    :end-before: #HIDDEN_END_FAQ12A

It creates hoppings between the second nearest neighbors of every sites of the system.

.. image:: ../images/FAQ9.*



How to create a lead with a lattice different from the scattering region?
===========================================================================

First of all, we need to plot the sites in different colors depending on the sites families.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAA
    :end-before: #HIDDEN_END_FAQAA


Then, we create the scattering region, here we take the example of the honeycomb lattice.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAB
    :end-before: #HIDDEN_END_FAQAB

.. image:: ../images/FAQAB.*

We now have the scattering region, however, we can't just make the lead and attach it if it doesn't have the same lattice. We want our lead lattice to be square, so we need to manually attach some sites from the same lead lattice to the scattering region.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAC
    :end-before: #HIDDEN_END_FAQAC

.. image:: ../images/FAQAC.*

Now that we have those sites, we can create the leads and attach it. We first begin with the top lead, considering the dimensions of the scattering region on the figure above, we create a shape for the lead and attach it.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAD
    :end-before: #HIDDEN_END_FAQAD

.. image:: ../images/FAQAD.*

We now do the same for the bottom lead.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAE
    :end-before: #HIDDEN_END_FAQAE

.. image:: ../images/FAQAE.*



How to cut a finite system out of a system with translationnal symmetries?
============================================================================

This can be achieved using the :doc:`fill()  <../reference/generated/kwant.builder.Builder>` method to fill a Builder with a Builder with higher symmetry.

An example of fill is given :doc:`here  <../tutorial/discretize>` for an undisctretised hamiltonian, but we want to see how it works with a discretised system.

When using the fill method, we need two systems, the template and the target (final sytem). The template is a part of the system which is periodically repeated in the desired shape to create the final system, so it needs to have translationnal symmetries.


We first create the system template with its symmetries.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQccc
    :end-before: #HIDDEN_END_FAQccc

We now define the shape of the scattering region and use the fill method to create the first part of the scattering region

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQddd
    :end-before: #HIDDEN_END_FAQddd

.. image:: ../images/FAQaaa.*

Finally, we create the template that will used to create the second part of the scattering region and the lead.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQeee
    :end-before: #HIDDEN_END_FAQeee

.. image:: ../images/FAQbbb.*

How are the physical quantities ordered within the output of the functions?
===========================================================================

We take the example of the ``wave_function()`` here. We have only one degree of freedom here, see `How are degrees of freedom ordered ?`_  for further explanations in case of multiple one.

So first, we define a system with a square lattice and 2 leads. We also plot the Fermi energy and the modes to see how much modes we can get with a given energy.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAF
    :end-before: #HIDDEN_END_FAQAF

.. image:: ../images/FAQBA.*
.. image:: ../images/FAQBB.*

Here we take this Fermi energy so we can have 2 modes. Note that the velocity is negative for the incoming modes.
We take the modes coming from the left lead (lead 0 in this case).

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAG
    :end-before: #HIDDEN_END_FAQAG

Then, we need to distinguish the different modes, it is indexed according to the decreasing k . For example here, we take the first mode and we want to know the wave function on the site at the position (6,2). To do that, we need to find the tag of this site using ``closest`` and then use ``id_by_site`` to find the index in the low-level system.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAH
    :end-before: #HIDDEN_END_FAQAH


How are degrees of freedom ordered ?
======================================

We take the example from :doc:`here <../tutorial/superconductors>` which has 2 degrees of freedom. We will focus on the general ordering of the  degrees of freedom with the example of the ``wave_function()``

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAO
    :end-before: #HIDDEN_END_FAQAO

.. image:: ../images/FAQTT.*

.. image:: ../images/FAQSS.*


With one degree of freedom, we had just to take the output of ``id_by_site`` to print the wave_function on a specific site.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAH
    :end-before: #HIDDEN_END_FAQAH

In the case of 2 degrees of freedom, it changes the way we have access to a given site at a given orbital.

.. literalinclude:: FAQ.py
    :start-after: #HIDDEN_BEGIN_FAQAP
    :end-before: #HIDDEN_END_FAQAP

The order of the orbitals and sites are completly modified. The N first orbitals of the first site are between the index 0 and N-1, for the second site, it is between N and 2N-1, etc...
