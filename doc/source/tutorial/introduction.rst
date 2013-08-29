Introduction
============

In this tutorial, the most important features of Kwant are explained using
simple, but still physically meaningful examples. Each of the examples is
commented extensively. In addition, you will find notes about more subtle,
technical details at the end of each example. At first reading, these notes may
be safely skipped.

Quantum transport
.................

This introduction to the software Kwant is written for people that already have
some experience with the theory of quantum transport.  Several introductions to
the field are available, the most widely known is probably the book "Electronic
transport in mesoscopic systems" by Supriyo Datta.

The Python programming language
...............................

Kwant is a library for `Python <http://python.org/>`_.  Care was taken to fit
well with the spirit of the language and to take advantage of its expressive
power.  If you do not know Python yet, do not fear: Python is widely regarded
as one of the most accessible programming languages.  For an introduction we
recommend the `official Python Tutorial <http://docs.python.org/2/tutorial/>`_.
The `Beginner's Guide to Python <http://wiki.python.org/moin/BeginnersGuide>`_
contains a wealth of links to other tutorials, guides and books including some
for absolute beginners.

Kwant
.....

There are two steps in obtaining a numerical solution to a problem: The first
is defining the problem in a computer-accessible way, the second solving it.
The aim of a software package like Kwant is to make both steps easier.

In Kwant, the definition of the problem amounts to the creation of a tight
binding system.  The solution of the problem, i.e. the calculation of the
values of physical observables, is achieved by passing the system to a
*solver*.

The definition of a tight binding system can be seen as nothing else than the
creation of a huge sparse matrix (the Hamiltonian).  Equivalently, the sparse
Hamiltonian matrix can be seen as an annotated *graph*: the nodes of the graph
are the sites of the tight binding system, the edges are the hoppings.  Sites
are annotated with the corresponding on-site Hamiltonian matrix, hoppings are
annotated with the corresponding hopping integral matrix.

One of the central goals of Kwant is to allow easy creation of such annotated
graphs that represent tight binding system.  Kwant can be made to know about
the general structure of a particular system, the involved lattices and
symmetries.  For example, a system with a 1D translational symmetry may be used
as a lead and attached to a another system.  If both systems have sites which
belong to the same lattices, the attaching can be done automatically, even if
the shapes of the systems are irregular.

Once a tight binding system has been created, solvers provided by Kwant can be
used to compute physical observables.  Solvers expect the system to be in a
different format than the one used for construction -- the system has to be
*finalized*.  In a finalized system the tight binding graph is fixed but the
matrix elements of the Hamiltonian may still change.  The finalized format is
both more efficient and simpler -- the solvers don't have to deal with the
various details which were facilitating the construction of the system.

The typical workflow with Kwant is as follows:

#. Create an "empty" tight binding system.

#. Set its matrix elements and hoppings.

#. Attach leads (tight binding systems with translational symmetry).

#. Pass the finalized system to a solver.

Please note that even though this tutorial only shows 2-d systems, Kwant is
completely general with respect to the number of dimensions.  Kwant does not
care in the least whether systems live in one, two, three, or any other number
of dimensions.  The only exception is plotting, which out-of-the-box only works
for up to three dimensions.  (But custom projections can be specified!)
