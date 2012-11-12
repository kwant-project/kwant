Introduction
============

There are two steps in obtaining a numerical solution to a problem: The first
is defining the problem in a computer-accessible way, the second solving it.
The aim of a software package like kwant is to make both steps easier.

In kwant, the definition of the problem amounts to the creation of a tight
binding system.  The solution of the problem, i.e. the calculation of the
values of physical observables, is achieved by passing the system to a
*solver*.

The definition of a tight binding system can be seen as nothing else than the
creation of a huge sparse matrix (the Hamiltonian).  Equivalently, the sparse
Hamiltonian matrix can be seen as an annotated *graph*: the nodes of the graph
are the sites of the tight binding system, the edges are the hoppings.  Sites
are annotated with the corresponding on-site Hamiltonian matrix, hoppings are
annotated with the corresponding hopping integral matrix.

One of the central goals of kwant is to allow easy creation of such annotated
graphs that represent tight binding system.  kwant can be made to know about
the general structure of a particular system, the involved lattices and
symmetries.  For example, a system with a 1D translational symmetry may be used
as a lead and attached to a another system.  If both systems have sites which
belong to the same lattices, the attaching can be done automatically, even if
the shapes of the systems are irregular.

kwant is a library for the `Python <http://python.org/>`_ programming language.
Care was taken to fit well with the spirit of the language and to take
advantage of its expressive power.

Once a tight binding system has been created, solvers provided by kwant can be
used to compute physical observables.  Solvers expect the system to be in a
different format than the one used for construction -- the system has to be
*finalized*.  In a finalized system the tight binding graph is fixed but the
matrix elements of the Hamiltonian may still change.  The finalized format is
both more efficient and simpler -- the solvers don't have to deal with the
various details which were facilitating the construction of the system.

The typical workflow with kwant is as follows:

#. Create an "empty" tight binding system.

#. Set its matrix elements and hoppings.

#. Attach leads (tight binding systems with translational symmetry).

#. Pass the finalized system to a solver.
