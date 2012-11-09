:mod:`kwant.solvers` -- Library of solvers
==========================================

Overview of the solver infrastructure
-------------------------------------

kwant offers a variety of solvers for computing the solution
to a quantum transport problem. These different solvers may either
depend on different external libraries, or use very different internal
algorithms for doing the actual computation.

Fortunately, all these different solvers still use the same interface
conventions. Thus, all solvers can be used without explicit knowledge
of the internals. On the other hand, many solvers allow for some
controls that can affect performance.

All of the solver modules implement the following functions:

- `~kwant.solvers.common.SparseSolver.solve`: Compute the Scattering matrix
  or Green's function between different leads. Can be used to compute
  conductances
- `~kwant.solvers.common.SparseSolver.ldos`: Compute the local density of
  states of the system.

For details see the respective function. (You might note that the above links
lead to the documentation of a method in a class, rather than a function
in the solver module. For details, see :ref:`below <details_of_solver>`.
In a nutshell, all of the solver functionality can be
accessed by functions on the module-level; internally they are encapsulated in
a class)

Due to the common interface you can thus write code that allows you to
switch solvers by changing one `import`-line::

    from kwant.solvers.some_solver import solve

    ...

    smatrix = solve(fsys)

In addition to the common interface, some solvers allow for additional
control parameters for performance-tuning. In this case they
have a function `options` on the module level. Code could then
look like this::

    import kwant.solvers.some_solver as solver

    solver.options(...)

    ...

    smatrix - solver.solve(fsys)


Summary of solver modules in kwant:
-----------------------------------

Right now, the following solvers are implemented in kwant:

- `~kwant.solvers.sparse`: A solver based on solving a sparse linear
  system, using the direct sparse solvers provided by scipy.
- `~kwant.solvers.mumps`: A solver based on solving a sparse linear
  system using the direct sparse solver MUMPS. To use it, the MUMPS
  library must be installed on the system. This solver typically
  gives the best performance of all sparse solvers for a single core.
  It allows for various solve options using `~kwant.solvers.mumps.options`.


.. _details_of_solver:

Details of the internal structure of a solver
---------------------------------------------

Each solver module implements a class `Solver` (e.g.
`kwant.solvers.sparse.Solver`), with methods implementing the solve
functionality of the solvers. This encapsulation in a class allows to
use solvers with different options concurrently.

Typically, one however does not need this flexibility, and will not want to
bother with the `Solver` class itself. For this reason, every solver module has
a default instance of the `Solver` class. Its methods are then made available
as functions on the module level.

In particular, the following::

    import kwant.solvers.sparse as slv

    ...

    smatrix = slv.solve(sys, energy=...)
    dos = slv.ldos(sys, energy=...)

is equivalent to::

    import kwant.solvers.sparse as slv

    ...

    smatrix = slv.default_solver.solve(sys, energy=...)
    dos = slv.default_solver.ldos(sys, energy=...)

where ``default_solver`` is an instance of `kwant.solvers.sparse.Solver`.


The default solver
------------------

Since computing conductance is the most basic task of kwant, the
`solve`-function of one of the solvers is provided via `kwant.solve`.
kwant chooses the solver which it considers best amongst the
available solvers. You can see by calling

>>> help(kwant.solve)

from which module it has been imported.


List of solver modules
----------------------

The modules of the solver package are listed in detail below. Note
that the solvers (with the exception of the module providing
`kwant.solve`) have to be imported explicitly.

.. module:: kwant.solvers

.. toctree::
   :maxdepth: 1

   kwant.solvers.sparse
   kwant.solvers.mumps
   kwant.solvers.common
