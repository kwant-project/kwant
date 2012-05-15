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

For details see the respective function.

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

Right now, the following solvers are implemented in kwant:

- `~kwant.solvers.sparse`: A solver based on solving a sparse linear
  system, using the direct sparse solvers provided by scipy.
- `~kwant.solvers.mumps`: A solver based on solving a sparse linear
  system using the direct sparse solver MUMPS. To use it, the MUMPS
  library must be installed on the system. This solver typically
  gives the best performance of all sparse solvers for a single core.
  it allows for various solve options using `~kwant.solvers.mumps.options`.

The default solver
------------------

Since computing conductance is the most basic task of kwant, the
`solve`-function of one of the solvers is provided via `kwant.solve`.
kwant chooses the solver which it considers best amongst the
available solvers. You can see by calling

>>> help(kwant.solve)

from whoch module it has been imported.

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
