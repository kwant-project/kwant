:mod:`kwant.solvers.mumps` -- High performance sparse solver based on MUMPS
===========================================================================

.. module:: kwant.solvers.mumps

A sparse solver that uses `MUMPS <http://graal.ens-lyon.fr/MUMPS/>`_.  (Only
the sequential, single core version is used.)

MUMPS is a very efficient direct sparse solver that can take advantage of
memory beyond 3GB for the solution of large problems.  Furthermore, it offers a
choice of several orderings of the input matrix from which can speed up a
calculation significantly.

Compared to the generic sparse solver framework, `mumps` adds the following
control options that may affect performance:

- `ordering`: a fill-in reducing ordering of the matrix
- `nrhs`: number of right hand sides that should be solved simultaneously
- `sparse_rhs`: whether to use dense or sparse right hand sides

For more details see `~Solver.options`.

.. autosummary::
   :toctree: generated/

   Solver
