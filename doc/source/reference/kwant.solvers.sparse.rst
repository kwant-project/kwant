:mod:`kwant.solvers.sparse` -- Basic sparse matrix solver
=========================================================

.. module:: kwant.solvers.sparse

A sparse solver that uses `scipy.sparse.linalg
<http://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_.

SciPy currently uses internally either the direct sparse solver UMFPACK or if
that is not installed, SuperLU. Often, SciPy's SuperLU will give quite poor
performance and you will be warned if only SuperLU is found.  The module
variable `uses_umfpack` can be checked to determine if UMFPACK is being used.

`sparse` does not introduce any additional options as compared to the generic
sparse solver framework.

.. autosummary::
   :toctree: generated/

   Solver
