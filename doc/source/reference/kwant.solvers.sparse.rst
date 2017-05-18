:mod:`kwant.solvers.sparse` -- Basic sparse matrix solver
=========================================================

.. module:: kwant.solvers.sparse

This solver uses SciPy's `scipy.sparse.linalg
<https://docs.scipy.org/doc/scipy/reference/sparse.linalg.html>`_.  The
interface is identical to that of the :mod:`default solver
<kwant.solvers.default>`.

``scipy.sparse.linalg`` currently uses internally either the direct sparse
solver UMFPACK or if that is not installed, SuperLU. Often, SciPy's SuperLU
will give quite poor performance and you will be warned if only SuperLU is
found.  The module variable ``uses_umfpack`` can be checked to determine if
UMFPACK is being used.
