:mod:`kwant.solvers.mumps` -- High performance sparse solver based on MUMPS
===========================================================================

.. module:: kwant.solvers.mumps

This solver uses `MUMPS <https://graal.ens-lyon.fr/MUMPS/>`_.  (Only the
sequential, single core version of MUMPS is used.)  MUMPS is a very efficient
direct sparse solver that can take advantage of memory beyond 3GiB for the
solution of large problems.  Furthermore, it offers a choice of several
orderings of the input matrix some of which can speed up a calculation
significantly.

Compared with the :mod:`default solver <kwant.solvers.default>`, this module
adds several options that may be used to fine-tune performance.  Otherwise the
interface is identical.  These options can be set and queried with the
following functions.

.. autofunction:: options

.. autofunction:: reset_options
