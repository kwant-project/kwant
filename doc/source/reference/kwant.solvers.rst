:mod:`kwant.solvers` -- Library of solvers
==========================================

Overview
--------

Kwant offers several modules for computing the solutions to quantum transport
problems, the so-called solvers. Each of these solvers may use different
internal algorithms and/or depend on different external libraries.  If the
libraries needed by one solver are not installed, trying to import it will
raise an ``ImportError`` exception.  The :doc:`Installation instructions
<../install>` list all the libraries that are required or can be used by Kwant
and its solvers.


:mod:`kwant.solvers.default` -- The default solver
--------------------------------------------------

.. module:: kwant.solvers.default

There is one solver, `kwant.solvers.default` that is always available.  For
each Kwant installation it combines the best functionality of the *available*
solvers into a single module.  We recommend to use it unless there are specific
reasons to use another.  The following functions are provided.

.. autosummary::
   :toctree: generated/

   smatrix
   greens_function
   wave_function
   ldos

``smatrix`` returns an object of the following type:

.. module:: kwant.solvers

.. autosummary::
   :toctree: generated/

   kwant.solvers.common.SMatrix

The analog of ``smatrix``, ``greens_function`` accordingly returns:

.. autosummary::
   :toctree: generated/

   kwant.solvers.common.GreensFunction

Being just a thin wrapper around other solvers, the default solver selectively
imports their functionality.  To find out the origin of any function in this
module, use Python's ``help``.  For example

>>> help(kwant.solvers.default.ldos)


Other solver modules
--------------------

Unlike the default one, other solvers have to be imported manually.  They
provide, whenever possible, exactly the same interface as the default.  Some
allow for specific tuning that can improve performance.  The differences to the
default solver are listed in the documentation of each module.

.. toctree::
   :maxdepth: 1

   kwant.solvers.sparse
   kwant.solvers.mumps

For Kwant-experts: detail of the internal structure of a solver
---------------------------------------------------------------

Each solver module (except the default one) contains a class ``Solver`` (e.g.
``kwant.solvers.sparse.Solver``), that actually implements that solver's
functionality.  For each module-level function provided by the solver, there is
a correspondent method in the ``Solver`` class.  The module-level functions are
simply the methods of a hidden ``Solver`` instance that is present in each
solver module.

The encapsulation in a class allows different solvers to easily share common
code.  It also makes it possible to use solvers with different options
concurrently.  Typically, one does not need this flexibility, and will not want
to bother with the ``Solver`` class itself.  Instead, one will use the
module-level functions as explained in the previous sections.
