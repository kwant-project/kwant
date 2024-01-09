:mod:`kwant` -- Top level package
=================================

.. module:: kwant

For convenience, short names are provided for a few widely used objects from
the sub-packages.
Otherwise, this package has only very limited functionality of its own.

Generic functionality
---------------------

.. autodata:: kwant.__version__

.. autosummary::
   :toctree: generated/

   KwantDeprecationWarning
   UserCodeError

.. currentmodule:: kwant.builder

From `kwant.builder`
--------------------
.. autosummary::

   Builder
   HoppingKind

.. currentmodule:: kwant.lattice

From `kwant.lattice`
--------------------
.. autosummary::

   TranslationalSymmetry

.. currentmodule:: kwant.plotter

From `kwant.plotter`
--------------------

.. autosummary::

   plot

.. currentmodule:: kwant.solvers.default

From `kwant.solvers.default`
----------------------------
.. autosummary::

   greens_function
   ldos
   smatrix
   wave_function
