:mod:`kwant` -- Top level package
=================================

.. module:: kwant

For convenience, short names are provided for a few widely used objects from
the sub-packages.
Otherwise, this package has only very limited functionality of its own.

Generic functionality
---------------------
..
   TODO: Once we depend on Sphinx 1.8, the documentation of __version__ can be
   put into the "docstring": https://github.com/sphinx-doc/sphinx/issues/344

The version of Kwant is available under the name ``__version__``.
This string respects `PEP 440 <https://www.python.org/dev/peps/pep-0440/>`_
and has the following format

- Released version: '1.3.0', '1.3.1', etc.
- Alpha version: '1.2.0a0', '1.2.0a1', etc.
- Beta version: '1.1.0b0', '1.1.0b1', etc.
- Development version (derived from ``git describe --first-parent --dirty``):
  '1.3.2.dev27+gdecf6893', '1.1.1.dev10+gabcd012.dirty', etc.
- Development version with incomplete information: 'unknown',
  'unknown+g0123abc', etc.

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
