What's new in the next release of Kwant
=======================================

This article lists the new features and enhancements that will be part of the
next release of Kwant.

Package cleanup
---------------
In order to modernize Kwant's codebase and improve maintainability, we have implemented
multiple

We have upgraded Kwant to use the modern format for providing metadata and
changed the build system to use ``meson-python``.  This makes the compilation of
Kwant faster and more reliable.

We adopted `SPEC-0 <https://scientific-python.org/specs/spec-0000/>`_ as the standard
for which dependency versions Kwant supports and updated the minimal required
versions of several dependencies.

Kwant used to provide a partial wrapper for the MUMPS library.  This wrapper has been
removed in favor of using the more general `python-mumps
<https://pypi.org/project/python-mumps/>`_ wrapper.
