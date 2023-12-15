=====================
Installation of Kwant
=====================

Ready-to-use Kwant packages are available for many platforms (like GNU/Linux,
Mac OS X, Microsoft Windows).  See the `installation page of the Kwant website
<https://kwant-project.org/install>`_ for instructions on how to install Kwant
on your platform.  This is the recommended way for new users.

The remainder of this section documents how to build Kwant from source.  This
information is mostly of interest to contributors and packagers.


********************
Generic instructions
********************

Obtaining the source code
=========================

Source distributions of Kwant (and Tinyarray) are available at the `downloads
section of the Kwant website <https://downloads.kwant-project.org/kwant/>`_ as well
as `PyPI <https://pypi.org/project/kwant>`_.  The sources may be also
cloned directly from the `official Kwant git repository
<https://gitlab.kwant-project.org/kwant/kwant>`_.


Prerequisites
=============

Building Kwant requires
 * `Python <https://www.python.org/>`_ 3.8 or above,
 * `NumPy <http://numpy.org/>`_ 1.11.0 or newer,
 * `SciPy <https://www.scipy.org/>`_ 0.17.0 or newer,
 * `LAPACK <http://netlib.org/lapack/>`_ and `BLAS <http://netlib.org/blas/>`_,
   (For best performance we recommend the free `OpenBLAS
   <https://www.openblas.net/>`_ or the nonfree `MKL
   <https://software.intel.com/en-us/intel-mkl>`_.)
 * `Tinyarray <https://gitlab.kwant-project.org/kwant/tinyarray>`_ 1.2 or newer,
a NumPy-like Python package optimized for very small arrays,
 * An environment which allows to compile Python extensions written in C and
   C++.

The following software is highly recommended though not strictly required:
 * `matplotlib <https://matplotlib.org/>`_ 3.0.2 or newer, for the module `kwant.plotter` and the tutorial,
 * `plotly <https://plotly.com/>`_ 3.6.1 or newer, for the module `kwant.plotter` and the tutorial,
 * `SymPy <https://sympy.org/>`_ 1.3.0 or newer, for the subpackage `kwant.continuum`.
 * `Qsymm <https://pypi.org/project/qsymm/>`_ 1.3.0 or newer, for the subpackage `kwant.qsymm`.
 * `MUMPS <https://graal.ens-lyon.fr/MUMPS/>`_, a sparse linear algebra library
   that will in many cases speed up Kwant several times and reduce the memory
   footprint.  (Kwant uses only the sequential, single core version
   of MUMPS.  The advantages due to MUMPS as used by Kwant are thus independent
   of the number of CPU cores of the machine on which Kwant runs.)
 * The `py.test testing framework <https://docs.pytest.org/>`_ 3.3.2 or newer for running the
   tests included with Kwant.

In addition, to build a copy of Kwant that has been checked-out directly from
version control, you will also need `Cython <https://cython.org/>`_ 0.26.1 or
newer.  You do not need Cython to build Kwant that has been unpacked from a
source .tar.gz-file.


Building and installing Kwant
=============================

Kwant can be built and installed following the `usual Python conventions
<https://docs.python.org/3/install/index.html>`_ by running the following
commands in the root directory of the Kwant distribution. ::

    python3 setup.py build
    python3 setup.py install

Depending on your system, you might have to run the second command with
administrator privileges (e.g. prefixing it with ``sudo``).

After installation, tests can be run with::

    python3 -c 'import kwant; kwant.test()'

The tutorial examples can be found in the directory ``tutorial`` inside the root
directory of the Kwant source distribution.

(Cython will be run automatically when the source tree has been checked out of
version control.  Kwant tarballs include the Cython-generated files, and
cythonization is disabled when building not from git.  If ever necessary, this
default can be overridden by giving the ``--cython`` option to setup.py.)


.. _build-configuration:

Build configuration
===================

Kwant contains several extension modules.  The compilation and linking of these
modules can be configured by editing a build configuration file.  By default,
this file is ``build.conf`` in the root directory of the Kwant distribution.  A
different path may be provided using the ``--configfile=PATH`` option.

This configuration file consists of
sections, one for each extension module that is contained in Kwant, led by a
``[section name]`` header and followed by ``key = value`` lines.

The sections bear the names of the extension modules, for example
``[kwant.operator]``.  There can be also a
``[DEFAULT]`` section that provides default values for all extensions, also
those not explicitly present in the file.

Possible keys are the keyword arguments for ``distutils.core.Extension`` (For a
complete list, see its `documentation
<https://docs.python.org/3/distutils/apiref.html#distutils.core.Extension>`_).
The corresponding values are whitespace-separated lists of strings.

Example ``build.conf`` for compiling Kwant with C assertions and Cython's line
trace feature::

    [DEFAULT]
    undef_macros = NDEBUG
    define_macros = CYTHON_TRACE=1

Kwant can optionally be linked against MUMPS.  The main
application of build configuration is adopting the build process to the various
deployments of MUMPS. MUMPS will be not linked
against by default, except on Debian-based systems when the package
``libmumps-scotch-dev`` is installed.

The section ``[kwant.linalg._mumps]`` may be used to adapt the build process.
(For simplicity and backwards compatibility, ``[mumps]`` is an aliases for the above.)


Example ``build.conf`` for linking Kwant against a self-compiled MUMPS, `SCOTCH
<https://www.labri.fr/perso/pelegrin/scotch/>`_ and `METIS
<http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_::

    [mumps]
    libraries = zmumps mumps_common pord metis esmumps scotch scotcherr mpiseq gfortran

The detailed syntax of ``build.conf`` is explained in the `documentation of
Python's configparser module
<https://docs.python.org/3/library/configparser.html#supported-ini-file-structure>`_.


Building the documentation
==========================

To build the documentation, the `Sphinx documentation generator
<https://www.sphinx-doc.org/en/stable/>`_ is required with ``numpydoc`` extension
(version 0.5 or newer), as well as ``jupyter-sphinx`` (version 0.2 or newer).
If PDF documentation is to be built, the tools
from the `libRSVG <https://wiki.gnome.org/action/show/Projects/LibRsvg>`_
(Debian/Ubuntu package ``librsvg2-bin``) and a Sphinx extension
``sphinxcontrib-svg2pdfconverter`` are needed to convert SVG drawings into the
PDF format.

As a prerequisite for building the documentation, Kwant must have been built
successfully using ``python3 setup.py build`` as described above (or Kwant must
be already installed in Python's search path).  HTML documentation is built by
entering the ``doc`` subdirectory of the Kwant package and executing ``make
html``.  PDF documentation is generated by executing ``make latex`` followed
by ``make all-pdf`` in ``doc/build/latex``.

Because of some quirks of how Sphinx works, it might be necessary to execute
``make clean`` between building HTML and PDF documentation.  If this is not
done, Sphinx may mistakenly use PNG files for PDF output or other problems may
appear.

****************************
Hints for specific platforms
****************************

Unix-like systems (GNU/Linux)
=============================

Kwant should run on all recent Unix-like systems.  The following instructions
have been verified to work on Debian 8 (Jessie) or newer, and on Ubuntu 14.04 or
newer.  For other distributions step 1 will likely have to be adapted.  If
Ubuntu-style ``sudo`` is not available, the respective command must be run as
root.

1. Install the required packages.  On Debian-based systems like Ubuntu this can
   be done by running the command ::

       sudo apt-get install python3-dev python3-setuptools python3-scipy python3-matplotlib python3-pytest python3-sympy g++ gfortran libmumps-scotch-dev

2. Unpack Tinyarray, enter its directory. To build and install, run ::

       python3 setup.py build
       sudo python3 setup.py install

3. Inside the Kwant source distribution's root directory run ::

       python3 setup.py build
       sudo python3 setup.py install

By default the package will be installed under ``/usr/local``.  Run ``python3
setup.py --help install`` for installation options.


Microsoft Windows
=================

Our efforts to compile Kwant on Windows using only free software (MinGW) were
only moderately successful.  At the end of a very complicated process we
obtained packages that worked, albeit unreliably.  As the only recommended way
to compile Python extensions on Windows is using Visual C++, it may well be that
there exists no easy solution.

It is possible to compile Kwant on Windows using non-free compilers, however we
(the authors of Kwant) have no experience with this.  The existing Windows
binary installers of Kwant and Tinyarray were kindly prepared by Christoph
Gohlke.
