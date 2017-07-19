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
as `PyPI <https://pypi.python.org/pypi/kwant>`_.  The sources may be also
cloned directly from the `official Kwant git repository
<https://gitlab.kwant-project.org/kwant/kwant>`_.


Prerequisites
=============

Building Kwant requires
 * `Python <https://www.python.org/>`_ 3.4 or above (Kwant 1.1 is the last
   version to support Python 2),
 * `NumPy <http://numpy.org/>`_ 1.8.1 or newer,
 * `SciPy <https://scipy.org/>`_ 0.14 or newer,
 * `LAPACK <http://netlib.org/lapack/>`_ and `BLAS <http://netlib.org/blas/>`_,
   (For best performance we recommend the free `OpenBLAS
   <http://www.openblas.net/>`_ or the nonfree `MKL
   <https://software.intel.com/en-us/intel-mkl>`_.)
 * `Tinyarray <https://gitlab.kwant-project.org/kwant/tinyarray>`_ 1.2 or newer,
a NumPy-like Python package optimized for very small arrays,
 * An environment which allows to compile Python extensions written in C and
   C++.

The following software is highly recommended though not strictly required:
 * `matplotlib <http://matplotlib.org/>`_ 1.4.2 or newer, for the module `kwant.plotter` and the tutorial,
 * `SymPy <http://sympy.org/>`_ 0.7.6 or newer, for the subpackage `kwant.continuum`.
 * `MUMPS <http://graal.ens-lyon.fr/MUMPS/>`_, a sparse linear algebra library
   that will in many cases speed up Kwant several times and reduce the memory
   footprint.  (Kwant uses only the sequential, single core version
   of MUMPS.  The advantages due to MUMPS as used by Kwant are thus independent
   of the number of CPU cores of the machine on which Kwant runs.)
 * The `py.test testing framework <http://pytest.org/>`_ 2.8 or newer for running the
   tests included with Kwant.

In addition, to build a copy of Kwant that has been checked-out directly from
version control, you will also need `Cython <http://cython.org/>`_ 0.22 or
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
``[kwant.operator]`` or ``[kwant.linalg.lapack]``.  There can be also a
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

Kwant must be linked against LAPACK & BLAS, and, optionally, MUMPS.  The main
application of build configuration is adopting the build process to the various
(deployment) variants of these libraries.  By default ``setup.py`` assumes that
LAPACK and BLAS can be found under their usual names.  MUMPS will be not linked
against by default, except on Debian-based systems when the package
``libmumps-scotch-dev`` is installed.

The sections ``[kwant.linalg.lapack]`` and ``[kwant.linalg._mumps]`` may be
used to adapt the build process.  (For simplicity and backwards compatibility,
``[lapack]`` and ``[mumps]`` are aliases for the above.)

The section ``[lapack]`` configures the linking against LAPACK _AND_ BLAS, the
section ``[mumps]`` against MUMPS.  The contents of ``[lapack]`` are
appended to the configuration for MUMPS itself needs LAPACK and BLAS as well.

Example ``build.conf`` for linking Kwant against a self-compiled MUMPS, `SCOTCH
<http://www.labri.fr/perso/pelegrin/scotch/>`_ and `METIS
<http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_::

    [mumps]
    libraries = zmumps mumps_common pord metis esmumps scotch scotcherr mpiseq gfortran

Example ``build.conf`` for linking Kwant with Intel MKL.::

    [lapack]
    libraries = mkl_intel_lp64 mkl_sequential mkl_core mkl_def
    library_dirs = /opt/intel/mkl/lib/intel64
    extra_link_args = -Wl,-rpath=/opt/intel/mkl/lib/intel64

The detailed syntax of ``build.conf`` is explained in the `documentation of
Python's configparser module
<https://docs.python.org/3/library/configparser.html#supported-ini-file-structure>`_.


Building the documentation
==========================

To build the documentation, the `Sphinx documentation generator
<http://www.sphinx-doc.org/en/stable/>`_ is required with ``numpydoc`` extension
(version 0.5 or newer).  If PDF documentation is to be built, the tools
from the `libRSVG <https://wiki.gnome.org/action/show/Projects/LibRsvg>`_ (Debian/Ubuntu package
``librsvg2-bin``) are needed to convert SVG drawings into the PDF format.

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

When ``make html`` is run, modified tutorial example scripts are executed to
update any figures that might have changed.  The machinery behind this works as
follows.  The canonical source for a tutorial script, say ``graphene.py`` is
the file ``doc/source/images/graphene.py.diff``.  This diff file contains the
information to recreate two versions of ``graphene.py``: a version that is
presented in the documentation (``doc/source/tutorial/graphene.py``), and a
version that is used to generate the figures for the documentation
(``doc/source/images/graphene.py``).  Both versions are related but differ
e.g. in the details of the plotting.  When ``make html`` is run, both versions
are extracted form the diff file.

The diff file may be modified directly.  Another possible way of working is to
directly modify either the tutorial script or the figure generation script.
Then ``make html`` will use the command line tool `wiggle
<http://neil.brown.name/wiggle/>`_ to propagate the modifications accordingly.
This will often just work, but may sometimes result in conflicts, in which case
a message will be printed.  The conflicts then have to be resolved much like
with a version control system.

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

       sudo apt-get install python3-dev python3-setuptools python3-scipy python3-matplotlib python3-pytest python3-sympy g++ gfortran libopenblas-dev liblapack-dev libmumps-scotch-dev

2. Unpack Tinyarray, enter its directory. To build and install, run ::

       python3 setup.py build
       sudo python3 setup.py install

3. Inside the Kwant source distribution's root directory run ::

       python3 setup.py build
       sudo python3 setup.py install

By default the package will be installed under ``/usr/local``.  Run ``python3
setup.py --help install`` for installation options.


Mac OS X: MacPorts
==================

The following instructions are valid for Kwant 1.1 with Python 2.7.  They need
to be updated for Kwant 1.2.  (Help is welcome.)

The required dependencies of Kwant are best installed with one of the packaging
systems. Here we only consider the case of `MacPorts
<https://www.macports.org>`_ in detail. Some remarks for homebrew are given
below.

1. Install a recent version of MacPorts, as explained in the `installation
   instructions of MacPorts <https://www.macports.org/install.php>`_.

2. Install the required dependencies::

       sudo port install gcc47 python27 py27-numpy py27-scipy py27-matplotlib mumps_seq
       sudo port select --set python python27

3. Unpack Tinyarray, enter its directory, build and install::

       python setup.py build
       sudo python setup.py install

p5. Unpack Kwant, go to the Kwant directory, and edit ``build.conf`` to read::

       [lapack]
       extra_link_args = -Wl,-framework -Wl,Accelerate
       [mumps]
       include_dirs = /opt/local/include
       library_dirs = /opt/local/lib
       libraries = zmumps_seq mumps_common_seq pord_seq esmumps scotch scotcherr mpiseq gfortran

6. Then, build and install Kwant. ::

       CC=gcc-mp-4.7 LDSHARED='gcc-mp-4.7 -shared -undefined dynamic_lookup' python setup.py build
       sudo python setup.py install

You might note that installing Kwant on Mac OS X is somewhat more involved than
installing on Linux. Part of the reason is that we need to mix Fortran and C
code in Kwant: While C code is usually compiled using Apple compilers,
Fortran code must be compiled with the Gnu Fortran compiler (there is
no Apple Fortran compiler). For this reason we force the Gnu compiler suite
with the environment variables ``CC`` and ``LDSHARED`` as shown above.


Mac OS X: homebrew
==================

The following instructions are valid for Kwant 1.1 with Python 2.7.  They need
to be updated for Kwant 1.2.  (Help is welcome.)

It is also possible to build Kwant using homebrew. The dependencies can be
installed as ::

    brew install gcc python
    brew tap homebrew/science
    brew tap homebrew/python
    brew tap kwant-project/kwant
    pip install pytest pytest-runner six
    brew install numpy scipy matplotlib

Note that during the installation you will be told which paths to add when you
want to compile/link against scotch/metis/mumps; you need to add these to the
build.conf file. Also, when linking against MUMPS, one needs also to link
against METIS (in addition to the libraries needed for MacPorts).


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
