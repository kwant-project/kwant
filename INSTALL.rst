=========================
Installation instructions
=========================

Kwant can be installed either using prepared packages (Debian and Ubuntu
variants of GNU/Linux, Mac OS X, and Windows), or it can be built and installed
from source.

In general, installation from packages is advisable, especially for novice
users.  Expert users may find it helpful to build Kwant from source, as this
will also allow them to customize Kwant to use certain optimized versions of
libraries.


************************
Installing from packages
************************

Ubuntu (and derivatives)
========================

Execute the following commands::

    sudo apt-add-repository ppa:kwant-project/ppa
    sudo apt-get update
    sudo apt-get install python-kwant python-kwant-doc

This should provide Kwant for all versions of Ubuntu >= 12.04.  The HTML
documentation will be installed locally in the directory
``/usr/share/doc/python-kwant-doc``.


Debian (and derivatives)
========================

The easiest way to install Kwant on a Debian system is using the pre-built
packages we provide.  Our packages are known to work with Debian "wheezy" and
Debian "jessie", but they may also work on many other recent Debian-derived
sytems as well.  (For example, the following works with recent Ubuntu versions.)

The lines prefixed with ``sudo`` have to be run as root.

1. Add the following lines to ``/etc/apt/sources.list``::

       deb http://downloads.kwant-project.org/debian/ stable main
       deb-src http://downloads.kwant-project.org/debian/ stable main

2. (Optional) Add the OpenPGP key used to sign the repositories by executing::

       sudo apt-key adv --keyserver pgp.mit.edu --recv-key C3F147F5980F3535

3. Update the package data, and install Kwant::

       sudo apt-get update
       sudo apt-get install python-kwant python-kwant-doc

   The ``python-kwant-doc`` package is optional and installs the HTML
   documentation of Kwant in the directory ``/usr/share/doc/python-kwant-doc``.

Should the last command (``apt-get install``) fail due to unresolved
dependencies, you can try to build and install your own packages, which is
surprisingly easy::

    cd /tmp

    sudo apt-get build-dep tinyarray
    apt-get source --compile tinyarray
    sudo dpkg -i python-tinyarray_*.deb

    sudo apt-get build-dep kwant
    apt-get source --compile kwant
    sudo dpkg -i python-kwant_*.deb python-kwant-doc_*.deb

This method should work for virtually all Debian-derived systems, even on exotic
architectures.


Windows
=======

There are multiple distributions of scientific Python software for Windows that
provide the prerequisites for Kwant.  We recommend to use the packages kindly
provided by Christoph Gohlke.  To install Kwant on Windows

1. Determine whether you have a 32-bit or 64-bit Windows installation by
   following these `instructions <http://support.microsoft.com/kb/827218>`_.

2. Download and install Python 2.7 for the appropriate architecture (32-bit or
   64-bit) from the official `Python download site
   <http://www.python.org/download/>`_.

3. Download and install ``scipy-stack``, ``tinyarray``, and ``kwant`` for Python
   2.7 from `Christoph Gohlke's page
   <http://www.lfd.uci.edu/~gohlke/pythonlibs/>`_.  Once again you should choose
   the architecture that is appropriate for your system.  ("win32" means 32-bit,
   "amd64" means 64-bit -- even if you have a processor from Intel.)  If the
   download from Gohlke's site is slow, try to download from `our mirror
   <http://downloads.kwant-project.org/gohlke-mirror/>`_.

   You may see a warning that says "The publisher could not be verified. Do you
   want to run this software?". Select "Run".


Mac OS X
========

There is a number of different package managers for bringing software from the
Unix/Linux world to Mac OS X. Since the community is quite split, we provide
Kwant and its dependencies both via the `homebrew <http://brew.sh>`_ and the
`MacPorts <http://www.macports.org>`_ systems.


Mac OS X: homebrew
==================

homebrew is a recent addition to the package managers on Mac OS X. It is
lightweight, tries to be as minimalistic as possible and give the user
freedom than Macports. We recommend this option if you have no preferences.

1. Open a terminal and install homebrew as described on the `homebrew
   homepage <http://brew.sh>`_ (instructions are towards the end of
   the page)

2. Run ::

       brew doctor

   and follow its directions. It will ask for a few prerequisites to be
   installed, in particular

  * the Xcode developer tools (compiler suite for Mac OS X) from
    `<http://developer.apple.com/downloads>`_. You will need an Apple ID to
    download. Note that if you have one already from using the App store on the
    Mac/Ipad/Iphone/... you can use that one. Downloading the command line
    tools (not the full Xcode suite) is sufficient. If you have the full Xcode
    suite installed, you might need to download the command line tools manually
    if you have version 4 or higher. In this case go to `Xcode->Preferences`,
    click on `Download`, go to `Components`, select `Command Line Tools` and
    click on `Install`.
  * although `brew doctor` might not complain about it right away, while we're
    at it, you should also install the X11 server from the `XQuartz project
    <http://xquartz.macosforge.org>`_ if you have Mac OS X 10.8 or higher.

3. Add permanently ``/usr/local/bin`` before ``/usr/bin/`` in the ``$PATH$``
   environment variable of your shell, for example by adding ::

       export PATH=/usr/local/bin:$PATH

   at the end of your ``.bash_profile`` or ``.profile``. Then close
   the terminal and reopen it again.

4. Install a few prerequisites ::

       brew install gfortran python

5. Add additional repositories ::

       brew tap homebrew/science
       brew tap samueljohn/python
       brew tap michaelwimmer/kwant

6. Install Kwant and its prerequisites ::

       pip install nose
       brew install numpy scipy matplotlib
       brew install kwant

Notes:

- If something does not work as expected, use ``brew doctor`` for
  instructions (it will find conflicts and things like that).
- As mentioned, homebrew allows for quite some freedom. In particular,
  if you are an expert, you don't need necessarily to install
  numpy/scipy/matplotlib from homebrew, but can use your own installation.
  The only prerequisite is that they are importable from python. (the
  Kwant installation will in any case complain if they are not)
- In principle, you need not install the homebrew python, but could use
  Apple's already installed python. Homebrew's python is more up-to-date,
  though.


Mac OS X: MacPorts
==================

MacPorts is a full-fledged package manager that recreates a whole Linux-like
environment on your Mac.

In order to install Kwant using MacPorts, you have to

1. Install a recent version of MacPorts, as explained in the
   `installation instructions of MacPorts
   <http://www.macports.org/install.php>`_.
   In particular, as explained there, you will have to install also a
   few prerequisites, namely

  * the Xcode developer tools (compiler suite for Mac OS X) from
    `<http://developer.apple.com/downloads>`_. You will need an Apple ID to
    download. Note that if you have one already from using the App store
    on the Mac/Ipad/Iphone/... you can use that one. You will also need the
    command line tools: Within Xcode 4, you have to download them by going to
    `Xcode->Preferences`, click on `Download`, go to `Components`,
    select `Command Line Tools` and click on `Install`. Alternatively, you can
    also directly download the command line tools from the
    Apple developer website.
  * if you have Mac OS X 10.8 or higher, the X11 server from the
    `XQuartz project <http://xquartz.macosforge.org>`_.

2. After the installation, open a terminal and execute ::

       echo http://downloads.kwant-project.org/macports/ports.tar |\
       sudo tee -a /opt/local/etc/macports/sources.conf >/dev/null

   (this adds the Kwant MacPorts download link
   `<http://downloads.kwant-project.org/macports/ports.tar>`_ at the end of the
   ``sources.conf`` file.)

3. Execute ::

       sudo port selfupdate

4. Now, install Kwant and its prerequisites ::

       sudo port install py27-kwant

5. Finally, we choose python 2.7 to be the default python ::

       sudo port select --set python python27

   After that, you will need to close and reopen the terminal to
   have all changes in effect.

Notes:

* If you have problems with macports because your institution's firewall
  blocks macports (more precisely, the `rsync` port), resulting in
  errors from ``sudo port selfupdate``, follow
  `these instructions <https://trac.macports.org/wiki/howto/PortTreeTarball>`_.
* Of course, if you already have macports installed, you can skip step 1
  and continue with step 2.


***********************************
Building and installing from source
***********************************

Prerequisites
=============

Building Kwant requires
 * `Python <http://python.org>`_ 2.6 or 2.7 (Python 3 is not supported yet),
 * `SciPy <http://scipy.org>`_ 0.9 or newer,
 * `LAPACK <http://netlib.org/lapack/>`_ and `BLAS <http://netlib.org/blas/>`_,
   (For best performance we recommend the free `OpenBLAS
   <http://xianyi.github.com/OpenBLAS/>`_ or the nonfree `MKL
   <http://software.intel.com/en-us/intel-mkl>`_.)
 * `Tinyarray <http://git.kwant-project.org/tinyarray/about/>`_, a NumPy-like
   Python package optimized for very small arrays,
 * An environment which allows to compile Python extensions written in C and
   C++.

The following software is highly recommended though not strictly required:
 * `matplotlib <http://matplotlib.sourceforge.net/>`_ 1.1 or newer, for Kwant's
   plotting module and the tutorial,
 * `MUMPS <http://graal.ens-lyon.fr/MUMPS/>`_, a sparse linear algebra library
   that will in many cases speed up Kwant several times and reduce the memory
   footprint.  (Kwant uses only the sequential, single core version
   of MUMPS.  The advantages due to MUMPS as used by Kwant are thus independent
   of the number of CPU cores of the machine on which Kwant runs.)
 * The `nose <http://nose.readthedocs.org/>`_ testing framework for running the
   tests included with Kwant.

In addition, to build a copy of Kwant that has been checked-out directly from
`its Git repository <http://git.kwant-project.org/kwant>`_, you will also need
`Cython <http://cython.org/>`_ 0.17.1 or newer.  You do not need Cython to build
Kwant that has been unpacked from a source .tar.gz-file.


Generic instructions
====================

Kwant can be built and installed following the `usual Python conventions
<http://docs.python.org/install/index.html>`_ by running the following commands
in the root directory of the Kwant distribution. ::

    python setup.py build
    python setup.py install

Depending on your system, you might have to run the second command with
administrator privileges (e.g. prefixing it with ``sudo``).

After installation, tests can be run with::

    python -c 'import kwant; kwant.test()'

The tutorial examples can be found in the directory ``tutorial`` inside the root
directory of the Kwant source distribution.


Unix-like systems (GNU/Linux)
=============================

Kwant should run on all recent Unix-like systems.  The following instructions
have been verified to work on Debian 7 (Wheezy) or newer, and on Ubuntu 12.04 or
newer.  For other distributions step 1 will likely have to be adapted.  If
Ubuntu-style ``sudo`` is not available, the respective command must be run as
root.

1. Install the required packages.  On Debian-based systems like Ubuntu this can
   be done by running the command ::

       sudo apt-get install python-dev python-scipy python-matplotlib python-nose g++ gfortran libopenblas-dev liblapack-dev libmumps-scotch-dev

2. Unpack Tinyarray, enter its directory. To build and install, run ::

       python setup.py build
       sudo python setup.py install

3. Inside the Kwant source distribution's root directory run ::

       python setup.py build
       sudo python setup.py install

By default the package will be installed under ``/usr/local``.  You can
change this using the ``--prefix`` option, e.g.::

    sudo python setup.py install --prefix=/opt

If you would like to install Kwant into your home directory only you can use ::

    python setup.py install --home=~

This does not require root privileges.  If you install Kwant in this way
be sure to tell python where to find it.  This can be done by setting the
``PYTHONPATH`` environment variable::

    export PYTHONPATH=$HOME/lib/python

You can make this setting permanent by adding this line to the file
``.bashrc`` (or equivalent) in your home directory.


Mac OS X: MacPorts
==================

The required dependencies of Kwant are best installed with one of the packaging
systems. Here we only consider the case of `MacPorts
<http://www.macports.org>`_ in detail. Some remarks for homebrew are given
below.

1. In order to set up MacPorts or homebrew, follow steps 1 - 3 of
   the respective instructions of `MacPorts`_

2. Install the required dependencies::

       sudo port install gcc47 python27 py27-numpy py27-scipy py27-matplotlib mumps_seq
       sudo port select --set python python27

3. Unpack Tinyarray, enter its directory, build and install::

       python setup.py build
       sudo python setup.py install

5. Unpack Kwant, go to the Kwant directory, and edit ``build.conf`` to read::

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

It is also possible to build Kwant using homebrew. The dependencies can be
installed as ::

    brew install gcc python
    brew tap homebrew/science
    brew tap homebrew/python
    brew tap michaelwimmer/kwant
    pip install nose six
    brew install numpy scipy matplotlib

Note that during the installation you will be told which paths to add when you
want to compile/link against scotch/metis/mumps; you need to add these to the
build.conf file. Also, when linking against mumps, one needs also to link
against metis (in addition to the libraries needed for MacPorts).


Windows
=======

Our efforts to compile Kwant on Windows using only free software (MinGW) were
only moderately successful.  At the end of a very complicated process we
obtained packages that worked, albeit unreliably.  As the only recommended way
to compile Python extensions on Windows is using Visual C++, it may well be that
there exists no easy solution.

It is possible to compile Kwant on Windows using non-free compilers, however we
(the authors of Kwant) have no experience with this.  The existing Windows
binary installers of Kwant and Tinyarray were kindly prepared by Christoph
Gohlke.


Build configuration
===================

The setup script of Kwant has to know how to link against LAPACK & BLAS, and,
optionally, MUMPS.  By default it will assume that LAPACK and BLAS can be found
under their usual names.  MUMPS will be not linked against by default, except
on Debian-based systems when the package ``libmumps-scotch-dev`` is installed.

All these settings can be configured by creating/editing the file
``build.conf`` in the root directory of the Kwant distribution.  This
configuration file consists of sections, one for each dependency, led by a
[dependency-name] header and followed by name = value entries.  Possible names
are keyword arguments for ``distutils.core.Extension`` (For a complete list,
see its `documentation
<http://docs.python.org/2/distutils/apiref.html#distutils.core.Extension>`_).
The corresponding values are whitespace-separated lists of strings.

The two currently possible sections are [lapack] and [mumps].  The former
configures the linking against LAPACK _AND_ BLAS, the latter against MUMPS
(without LAPACK and BLAS).

Example ``build.conf`` for linking Kwant against a self-compiled MUMPS, `SCOTCH
<http://www.labri.fr/perso/pelegrin/scotch/>`_ and `METIS
<http://glaros.dtc.umn.edu/gkhome/metis/metis/overview>`_::

    [mumps]
    libraries = zmumps mumps_common pord metis esmumps scotch scotcherr mpiseq
        gfortran

Example ``build.conf`` for linking Kwant with Intel MKL.::

    [lapack]
    libraries = mkl_intel_lp64 mkl_sequential mkl_core mkl_def
    library_dirs = /opt/intel/mkl/lib/intel64
    extra_link_args = -Wl,-rpath=/opt/intel/mkl/lib/intel64

The detailed syntax of ``build.conf`` is explained in the `documentation of
Python's configparser module
<http://docs.python.org/3/library/configparser.html#supported-ini-file-structure>`_.


Building the documentation
==========================

To build the documentation, the `Sphinx documentation generator
<http://sphinx.pocoo.org/>`_ is required with ``numpydoc`` extension
(version 0.5 or newer).  If PDF documentation is to be built, the tools
from the `libRSVG <http://live.gnome.org/LibRsvg>`_ (Debian/Ubuntu package
``librsvg2-bin``) are needed to convert SVG drawings into the PDF format.

As a prerequisite for building the documentation, Kwant must have been built
successfully using ``./setup.py build`` as described above (or Kwant must be
already installed in Python's search path).  HTML documentation is built by
entering the ``doc`` subdirectory of the Kwant package and executing ``make
html``.  PDF documentation is generated by executing ``make latex`` followed by
``make all-pdf`` in ``doc/build/latex``.

Because of some quirks of how Sphinx works, it might be necessary to execute
``make clean`` between building HTML and PDF documentation.  If this is not
done, Sphinx may mistakenly use PNG files for PDF output or other problems may
appear.
