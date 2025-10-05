=====================
Installation of Kwant
=====================

.. note::

  If you just want to use Kwant, see the `installation page of the Kwant
  website <https://kwant-project.org/install>`_ for instructions on how to install Kwant
  on your platform.  This is the recommended way for new users.

The remainder of this section documents how to build Kwant from source.  This
information is mostly of interest to contributors and packagers.

*************************
Obtaining the source code
*************************

Source distributions of Kwant (and Tinyarray) are available at the `downloads
section of the Kwant website <https://downloads.kwant-project.org/kwant/>`_ as well
as `PyPI <https://pypi.org/project/kwant>`_.  The sources may be also
cloned directly from the `official Kwant git repository
<https://gitlab.kwant-project.org/kwant/kwant>`_.


***********************
Development environment
***********************

The fastest way to set up a development environment for Kwant is to use `pixi
<https://pixi.sh/>`_.

After installing pixi and obtaining the Kwant source code, run::

    pixi run test-all

This will:

* Configure isolated environments for all supported Kwant dependencies
* Build and install Kwant in each environment
* Run all tests in each environment

The environments will be cached by pixi, so subsequent runs will be much
faster.

Similarly, to build the documentation, run::

    pixi run build-docs

.. _build-configuration:

********************
Generic instructions
********************

Prerequisites
=============

System dependencies
-------------------

Building Kwant from source requires a standard scientific Python toolchain with
the following system components:

* A supported Python interpreter together with its development headers.  The
  supported versions are listed in ``pyproject.toml``.
* A C compiler with C11 support and a C++ compiler with C++14 support (for
  example GCC or Clang), plus the usual build utilities such as ``pkg-config``.
* The Meson build system and Ninja build tool available on ``PATH``.  Meson is
  also responsible for driving Cython compilation as part of the build.
* `Cython <https://cython.org/>`_ is needed when building from a version control
  checkout.  Release tarballs already include the generated C files.

Python requirements
-------------------

All Python runtime dependencies, optional extras, and build requirements are
declared in ``pyproject.toml``.  Install the sets you need with your preferred
environment manager (``pip``, ``pixi``, ``conda``, ...).

Optional components such as plotting support, continuum helpers, symmetry
tools, or the MUMPS-based linear solver are provided through extras also listed
in ``pyproject.toml``.  For MUMPS support install the ``python-mumps`` package
from PyPI (or your distribution, if available) and ensure that the MUMPS
libraries are present on your machine.


Building and installing Kwant
=============================

The recommended way to build Kwant in a generic environment is to let
``pip`` drive the ``meson-python`` backend.  Ensure that the required
dependencies from ``pyproject.toml`` are provided by your environment, then run
in the root of the source tree::

  python -m pip install --no-build-isolation .

The ``--no-build-isolation`` flag makes ``pip`` reuse the packages available in
your environment (in particular it is important to build Kwant using the
correct ``numpy`` version), so please install them beforehand. Add extras such
as ``.[plotting,continuum]`` if you need optional features.

For development work, perform an editable install instead::

  python -m pip install --no-build-isolation -e .

Editable installs keep the source tree and the installed package in sync.  Add
optional extras explicitly (for example ``-e .[plotting]``) based on the groups
declared in ``pyproject.toml``.  Install testing tools such as ``pytest`` if you
intend to run the test suite.

If you prefer to invoke Meson manually, activate the Python environment into
which Kwant should be installed and run::

  meson setup build --prefix="$(python -c 'import sys; print(sys.prefix)')" --buildtype=release
  meson compile -C build
  meson install -C build

This uses the same interpreter Meson detected during configuration, installs
into its prefix (for example a virtual environment), and mirrors the layout
produced by ``pip``/``meson-python``.

After installing, run the test suite from the repository root with::

    python -m pytest src/kwant


Building the documentation
==========================

To build the documentation install the packages listed under
``[tool.pixi.feature.docs.dependencies]`` in ``pyproject.toml`` (``make``,
``sphinx``, ``sphinx-book-theme``, ``numpydoc``, ``requests``,
``sphinx-togglebutton``, and ``jupyter-sphinx``).  The
``sphinxcontrib-svg2pdfconverter`` package listed in
``[tool.pixi.feature.docs.pypi-dependencies]`` is required for PDF builds.  If
PDF documentation is to be generated you also need the tools from the
`libRSVG <https://wiki.gnome.org/Projects/LibRsvg>`_
(Debian/Ubuntu package ``librsvg2-bin``) to convert SVG drawings into PDF.

As a prerequisite for building the documentation, Kwant must already be built
successfully (for example via ``python -m pip install .`` or a manual Meson
build) or otherwise installed on Python's search path.  HTML documentation is
built by
entering the ``doc`` subdirectory of the Kwant package and executing ``make
html``.  PDF documentation is generated by executing ``make latex`` followed
by ``make all-pdf`` in ``doc/build/latex``.

*****************************
Unix-like systems (GNU/Linux)
*****************************

Kwant should run on all recent Unix-like systems.  The following instructions
have been verified to work on Debian 13 (Trixie) or newer, and on Ubuntu 25.04 or
newer.  For other distributions step 1 will likely have to be adapted.

1. Install the required packages (these are both runtime and build dependencies):

   .. code-block:: console

      sudo apt-get update && sudo apt-get install --no-install-recommends \
         build-essential python3 python3-venv python3-pip python3-dev pkg-config \
         ninja-build meson python3-mesonpy cmake python3-numpy python3-scipy \
         python3-tinyarray python3-packaging cython3 python3-setuptools-scm \
         python3-pytest

   Install optional packages such as plotting libraries if you need them.

2. Create and activate a virtual environment that can access the
   distribution-provided packages:

   .. code-block:: console

      python3 -m venv --system-site-packages kwant-venv
      source kwant-venv/bin/activate

3. From the root of the Kwant source tree install the package using the
   pre-installed dependencies:

   .. code-block:: console

      pip install --no-build-isolation .

   Use ``-e .`` instead if you need an editable install, and add extras such as
   ``.[docs]`` when necessary.  This approach avoids downloading wheels because
   ``pip`` reuses the packages supplied by Debian.

4. (Optional) Verify the installation and run the tests:

   .. code-block:: console

      python -c "import kwant; print(kwant.__version__)"
      python -m pytest src/kwant

By default the package is installed inside the virtual environment.  Adjust the
commands above if you prefer a different layout.
