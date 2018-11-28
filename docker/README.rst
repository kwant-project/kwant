====================
Kwant Testing Images
====================
This directory contains files for building Docker images used in
testing Kwant. The Dockerfile for a given platforms can also
be understood as recipes for how to install the prerequesites of
Kwant on that platform. To build an image it is enough to execute::

    docker build -f <dockerfile>

from this directory (where ``<dockerfile>`` should be replaced with the
filename of the appropriate dockerfile to build).

Dockerfiles
===========

Dockerfile.debian
  Builds an environment based on Debian stable. Python 3 and all the dependencies
  of Kwant are installed using the Debian repositories. Includes all optional
  dependencies (for plotting and the continuum module).

Dockerfile.ubuntu
  Builds an environment based on the latest Ubuntu LTS release that is supported
  by Kwant (we typically wait until 6 months after a new LTS release to upgrade
  Kwant's requirements, to give people time to switch). Python 3 and
  all the dependencies of Kwant are installed using the Ubuntu repositories.
  Includes all optional dependencies (for plotting and the continuum module).

Dockerfile.conda
  Builds an environment that contains Miniconda and a minimal number of
  system-installed packages (mainly to make Git and documentation building work).
  Individual testing environments are installed as conda environments (see
  the next section for details).

Conda Environments
==================
These conda environments are specified in Yaml files, and are loaded into
the conda-based Docker image. All the dependencies are loaded from the
`conda-forge <https://anaconda.org/conda-forge/>`_ Anaconda channel.

kwant-stable-no-extras
  The minimal environment in which Kwant can run. Pins all the dependencies to
  the oldest supported versions, and does not include optional dependencies.

kwant-stable
  Pins all the dependencies to the oldest supported versions, including all
  optional dependencies.

kwant-latest
  References all Kwant dependencies, but does not pin any versions, meaning
  that the latest released versions of the dependencies are used (at least,
  the latest released versions on conda-forge)
