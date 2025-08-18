# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.


# For obtaining version information
from importlib.metadata import version as _pkg_version
from packaging.version import Version

from ._common import KwantDeprecationWarning, UserCodeError
# Make selected functionality available directly in the root namespace.
from . import builder, digest, kpm, lattice, operator, rmt, solvers, system, wraparound
from .builder import Builder, HoppingKind
from .lattice import TranslationalSymmetry
from .solvers.default import greens_function, ldos, smatrix, wave_function
# Lazy import continuum package for backwards compatibility
from ._common import lazy_import

# Importing plotter might not work, but this does not have to be a problem --
# only no plotting will be available.
try:
    from . import plotter
    from .plotter import plot
except ImportError:
    _plotting_available = False
else:
    _plotting_available = True

continuum = lazy_import('continuum', deprecation_warning=True)
del lazy_import

__version__ = _pkg_version('kwant')
_v = Version(__version__)
__version_tuple__ = _v.release + (_v.pre if _v.pre else ())


def test(verbose=True):
    import os.path

    from pytest import main

    return main([os.path.dirname(os.path.abspath(__file__)),
                     "-s"] + (['-v'] if verbose else []))

test.__test__ = False


__all__ = [
    "system",
    "builder",
    "lattice",
    "solvers",
    "digest",
    "rmt",
    "operator",
    "kpm",
    "wraparound",
    "KwantDeprecationWarning",
    "UserCodeError",
    "__version__",
    "__version_tuple__",
    "Builder",
    "HoppingKind",
    "TranslationalSymmetry",
    "smatrix",
    "greens_function",
    "ldos",
    "wave_function",
    "plotter",
    "plot",
    "continuum",
    "test"
]

if not _plotting_available:
    __all__.remove('plotter')
    __all__.remove('plot')
