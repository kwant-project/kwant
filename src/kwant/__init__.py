# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.

__all__ = []

# Handle version information
try:
    # Try to get version from the generated _version.py file
    from ._version import __version__, __version_tuple__
except ImportError:
    # Fallback if _version.py is not available
    try:
        from importlib.metadata import version as _version
        from packaging.version import parse as _parse_version

        __version__ = _version("kwant")

        # Use packaging to parse the version string
        parsed_version = _parse_version(__version__)

        # Build version tuple from the parsed components
        # Start with the release segment (major, minor, micro)
        release_tuple = parsed_version.release

        # Handle pre-releases (alpha, beta, rc)
        if parsed_version.pre is not None:
            pre_type, pre_num = parsed_version.pre
            # Convert 'a', 'b', 'rc' to full words
            pre_type_map = {'a': 'alpha', 'b': 'beta', 'c': 'rc'}
            pre_type_str = pre_type_map.get(pre_type, pre_type)
            release_tuple = release_tuple + (pre_type_str, pre_num)

        # Handle dev releases
        if parsed_version.dev is not None:
            release_tuple = release_tuple + ('dev', parsed_version.dev)

        # Handle local version (like git hash)
        if parsed_version.local is not None:
            release_tuple = release_tuple + ('local', parsed_version.local)

        __version_tuple__ = release_tuple
    except Exception:
        # Last resort fallback
        __version__ = "unknown"
        __version_tuple__ = (0, 0, 0, "unknown")

import numpy                    # Needed by C. Gohlke's Windows package.
import warnings

try:
    from . import _system
except ImportError:
    import inspect
    if len(inspect.trace()) == 1:
        msg = """Error importing Kwant:
        You should not try to import Kwant from its source directory.
        Please exit the Kwant source distribution directory, and relaunch
        your Python intepreter from there."""
        raise ImportError(msg)
    else:
        raise

from ._common import KwantDeprecationWarning, UserCodeError
__all__.extend(['KwantDeprecationWarning', 'UserCodeError'])

# Pre-import most submodules.  (We make exceptions for submodules that have
# special dependencies or where that would take too long.)
from . import system
from . import builder
from . import lattice
from . import solvers
from . import digest
from . import rmt
from . import operator
from . import kpm
from . import wraparound
__all__.extend(['system', 'builder', 'lattice', 'solvers', 'digest', 'rmt',
                'operator', 'kpm', 'wraparound'])

# Make selected functionality available directly in the root namespace.
from .builder import Builder, HoppingKind
__all__.extend(['Builder', 'HoppingKind'])
from .lattice import TranslationalSymmetry
__all__.extend(['TranslationalSymmetry'])
from .solvers.default import smatrix, greens_function, ldos, wave_function
__all__.extend(['smatrix', 'greens_function', 'ldos', 'wave_function'])

# Importing plotter might not work, but this does not have to be a problem --
# only no plotting will be available.
try:
    from . import plotter
    from .plotter import plot
except:
    pass
else:
    __all__.extend(['plotter', 'plot'])

# Lazy import continuum package for backwards compatibility
from ._common import lazy_import

continuum = lazy_import('continuum', deprecation_warning=True)
__all__.append('continuum')
del lazy_import


def test(verbose=True):
    from pytest import main
    import os.path

    return main([os.path.dirname(os.path.abspath(__file__)),
                     "-s"] + (['-v'] if verbose else []))

test.__test__ = False
