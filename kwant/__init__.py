# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.

__all__ = []

from . import version
version.ensure_python()
__version__ = version.version

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
