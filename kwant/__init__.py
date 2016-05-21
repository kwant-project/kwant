# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = []

import numpy                    # Needed by C. Gohlke's Windows package.

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

from ._common import version as __version__

for module in ['system', 'builder', 'lattice', 'solvers', 'digest', 'rmt']:
    exec('from . import {0}'.format(module))
    __all__.append(module)

# Make selected functionality available directly in the root namespace.
available = [('builder', ['Builder', 'HoppingKind']),
             ('lattice', ['TranslationalSymmetry']),
             ('solvers.default',
              ['smatrix', 'greens_function', 'ldos', 'wave_function'])]
for module, names in available:
    exec('from .{0} import {1}'.format(module, ', '.join(names)))
    __all__.extend(names)

# Importing plotter might not work, but this does not have to be a problem --
# only no plotting will be available.
try:
    from . import plotter
    from .plotter import plot
except:
    pass
else:
    __all__.extend(['plotter', 'plot'])


def test(verbose=1):
    from nose.core import run
    import os.path

    return run(argv=[__file__, os.path.dirname(os.path.abspath(__file__)),
                     "-s", "--verbosity="+str(verbose)])

test.__test__ = False
