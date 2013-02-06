# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['system', 'version', 'builder', 'lattice', 'solvers']
for module in __all__:
    exec 'from . import {0}'.format(module)

from .version import version as __version__

from .builder import Builder
__all__.append('Builder')

from .lattice import make_lattice, TranslationalSymmetry
__all__.extend(['make_lattice', 'TranslationalSymmetry'])

# Make kwant.solvers.default.solve available as kwant.solve.
solve = solvers.default.solve
__all__.extend(['solvers', 'solve'])

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
