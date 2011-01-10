__all__ = ['system', 'version', 'builder', 'lattice', 'run']
for module in __all__:
    exec 'from . import {0}'.format(module)

from .builder import Builder
__all__.append('Builder')

from .lattice import make_lattice, TranslationalSymmetry
__all__.extend(['make_lattice', 'TranslationalSymmetry'])

# Importing plotter might not work, but this does not have to be a problem --
# only no plotting will be available.
try:
    from . import plotter
    from .plotter import plot
except:
    pass
else:
    __all__.extend(['plotter', 'plot'])

from .solvers.sparse import solve
__all__.append('solve')
