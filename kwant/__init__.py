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
