__all__ = ['system', 'version', 'builder', 'lattice', 'run']
for module in __all__:
    exec 'from . import {0}'.format(module)

from .version import version as __version__

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

# Mumps usally works best, use sparse as fallback
try:
    from .solvers.mumps import solve
    __all__.append('solve')
except ImportError:
    from .solvers.sparse import solve
    __all__.append('solve')

def runtest():
    """Run the tests included with kwant."""
    import os.path
    try:
        import nose
    except ImportError:
        msg = 'No module named nose.\n(See INSTALL.txt in ' \
            'the top-level directory of the kwant distribution.)'
        raise ImportError(msg)
    kwant_dir = os.path.dirname(__file__)
    if os.path.basename(kwant_dir) == kwant_dir:
        msg = 'Do not call kwant.test from within the kwant source ' \
        'distribution directory.\nYou can run "nosetests -s" instead.'
        raise RuntimeError(msg)
    nose.run(defaultTest=kwant_dir)

# This trick avoids recursive execution of tests by nosetests.
test = runtest
