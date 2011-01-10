__all__ = ['lapack']
from . import lapack

# Merge the public interface of the other submodules.
for module in ['decomp_lu', 'decomp_ev', 'decomp_schur']:
    exec 'from . import {0}'.format(module)
    exec 'from .{0} import *'.format(module)
    exec '__all__.extend({0}.__all__)'.format(module)
