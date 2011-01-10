"""Physics-related algorithms"""

# Merge the public interface of all submodules.
__all__ = []
for module in ['selfenergy']:
    exec 'from . import {0}'.format(module)
    exec 'from .{0} import *'.format(module)
    exec '__all__.extend({0}.__all__)'.format(module)
