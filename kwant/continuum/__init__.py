# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import sys

from .._common import ExtensionUnavailable

try:
    from .discretizer import discretize, discretize_symbolic, build_discretized
    from ._common import sympify, lambdify
    from ._common import momentum_operators, position_operators
except ImportError:
    sys.modules[__name__] = ExtensionUnavailable(__name__, ('sympy',))

del sys, ExtensionUnavailable

__all__ = ['discretize', 'discretize_symbolic', 'build_discretized',
           'sympify', 'lambdify']
