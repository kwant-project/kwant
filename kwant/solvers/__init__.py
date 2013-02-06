# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""This package contains solvers in individual sub-packages.

One of the solvers will be imported as `default`.  Please import other
sub-packages explicitly if you need them.
"""

__all__ = ['default']
from . import default
