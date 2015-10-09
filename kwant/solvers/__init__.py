# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""This package contains solvers in individual sub-packages.

One of the solvers will be imported as `default`.  Please import other
sub-packages explicitly if you need them.
"""

__all__ = ['default']
from . import default
