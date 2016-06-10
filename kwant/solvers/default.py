# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['smatrix', 'ldos', 'wave_function', 'greens_function']

# MUMPS usually works best.  Use SciPy as fallback.
import warnings
try:
    from . import mumps as smodule
except ImportError:
    warnings.warn("MUMPS is not available, "
                  "SciPy built-in solver will be used as a fallback. "
                  "Performance can be very poor in this case.", RuntimeWarning)
    from . import sparse as smodule

hidden_instance = smodule.Solver()

smatrix = hidden_instance.smatrix
ldos = hidden_instance.ldos
wave_function = hidden_instance.wave_function
greens_function = hidden_instance.greens_function
