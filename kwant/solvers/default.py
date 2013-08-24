# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['smatrx', 'ldos', 'wave_function', 'greens_function']

# MUMPS usually works best.  Use SciPy as fallback.
try:
    from . import mumps as smodule
except ImportError:
    from . import sparse as smodule

hidden_instance = smodule.Solver()

smatrix = hidden_instance.smatrix
ldos = hidden_instance.ldos
wave_function = hidden_instance.wave_function
greens_function = hidden_instance.greens_function
