# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.

__all__ = ['smatrix', 'greens_function', 'ldos', 'wave_function', 'Solver']

import numpy as np
import scipy.sparse as sp
from scipy.sparse.linalg import factorized

from . import common


class Solver(common.SparseSolver):
    "Sparse Solver class based on the sparse direct solvers provided by SciPy."
    lhsformat = 'csc'
    rhsformat = 'csc'
    nrhs = 1

    def _factorized(self, a):
        a = sp.csc_matrix(a)
        return factorized(a)

    def _solve_linear_sys(self, factorized_a, b, kept_vars):
        if b.shape[1] == 0:
            return b[kept_vars]

        sols = []
        vec = np.empty(b.shape[0], complex)
        for j in range(b.shape[1]):
            vec[:] = b[:, j].toarray().flatten()
            sols.append(factorized_a(vec)[kept_vars])

        return np.asarray(sols).transpose()


default_solver = Solver()

smatrix = default_solver.smatrix
greens_function = default_solver.greens_function
ldos = default_solver.ldos
wave_function = default_solver.wave_function
