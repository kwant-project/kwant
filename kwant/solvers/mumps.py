# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['smatrix', 'ldos', 'wave_function', 'greens_function', 'options',
           'Solver']

import numpy as np
from . import common
from ..linalg import mumps


class Solver(common.SparseSolver):
    """Sparse Solver class based on the sparse direct solver MUMPS."""

    lhsformat = 'coo'
    rhsformat = 'csc'

    def __init__(self):
        self.nrhs = self.ordering = self.sparse_rhs = None
        self.reset_options()

    def reset_options(self):
        """Set the options to default values.  Return the old options."""
        return self.options(nrhs=6, ordering='kwant_decides', sparse_rhs=False)

    def options(self, nrhs=None, ordering=None, sparse_rhs=None):
        """
        Modify some options.  Return the old options.

        Parameters
        ----------
        nrhs : number
            number of right hand sides that should be solved simultaneously. A
            value around 5-10 gives optimal performance on many machines.  If
            memory is an issue, it can be set to 1, to minimize memory usage
            (at the cost of slower performance). Default value is 6.
        ordering : string
            one of the ordering methods supported by the MUMPS solver (see
            ``kwant.linalg.mumps``. The availability of certain orderings
            depends on the MUMPS installation.), or 'kwant_decides'.  If
            ``ordering=='kwant_decides'``, the ordering that typically gives
            the best performance is chosen from the available ones.  One can
            also defer the choice of ordering to MUMPS by specifying 'auto', in
            some cases MUMPS however chooses poorly.

            The choice of ordering can significantly influence the performance
            and memory impact of the solve phase. Typically the nested
            dissection orderings 'metis' and 'scotch' are most suited for
            physical systems. Default is 'kwant_decides'
        sparse_rhs : True or False
            whether to use a sparse right hand side in the solve phase of
            MUMPS. Preliminary tests have not shown a significant performance
            increase when this feature is used, but this needs more looking
            into. Default value is False.

        Returns
        -------
        old_options: dict
            dictionary containing the previous options.

        Notes
        -----
        Thanks to this method returning the old options as a dictionary it is
        easy to change some options temporarily:

        >>> saved_options = kwant.solvers.mumps.options(nrhs=12)
        >>> some_code()
        >>> kwant.solvers.mumps.options(**saved_options)
        """

        old_opts = {'nrhs': self.nrhs,
                    'ordering': self.ordering,
                    'sparse_rhs': self.sparse_rhs}

        if nrhs is not None:
            if nrhs < 1 and int(nrhs) != nrhs:
                raise ValueError("nrhs must be an integer bigger than zero")
            nrhs = int(nrhs)
            self.nrhs = nrhs

        if ordering is not None:
            if ordering == 'kwant_decides':
                # Choose what is considered to be the best ordering.
                sorted_orderings = [order
                                    for order in ['metis', 'scotch', 'auto']
                                    if order in mumps.possible_orderings()]
                ordering = sorted_orderings[0]
            elif ordering not in mumps.orderings:
                raise ValueError("Invalid ordering: " + ordering)
            self.ordering = ordering

        if sparse_rhs is not None:
            self.sparse_rhs = bool(sparse_rhs)

        return old_opts

    def _factorized(self, a):
        inst = mumps.MUMPSContext()
        inst.factor(a, ordering=self.ordering)
        return inst

    def _solve_linear_sys(self, factorized_a, b, kept_vars):
        if b.shape[1] == 0:
            return b[kept_vars]

        solve = factorized_a.solve
        sols = []

        for j in range(0, b.shape[1], self.nrhs):
            tmprhs = b[:, j:min(j + self.nrhs, b.shape[1])]

            if not self.sparse_rhs:
                tmprhs = tmprhs.toarray()
            sols.append(solve(tmprhs)[kept_vars, :])

        return np.concatenate(sols, axis=1)


default_solver = Solver()

smatrix = default_solver.smatrix
greens_function = default_solver.greens_function
ldos = default_solver.ldos
wave_function = default_solver.wave_function
options = default_solver.options
reset_options = default_solver.reset_options
