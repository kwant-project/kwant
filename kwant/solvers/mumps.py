"""Implementation of the sparse solver framework using the direct sparse solver
MUMPS (http://graal.ens-lyon.fr/MUMPS/, only the sequential, single core
version is used).

MUMPS is a very efficient direct sparse solver that can take advantage of
memory beyond 3GB for the solution of large problems.  Furthermore, it offers a
choice of several orderings of the input matrix from which can speed up a
calculation significantly.

Compared to the generic sparse solver framework, `mumps` adds the following
control options that may affect performance:

- `ordering`: a fill-in reducing ordering of the matrix
- `nrhs`: number of right hand sides that should be solved simultaneously
- `sparse_rhs`: whether to use dense or sparse right hand sides

For more details see `~Solver.options`.
"""

__all__ = ['solve', 'ldos', 'options', 'Solver']

import numpy as np
import scipy.sparse as sp
from . import common
from ..linalg import mumps


class Solver(common.SparseSolver):
    """Sparse Solver class based on the sparse direct solver MUMPS.
    """

    lhsformat = 'coo'
    rhsformat = 'csc'
    nrhs = 1     # later governed by options(), reset_options()

    def __init__(self):
        self.nrhs = self.ordering = self.sparse_rhs = None
        self.reset_options()

    def reset_options(self):
        """Set the options to default values."""
        self.options(nrhs=6, ordering='kwant_decides', sparse_rhs=False)

    def options(self, nrhs=None, ordering=None, sparse_rhs=None):
        """
        Parameters
        ----------
        nrhs : number
            number of right hand sides that should be solved simultaneously. A
            value around 5-10 gives optimal performance on many machines.  If
            memory is an issue, it can be set to 1, to minimize memory usage
            (at the cost of slower performance). Default value is 6.
        ordering : string
            one of the ordering methods supported by the MUMPS solver (see
            `~kwant.linalg.mumps`. The availability of certain orderings
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
            if ordering not in mumps.orderings.keys() + ['kwant_decides']:
                raise ValueError("Invalid ordering: " + ordering)
            if ordering == 'kwant_decides':
                # Choose what is considered to be the best ordering.
                sorted_orderings = [ordering
                                    for ordering in ['metis', 'scotch', 'auto']
                                    if ordering in mumps.possible_orderings()]
                ordering = sorted_orderings[0]
            self.ordering = ordering

        if sparse_rhs is not None:
            self.sparse_rhs = bool(sparse_rhs)

        return old_opts

    def solve_linear_sys(self, a, b, kept_vars=None, factored=None):

        """
        Solve matrix system of equations a x = b with sparse input,
        using MUMPS.

        Parameters
        ----------
        a : a scipy.sparse.coo_matrix sparse matrix.
        b : a list of scipy.sparse.csc_matrices.
        kept_vars : list of integers
            a list of numbers of variables to keep in the solution.
        factored : a factorized lhs as returned by a previous call
            to solve_linear_sys with the same lhs.

        Returns
        -------
        output : a numpy matrix
            solution to the system of equations.
        factored : factorized lhs. Can be reused in later solves with
            the same lhs, but different rhs.
        """

        if kept_vars == None:
            kept_vars = [range(a.shape[1])]

        sols = []

        if not factored:
            inst = mumps.MUMPSContext()
            inst.factor(a, ordering=self.ordering)
        else:
            inst = factored

        for mat in b:
            if mat.shape[1] != 0:
                # See comment about zero-shaped sparse matrices at the top
                # of sparse.
                mat = sp.csr_matrix(mat)

            for j in xrange(0, mat.shape[1], self.nrhs):
                jend = min(j + self.nrhs, mat.shape[1])

                if self.sparse_rhs:
                    sols.append(inst.solve(mat[:, j:jend])[kept_vars, :])
                else:
                    sols.append(inst.solve(mat[:, j:jend].todense())
                                [kept_vars, :])

        if len(sols):
            return np.concatenate(sols, axis=1), inst
        else:
            return np.zeros(shape=(len(kept_vars), 0)), inst


default_solver = Solver()

solve = default_solver.solve
ldos = default_solver.ldos
options = default_solver.options
reset_options = default_solver.reset_options
