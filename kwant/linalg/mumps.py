# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Interface to the MUMPS sparse solver library"""

__all__ = ['MUMPSContext', 'schur_complement', 'AnalysisStatistics',
           'FactorizationStatistics', 'MUMPSError']

import time
import numpy as np
import scipy.sparse
import warnings
from . import _mumps
from .fortran_helpers import prepare_for_fortran

orderings = { 'amd' : 0, 'amf' : 2, 'scotch' : 3, 'pord' : 4, 'metis' : 5,
              'qamd' : 6, 'auto' : 7 }

ordering_name = [ 'amd', 'user-defined', 'amf',
                  'scotch', 'pord', 'metis', 'qamd']


def possible_orderings():
    """Return the ordering options that are available in the current
    installation of MUMPS.

    Which ordering options are actually available depends how MUMPs was
    compiled. Note that passing an ordering that is not avaialble in the
    current installation of MUMPS will not fail, instead MUMPS will fall back
    to a supported one.

    Returns
    -------
    orderings : list of strings
       A list of installed orderings that can be used in the `ordering` option
       of MUMPS.
    """

    if not possible_orderings.cached:
        # Try all orderings on a small test matrix, and check which one was
        # actually used.

        possible_orderings.cached = ['auto']
        for ordering in [0, 2, 3, 4, 5, 6]:
            data = np.asfortranarray([1, 1], dtype=np.complex128)
            row = np.asfortranarray([1, 2], dtype=_mumps.int_dtype)
            col = np.asfortranarray([1, 2], dtype=_mumps.int_dtype)

            instance = _mumps.zmumps()
            instance.set_assembled_matrix(2, row, col, data)
            instance.icntl[7] = ordering
            instance.job = 1
            instance.call()

            if instance.infog[7] == ordering:
                possible_orderings.cached.append(ordering_name[ordering])

    return possible_orderings.cached

possible_orderings.cached = None


error_messages = {
    -5 : "Not enough memory during analysis phase",
    -6 : "Matrix is singular in structure",
    -7 : "Not enough memory during analysis phase",
    -10 : "Matrix is numerically singular",
    -11 : "The authors of MUMPS would like to hear about this",
    -12 : "The authors of MUMPS would like to hear about this",
    -13 : "Not enough memory"
}

class MUMPSError(RuntimeError):
    def __init__(self, infog):
        self.error = infog[1]
        if self.error in error_messages:
            msg = "{}. (MUMPS error {})".format(
                error_messages[self.error], self.error)
        else:
            msg = "MUMPS failed with error {}.".format(self.error)

        RuntimeError.__init__(self, msg)


class AnalysisStatistics:
    def __init__(self, inst, time=None):
        self.est_mem_incore = inst.infog[17]
        self.est_mem_ooc = inst.infog[27]
        self.est_nonzeros = (inst.infog[20] if inst.infog[20] > 0 else
                             -inst.infog[20] * 1000000)
        self.est_flops = inst.rinfog[1]
        self.ordering = ordering_name[inst.infog[7]]
        self.time = time

    def __str__(self):
        parts = ["estimated memory for in-core factorization:",
                 str(self.est_mem_incore), "mbytes\n",
                 "estimated memory for out-of-core factorization:",
                 str(self.est_mem_ooc), "mbytes\n",
                 "estimated number of nonzeros in factors:",
                 str(self.est_nonzeros), "\n",
                 "estimated number of flops:", str(self.est_flops), "\n",
                 "ordering used:", self.ordering]
        if hasattr(self, "time"):
            parts.extend(["\n analysis time:", str(self.time), "secs"])
        return " ".join(parts)


class FactorizationStatistics:
    def __init__(self, inst, time=None, include_ordering=False):
        # information about pivoting
        self.offdiag_pivots = inst.infog[12] if inst.sym == 0 else 0
        self.delayed_pivots = inst.infog[13]
        self.tiny_pivots = inst.infog[25]

        # possibly include ordering (used in schur_complement)
        if include_ordering:
            self.ordering = ordering_name[inst.infog[7]]

        # information about runtime effiency
        self.memory = inst.infog[22]
        self.nonzeros = (inst.infog[29] if inst.infog[29] > 0 else
                         -inst.infog[29] * 1000000)
        self.flops = inst.rinfog[3]
        if time:
            self.time = time

    def __str__(self):
        parts = ["off-diagonal pivots:", str(self.offdiag_pivots), "\n",
                 "delayed pivots:", str(self.delayed_pivots), "\n",
                 "tiny pivots:", str(self.tiny_pivots), "\n"]
        if hasattr(self, "ordering"):
            parts.extend(["ordering used:", self.ordering, "\n"])
        parts.extend(["memory used during factorization:", str(self.memory),
                      "mbytes\n",
                      "nonzeros in factored matrix:", str(self.nonzeros), "\n",
                      "floating point operations:", str(self.flops)])
        if hasattr(self, "time"):
            parts.extend(["\n factorization time:", str(self.time), "secs"])
        return " ".join(parts)


class MUMPSContext:
    """MUMPSContext contains the internal data structures needed by the
    MUMPS library and contains a user-friendly interface.

    WARNING: Only complex numbers supported.

    Examples
    --------

    Solving a small system of equations.

    >>> import scipy.sparse as sp
    >>> a = sp.coo_matrix([[1.,0],[0,2.]], dtype=complex)
    >>> ctx = kwant.linalg.mumps.MUMPSContext()
    >>> ctx.factor(a)
    >>> ctx.solve([1., 1.])
    array([ 1.0+0.j,  0.5+0.j])

    Instance variables
    ------------------

    analysis_stats : `AnalysisStatistics`
        contains MUMPS statistics after an analysis step (i.e.  after a call to
        `analyze` or `factor`)
    factor_stats : `FactorizationStatistics`
        contains MUMPS statistics after a factorization step (i.e.  after a
        call to `factor`)

    """

    def __init__(self, verbose=False):
        """Init the MUMPSContext class

        Parameters
        ----------

        verbose : True or False
            control whether MUMPS prints lots of internal statistics
            and debug information to screen.
        """
        self.mumps_instance = None
        self.dtype = None
        self.verbose = verbose
        self.factored = False

    def analyze(self, a, ordering='auto', overwrite_a=False):
        """Perform analysis step of MUMPS.

        In the analyis step, MUMPS figures out a reordering for the matrix and
        estimates number of operations and memory needed for the factorization
        time. This step usually needs not be called separately (it is done
        automatically by `factor`), but it can be useful to test which ordering
        would give best performance in the actual factorization, as MUMPS
        estimates are available in `analysis_stats`.

        Parameters
        ----------

        a : sparse SciPy matrix
            input matrix. Internally, the matrix is converted to `coo` format
            (so passing this format is best for performance)
        ordering : { 'auto', 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd' }
            ordering to use in the factorization. The availability of a
            particular ordering depends on the MUMPS installation.  Default is
            'auto'.
        overwrite_a : True or False
            whether the data in a may be overwritten, which can lead to a small
            performance gain. Default is False.
        """

        a = a.tocoo()

        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("Input matrix must be square!")

        if not ordering in orderings.keys():
            raise ValueError("Unknown ordering '"+ordering+"'!")

        dtype, row, col, data = _make_assembled_from_coo(a, overwrite_a)

        if dtype != self.dtype:
            self.mumps_instance = getattr(_mumps, dtype+"mumps")(self.verbose)
            self.dtype = dtype

        self.n = a.shape[0]
        self.row = row
        self.col = col
        self.data = data
        # Note: if I don't store them, they go out of scope and are
        #       deleted. I however need the memory to stay around!

        self.mumps_instance.set_assembled_matrix(a.shape[0], row, col, data)
        self.mumps_instance.icntl[7] = orderings[ordering]
        self.mumps_instance.job = 1
        t1 = time.process_time()
        self.mumps_instance.call()
        t2 = time.process_time()
        self.factored = False

        if self.mumps_instance.infog[1] < 0:
            raise MUMPSError(self.mumps_instance.infog)

        self.analysis_stats = AnalysisStatistics(self.mumps_instance,
                                                 t2 - t1)

    def factor(self, a, ordering='auto', ooc=False, pivot_tol=0.01,
               reuse_analysis=False, overwrite_a=False):
        """Perform the LU factorization of the matrix.

        This LU factorization can then later be used to solve a linear system
        with `solve`. Statistical data of the factorization is stored in
        `factor_stats`.

        Parameters
        ----------

        a : sparse SciPy matrix
            input matrix. Internally, the matrix is converted to `coo` format
            (so passing this format is best for performance)
        ordering : { 'auto', 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd' }
            ordering to use in the factorization. The availability of a
            particular ordering depends on the MUMPS installation.  Default is
            'auto'.
        ooc : True or False
            whether to use the out-of-core functionality of MUMPS.
            (out-of-core means that data is written to disk to reduce memory
            usage.) Default is False.
        pivot_tol: number in the range [0, 1]
            pivoting threshold. Pivoting is typically limited in sparse
            solvers, as too much pivoting destroys sparsity. 1.0 means full
            pivoting, whereas 0.0 means no pivoting. Default is 0.01.
        reuse_analysis: True or False
            whether to reuse the analysis done in a previous call to `analyze`
            or `factor`. If the structure of the matrix stays the same, and the
            numerical values do not change much, the previous analysis can be
            reused, saving some time.  WARNING: There is no check whether the
            structure of your matrix is compatible with the previous
            analysis. Also, if the values are not similar enough, there might
            be loss of accuracy, without a warning. Default is False.
        overwrite_a : True or False
            whether the data in a may be overwritten, which can lead to a small
            performance gain. Default is False.
        """
        a = a.tocoo()

        if a.ndim != 2 or a.shape[0] != a.shape[1]:
            raise ValueError("Input matrix must be square!")

        # Analysis phase must be done before factorization
        # Note: previous analysis is reused only if reuse_analysis == True

        if reuse_analysis:
            if self.mumps_instance is None:
                warnings.warn("Missing analysis although reuse_analysis=True. "
                              "New analysis is performed.",
                              RuntimeWarning,
                              stacklevel=2)
                self.analyze(a, ordering=ordering, overwrite_a=overwrite_a)
            else:
                dtype, row, col, data = _make_assembled_from_coo(a,
                                                                 overwrite_a)
                if self.dtype != dtype:
                    raise ValueError("MUMPSContext dtype and matrix dtype "
                                     "incompatible!")

                self.n = a.shape[0]
                self.row = row
                self.col = col
                self.data = data
                self.mumps_instance.set_assembled_matrix(a.shape[0],
                                                         row, col, data)
        else:
            self.analyze(a, ordering=ordering, overwrite_a=overwrite_a)

        self.mumps_instance.icntl[22] = 1 if ooc else 0
        self.mumps_instance.job = 2
        self.mumps_instance.cntl[1] = pivot_tol

        done = False
        while not done:
            t1 = time.process_time()
            self.mumps_instance.call()
            t2 = time.process_time()

            # error -8, -9 (not enough allocated memory) is treated
            # specially, by increasing the memory relaxation parameter
            if self.mumps_instance.infog[1] < 0:
                if self.mumps_instance.infog[1] in (-8, -9):
                    # double the additional memory
                    self.mumps_instance.icntl[14] *= 2
                else:
                    raise MUMPSError(self.mumps_instance.infog)
            else:
                done = True

        self.factored = True
        self.factor_stats = FactorizationStatistics(self.mumps_instance,
                                                    t2 - t1)

    def _solve_sparse(self, b):
        b = b.tocsc()
        x = np.empty((b.shape[0], b.shape[1]),
                     order='F', dtype=self.data.dtype)

        dtype, col_ptr, row_ind, data = _make_sparse_rhs_from_csc(
            b, self.data.dtype)

        if b.shape[0] != self.n:
            raise ValueError("Right hand side has wrong size")

        if self.dtype != dtype:
            raise ValueError("Data type of right hand side is not "
                             "compatible with the dtype of the "
                             "linear system")

        self.mumps_instance.set_sparse_rhs(col_ptr, row_ind, data)
        self.mumps_instance.set_dense_rhs(x)
        self.mumps_instance.job = 3
        self.mumps_instance.icntl[20] = 1
        self.mumps_instance.call()

        return x

    def _solve_dense(self, b, overwrite_b=False):
        dtype, b = prepare_for_fortran(overwrite_b, b,
                                       np.zeros(1, dtype=self.data.dtype))[:2]

        if b.shape[0] != self.n:
            raise ValueError("Right hand side has wrong size")

        if self.dtype != dtype:
            raise ValueError("Data type of right hand side is not "
                             "compatible with the dtype of the "
                             "linear system")

        self.mumps_instance.set_dense_rhs(b)
        self.mumps_instance.job = 3
        self.mumps_instance.call()

        return b

    def solve(self, b, overwrite_b=False):
        """Solve a linear system after the LU factorization has previously
        been performed by `factor`.

        Supports both dense and sparse right hand sides.

        Parameters
        ----------

        b : dense (NumPy) matrix or vector or sparse (SciPy) matrix
            the right hand side to solve. Accepts both dense and sparse input;
            if the input is sparse 'csc' format is used internally (so passing
            a 'csc' matrix gives best performance).
        overwrite_b : True or False
            whether the data in b may be overwritten, which can lead to a small
            performance gain. Default is False.

        Returns
        -------

        x : NumPy array
            the solution to the linear system as a dense matrix (a vector is
            returned if b was a vector, otherwise a matrix is returned).
        """

        if not self.factored:
            raise RuntimeError("Factorization must be done before solving!")

        if scipy.sparse.isspmatrix(b):
            return self._solve_sparse(b)
        else:
            return self._solve_dense(b, overwrite_b)


def schur_complement(a, indices, ordering='auto', ooc=False, pivot_tol=0.01,
                     calc_stats=False, overwrite_a=False):
    """Compute the Schur complement block of matrix a using MUMPS.

    Parameters:
    a : sparse matrix
        input matrix. Internally, the matrix is converted to `coo` format (so
        passing this format is best for performance)
    indices : 1d array
        indices (row and column) of the desired Schur complement block.  (The
        Schur complement block is square, so that the indices are both row and
        column indices.)
    ordering : { 'auto', 'amd', 'amf', 'scotch', 'pord', 'metis', 'qamd' }
        ordering to use in the factorization. The availability of a particular
        ordering depends on the MUMPS installation.  Default is 'auto'.
    ooc : True or False
        whether to use the out-of-core functionality of MUMPS.  (out-of-core
        means that data is written to disk to reduce memory usage.) Default is
        False.
    pivot_tol: number in the range [0, 1]
        pivoting threshold. Pivoting is typically limited in sparse solvers, as
        too much pivoting destroys sparsity. 1.0 means full pivoting, whereas
        0.0 means no pivoting. Default is 0.01.
    calc_stats: True or False
        whether to return the analysis and factorization statistics collected
        by MUMPS. Default is False.
    overwrite_a : True or False
        whether the data in a may be overwritten, which can lead to a small
        performance gain. Default is False.

    Returns
    -------

    s : NumPy array
        Schur complement block
    factor_stats: `FactorizationStatistics`
        statistics of the factorization as collected by MUMPS.  Only returned
        if ``calc_stats==True``.
    """

    if not scipy.sparse.isspmatrix(a):
        raise ValueError("a must be a sparse SciPy matrix!")

    a = a.tocoo()

    if a.ndim != 2 or a.shape[0] != a.shape[1]:
        raise ValueError("Input matrix must be square!")

    indices = np.asanyarray(indices)

    if indices.ndim != 1:
        raise ValueError("Schur indices must be specified in a 1d array!")

    if not ordering in orderings.keys():
        raise ValueError("Unknown ordering '"+ordering+"'!")

    dtype, row, col, data = _make_assembled_from_coo(a, overwrite_a)
    indices = _make_mumps_index_array(indices)

    mumps_instance = getattr(_mumps, dtype+"mumps")()

    mumps_instance.set_assembled_matrix(a.shape[0], row, col, data)
    mumps_instance.icntl[7] = orderings[ordering]
    mumps_instance.icntl[19] = 1
    mumps_instance.icntl[31] = 1  # discard factors, from 4.10.0
                                  # has no effect in earlier versions

    schur_compl = np.empty((indices.size, indices.size),
                           order='C', dtype=data.dtype)
    mumps_instance.set_schur(schur_compl, indices)

    mumps_instance.job = 4   # job=4 -> 1 and 2 after each other
    t1 = time.process_time()
    mumps_instance.call()
    t2 = time.process_time()

    if not calc_stats:
        return schur_compl
    else:
        return (schur_compl, FactorizationStatistics(
            mumps_instance, time=t2 - t1, include_ordering=True))


# Some internal helper functions
def _make_assembled_from_coo(a, overwrite_a):
    dtype, data = prepare_for_fortran(overwrite_a, a.data)

    row = np.asfortranarray(a.row.astype(_mumps.int_dtype))
    col = np.asfortranarray(a.col.astype(_mumps.int_dtype))

    # MUMPS uses Fortran indices.
    row += 1
    col += 1

    return dtype, row, col, data


def _make_sparse_rhs_from_csc(b, dtype):
    dtype, data = prepare_for_fortran(True, b.data,
                                      np.zeros(1, dtype=dtype))[:2]

    col_ptr = np.asfortranarray(b.indptr.astype(_mumps.int_dtype))
    row_ind = np.asfortranarray(b.indices.astype(_mumps.int_dtype))

    # MUMPS uses Fortran indices.
    col_ptr += 1
    row_ind += 1

    return dtype, col_ptr, row_ind, data


def _make_mumps_index_array(a):
    a = np.asfortranarray(a.astype(_mumps.int_dtype))
    a += 1                      # Fortran indices

    return a
