# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np


def prepare_for_fortran(overwrite, *args):
    """Convert arrays to Fortran format.

    This function takes a number of array objects in `args` and converts them
    to a format that can be directly passed to a Fortran function (Fortran
    contiguous NumPy array). If the arrays have different data type, they
    converted arrays are cast to a common compatible data type (one of NumPy's
    `float32`, `float64`, `complex64`, `complex128` data types).

    If `overwrite` is ``False``, an NumPy array that would already be in the
    correct format (Fortran contiguous, right data type) is neverthelessed
    copied. (Hence, overwrite = True does not imply that acting on the
    converted array in the return values will overwrite the original array in
    all cases -- it does only so if the original array was already in the
    correct format. The conversions require copying. In fact, that's the same
    behavior as in SciPy, it's just not explicitly stated there)

    If an argument is ``None``, it is just passed through and not used to
    determine the proper data type.

    `prepare_for_lapack` returns a character indicating the proper
    data type in LAPACK style ('s', 'd', 'c', 'z') and a list of
    properly converted arrays.
    """

    # Make sure we have NumPy arrays
    mats = [None]*len(args)
    for i in range(len(args)):
        if args[i] is not None:
            arr = np.asanyarray(args[i])
            if not np.issubdtype(arr.dtype, np.number):
                raise ValueError("Argument cannot be interpreted "
                                 "as a numeric array")

            mats[i] = (arr, arr is not args[i] or overwrite)
        else:
            mats[i] = (None, True)

    # First figure out common dtype
    # Note: The return type of common_type is guaranteed to be a floating point
    #       kind.
    dtype = np.common_type(*[arr for arr, ovwrt in mats if arr is not None])

    if dtype == np.float32:
        lapacktype = 's'
    elif dtype == np.float64:
        lapacktype = 'd'
    elif dtype == np.complex64:
        lapacktype = 'c'
    elif dtype == np.complex128:
        lapacktype = 'z'
    else:
        raise AssertionError("Unexpected data type from common_type")

    ret = [ lapacktype ]
    for npmat, ovwrt in mats:
        # Now make sure that the array is contiguous, and copy if necessary.
        if npmat is not None:
            if npmat.ndim == 2:
                if not npmat.flags["F_CONTIGUOUS"]:
                    npmat = np.asfortranarray(npmat, dtype = dtype)
                elif npmat.dtype != dtype:
                    npmat = npmat.astype(dtype)
                elif not ovwrt:
                    # ugly here: copy makes always C-array, no way to tell it
                    # to make a Fortran array.
                    npmat = np.asfortranarray(npmat.copy())
            elif npmat.ndim == 1:
                if not npmat.flags["C_CONTIGUOUS"]:
                    npmat = np.ascontiguousarray(npmat, dtype = dtype)
                elif npmat.dtype != dtype:
                    npmat = npmat.astype(dtype)
                elif not ovwrt:
                    npmat = np.asfortranarray(npmat.copy())
            else:
                raise ValueError("Dimensionality of array is not 1 or 2")

        ret.append(npmat)

    return tuple(ret)


def assert_fortran_mat(*mats):
    """Check if the input ndarrays are all proper Fortran matrices."""

    # This is a workaround for a bug in NumPy version < 2.0,
    # where 1x1 matrices do not have the F_Contiguous flag set correctly.
    for mat in mats:
        if (mat is not None and (mat.shape[0] > 1 or mat.shape[1] > 1) and
            not mat.flags["F_CONTIGUOUS"]):
            raise ValueError("Input matrix must be Fortran contiguous")


def assert_fortran_matvec(*arrays):
    """Check if the input ndarrays are all proper Fortran matrices
    or vectors."""

    # This is a workaround for a bug in NumPy version < 2.0,
    # where 1x1 matrices do not have the F_Contiguous flag set correctly.
    for arr in arrays:
        if not arr.ndim in (1, 2):
            raise ValueError("Input must be either a vector "
                             "or a matrix.")

        if (not arr.flags["F_CONTIGUOUS"] or
            (arr.ndim == 2 and arr.shape[0] == 1 and arr.shape[1] == 1) ):
            raise ValueError("Input must be a Fortran ordered "
                             "NumPy array")
