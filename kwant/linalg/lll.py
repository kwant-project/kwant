# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['lll', 'cvp', 'voronoi']

import numpy as np
from scipy import linalg as la
from itertools import product


def gs_coefficient(a, b):
    """Gram-Schmidt coefficient."""
    return np.dot(a, b) / np.linalg.norm(b)**2


def gs(mat):
    """Compute Gram-Schmidt decomposition on a matrix."""
    mat = np.copy(mat)
    for i in range(len(mat)):
        for j in range(i):
            mat[i] -= gs_coefficient(mat[i], mat[j]) * mat[j]
    return mat


def is_c_reduced(vecs, c):
    """Check if a basis is c-reduced."""
    vecs = gs(vecs)
    r = np.apply_along_axis(lambda x: np.linalg.norm(x)**2, 1, vecs)
    return np.all((r[: -1] / r[1:]) < c)


def lll(basis, c=1.34):
    """
    Calculate a reduced lattice basis using LLL algorithm.

    Reduce a basis of a lattice to an almost orthonormal form. For details see
    e.g. http://en.wikipedia.org/wiki/LLL-algorithm.

    Parameters
    ----------
    basis : 2d array-like of floats
        The lattice basis vectors to be reduced.
    c : float
        Reduction parameter for the algorithm. Must be larger than 1 1/3,
        since otherwise a solution is not guaranteed to exist.

    Returns
    -------
    reduced_basis : numpy array
        The basis vectors of the LLL-reduced basis.
    transformation : numpy integer array
        Coefficient matrix for tranforming from the reduced basis to the
        original one.
    """
    vecs_orig = np.asarray(basis, float)
    if vecs_orig.ndim != 2:
        raise ValueError('"basis" must be a 2d array-like object.')
    if vecs_orig.shape[0] > vecs_orig.shape[1]:
        raise ValueError('The number of basis vectors exceeds the '
                         'space dimensionality.')
    vecs = vecs_orig.copy()
    vecsstar = vecs_orig.copy()
    m = vecs.shape[0]
    u = np.identity(m)
    def ll_reduce(i):
        for j in reversed(range(i)):
            vecs[i] -= np.round(u[i, j]) * vecs[j]
            u[i] -= np.round(u[i, j]) * u[j]

    # Initialize values.
    for i in range(m):
        for j in range(i):
            u[i, j] = gs_coefficient(vecs[i], vecsstar[j])
            vecsstar[i] -= u[i, j] * vecsstar[j]
        ll_reduce(i)

    # Main part of LLL algorithm.
    i = 0
    while i < m-1:
        if (np.linalg.norm(vecsstar[i]) ** 2 <
           c * np.linalg.norm(vecsstar[i+1]) ** 2):
            i += 1
        else:
            vecsstar[i+1] += u[i+1, i] * vecsstar[i]

            u[i, i] = gs_coefficient(vecs[i], vecsstar[i+1])
            u[i, i+1] = u[i+1, i] = 1
            u[i+1, i+1] = 0
            vecsstar[i] -= u[i, i] * vecsstar[i+1]
            vecs[[i, i+1]] = vecs[[i+1, i]]
            vecsstar[[i, i+1]] = vecsstar[[i+1, i]]
            u[[i, i+1]] = u[[i+1, i]]
            for j in range(i+2, m):
                u[j, i] = gs_coefficient(vecs[j], vecsstar[i])
                u[j, i+1] = gs_coefficient(vecs[j], vecsstar[i+1])
            if abs(u[i+1, i]) > 0.5:
                ll_reduce(i+1)
            i = max(i-1, 0)
    # TODO: change to rcond=None once we depend on numpy >= 1.14.
    coefs = np.linalg.lstsq(vecs_orig.T, vecs.T, rcond=-1)[0]
    if not np.allclose(np.round(coefs), coefs, atol=1e-6):
        raise RuntimeError('LLL algorithm instability.')
    if not is_c_reduced(vecs, c):
        raise RuntimeError('LLL algorithm instability.')
    return vecs, np.array(np.round(coefs), int)


def cvp(vec, basis, n=1, group_by_length=False, rtol=1e-09):
    """
    Solve the n-closest vector problem for a vector, given a basis.

    This algorithm performs poorly in general, so it should be supplied
    with LLL-reduced basis.

    Parameters
    ----------
    vec : 1d array-like of floats
        The lattice vectors closest to this vector are to be found.
    basis : 2d array-like of floats
        Sequence of basis vectors
    n : int
        Number of lattice vectors closest to the point that need to be found. If
        `group_by_length=True`, all vectors with the `n` shortest distinct distances
        are found.
    group_by_length : bool
        Whether to count points with distances that are within `rtol` as one
        towards `n`. Useful for finding `n`-th nearest neighbors of high symmetry points.
    rtol : float
        If `group_by_length=True`, relative tolerance when deciding whether
        distances are equal.

    Returns
    -------
    coords : numpy array
        An array with the coefficients of the `n` closest lattice vectors, or all
        the lattice vectors with the `n` shortest distances from the
        requested point.

    Notes
    -----
    This function can also be used to solve the `n` shortest lattice vector
    problem if the `vec` is zero, and `n+1` points are requested
    (and the first output is ignored).
    """
    # Calculate coordinates of the starting point in this basis.
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError('`basis` must be a 2d array-like object.')
    vec = np.asarray(vec)
    # Project the coordinates on the space spanned by `basis`, in
    # units of `basis` vectors.
    vec_bas = la.lstsq(basis.T, vec)[0]
    # Round the coordinates, this finds the lattice point which is
    # the center of the parallelepiped that contains `vec`.
    # `vec` is surely in the parallelepiped with edges `basis`
    # centered at `center_coords`.
    center_coords = np.array(np.round(vec_bas), int)
    # Make `vec` the projection of `vec` onto the space spanned by `basis`.
    vec = vec_bas @ basis
    bbt = basis @ basis.T
    # The volume of a parallelepiped spanned by `basis` is `sqrt(det(bbt))`.
    # (We use this instead of `det(basis)` because basis does not necessarily
    # span the full space, hence not always a square matrix.)
    # We calculate the radius `rad` of the largest sphere that can be inscribed
    # in the parallelepiped spanned by `basis`. The area of each face spanned
    # by one less vector in `basis` is given by the above formula applied to `bbt`
    # with one row and column discarded (first minor). We utilizie the relation
    # between entries of the inverse matrix and first minors. The perpendicular
    # size of the parallelepiped is given by the volume of the parallelepiped
    # divided by the area of the face. We choose the smallest perpendicular
    # size as the diameter of the largest sphere inside.
    rad = 0.5 / np.sqrt(np.max(np.diag(la.inv(bbt))))

    l = 1
    while True:
        # Make all the lattice points in and on the edges of a parallelepiped
        # of `2*l` size around `center_coords`.
        points = np.mgrid[tuple(slice(i - l, i + l + 1) for i in center_coords)]
        points = points.reshape(basis.shape[0], -1).T
        # If there are less than `n` points, make more.
        if len(points) < n:
            l += 1
            continue
        point_coords = points @ basis
        point_coords = point_coords - vec.T
        distances = la.norm(point_coords, axis = 1)
        order = np.argsort(distances)
        distances = distances[order]
        points = points[order]
        if group_by_length:
            # Group points that are equidistant within `rtol * rad`.
            group_boundaries = np.where(np.diff(distances) > rtol * rad)[0]
            if len(group_boundaries) + 1 < n:
                # Make more points if there are less than `n` groups.
                l += 1
                continue
            elif len(group_boundaries) == n - 1:
                # If there are exactly `n` groups, choose largest distance.
                dist_n = distances[-1]
                points_keep = points
            else:
                # If there are more than `n` groups, choose the largest
                # distance in the `n`-th group.
                dist_n = distances[group_boundaries[n-1]]
                # Only return points that are in the first `n` groups.
                points_keep = points[:group_boundaries[n-1] + 1]
        else:
            rtol = 0
            # Choose the `n`-th distance.
            dist_n = distances[n-1]
            # Only return the first `n` points.
            points_keep = points[:n]
        # We covered all points within radius `(2*l-1) * rad` from the
        # parallelepiped spanned by `basis` centered at `center_coords`.
        # Because `vec` is guaranteed to be in this parallelepiped, if
        # the current `dist_n` is smaller than `(2*l-1) * rad`, we surely
        # found all the smaller vectors.
        if dist_n < (2*l - 1 - rtol) * rad:
            return np.array(points_keep, int)
        else:
            # Otherwise there may be smaller vectors we haven't found,
            # increase `l`.
            l += 1
            continue


def voronoi(basis, reduced=False, rtol=1e-09):
    """
    Return an array of lattice vectors forming its voronoi cell.

    Parameters
    ----------
    basis : 2d array-like of floats
        Basis vectors for which the Voronoi neighbors have to be found.
    reduced : bool
        If False, exactly `2 (2^D - 1)` vectors are returned (where `D`
        is the number of lattice vectors), these are not always the minimal
        set of Voronoi vectors.
        If True, only the minimal set of Voronoi vectors are returned.
    rtol : float
        Tolerance when deciding whether a vector is in the minimal set,
        vectors associated with small faces compared to the size of the
        unit cell are discarded.

    Returns
    -------
    voronoi_neighbors : numpy array of ints
        All the lattice vectors that (potentially) neighbor the origin.
        List of length `2n` where the second half of the list contains
        is `-1` times the vectors in the first half.
    """
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError('`basis` must be a 2d array-like object.')
    # Find halved lattice points, every face of the VC contains half
    # of the lattice vector which is the normal of the face.
    # These points are all potentially on a face a VC,
    # but not necessarily the VC centered at the origin.
    displacements = list(product(*(len(basis) * [[0, .5]])))[1:]
    # Find the nearest lattice point, this is the lattice point whose
    # VC this face belongs to.
    vertices = np.array([cvp(vec @ basis, basis)[0] for vec in
                         displacements])
    # The lattice vector for a face is exactly twice the vector to the
    # closest lattice point from the halved lattice point on the face.
    vertices = np.array(np.round((vertices - displacements) * 2), int)
    if reduced:
        # Discard vertices that are not associated with a face of the VC.
        # This happens if half of the lattice vector is outside the convex
        # polytope defined by the rest of the vertices in `keep`.
        bbt = basis @ basis.T
        products = vertices @ bbt @ vertices.T
        keep = np.array([True] * len(vertices))
        for i in range(len(vertices)):
            # Relevant if the projection of half of the lattice vector `vertices[i]`
            # onto every other lattice vector in `veritces[keep]` is less than `0.5`.
            mask = np.array([False if k == i else b for k, b in enumerate(keep)])
            projections = 0.5 * products[i, mask] / np.diag(products)[mask]
            if not np.all(np.abs(projections) < 0.5 - rtol):
                keep[i] = False
        vertices = vertices[keep]

    vertices = np.concatenate([vertices, -vertices])
    return vertices
