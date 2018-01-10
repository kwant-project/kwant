# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['lll', 'cvp', 'voronoi']

import numpy as np
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


def cvp(vec, basis, n=1):
    """
    Solve the n-closest vector problem for a vector, given a basis.

    This algorithm performs poorly in general, so it should be supplied
    with LLL-reduced bases.

    Parameters
    ----------
    vec : 1d array-like of floats
        The lattice vectors closest to this vector are to be found.
    basis : 2d array-like of floats
        Sequence of basis vectors
    n : int
        Number of lattice vectors closest to the point that need to be found.

    Returns
    -------
    coords : numpy array
        An array with the coefficients of the lattice vectors closest to the
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
    center_coords = np.array(np.round(np.linalg.lstsq(basis.T, vec)[0]), int)
    # Cutoff radius for n-th nearest neighbor.
    rad = 1
    nth_dist = np.inf
    while True:
        r = np.round(rad * np.linalg.cond(basis)) + 1
        points = np.mgrid[tuple(slice(i - r, i + r) for i in center_coords)]
        points = points.reshape(basis.shape[0], -1).T
        if len(points) < n:
            rad += 1
            continue
        point_coords = np.dot(points, basis)
        point_coords -= vec.T
        distances = np.sqrt(np.sum(point_coords**2, 1))
        order = np.argsort(distances)
        new_nth_dist = distances[order[n - 1]]
        if new_nth_dist < nth_dist:
            nth_dist = new_nth_dist
            rad += 1
        else:
            return np.array(points[order[:n]], int)


def voronoi(basis):
    """
    Return an array of lattice vectors forming its voronoi cell.

    Parameters
    ----------
    basis : 2d array-like of floats
        Basis vectors for which the Voronoi neighbors have to be found.

    Returns
    -------
    voronoi_neighbors : numpy array of ints
        All the lattice vectors that may potentially neighbor the origin.

    Notes
    -----
    This algorithm does not calculate the minimal Voronoi cell of the lattice
    and can be optimized. Its main aim is flood-fill, however, and better
    safe than sorry.
    """
    basis = np.asarray(basis)
    if basis.ndim != 2:
        raise ValueError('`basis` must be a 2d array-like object.')
    displacements = list(product(*(len(basis) * [[0, .5]])))[1:]
    vertices = np.array([cvp(np.dot(vec, basis), basis)[0] for vec in
                         displacements])
    vertices = np.array(np.round((vertices - displacements) * 2), int)
    for i in range(len(vertices)):
        if not np.any(vertices[i]):
            vertices[i] += 2 * np.array(displacements[i])
    vertices = np.concatenate([vertices, -vertices])
    return vertices
