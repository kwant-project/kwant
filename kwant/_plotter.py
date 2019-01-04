# -*- coding: utf-8 -*-
# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

# This module is imported by plotter.py. It contains all the expensive imports
# that we want to remove from plotter.py

# All matplotlib imports must be isolated in a try, because even without
# matplotlib iterators remain useful.  Further, mpl_toolkits used for 3D
# plotting are also imported separately, to ensure that 2D plotting works even
# if 3D does not.

import warnings
from math import sqrt, pi
import numpy as np

try:
    import matplotlib
    import matplotlib.colors
    import matplotlib.cm
    from matplotlib.figure import Figure
    from matplotlib import collections
    from . import _colormaps
    mpl_available = True
    try:
        from mpl_toolkits import mplot3d
        has3d = True
    except ImportError:
        warnings.warn("3D plotting not available.", RuntimeWarning)
        has3d = False
except ImportError:
    warnings.warn("matplotlib is not available, only iterator-providing "
                  "functions will work.", RuntimeWarning)
    mpl_available = False


# Collections that allow for symbols and linewiths to be given in data space
# (not for general use, only implement what's needed for plotter)
def isarray(var):
    if hasattr(var, '__getitem__') and not isinstance(var, str):
        return True
    else:
        return False


def nparray_if_array(var):
    return np.asarray(var) if isarray(var) else var


if mpl_available:
    class LineCollection(collections.LineCollection):
        def __init__(self, segments, reflen=None, **kwargs):
            super().__init__(segments, **kwargs)
            self.reflen = reflen

        def set_linewidths(self, linewidths):
            self.linewidths_orig = nparray_if_array(linewidths)

        def draw(self, renderer):
            if self.reflen is not None:
                # Note: only works for aspect ratio 1!
                #       72.0 - there is 72 points in an inch
                factor = (self.axes.transData.frozen().to_values()[0] * 72.0 *
                          self.reflen / self.figure.dpi)
            else:
                factor = 1

            super().set_linewidths(self.linewidths_orig *
                                                       factor)
            return super().draw(renderer)


    class PathCollection(collections.PathCollection):
        def __init__(self, paths, sizes=None, reflen=None, **kwargs):
            super().__init__(paths, sizes=sizes, **kwargs)

            self.reflen = reflen
            self.linewidths_orig = nparray_if_array(self.get_linewidths())

            self.transforms = np.array(
                [matplotlib.transforms.Affine2D().scale(x).get_matrix()
                 for x in sizes])

        def get_transforms(self):
            return self.transforms

        def get_transform(self):
            Affine2D = matplotlib.transforms.Affine2D
            if self.reflen is not None:
                # For the paths, use the data transformation but strip the
                # offset (will be added later with offsets)
                args = self.axes.transData.frozen().to_values()[:4] + (0, 0)
                return Affine2D().from_values(*args).scale(self.reflen)
            else:
                return Affine2D().scale(self.figure.dpi / 72.0)

        def draw(self, renderer):
            if self.reflen:
                # Note: only works for aspect ratio 1!
                factor = (self.axes.transData.frozen().to_values()[0] /
                          self.figure.dpi * 72.0 * self.reflen)
                self.set_linewidths(self.linewidths_orig * factor)

            return collections.Collection.draw(self, renderer)


    if has3d:
        # Sorting is optional.
        sort3d = True

        # Compute the projection of a 3D length into 2D data coordinates
        # for this we use 2 3D half-circles that are projected into 2D.
        # (This gives the same length as projecting the full unit sphere.)

        phi = np.linspace(0, pi, 21)
        xyz = np.c_[np.cos(phi), np.sin(phi), 0 * phi].T.reshape(-1, 1, 21)
        # TODO: use np.block once we depend on numpy >= 1.13.
        unit_sphere = np.vstack([
            np.hstack([xyz[0], xyz[2]]),
            np.hstack([xyz[1], xyz[0]]),
            np.hstack([xyz[2], xyz[1]]),
        ])

        def projected_length(ax, length):
            rc = np.array([ax.get_xlim3d(), ax.get_ylim3d(), ax.get_zlim3d()])
            rc = np.apply_along_axis(np.sum, 1, rc) / 2.

            rs = unit_sphere * length + rc.reshape(-1, 1)

            transform = mplot3d.proj3d.proj_transform
            rp = np.asarray(transform(*(list(rs) + [ax.get_proj()]))[:2])
            rc[:2] = transform(*(list(rc) + [ax.get_proj()]))[:2]

            coords = rp - np.repeat(rc[:2].reshape(-1, 1), len(rs[0]), axis=1)
            return sqrt(np.sum(coords**2, axis=0).max())


        # Auxiliary array for calculating corners of a cube.
        corners = np.zeros((3, 8, 6), np.float_)
        corners[0, [0, 1, 2, 3], 0] = corners[0, [4, 5, 6, 7], 1] = \
        corners[0, [0, 1, 4, 5], 2] = corners[0, [2, 3, 6, 7], 3] = \
        corners[0, [0, 2, 4, 6], 4] = corners[0, [1, 3, 5, 7], 5] = 1.0


        class Line3DCollection(mplot3d.art3d.Line3DCollection):
            def __init__(self, segments, reflen=None, zorder=0, **kwargs):
                super().__init__(segments, **kwargs)
                self.reflen = reflen
                self.zorder3d = zorder

            def set_linewidths(self, linewidths):
                self.linewidths_orig = nparray_if_array(linewidths)

            def do_3d_projection(self, renderer):
                super().do_3d_projection(renderer)
                # The whole 3D ordering is flawed in mplot3d when several
                # collections are added. We just use normal zorder. Note the
                # "-" due to the different logic in the 3d plotting, we still
                # want larger zorder values to be plotted on top of smaller
                # ones.
                return -self.zorder3d

            def draw(self, renderer):
                if self.reflen:
                    proj_len = projected_length(self.axes, self.reflen)
                    args = self.axes.transData.frozen().to_values()
                    # Note: unlike in the 2D case, where we can enforce equal
                    #       aspect ratio, this (currently) does not work with
                    #       3D plots in matplotlib. As an approximation, we
                    #       thus scale with the average of the x- and y-axis
                    #       transformation.
                    factor = proj_len * (args[0] +
                                         args[3]) * 0.5 * 72.0 / self.figure.dpi
                else:
                    factor = 1

                super().set_linewidths(
                                                self.linewidths_orig * factor)
                super().draw(renderer)


        class Path3DCollection(mplot3d.art3d.Patch3DCollection):
            def __init__(self, paths, sizes, reflen=None, zorder=0,
                         offsets=None, **kwargs):
                paths = [matplotlib.patches.PathPatch(path) for path in paths]

                if offsets is not None:
                    kwargs['offsets'] = offsets[:, :2]

                super().__init__(paths, **kwargs)

                if offsets is not None:
                    self.set_3d_properties(zs=offsets[:, 2], zdir="z")

                self.reflen = reflen
                self.zorder3d = zorder

                self.paths_orig = np.array(paths, dtype='object')
                self.linewidths_orig = nparray_if_array(self.get_linewidths())
                self.linewidths_orig2 = self.linewidths_orig
                self.array_orig = nparray_if_array(self.get_array())
                self.facecolors_orig = nparray_if_array(self.get_facecolors())
                self.edgecolors_orig = nparray_if_array(self.get_edgecolors())

                Affine2D = matplotlib.transforms.Affine2D
                self.orig_transforms = np.array(
                    [Affine2D().scale(x).get_matrix() for x in sizes])
                self.transforms = self.orig_transforms

            def set_array(self, array):
                self.array_orig = nparray_if_array(array)
                super().set_array(array)

            def set_color(self, colors):
                self.facecolors_orig = nparray_if_array(colors)
                self.edgecolors_orig = self.facecolors_orig
                super().set_color(colors)

            def set_edgecolors(self, colors):
                colors = matplotlib.colors.colorConverter.to_rgba_array(colors)
                self.edgecolors_orig = nparray_if_array(colors)
                super().set_edgecolors(colors)

            def get_transforms(self):
                # this is exact only for an isometric projection, for the
                # perspective projection used in mplot3d it's an approximation
                return self.transforms

            def get_transform(self):
                Affine2D = matplotlib.transforms.Affine2D
                if self.reflen:
                    proj_len = projected_length(self.axes, self.reflen)

                    # For the paths, use the data transformation but strip the
                    # offset (will be added later with the offsets).
                    args = self.axes.transData.frozen().to_values()[:4] + (0, 0)
                    return Affine2D().from_values(*args).scale(proj_len)
                else:
                    return Affine2D().scale(self.figure.dpi / 72.0)

            def do_3d_projection(self, renderer):
                xs, ys, zs = self._offsets3d

                # numpy complains about zero-length index arrays
                if len(xs) == 0:
                    return -self.zorder3d

                proj = mplot3d.proj3d.proj_transform_clip
                vs = np.array(proj(xs, ys, zs, renderer.M)[:3])

                if sort3d:
                    indx = vs[2].argsort()[::-1]

                    self.set_offsets(vs[:2, indx].T)

                    if len(self.paths_orig) > 1:
                        paths = np.resize(self.paths_orig, (vs.shape[1],))
                        self.set_paths(paths[indx])

                    if len(self.orig_transforms) > 1:
                        self.transforms = self.transforms[indx]

                    lw_orig = self.linewidths_orig
                    if (isinstance(lw_orig, np.ndarray) and len(lw_orig) > 1):
                        self.linewidths_orig2 = np.resize(lw_orig,
                                                           (vs.shape[1],))[indx]

                    # Note: here array, facecolors and edgecolors are
                    #       guaranteed to be 2d numpy arrays or None.  (And
                    #       array is the same length as the coordinates)

                    if self.array_orig is not None:
                        super(Path3DCollection,
                              self).set_array(self.array_orig[indx])

                    if (self.facecolors_orig is not None and
                        self.facecolors_orig.shape[0] > 1):
                        shape = list(self.facecolors_orig.shape)
                        shape[0] = vs.shape[1]
                        super().set_facecolors(
                            np.resize(self.facecolors_orig, shape)[indx])

                    if (self.edgecolors_orig is not None and
                        self.edgecolors_orig.shape[0] > 1):
                        shape = list(self.edgecolors_orig.shape)
                        shape[0] = vs.shape[1]
                        super().set_edgecolors(
                                                np.resize(self.edgecolors_orig,
                                                          shape)[indx])
                else:
                    self.set_offsets(vs[:2].T)

                # the whole 3D ordering is flawed in mplot3d when several
                # collections are added. We just use normal zorder, but correct
                # by the projected z-coord of the "center of gravity",
                # normalized by the projected z-coord of the world coordinates.
                # In doing so, several Path3DCollections are plotted probably
                # in the right order (it's not exact) if they have the same
                # zorder. Still, smaller and larger integer zorders are plotted
                # below or on top.

                bbox = np.asarray(self.axes.get_w_lims())

                proj = mplot3d.proj3d.proj_transform_clip
                cz = proj(*(list(np.dot(corners, bbox)) + [renderer.M]))[2]

                return -self.zorder3d + vs[2].mean() / cz.ptp()

            def draw(self, renderer):
                if self.reflen:
                    proj_len = projected_length(self.axes, self.reflen)
                    args = self.axes.transData.frozen().to_values()
                    factor = proj_len * (args[0] +
                                         args[3]) * 0.5 * 72.0 / self.figure.dpi

                    self.set_linewidths(self.linewidths_orig2 * factor)

                super().draw(renderer)
