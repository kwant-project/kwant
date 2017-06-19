# -*- coding: utf-8 -*-
# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Plotter module for Kwant.

This module provides iterators useful for any plotter routine, such as a list
of system sites, their coordinates, lead sites at any lead unit cell, etc.  If
`matplotlib` is available, it also provides simple functions for plotting the
system in two or three dimensions.
"""

from collections import defaultdict
import itertools
import functools
import warnings
import cmath
import numpy as np
import tinyarray as ta
from scipy import spatial, interpolate
from math import cos, sin, pi, sqrt

# All matplotlib imports must be isolated in a try, because even without
# matplotlib iterators remain useful.  Further, mpl_toolkits used for 3D
# plotting are also imported separately, to ensure that 2D plotting works even
# if 3D does not.
try:
    import matplotlib
    import matplotlib.colors
    import matplotlib.cm
    from matplotlib.figure import Figure
    from matplotlib import collections
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from . import _colormaps
    mpl_enabled = True
    try:
        from mpl_toolkits import mplot3d
        has3d = True
    except ImportError:
        warnings.warn("3D plotting not available.", RuntimeWarning)
        has3d = False
except ImportError:
    warnings.warn("matplotlib is not available, only iterator-providing "
                  "functions will work.", RuntimeWarning)
    mpl_enabled = False

from . import system, builder, _common


__all__ = ['plot', 'map', 'bands', 'spectrum', 'current',
           'interpolate_current', 'streamplot',
           'sys_leads_sites', 'sys_leads_hoppings', 'sys_leads_pos',
           'sys_leads_hopping_pos', 'mask_interpolate']


# TODO: Remove the following once we depend on matplotlib >= 1.4.1.
def matplotlib_chores():
    global pre_1_4_matplotlib
    ver = matplotlib.__version__

    if ver == "1.4.0":
        warnings.warn("Matplotlib 1.4.0 has a bug that makes 3D plotting "
                      "unusable (2D plotting is not affected). Please "
                      "consider using a different version of matplotlib.",
                      RuntimeWarning)

    pre_1_4_matplotlib = [int(x) for x in ver.split('.')[:2]] < [1, 4]


if mpl_enabled:
    matplotlib_chores()


# Collections that allow for symbols and linewiths to be given in data space
# (not for general use, only implement what's needed for plotter)
def isarray(var):
    if hasattr(var, '__getitem__') and not isinstance(var, str):
        return True
    else:
        return False


def nparray_if_array(var):
    return np.asarray(var) if isarray(var) else var


def _sample_array(array, n_samples, rng=None):
    rng = _common.ensure_rng(rng)
    la = len(array)
    return array[rng.choice(range(la), min(n_samples, la))]


if mpl_enabled:
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

            if pre_1_4_matplotlib:
                self.transforms = [matplotlib.transforms.Affine2D().scale(x)
                                   for x in sizes]
            else:
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
        unit_sphere = np.bmat([[xyz[0], xyz[2]], [xyz[1], xyz[0]],
                                [xyz[2], xyz[1]]])
        unit_sphere = np.asarray(unit_sphere)

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
                if pre_1_4_matplotlib:
                    self.orig_transforms = np.array(
                        [Affine2D().scale(x) for x in sizes], dtype='object')
                else:
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
                        self.transforms = np.resize(self.orig_transforms,
                                                     (vs.shape[1],))
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


# matplotlib helper functions.

def _make_figure(dpi, fig_size):
    fig = Figure()
    if dpi is not None:
        fig.set_dpi(dpi)
    if fig_size is not None:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])
    return fig


def set_colors(color, collection, cmap, norm=None):
    """Process a color specification to a format accepted by collections.

    Parameters
    ----------
    color : color specification
    collection : instance of a subclass of ``matplotlib.collections.Collection``
        Collection to which the color is added.
    cmap : ``matplotlib`` color map specification or None
        Color map to be used if colors are specified as floats.
    norm : ``matplotlib`` color norm
        Norm to be used if colors are specified as floats.
    """

    length = max(len(collection.get_paths()), len(collection.get_offsets()))

    # matplotlib gets confused if dtype='object'
    if (isinstance(color, np.ndarray) and color.dtype == np.dtype('object')):
        color = tuple(color)

    if isinstance(collection, mplot3d.art3d.Line3DCollection):
        length = len(collection._segments3d)  # Once again, matplotlib fault!

    if isarray(color) and len(color) == length:
        try:
            # check if it is an array of floats for color mapping
            color = np.asarray(color, dtype=float)
            if color.ndim == 1:
                collection.set_array(color)
                collection.set_cmap(cmap)
                collection.set_norm(norm)
                collection.set_color(None)
                return
        except (TypeError, ValueError):
            pass

    colors = matplotlib.colors.colorConverter.to_rgba_array(color)
    collection.set_color(colors)


symbol_dict = {'O': 'o', 's': ('p', 4, 45), 'S': ('P', 4, 45)}

def get_symbol(symbols):
    """Return the path corresponding to the description in ``symbols``"""
    # Figure out if list of symbols or single symbol.
    if not hasattr(symbols, '__getitem__'):
        symbols = [symbols]
    elif len(symbols) == 3 and symbols[0] in ('p', 'P'):
        # Most likely a polygon specification (at least not a valid other
        # symbol).
        symbols = [symbols]

    symbols = [symbol_dict[symbol] if symbol in symbol_dict else symbol for
               symbol in symbols]

    paths = []
    for symbol in symbols:
        if isinstance(symbol, matplotlib.path.Path):
            return symbol
        elif hasattr(symbol, '__getitem__') and len(symbol) == 3:
            kind, n, angle = symbol

            if kind in ['p', 'P']:
                if kind == 'p':
                    radius = 1. / cos(pi / n)
                else:
                    # make the polygon such that it has area equal
                    # to a unit circle
                    radius = sqrt(2 * pi / (n * sin(2 * pi / n)))

                angle = pi * angle / 180
                patch = matplotlib.patches.RegularPolygon((0, 0), n,
                                                          radius=radius,
                                                          orientation=angle)
            else:
                raise ValueError("Unknown symbol definition " + str(symbol))
        elif symbol == 'o':
            patch = matplotlib.patches.Circle((0, 0), 1)

        paths.append(patch.get_path().transformed(patch.get_transform()))

    return paths


def symbols(axes, pos, symbol='o', size=1, reflen=None, facecolor='k',
            edgecolor='k', linewidth=None, cmap=None, norm=None, zorder=0,
            **kwargs):
    """Add a collection of symbols (2D or 3D) to an axes instance.

    Parameters
    ----------
    axes : matplotlib.axes.Axes instance
        Axes to which the lines have to be added.
    pos0 : 2d or 3d array_like
        Coordinates of each symbol.
    symbol: symbol definition.
        TODO To be written.
    size: float or 1d array
        Size(s) of the symbols. Defaults to 1.
    reflen: float or None, optional
        If ``reflen`` is ``None``, the symbol sizes and linewidths are
        given in points (absolute size in the figure space). If
        ``reflen`` is a number, the symbol sizes and linewidths are
        given in units of ``reflen`` in data space (i.e. scales with the
        scale of the plot). Defaults to ``None``.
    facecolor: color definition, optional
    edgecolor: color definition, optional
        Defines the fill and edge color of the symbol, repsectively.
        Either a single object that is a proper matplotlib color
        definition or a sequence of such objects of appropriate
        length.  Defaults to all black.
    cmap : ``matplotlib`` color map specification or None
        Color map to be used if colors are specified as floats.
    norm : ``matplotlib`` color norm
        Norm to be used if colors are specified as floats.
    zorder: int
        Order in which different collections are drawn: larger
        ``zorder`` means the collection is drawn over collections with
        smaller ``zorder`` values.
    **kwargs : dict keyword arguments to
        pass to `PathCollection` or `Path3DCollection`, respectively.

    Returns
    -------
    `PathCollection` or `Path3DCollection` instance containing all the
    symbols that were added.
    """

    dim = pos.shape[1]
    assert dim == 2 or dim == 3

    #internally, size must be array_like
    try:
        size[0]
    except TypeError:
        size = (size, )

    if dim == 2:
        Collection = PathCollection
    else:
        Collection = Path3DCollection

    if len(pos) == 0 or np.all(symbol == 'no symbol') or np.all(size == 0):
        paths = []
        pos = np.empty((0, dim))
    else:
        paths = get_symbol(symbol)

    coll = Collection(paths, sizes=size, reflen=reflen, linewidths=linewidth,
                      offsets=pos, transOffset=axes.transData, zorder=zorder)

    set_colors(facecolor, coll, cmap, norm)
    coll.set_edgecolors(edgecolor)

    coll.update(kwargs)

    if dim == 2:
        axes.add_collection(coll)
    else:
        axes.add_collection3d(coll)

    return coll


def lines(axes, pos0, pos1, reflen=None, colors='k', linestyles='solid',
          cmap=None, norm=None, zorder=0, **kwargs):
    """Add a collection of line segments (2D or 3D) to an axes instance.

    Parameters
    ----------
    axes : matplotlib.axes.Axes instance
        Axes to which the lines have to be added.
    pos0 : 2d or 3d array_like
        Starting coordinates of each line segment
    pos1 : 2d or 3d array_like
        Ending coordinates of each line segment
    reflen: float or None, optional
        If `reflen` is `None`, the linewidths are given in points (absolute
        size in the figure space). If `reflen` is a number, the linewidths
        are given in units of `reflen` in data space (i.e. scales with
        the scale of the plot). Defaults to `None`.
    colors : color definition, optional
        Either a single object that is a proper matplotlib color definition
        or a sequence of such objects of appropriate length.  Defaults to all
        segments black.
    linestyles :linestyle definition, optional
        Either a single object that is a proper matplotlib line style
        definition or a sequence of such objects of appropriate length.
        Defaults to all segments solid.
    cmap : ``matplotlib`` color map specification or None
        Color map to be used if colors are specified as floats.
    norm : ``matplotlib`` color norm
        Norm to be used if colors are specified as floats.
    zorder: int
        Order in which different collections are drawn: larger
        `zorder` means the collection is drawn over collections with
        smaller `zorder` values.
    **kwargs : dict keyword arguments to
        pass to `LineCollection` or `Line3DCollection`, respectively.

    Returns
    -------
    `LineCollection` or `Line3DCollection` instance containing all the
    segments that were added.
    """

    if not pos0.shape == pos1.shape:
        raise ValueError('Incompatible lengths of coordinate arrays.')

    dim = pos0.shape[1]
    assert dim == 2 or dim == 3
    if dim == 2:
        Collection = LineCollection
    else:
        Collection = Line3DCollection

    if (len(pos0) == 0 or
        ('linewidths' in kwargs and kwargs['linewidths'] == 0)):
        coll = Collection([], reflen=reflen, linestyles=linestyles,
                          zorder=zorder)
        coll.update(kwargs)
        if dim == 2:
            axes.add_collection(coll)
        else:
            axes.add_collection3d(coll)
        return coll

    segments = np.c_[pos0, pos1].reshape(pos0.shape[0], 2, dim)

    coll = Collection(segments, reflen=reflen, linestyles=linestyles,
                      zorder=zorder)
    set_colors(colors, coll, cmap, norm)
    coll.update(kwargs)

    if dim == 2:
        axes.add_collection(coll)
    else:
        axes.add_collection3d(coll)

    return coll


def output_fig(fig, output_mode='auto', file=None, savefile_opts=None,
               show=True):
    """Output a matplotlib figure using a given output mode.

    Parameters
    ----------
    fig : matplotlib.figure.Figure instance
        The figure to be output.
    output_mode : string
        The output mode to be used.  Can be one of the following:
        'pyplot' : attach the figure to pyplot, with the same behavior as if
        pyplot.plot was called to create this figure.
        'ipython' : attach a `FigureCanvasAgg` to the figure and return it.
        'return' : return the figure.
        'file' : same as 'ipython', but also save the figure into a file.
        'auto' : if fname is given, save to a file, else if pyplot
        is imported, attach to pyplot, otherwise just return.  See also the
        notes below.
    file : string or a file object
        The name of the target file or the target file itself
        (opened for writing).
    savefile_opts : (list, dict) or None
        args and kwargs passed to `print_figure` of ``matplotlib``
    show : bool
        Whether to call ``matplotlib.pyplot.show()``.  Only has an effect if the
        output uses pyplot.

    Notes
    -----
    The behavior of this function producing a file is different from that of
    matplotlib in that the `dpi` attribute of the figure is used by defaul
    instead of the matplotlib config setting.
    """
    if not mpl_enabled:
        raise RuntimeError('matplotlib is not installed.')
    if output_mode == 'auto':
        if file is not None:
            output_mode = 'file'
        else:
            try:
                matplotlib.pyplot.get_backend()
                output_mode = 'pyplot'
            except AttributeError:
                output_mode = 'pyplot'
    if output_mode == 'pyplot':
        try:
            fake_fig = matplotlib.pyplot.figure()
        except AttributeError:
            msg = ('matplotlib.pyplot is unavailable.  Execute `import '
                   'matplotlib.pyplot` or use a different output mode.')
            raise RuntimeError(msg)
        fake_fig.canvas.figure = fig
        fig.canvas = fake_fig.canvas
        for ax in fig.axes:
            try:
                ax.mouse_init()  # Make 3D interface interactive.
            except AttributeError:
                pass
        if show:
            matplotlib.pyplot.show()
        return fig
    elif output_mode == 'return':
        canvas = FigureCanvasAgg(fig)
        fig.canvas = canvas
        return fig
    elif output_mode == 'file':
        canvas = FigureCanvasAgg(fig)
        if savefile_opts is None:
            savefile_opts = ([], {})
        if 'dpi' not in savefile_opts[1]:
            savefile_opts[1]['dpi'] = fig.dpi
        canvas.print_figure(file, *savefile_opts[0], **savefile_opts[1])
        return fig
    else:
        assert False, 'Unknown output_mode'


# Extracting necessary data from the system.

def sys_leads_sites(sys, num_lead_cells=2):
    """Return all the sites of the system and of the leads as a list.

    Parameters
    ----------
    sys : kwant.builder.Builder or kwant.system.System instance
        The system, sites of which should be returned.
    num_lead_cells : integer
        The number of times lead sites from each lead should be returned.
        This is useful for showing several unit cells of the lead next to the
        system.

    Returns
    -------
    sites : list of (site, lead_number, copy_number) tuples
        A site is a `~kwant.builder.Site` instance if the system is not finalized,
        and an integer otherwise.  For system sites `lead_number` is `None` and
        `copy_number` is `0`, for leads both are integers.
    lead_cells : list of slices
        `lead_cells[i]` gives the position of all the coordinates of lead
        `i` within `sites`.

    Notes
    -----
    Leads are only supported if they are of the same type as the original
    system, i.e.  sites of `~kwant.builder.BuilderLead` leads are returned with an
    unfinalized system, and sites of ``system.InfiniteSystem`` leads are
    returned with a finalized system.
    """
    syst = sys  # for naming consistency within function bodies
    lead_cells = []
    if isinstance(syst, builder.Builder):
        sites = [(site, None, 0) for site in syst.sites()]
        for leadnr, lead in enumerate(syst.leads):
            start = len(sites)
            if hasattr(lead, 'builder') and len(lead.interface):
                sites.extend(((site, leadnr, i) for site in
                              lead.builder.sites() for i in
                              range(num_lead_cells)))
            lead_cells.append(slice(start, len(sites)))
    elif isinstance(syst, system.FiniteSystem):
        sites = [(i, None, 0) for i in range(syst.graph.num_nodes)]
        for leadnr, lead in enumerate(syst.leads):
            start = len(sites)
            # We will only plot leads with a graph and with a symmetry.
            if (hasattr(lead, 'graph') and hasattr(lead, 'symmetry') and
                len(syst.lead_interfaces[leadnr])):
                sites.extend(((site, leadnr, i) for site in
                              range(lead.cell_size) for i in
                              range(num_lead_cells)))
            lead_cells.append(slice(start, len(sites)))
    else:
        raise TypeError('Unrecognized system type.')
    return sites, lead_cells


def sys_leads_pos(sys, site_lead_nr):
    """Return an array of positions of sites in a system.

    Parameters
    ----------
    sys : `kwant.builder.Builder` or `kwant.system.System` instance
        The system, coordinates of sites of which should be returned.
    site_lead_nr : list of `(site, leadnr, copynr)` tuples
        Output of `sys_leads_sites` applied to the system.

    Returns
    -------
    coords : numpy.ndarray of floats
        Array of coordinates of the sites.

    Notes
    -----
    This function uses `site.pos` property to get the position of a builder
    site and `sys.pos(sitenr)` for finalized systems.  This function requires
    that all the positions of all the sites have the same dimensionality.
    """

    # Note about efficiency (also applies to sys_leads_hoppings_pos)
    # NumPy is really slow when making a NumPy array from a tinyarray
    # (buffer interface seems very slow). It's much faster to first
    # convert to a tuple and then to convert to numpy array ...

    syst = sys  # for naming consistency inside function bodies
    is_builder = isinstance(syst, builder.Builder)
    num_lead_cells = site_lead_nr[-1][2] + 1
    if is_builder:
        pos = np.array(ta.array([i[0].pos for i in site_lead_nr]))
    else:
        syst_from_lead = lambda lead: (syst if (lead is None)
                                      else syst.leads[lead])
        pos = np.array(ta.array([syst_from_lead(i[1]).pos(i[0])
                                 for i in site_lead_nr]))
    if pos.dtype == object:  # Happens if not all the pos are same length.
        raise ValueError("pos attribute of the sites does not have consistent"
                         " values.")
    dim = pos.shape[1]

    def get_vec_domain(lead_nr):
        if lead_nr is None:
            return np.zeros((dim,)), 0
        if is_builder:
            sym = syst.leads[lead_nr].builder.symmetry
            try:
                site = syst.leads[lead_nr].interface[0]
            except IndexError:
                return (0, 0)
        else:
            try:
                sym = syst.leads[lead_nr].symmetry
                site = syst.sites[syst.lead_interfaces[lead_nr][0]]
            except (AttributeError, IndexError):
                # empty leads, or leads without symmetry aren't drawn anyways
                return (0, 0)
        dom = sym.which(site)[0] + 1
        # Conversion to numpy array here useful for efficiency
        vec = np.array(sym.periods)[0]
        return vec, dom
    vecs_doms = dict((i, get_vec_domain(i)) for i in range(len(syst.leads)))
    vecs_doms[None] = np.zeros((dim,)), 0
    for k, v in vecs_doms.items():
        vecs_doms[k] = [v[0] * i for i in range(v[1], v[1] + num_lead_cells)]
    pos += [vecs_doms[i[1]][i[2]] for i in site_lead_nr]
    return pos


def sys_leads_hoppings(sys, num_lead_cells=2):
    """Return all the hoppings of the system and of the leads as an iterator.

    Parameters
    ----------
    sys : kwant.builder.Builder or kwant.system.System instance
        The system, sites of which should be returned.
    num_lead_cells : integer
        The number of times lead sites from each lead should be returned.
        This is useful for showing several unit cells of the lead next to the
        system.

    Returns
    -------
    hoppings : list of (hopping, lead_number, copy_number) tuples
        A site is a `~kwant.builder.Site` instance if the system is not finalized,
        and an integer otherwise.  For system sites `lead_number` is `None` and
        `copy_number` is `0`, for leads both are integers.
    lead_cells : list of slices
        `lead_cells[i]` gives the position of all the coordinates of lead
        `i` within `hoppings`.

    Notes
    -----
    Leads are only supported if they are of the same type as the original
    system, i.e.  hoppings of `~kwant.builder.BuilderLead` leads are returned with an
    unfinalized system, and hoppings of `~kwant.system.InfiniteSystem` leads are
    returned with a finalized system.
    """

    syst = sys  # for naming consistency inside function bodies
    hoppings = []
    lead_cells = []
    if isinstance(syst, builder.Builder):
        hoppings.extend(((hop, None, 0) for hop in syst.hoppings()))

        def lead_hoppings(lead):
            sym = lead.symmetry
            for site2, site1 in lead.hoppings():
                shift1 = sym.which(site1)[0]
                shift2 = sym.which(site2)[0]
                # We need to make sure that the hopping is between a site in a
                # fundamental domain and a site with a negative domain.  The
                # direction of the hopping is chosen arbitrarily
                # NOTE(Anton): This may need to be revisited with the future
                # builder format changes.
                shift = max(shift1, shift2)
                yield sym.act([-shift], site2), sym.act([-shift], site1)

        for leadnr, lead in enumerate(syst.leads):
            start = len(hoppings)
            if hasattr(lead, 'builder') and len(lead.interface):
                hoppings.extend(((hop, leadnr, i) for hop in
                                 lead_hoppings(lead.builder) for i in
                                 range(num_lead_cells)))
            lead_cells.append(slice(start, len(hoppings)))
    elif isinstance(syst, system.System):
        def ll_hoppings(syst):
            for i in range(syst.graph.num_nodes):
                for j in syst.graph.out_neighbors(i):
                    if i < j:
                        yield i, j

        hoppings.extend(((hop, None, 0) for hop in ll_hoppings(syst)))
        for leadnr, lead in enumerate(syst.leads):
            start = len(hoppings)
            # We will only plot leads with a graph and with a symmetry.
            if (hasattr(lead, 'graph') and hasattr(lead, 'symmetry') and
                len(syst.lead_interfaces[leadnr])):
                hoppings.extend(((hop, leadnr, i) for hop in ll_hoppings(lead)
                                 for i in range(num_lead_cells)))
            lead_cells.append(slice(start, len(hoppings)))
    else:
        raise TypeError('Unrecognized system type.')
    return hoppings, lead_cells


def sys_leads_hopping_pos(sys, hop_lead_nr):
    """Return arrays of coordinates of all hoppings in a system.

    Parameters
    ----------
    sys : ``~kwant.builder.Builder`` or ``~kwant.system.System`` instance
        The system, coordinates of sites of which should be returned.
    hoppings : list of ``(hopping, leadnr, copynr)`` tuples
        Output of `sys_leads_hoppings` applied to the system.

    Returns
    -------
    coords : (end_site, start_site): tuple of NumPy arrays of floats
        Array of coordinates of the hoppings.  The first half of coordinates
        in each array entry are those of the first site in the hopping, the
        last half are those of the second site.

    Notes
    -----
    This function uses ``site.pos`` property to get the position of a builder
    site and ``sys.pos(sitenr)`` for finalized systems.  This function requires
    that all the positions of all the sites have the same dimensionality.
    """

    syst = sys  # for naming consistency inside function bodies
    is_builder = isinstance(syst, builder.Builder)
    if len(hop_lead_nr) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    num_lead_cells = hop_lead_nr[-1][2] + 1
    if is_builder:
        pos = np.array(ta.array([ta.array(tuple(i[0][0].pos) +
                                          tuple(i[0][1].pos)) for i in
                                 hop_lead_nr]))
    else:
        syst_from_lead = lambda lead: (syst if (lead is None) else
                                      syst.leads[lead])
        pos = ta.array([ta.array(tuple(syst_from_lead(i[1]).pos(i[0][0])) +
                                 tuple(syst_from_lead(i[1]).pos(i[0][1]))) for i
                        in hop_lead_nr])
        pos = np.array(pos)
    if pos.dtype == object:  # Happens if not all the pos are same length.
        raise ValueError("pos attribute of the sites does not have consistent"
                         " values.")
    dim = pos.shape[1]

    def get_vec_domain(lead_nr):
        if lead_nr is None:
            return np.zeros((dim,)), 0
        if is_builder:
            sym = syst.leads[lead_nr].builder.symmetry
            try:
                site = syst.leads[lead_nr].interface[0]
            except IndexError:
                return (0, 0)
        else:
            try:
                sym = syst.leads[lead_nr].symmetry
                site = syst.sites[syst.lead_interfaces[lead_nr][0]]
            except (AttributeError, IndexError):
                # empyt leads or leads without symmetry are not drawn anyways
                return (0, 0)
        dom = sym.which(site)[0] + 1
        vec = np.array(sym.periods)[0]
        return np.r_[vec, vec], dom

    vecs_doms = dict((i, get_vec_domain(i)) for i in range(len(syst.leads)))
    vecs_doms[None] = np.zeros((dim,)), 0
    for k, v in vecs_doms.items():
        vecs_doms[k] = [v[0] * i for i in range(v[1], v[1] + num_lead_cells)]
    pos += [vecs_doms[i[1]][i[2]] for i in hop_lead_nr]
    return np.copy(pos[:, : dim // 2]), np.copy(pos[:, dim // 2:])


# Useful plot functions (to be extended).

defaults = {'site_symbol': {2: 'o', 3: 'o'},
            'site_size': {2: 0.25, 3: 0.5},
            'site_color': {2: 'black', 3: 'white'},
            'site_edgecolor': {2: 'black', 3: 'black'},
            'site_lw': {2: 0, 3: 0.1},
            'hop_color': {2: 'black', 3: 'black'},
            'hop_lw': {2: 0.1, 3: 0},
            'lead_color': {2: 'red', 3: 'red'}}


def plot(sys, num_lead_cells=2, unit='nn',
         site_symbol=None, site_size=None,
         site_color=None, site_edgecolor=None, site_lw=None,
         hop_color=None, hop_lw=None,
         lead_site_symbol=None, lead_site_size=None, lead_color=None,
         lead_site_edgecolor=None, lead_site_lw=None,
         lead_hop_lw=None, pos_transform=None,
         cmap='gray', colorbar=True, file=None,
         show=True, dpi=None, fig_size=None, ax=None):
    """Plot a system in 2 or 3 dimensions.

    Parameters
    ----------
    sys : kwant.builder.Builder or kwant.system.FiniteSystem
        A system to be plotted.
    num_lead_cells : int
        Number of lead copies to be shown with the system.
    unit : 'nn', 'pt', or float
        The unit used to specify symbol sizes and linewidths.
        Possible choices are:

        - 'nn': unit is the shortest hopping or a typical nearst neighbor
          distance in the system if there are no hoppings.  This means that
          symbol sizes/linewidths will scale as the zoom level of the figure is
          changed.  Very short distances are discarded before searching for the
          shortest.  This choice means that the symbols will scale if the
          figure is zoomed.
        - 'pt': unit is points (point = 1/72 inch) in figure space.  This means
          that symbols and linewidths will always be drawn with the same size
          independent of zoom level of the plot.
        - float: sizes are given in units of this value in real (system) space,
          and will accordingly scale as the plot is zoomed.

        The default value is 'nn', which allows to ensure that the images
        neighboring sites do not overlap.

    site_symbol : symbol specification, function, array, or `None`
        Symbol used for representing a site in the plot. Can be specified as

        - 'o': circle with radius of 1 unit.
        - 's': square with inner circle radius of 1 unit.
        - ``('p', nvert, angle)``: regular polygon with ``nvert`` vertices,
          rotated by ``angle``. ``angle`` is given in degrees, and ``angle=0``
          corresponds to one edge of the polygon pointing upward. The
          radius of the inner circle is 1 unit.
        - 'no symbol': no symbol is plotted.
        - 'S', `('P', nvert, angle)`: as the lower-case variants described
          above, but with an area equal to a circle of radius 1. (Makes
          the visual size of the symbol equal to the size of a circle with
          radius 1).
        - matplotlib.path.Path instance.

        Instead of a single symbol, different symbols can be specified
        for different sites by passing a function that returns a valid
        symbol specification for each site, or by passing an array of
        symbols specifications (only for kwant.system.FiniteSystem).
    site_size : number, function, array, or `None`
        Relative (linear) size of the site symbol.
    site_color : ``matplotlib`` color description, function, array, or `None`
        A color used for plotting a site in the system. If a colormap is used,
        it should be a function returning single floats or a one-dimensional
        array of floats.
    site_edgecolor : ``matplotlib`` color description, function, array, or `None`
        Color used for plotting the edges of the site symbols. Only
        valid matplotlib color descriptions are allowed (and no
        combination of floats and colormap as for site_color).
    site_lw : number, function, array, or `None`
        Linewidth of the site symbol edges.
    hop_color : ``matplotlib`` color description or a function
        Same as `site_color`, but for hoppings.  A function is passed two sites
        in this case. (arrays are not allowed in this case).
    hop_lw : number, function, or `None`
        Linewidth of the hoppings.
    lead_site_symbol : symbol specification or `None`
        Symbol to be used for the leads. See `site_symbol` for allowed
        specifications. Note that for leads, only constants
        (i.e. no functions or arrays) are allowed. If None, then
        `site_symbol` is used if it is constant (i.e. no function or array),
        the default otherwise. The same holds for the other lead properties
        below.
    lead_site_size : number or `None`
        Relative (linear) size of the lead symbol
    lead_color : ``matplotlib`` color description or `None`
        For the leads, `num_lead_cells` copies of the lead unit cell
        are plotted. They are plotted in color fading from `lead_color`
        to white (alpha values in `lead_color` are supported) when moving
        from the system into the lead. Is also applied to the
        hoppings.
    lead_site_edgecolor : ``matplotlib`` color description or `None`
        Color of the symbol edges (no fading done).
    lead_site_lw : number or `None`
        Linewidth of the lead symbols.
    lead_hop_lw : number or `None`
        Linewidth of the lead hoppings.
    cmap : ``matplotlib`` color map or a sequence of two color maps or `None`
        The color map used for sites and optionally hoppings.
    pos_transform : function or `None`
        Transformation to be applied to the site position.
    colorbar : bool
        Whether to show a colorbar if colormap is used. Ignored if `ax` is
        provided.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether ``matplotlib.pyplot.show()`` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float or `None`
        Number of pixels per inch.  If not set the ``matplotlib`` default is
        used.
    fig_size : tuple or `None`
        Figure size `(width, height)` in inches.  If not set, the default
        ``matplotlib`` value is used.
    ax : ``matplotlib.axes.Axes`` instance or `None`
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ignored.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.

    Notes
    -----
    - If `None` is passed for a plot property, a default value depending on
      the dimension is chosen. Typically, the default values result in
      acceptable plots.

    - The meaning of "site" depends on whether the system to be plotted is a
      builder or a low level system.  For builders, a site is a
      kwant.builder.Site object.  For low level systems, a site is an integer
      -- the site number.

    - color and symbol definitions may be tuples, but not lists or arrays.
      Arrays of values (linewidths, colors, sizes) may not be tuples.

    - The dimensionality of the plot (2D vs 3D) is inferred from the coordinate
      array.  If there are more than three coordinates, only the first three
      are used.  If there is just one coordinate, the second one is padded with
      zeros.

    - The system is scaled to fit the smaller dimension of the figure, given
      its aspect ratio.

    """
    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plot()")

    syst = sys  # for naming consistency inside function bodies
    # Generate data.
    sites, lead_sites_slcs = sys_leads_sites(syst, num_lead_cells)
    n_syst_sites = sum(i[1] is None for i in sites)
    sites_pos = sys_leads_pos(syst, sites)
    hops, lead_hops_slcs = sys_leads_hoppings(syst, num_lead_cells)
    n_syst_hops = sum(i[1] is None for i in hops)
    end_pos, start_pos = sys_leads_hopping_pos(syst, hops)

    # Choose plot type.
    def resize_to_dim(array):
        if array.shape[1] != dim:
            ar = np.zeros((len(array), dim), dtype=float)
            ar[:, : min(dim, array.shape[1])] = array[
                :, : min(dim, array.shape[1])]
            return ar
        else:
            return array

    loc = locals()

    def check_length(name):
        value = loc[name]
        if name in ('site_size', 'site_lw') and isinstance(value, tuple):
            raise TypeError('{0} may not be a tuple, use list or '
                            'array instead.'.format(name))
        if isinstance(value, (str, tuple)):
            return
        try:
            if len(value) != n_syst_sites:
                raise ValueError('Length of {0} is not equal to number of '
                                 'system sites.'.format(name))
        except TypeError:
            pass

    for name in ['site_symbol', 'site_size', 'site_color', 'site_edgecolor',
                 'site_lw']:
        check_length(name)

    dim = 3 if (sites_pos.shape[1] == 3) else 2
    if dim == 3 and not has3d:
        raise RuntimeError("Installed matplotlib does not support 3d plotting")
    sites_pos = resize_to_dim(sites_pos)
    end_pos = resize_to_dim(end_pos)
    start_pos = resize_to_dim(start_pos)

    # Apply transformations to the data
    if pos_transform is not None:
        sites_pos = np.apply_along_axis(pos_transform, 1, sites_pos)
        end_pos = np.apply_along_axis(pos_transform, 1, end_pos)
        start_pos = np.apply_along_axis(pos_transform, 1, start_pos)

    # Determine the reference length.
    if unit == 'pt':
        reflen = None
    elif unit == 'nn':
        if n_syst_hops:
            # If hoppings are present use their lengths to determine the
            # minimal one.
            distances = end_pos - start_pos
        else:
            # If no hoppings are present, use for the same purpose distances
            # from ten randomly selected points to the remaining points in the
            # system.
            points = _sample_array(sites_pos, 10).T
            distances = (sites_pos.T.reshape(1, -1, dim) -
                         points.reshape(-1, 1, dim)).reshape(-1, dim)
        distances = np.sort(np.sum(distances**2, axis=1))
        # Then check if distances are present that are way shorter than the
        # longest one. Then take first distance longer than these short
        # ones. This heuristic will fail for too large systems, or systems with
        # hoppings that vary by orders and orders of magnitude, but for sane
        # cases it will work.
        long_dist_coord = np.searchsorted(distances, 1e-16 * distances[-1])
        reflen = sqrt(distances[long_dist_coord])

    else:
        # The last allowed value is float-compatible.
        try:
            reflen = float(unit)
        except:
            raise ValueError('Invalid value of unit argument.')

    # make all specs proper: either constant or lists/np.arrays:
    def make_proper_site_spec(spec, fancy_indexing=False):
        if callable(spec):
            spec = [spec(i[0]) for i in sites if i[1] is None]
        if (fancy_indexing and isarray(spec)
            and not isinstance(spec, np.ndarray)):
            try:
                spec = np.asarray(spec)
            except:
                spec = np.asarray(spec, dtype='object')
        return spec

    def make_proper_hop_spec(spec, fancy_indexing=False):
        if callable(spec):
            spec = [spec(*i[0]) for i in hops if i[1] is None]
        if (fancy_indexing and isarray(spec)
            and not isinstance(spec, np.ndarray)):
            try:
                spec = np.asarray(spec)
            except:
                spec = np.asarray(spec, dtype='object')
        return spec

    site_symbol = make_proper_site_spec(site_symbol)
    if site_symbol is None: site_symbol = defaults['site_symbol'][dim]
    # separate different symbols (not done in 3D, the separation
    # would mess up sorting)
    if (isarray(site_symbol) and dim != 3 and
        (len(site_symbol) != 3 or site_symbol[0] not in ('p', 'P'))):
        symbol_dict = defaultdict(list)
        for i, symbol in enumerate(site_symbol):
            symbol_dict[symbol].append(i)
        symbol_slcs = []
        for symbol, indx in symbol_dict.items():
            symbol_slcs.append((symbol, np.array(indx)))
        fancy_indexing = True
    else:
        symbol_slcs = [(site_symbol, slice(n_syst_sites))]
        fancy_indexing = False

    site_size = make_proper_site_spec(site_size, fancy_indexing)
    site_color = make_proper_site_spec(site_color, fancy_indexing)
    site_edgecolor = make_proper_site_spec(site_edgecolor, fancy_indexing)
    site_lw = make_proper_site_spec(site_lw, fancy_indexing)

    hop_color = make_proper_hop_spec(hop_color)
    hop_lw = make_proper_hop_spec(hop_lw)

    # Choose defaults depending on dimension, if None was given
    if site_size is None: site_size = defaults['site_size'][dim]
    if site_color is None: site_color = defaults['site_color'][dim]
    if site_edgecolor is None:
        site_edgecolor = defaults['site_edgecolor'][dim]
    if site_lw is None: site_lw = defaults['site_lw'][dim]

    if hop_color is None: hop_color = defaults['hop_color'][dim]
    if hop_lw is None: hop_lw = defaults['hop_lw'][dim]

    # if symbols are split up into different collections,
    # the colormapping will fail without normalization
    norm = None
    if len(symbol_slcs) > 1:
        try:
            if site_color.ndim == 1 and len(site_color) == n_syst_sites:
                site_color = np.asarray(site_color, dtype=float)
                norm = matplotlib.colors.Normalize(site_color.min(),
                                                   site_color.max())
        except:
            pass

    # take spec also for lead, if it's not a list/array, default, otherwise
    if lead_site_symbol is None:
        lead_site_symbol = (site_symbol if not isarray(site_symbol)
                            else defaults['site_symbol'][dim])
    if lead_site_size is None:
        lead_site_size = (site_size if not isarray(site_size)
                          else defaults['site_size'][dim])
    if lead_color is None:
        lead_color = defaults['lead_color'][dim]
    lead_color = matplotlib.colors.colorConverter.to_rgba(lead_color)

    if lead_site_edgecolor is None:
        lead_site_edgecolor = (site_edgecolor if not isarray(site_edgecolor)
                               else defaults['site_edgecolor'][dim])
    if lead_site_lw is None:
        lead_site_lw = (site_lw if not isarray(site_lw)
                        else defaults['site_lw'][dim])
    if lead_hop_lw is None:
        lead_hop_lw = (hop_lw if not isarray(hop_lw)
                       else defaults['hop_lw'][dim])

    hop_cmap = None
    if not isinstance(cmap, str):
        try:
            cmap, hop_cmap = cmap
        except TypeError:
            pass

    # make a new figure unless axes specified
    if not ax:
        fig = _make_figure(dpi, fig_size)
        if dim == 2:
            ax = fig.add_subplot(1, 1, 1, aspect='equal')
            ax.set_xmargin(0.05)
            ax.set_ymargin(0.05)
        else:
            warnings.filterwarnings('ignore', message=r'.*rotation.*')
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            warnings.resetwarnings()
    else:
        fig = None

    # plot system sites and hoppings
    for symbol, slc in symbol_slcs:
        size = site_size[slc] if isarray(site_size) else site_size
        col = site_color[slc] if isarray(site_color) else site_color
        edgecol = (site_edgecolor[slc] if isarray(site_edgecolor) else
                   site_edgecolor)
        lw = site_lw[slc] if isarray(site_lw) else site_lw

        symbol_coll = symbols(ax, sites_pos[slc], size=size,
                              reflen=reflen, symbol=symbol,
                              facecolor=col, edgecolor=edgecol,
                              linewidth=lw, cmap=cmap, norm=norm, zorder=2)

    end, start = end_pos[: n_syst_hops], start_pos[: n_syst_hops]
    line_coll = lines(ax, end, start, reflen, hop_color, linewidths=hop_lw,
                      zorder=1, cmap=hop_cmap)

    # plot lead sites and hoppings
    norm = matplotlib.colors.Normalize(-0.5, num_lead_cells - 0.5)
    cmap_from_list = matplotlib.colors.LinearSegmentedColormap.from_list
    lead_cmap = cmap_from_list(None, [lead_color, (1, 1, 1, lead_color[3])])

    for sites_slc, hops_slc in zip(lead_sites_slcs, lead_hops_slcs):
        lead_site_colors = np.array([i[2] for i in sites[sites_slc]],
                                    dtype=float)

        # Note: the previous version of the code had in addition this
        # line in the 3D case:
        # lead_site_colors = 1 / np.sqrt(1. + lead_site_colors)
        symbols(ax, sites_pos[sites_slc], size=lead_site_size, reflen=reflen,
                symbol=lead_site_symbol, facecolor=lead_site_colors,
                edgecolor=lead_site_edgecolor, linewidth=lead_site_lw,
                cmap=lead_cmap, zorder=2, norm=norm)

        lead_hop_colors = np.array([i[2] for i in hops[hops_slc]], dtype=float)

        # Note: the previous version of the code had in addition this
        # line in the 3D case:
        # lead_hop_colors = 1 / np.sqrt(1. + lead_hop_colors)
        end, start = end_pos[hops_slc], start_pos[hops_slc]
        lines(ax, end, start, reflen, lead_hop_colors, linewidths=lead_hop_lw,
              cmap=lead_cmap, norm=norm, zorder=1)

    min_ = np.min(sites_pos, 0)
    max_ = np.max(sites_pos, 0)
    m = (min_ + max_) / 2
    if dim == 2:
        w = np.max([(max_ - min_) / 2, (reflen, reflen)], axis=0)
        ax.update_datalim((m - w, m + w))
        ax.autoscale_view(tight=True)
    else:
        # make axis limits the same in all directions
        # (3D only works decently for equal aspect ratio. Since
        #  this doesn't work out of the box in mplot3d, this is a
        #  workaround)
        w = np.max(max_ - min_) / 2
        ax.auto_scale_xyz(*[(i - w, i + w) for i in m], had_data=True)

    # add separate colorbars for symbols and hoppings if ncessary
    if symbol_coll.get_array() is not None and colorbar and fig is not None:
        fig.colorbar(symbol_coll)
    if line_coll.get_array() is not None and colorbar and fig is not None:
        fig.colorbar(line_coll)

    if fig is not None:
        return output_fig(fig, file=file, show=show)


def mask_interpolate(coords, values, a=None, method='nearest', oversampling=3):
    """Interpolate a scalar function in vicinity of given points.

    Create a masked array corresponding to interpolated values of the function
    at points lying not further than a certain distance from the original
    data points provided.

    Parameters
    ----------
    coords : np.ndarray
        An array with site coordinates.
    values : np.ndarray
        An array with the values from which the interpolation should be built.
    a : float, optional
        Reference length.  If not given, it is determined as a typical
        nearest neighbor distance.
    method : string, optional
        Passed to ``scipy.interpolate.griddata``: "nearest" (default), "linear",
        or "cubic"
    oversampling : integer, optional
        Number of pixels per reference length.  Defaults to 3.

    Returns
    -------
    array : 2d NumPy array
        The interpolated values.
    min, max : vectors
        The real-space coordinates of the two extreme ([0, 0] and [-1, -1])
        points of ``array``.

    Notes
    -----
    - `min` and `max` are chosen such that when plotting a system on a square
      lattice and `oversampling` is set to an odd integer, each site will lie
      exactly at the center of a pixel of the output array.

    - When plotting a system on a square lattice and `method` is "nearest", it
      makes sense to set `oversampling` to ``1``.  Then, each site will
      correspond to exactly one pixel in the resulting array.
    """
    # Build the bounding box.
    cmin, cmax = coords.min(0), coords.max(0)

    tree = spatial.cKDTree(coords)

    # Select 10 sites to compare -- comparing them all is too costly.
    points = _sample_array(coords, 10)
    min_dist = np.min(tree.query(points, 2)[0][:, 1])
    if min_dist < 1e-6 * np.linalg.norm(cmax - cmin):
        warnings.warn("Some sites have nearly coinciding positions, "
                      "interpolation may be confusing.",
                      RuntimeWarning)

    if a is None:
        a = min_dist

    if a < 1e-6 * np.linalg.norm(cmax - cmin):
        raise ValueError("The reference distance a is too small.")

    if len(coords) != len(values):
        raise ValueError("The number of sites doesn't match the number of"
                         "provided values.")

    shape = (((cmax - cmin) / a + 1) * oversampling).round()
    delta = 0.5 * (oversampling - 1) * a / oversampling
    cmin -= delta
    cmax += delta
    dims = tuple(slice(cmin[i], cmax[i], 1j * shape[i]) for i in
                 range(len(cmin)))
    grid = tuple(np.ogrid[dims])
    img = interpolate.griddata(coords, values, grid, method)
    mask = np.mgrid[dims].reshape(len(cmin), -1).T
    # The numerical values in the following line are optimized for the common
    # case of a square lattice:
    # * 0.99 makes sure that non-masked pixels and sites correspond 1-by-1 to
    #   each other when oversampling == 1.
    # * 0.4 (which is just below sqrt(2) - 1) makes tree.query() exact.
    mask = tree.query(mask, eps=0.4)[0] > 0.99 * a

    return np.ma.masked_array(img, mask), cmin, cmax


def map(sys, value, colorbar=True, cmap=None, vmin=None, vmax=None, a=None,
        method='nearest', oversampling=3, num_lead_cells=0, file=None,
        show=True, dpi=None, fig_size=None, ax=None, pos_transform=None,
        background='#e0e0e0'):
    """Show interpolated map of a function defined for the sites of a system.

    Create a pixmap representation of a function of the sites of a system by
    calling `~kwant.plotter.mask_interpolate` and show this pixmap using
    matplotlib.

    Parameters
    ----------
    sys : kwant.system.FiniteSystem or kwant.builder.Builder
        The system for whose sites `value` is to be plotted.
    value : function or list
        Function which takes a site and returns a value if the system is a
        builder, or a list of function values for each system site of the
        finalized system.
    colorbar : bool, optional
        Whether to show a color bar if numerical data has to be plotted.
        Defaults to `True`. If `ax` is provided, the colorbar is never plotted.
    cmap : ``matplotlib`` color map or `None`
        The color map used for sites and optionally hoppings, if `None`,
        ``matplotlib`` default is used.
    vmin : float, optional
        The lower saturation limit for the colormap; values returned by
        `value` which are smaller than this will saturate
    vmax : float, optional
        The upper saturation limit for the colormap; valued returned by
        `value` which are larger than this will saturate
    a : float, optional
        Reference length.  If not given, it is determined as a typical
        nearest neighbor distance.
    method : string, optional
        Passed to ``scipy.interpolate.griddata``: "nearest" (default), "linear",
        or "cubic"
    oversampling : integer, optional
        Number of pixels per reference length.  Defaults to 3.
    num_lead_cells : integer, optional
        number of lead unit cells that should be plotted to indicate
        the position of leads. Defaults to 0.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether ``matplotlib.pyplot.show()`` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    ax : ``matplotlib.axes.Axes`` instance or `None`
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ignored.
    pos_transform : function or `None`
        Transformation to be applied to the site position.
    background : matplotlib color spec
        Areas without sites are filled with this color.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.

    Notes
    -----
    - When plotting a system on a square lattice and `method` is "nearest", it
      makes sense to set `oversampling` to ``1``.  Then, each site will
      correspond to exactly one pixel.
    """

    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for map()")

    syst = sys  # for naming consistency inside function bodies
    sites = sys_leads_sites(syst, 0)[0]
    coords = sys_leads_pos(syst, sites)

    if pos_transform is not None:
        coords = np.apply_along_axis(pos_transform, 1, coords)

    if coords.shape[1] != 2:
        raise ValueError('Only 2D systems can be plotted this way.')

    if callable(value):
        value = [value(site[0]) for site in sites]
    else:
        if not isinstance(syst, system.FiniteSystem):
            raise ValueError('List of values is only allowed as input '
                             'for finalized systems.')
    value = np.array(value)
    img, min, max = mask_interpolate(coords, value, a, method, oversampling)
    border = 0.5 * (max - min) / (np.asarray(img.shape) - 1)
    min -= border
    max += border
    if ax is None:
        fig = _make_figure(dpi, fig_size)
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
    else:
        fig = None

    if cmap is None:
        cmap = _colormaps.kwant_red

    # Note that we tell imshow to show the array created by mask_interpolate
    # faithfully and not to interpolate by itself another time.
    image = ax.imshow(img.T, extent=(min[0], max[0], min[1], max[1]),
                      origin='lower', interpolation='none', cmap=cmap,
                      vmin=vmin, vmax=vmax)
    if num_lead_cells:
        plot(syst, num_lead_cells, site_symbol='no symbol', hop_lw=0,
             lead_site_symbol='s', lead_site_size=0.501, lead_site_lw=0,
             lead_hop_lw=0, lead_color='black', colorbar=False, ax=ax)

    ax.patch.set_facecolor(background)

    if colorbar and fig is not None:
        fig.colorbar(image)

    if fig is not None:
        return output_fig(fig, file=file, show=show)


def bands(sys, args=(), momenta=65, file=None, show=True, dpi=None,
          fig_size=None, ax=None, *, params=None):
    """Plot band structure of a translationally invariant 1D system.

    Parameters
    ----------
    sys : kwant.system.InfiniteSystem
        A system bands of which are to be plotted.
    args : tuple, defaults to empty
        Positional arguments to pass to the ``hamiltonian`` method.
        Mutally exclusive with 'params'.
    momenta : int or 1D array-like
        Either a number of sampling points on the interval [-pi, pi], or an
        array of points at which the band structure has to be evaluated.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether ``matplotlib.pyplot.show()`` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float
        Number of pixels per inch.  If not set the ``matplotlib`` default is
        used.
    fig_size : tuple
        Figure size `(width, height)` in inches.  If not set, the default
        ``matplotlib`` value is used.
    ax : ``matplotlib.axes.Axes`` instance or `None`
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ignored.
    params : dict, optional
        Dictionary of parameter names and their values. Mutually exclusive
        with 'args'.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.

    Notes
    -----
    See `~kwant.physics.Bands` for the calculation of dispersion without plotting.
    """

    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for bands()")

    syst = sys  # for naming consistency inside function bodies
    _common.ensure_isinstance(syst, system.InfiniteSystem)

    momenta = np.array(momenta)
    if momenta.ndim != 1:
        momenta = np.linspace(-np.pi, np.pi, momenta)

    # expand out the contents of 'physics.Bands' to get the H(k),
    # because 'spectrum' already does the diagonalisation.
    ham = syst.cell_hamiltonian(args, params=params)
    if not np.allclose(ham, ham.conjugate().transpose()):
        raise ValueError('The cell Hamiltonian is not Hermitian.')
    _hop = syst.inter_cell_hopping(args, params=params)
    hop = np.empty(ham.shape, dtype=complex)
    hop[:, :_hop.shape[1]] = _hop
    hop[:, _hop.shape[1]:] = 0

    def h_k(k):
        # H_k = H_0 + V e^-ik + V^\dagger e^ik
        mat = hop * cmath.exp(-1j * k)
        mat +=  mat.conjugate().transpose() + ham
        return mat

    return spectrum(h_k, ('k', momenta), file=file, show=show, dpi=dpi,
                    fig_size=fig_size, ax=ax)


def spectrum(syst, x, y=None, params=None, mask=None, file=None,
             show=True, dpi=None, fig_size=None, ax=None):
    """Plot the spectrum of a Hamiltonian as a function of 1 or 2 parameters

    Parameters
    ----------
    syst : `kwant.system.FiniteSystem` or callable
        If a function, then it must take named parameters and return the
        Hamiltonian as a dense matrix.
    x : pair ``(name, values)``
        Parameter to ``ham`` that will be varied. Consists of the
        parameter name, and a sequence of parameter values.
    y : pair ``(name, values)``, optional
        Used for 3D plots (same as ``x``). If provided, then the cartesian
        product of the ``x`` values and these values will be used as a grid
        over which to evaluate the spectrum.
    params : dict, optional
        The rest of the parameters to ``ham``, which will be kept constant.
    mask : callable, optional
        Takes the parameters specified by ``x`` and ``y`` and returns True
        if the spectrum should not be calculated for the given parameter
        values.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether ``matplotlib.pyplot.show()`` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float
        Number of pixels per inch.  If not set the ``matplotlib`` default is
        used.
    fig_size : tuple
        Figure size `(width, height)` in inches.  If not set, the default
        ``matplotlib`` value is used.
    ax : ``matplotlib.axes.Axes`` instance or `None`
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ignored.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.
    """

    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for plot_spectrum()")
    if y is not None and not has3d:
        raise RuntimeError("Installed matplotlib does not support 3d plotting")

    if isinstance(syst, system.FiniteSystem):
        def ham(**kwargs):
            return syst.hamiltonian_submatrix(params=kwargs, sparse=False)
    elif callable(syst):
        ham = syst
    else:
        raise TypeError("Expected 'syst' to be a finite Kwant system "
                        "or a function.")

    params = params or dict()
    keys = (x[0],) if y is None else (x[0], y[0])
    array_values = (x[1],) if y is None else (x[1], y[1])

    # calculate spectrum on the grid of points
    spectrum = []
    bound_ham = functools.partial(ham, **params)
    for point in itertools.product(*array_values):
        p = dict(zip(keys, point))
        if mask and mask(**p):
            spectrum.append(None)
        else:
            h_p = np.atleast_2d(bound_ham(**p))
            spectrum.append(np.linalg.eigvalsh(h_p))
    # massage masked grid points into a list of NaNs of the appropriate length
    n_eigvals = len(next(filter(lambda s: s is not None, spectrum)))
    nan_list = [np.nan] * n_eigvals
    spectrum = [nan_list if s is None else s for s in spectrum]
    # make into a numpy array and reshape
    new_shape = [len(v) for v in array_values] + [-1]
    spectrum = np.array(spectrum).reshape(new_shape)

    # set up axes
    if ax is None:
        fig = Figure()
        if dpi is not None:
            fig.set_dpi(dpi)
        if fig_size is not None:
            fig.set_figwidth(fig_size[0])
            fig.set_figheight(fig_size[1])
        if y is None:
            ax = fig.add_subplot(1, 1, 1)
        else:
            warnings.filterwarnings('ignore',
                                    message=r'.*mouse rotation disabled.*')
            ax = fig.add_subplot(1, 1, 1, projection='3d')
            warnings.resetwarnings()
        ax.set_xlabel(keys[0])
        if y is None:
            ax.set_ylabel('Energy')
        else:
            ax.set_ylabel(keys[1])
            ax.set_zlabel('Energy')
        ax.set_title(', '.join('{} = {}'.format(*kv) for kv in params.items()))
    else:
        fig = None

    # actually do the plot
    if y is None:
        ax.plot(array_values[0], spectrum)
    else:
        if not hasattr(ax, 'plot_surface'):
            msg = ("When providing an axis for plotting over a 2D domain the "
                   "axis should be created with 'projection=\"3d\"")
            raise TypeError(msg)
        # plot_surface cannot directly handle rank-3 values, so we
        # explicitly loop over the last axis
        grid = np.meshgrid(*array_values)
        for i in range(spectrum.shape[-1]):
            spec = spectrum[:, :, i].transpose()  # row-major to x-y ordering
            ax.plot_surface(*(grid + [spec]), cstride=1, rstride=1)

    if fig is not None:
        return output_fig(fig, file=file, show=show)


def interpolate_current(syst, current, relwidth=None, abswidth=None, n=9):
    """Interpolate currents in a system onto a regular grid.

    The system graph together with current intensities defines a "discrete"
    current density field where the current density is non-zero only on the
    straight lines that connect sites that are coupled by a hopping term.

    To make this vector field easier to visualize and interpret at different
    length scales, it is smoothed by convoluting it with the bell-shaped bump
    function ``f(r) = max(1 - (2*r / width)**2, 0)**2``.  The bump width is
    determined by the `max_res` and `width` parameters.

    This routine samples the smoothed field on a regular (square or cubic)
    grid.

    Parameters
    ----------
    syst : A finalized system
        The system on which we are going to calculate the field.
    current : '1D array of float'
        Must contain the intensity on each hoppings in the same order that they
        appear in syst.graph.
    relwidth : float or `None`
        Relative width of the bumps used to generate the field, as a fraction
        of the length of the longest side of the bounding box.  This argument
        is only used if `abswidth` is not given.
    abswidth : float or `None`
        Absolute width ot the bumps used to generate the field.  Takes
        precedence over `relwidth`.  If neither is given, the bump width is set
        to four times the length of the shortest hopping.
    n : int
        Number of points the grid must have over the width of the bump.

    Returns
    -------
    field : n-d arraylike of float
        n-d array of n-d vectors.
    box : sequence of 2-sequences of float
        the extents of `field`: ((x0, x1), (y0, y1), ...)

    """
    if not isinstance(syst, builder.FiniteSystem):
        raise TypeError("The system needs to be finalized.")

    if len(current) != syst.graph.num_edges:
        raise ValueError("Current and hoppings arrays do not have the same"
                         " length.")

    # hops: hoppings (pairs of points)
    dim = len(syst.sites[0].pos)
    hops = np.empty((syst.graph.num_edges // 2, 2, dim))
    # Take the average of the current flowing each way along the hoppings
    current_one_way = np.empty(syst.graph.num_edges // 2)
    seen_hoppings = dict()
    kprime = 0
    for k, (i, j) in enumerate(syst.graph):
        if (j, i) in seen_hoppings:
            current_one_way[seen_hoppings[j, i]] -= current[k]
        else:
            current_one_way[kprime] = current[k]
            hops[kprime][0] = syst.sites[j].pos
            hops[kprime][1] = syst.sites[i].pos
            seen_hoppings[i, j] = kprime
            kprime += 1
    current = current_one_way / 2

    min_hops = np.min(hops, 1)
    max_hops = np.max(hops, 1)
    bbox_min = np.min(min_hops, 0)
    bbox_max = np.max(max_hops, 0)
    bbox_size = bbox_max - bbox_min

    # lens: scaled lengths of hoppings
    # dirs: normalized directions of hoppings
    dirs = hops[:, 1] - hops[:, 0]
    lens = np.sqrt(np.sum(dirs * dirs, -1))
    dirs /= lens[:, None]

    if abswidth is None:
        if relwidth is None:
            unique_lens = np.unique(lens)
            longest = unique_lens[-1]
            for shortest_nonzero in unique_lens:
                if shortest_nonzero / longest < 1e-5:
                    break
            width = 4 * shortest_nonzero
        else:
            width = relwidth * np.max(bbox_size)
    else:
        width = abswidth

    # TODO: Generalize 'factor' prefactor to arbitrary dimensions and remove
    #       this check. This check is done here to keep changes local
    if dim != 2:
        raise ValueError("'interpolate_current' only works for 2D systems.")
    factor = (3 / np.pi) / (width / 2)
    scale = 2 / width
    lens *= scale

    # Create field array.
    field_shape = np.zeros(dim + 1, int)
    field_shape[dim] = dim
    for d in range(dim):
        field_shape[d] = int(bbox_size[d] * n / width + 1.5*n)
        if field_shape[d] % 2:
            field_shape[d] += 1
    field = np.zeros(field_shape)

    region = [np.linspace(bbox_min[d] - 0.75*width,
                          bbox_max[d] + 0.75*width,
                          field_shape[d])
              for d in range(dim)]

    grid_density = (field_shape[:dim] - 1) / (bbox_max + 1.5*width - bbox_min)
    slices = np.empty((len(hops), dim, 2), int)
    slices[:, :, 0] = np.floor((min_hops - bbox_min) * grid_density)
    slices[:, :, 1] = np.ceil((max_hops + 1.5*width - bbox_min) * grid_density)

    # F is the antiderivative of the smoothing function
    # f(ρ, z) = (1 - ρ^2 - z^2)^2 Θ(1 - ρ^2 - z^2), with respect to
    # z,with F(ρ, -∞) = 0 and where Θ is the heaviside function.
    def F(rho, z):
        r = 1 - rho * rho
        r[r < 0] = 0
        r = np.sqrt(r)
        m = np.clip(z, -r, r)
        rr = r * r
        rrrr = rr * rr
        mm = m * m
        return m * (mm * (mm/5 - (2/3) * rr) + rrrr) + (8 / 15) * rrrr * r

    # Interpolate the field for each hopping.
    for i in range(len(current)):

        if not np.diff(slices[i]).all():
            # Zero volume: nothing to do.
            continue

        field_slice = [slice(*slices[i, d]) for d in range(dim)]

        # Coordinates of the grid points that are within range of the current
        # hopping.
        coords = np.meshgrid(*[region[d][field_slice[d]] for d in range(dim)],
                             sparse=True, indexing='ij')
        coords -= hops[i, 0]

        # Convert "coords" into scaled cylindrical coordinates with regard to
        # the hopping.
        z = np.dot(coords, dirs[i])
        rho = np.sqrt(np.abs(np.sum(coords * coords) - z*z))
        z *= scale
        rho *= scale

        magns = F(rho, z) - F(rho, z - lens[i])
        magns *= current[i] * factor

        field[field_slice] += dirs[i] * magns[..., None]

    # 'field' contains contributions from both hoppings (i, j) and (j, i)
    return field, ((region[0][0], region[0][-1]), (region[1][0], region[1][-1]))


def _gamma_compress(linear):
    """Compress linear sRGB into sRGB."""
    if linear <= 0.0031308:
        return 12.92 * linear
    else:
        a = 0.055
        return (1 + a) * linear ** (1 / 2.4) - a

_gamma_compress = np.vectorize(_gamma_compress, otypes=[float])


def _gamma_expand(corrected):
    """Expand sRGB into linear sRGB."""
    if corrected <= 0.04045:
        return corrected / 12.92
    else:
        a = 0.055
        return ((corrected + a) / (1 + a))**2.4

_gamma_expand = np.vectorize(_gamma_expand, otypes=[float])


def _linear_cmap(a, b):
    """Make a colormap that linearly interpolates between the colors a and b."""
    a = matplotlib.colors.colorConverter.to_rgb(a)
    b = matplotlib.colors.colorConverter.to_rgb(b)
    a_linear = _gamma_expand(a)
    b_linear = _gamma_expand(b)
    color_diff = a_linear - b_linear
    palette = (np.linspace(0, 1, 256).reshape((-1, 1))
               * color_diff.reshape((1, -1)))
    palette += b_linear
    palette = _gamma_compress(palette)
    return matplotlib.colors.ListedColormap(palette)


def streamplot(field, box, cmap=None, bgcolor=None, linecolor='k',
               max_linewidth=3, min_linewidth=1, density=2/9,
               colorbar=True, file=None,
               show=True, dpi=None, fig_size=None, ax=None):
    """Draw streamlines of a flow field in Kwant style

    Solid colored streamlines are drawn, superimposed on a color plot of the
    flow speed that may be disabled by setting `bgcolor`.  The width of the
    streamlines is proportional to the flow speed.  Lines that would be thinner
    than `min_linewidth` are blended into the background color in order to
    create the illusion of arbitrarily thin lines.  (This is done because some
    plot backends like PDF do not support lines of arbitrary width.)

    Internally, this routine uses ``matplotlib.pyplot.streamplot``.

    Parameters
    ----------
    field : 3d arraylike of float
        2d array of 2d vectors.
    box : 2-sequence of 2-sequences of float
        the extents of `field`: ((x0, x1), (y0, y1))
    cmap : colormap, optional
        Colormap for the background color plot.  When not set the colormap
        "kwant_red" is used by default, unless `bgcolor` is set.
    bgcolor : color definition, optional
        The solid color of the background.  Mutually exclusive with `cmap`.
    linecolor : color definition
        Color of the flow lines.
    max_linewidth : float
        Width of lines at maximum flow speed.
    min_linewidth : float
        Minimum width of lines before blending into the background color begins.
    density : float
        Number of flow lines per point of the field.  The default value is
        chosen to show two lines per default width of the interpolation bump of
        `~kwant.plotter.interpolate_current`.
    colorbar : bool
        Whether to show a colorbar if a colormap is used. Ignored if `ax` is
        provided.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether ``matplotlib.pyplot.show()`` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float or `None`
        Number of pixels per inch.  If not set the ``matplotlib`` default is
        used.
    fig_size : tuple or `None`
        Figure size `(width, height)` in inches.  If not set, the default
        ``matplotlib`` value is used.
    ax : ``matplotlib.axes.Axes`` instance or `None`
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ignored.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.

    """
    if not mpl_enabled:
        raise RuntimeError("matplotlib was not found, but is required "
                           "for current()")

    # Matplotlib's "density" is in units of 30 streamlines...
    density *= 1 / 30 * ta.array(field.shape[:2], int)

    # Matplotlib plots images like matrices: image[y, x].  We use the opposite
    # convention: image[x, y].  Hence, it is necessary to transpose.
    field = field.transpose(1, 0, 2)

    if field.shape[-1] != 2 or field.ndim != 3:
        raise ValueError("Only 2D field can be plotted.")

    if bgcolor is None:
        if cmap is None:
            cmap = _colormaps.kwant_red
        cmap = matplotlib.cm.get_cmap(cmap)
        bgcolor = cmap(0)[:3]
    elif cmap is not None:
        raise ValueError("The parameters 'cmap' and 'bgcolor' are "
                         "mutually exclusive.")

    if ax is None:
        fig = _make_figure(dpi, fig_size)
        ax = fig.add_subplot(1, 1, 1, aspect='equal')
    else:
        fig = None

    X = np.linspace(*box[0], num=field.shape[1])
    Y = np.linspace(*box[1], num=field.shape[0])

    speed = np.linalg.norm(field, axis=-1)

    if cmap is None:
        ax.set_axis_bgcolor(bgcolor)
    else:
        image = ax.imshow(speed, cmap=cmap,
                          interpolation='bicubic',
                          extent=[e for c in box for e in c],
                          origin='lower')

    linewidth = max_linewidth / (np.max(speed) or 1) * speed
    color = linewidth / min_linewidth
    linewidth[linewidth < min_linewidth] = min_linewidth
    color[color > 1] = 1

    line_cmap = _linear_cmap(linecolor, cmap(0))

    ax.streamplot(X, Y, field[:,:,0], field[:,:,1],
                  density=density, linewidth=linewidth,
                  color=color, cmap=line_cmap, arrowstyle='->')

    ax.set_xlim(*box[0])
    ax.set_ylim(*box[1])

    if fig is not None:
        if colorbar and bgcolor is None:
            fig.colorbar(image)
        return output_fig(fig, file=file, show=show)


def current(syst, current, relwidth=0.05, **kwargs):
    """Show an interpolated current defined for the hoppings of a system.

    The system graph together with current intensities defines a "discrete"
    current density field where the current density is non-zero only on the
    straight lines that connect sites that are coupled by a hopping term.

    To make this vector field easier to visualize and interpret at different
    length scales, it is smoothed by convoluting it with the bell-shaped bump
    function ``f(r) = max(1 - (2*r / width)**2, 0)**2``.  The bump width is
    determined by the `limit` and `width` parameters.

    This routine samples the smoothed field on a regular (square or cubic) grid
    and displays it using an enhanced variant of matplotlib's streamplot.

    This is a convenience function that is equivalent to
    ``streamplot(*interpolate_current(syst, current, relwidth), **kwargs)``.
    The longer form makes it possible to tweak additional options of
    `~kwant.plotter.interpolate_current`.

    Parameters
    ----------
    syst : `kwant.system.FiniteSystem`
        The system for which to plot the ``current``.
    current : sequence of float
        Sequence of values defining currents on each hopping of the system.
        Ordered in the same way as ``syst.graph``. This typically will be
        the result of evaluating a `~kwant.operator.Current` operator.
    relwidth : float or `None`
        Relative width of the bumps used to generate the field, as a fraction
        of the length of the longest side of the bounding box.
    **kwargs : various
        Keyword args to be passed verbatim to `~kwant.plotter.streamplot`.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.

    """
    return streamplot(*interpolate_current(syst, current, relwidth),
                      **kwargs)


# TODO (Anton): Fix plotting of parts of the system using color = np.nan.
# Not plotting sites currently works, not plotting hoppings does not.
# TODO (Anton): Allow a more flexible treatment of position than pos_transform
# (an interface for user-defined pos).
