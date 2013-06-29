# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Plotter module for kwant.

This module provides iterators useful for any plotter routine, such as a list
of system sites, their coordinates, lead sites at any lead unit cell, etc.  If
`matplotlib` is available, it also provides simple functions for plotting the
system in two or three dimensions.
"""

from collections import defaultdict
import warnings
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
    from matplotlib.figure import Figure
    from matplotlib import collections
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    _mpl_enabled = True
    try:
        from mpl_toolkits import mplot3d
        has3d = True
    except ImportError:
        warnings.warn("3D plotting not available.", RuntimeWarning)
except ImportError:
    warnings.warn("matplotlib is not available, only iterator-providing"
                  "functions will work.", RuntimeWarning)
    _mpl_enabled = False

from . import system, builder, physics

__all__ = ['plot', 'map', 'bands', 'sys_leads_sites', 'sys_leads_hoppings',
           'sys_leads_pos', 'sys_leads_hopping_pos', 'mask_interpolate']

# Collections that allow for symbols and linewiths to be given in data space
# (not for general use, only implement what's needed for plotter)

def _isarray(var):
    if hasattr(var, '__getitem__') and not isinstance(var, basestring):
        return True
    else:
        return False

def _nparray_if_array(var):
    return np.asarray(var) if _isarray(var) else var

class LineCollection(collections.LineCollection):
    def __init__(self, segments, reflen=None, **kwargs):
        collections.LineCollection.__init__(self, segments,
                                                       **kwargs)
        self.reflen = reflen


    def set_linewidths(self, linewidths):
        self._linewidths_orig = _nparray_if_array(linewidths)


    def draw(self, renderer):
        if self.reflen:
            # Note: only works for aspect ratio 1!
            #       72.0 - there is 72 points in an inch
            factor = (self.axes.transData.frozen().to_values()[0] /
                      self.figure.dpi * 72.0 * self.reflen)
            collections.LineCollection.set_linewidths(self,
                                         self._linewidths_orig * factor)
        else:
            collections.LineCollection.set_linewidths(self,
                                                  self._linewidths_orig)

        return collections.LineCollection.draw(self, renderer)


class PathCollection(collections.PathCollection):
    def __init__(self, paths, sizes=None, reflen=None,
                 **kwargs):
        collections.PathCollection.__init__(self, paths,
                                                       sizes=sizes,
                                                       **kwargs)

        self.reflen = reflen
        self._linewidths_orig = _nparray_if_array(self.get_linewidths())

        self._transforms = [
            matplotlib.transforms.Affine2D().scale(x)
            for x in sizes]

    def get_transforms(self):
        return self._transforms

    def get_transform(self):
        if self.reflen:
            # For the paths, use the data transformation but
            # strip the offset (will be added later with offsets)
            a, b, c, d, e, f = self.axes.transData.frozen().to_values()
            return matplotlib.transforms.Affine2D().from_values(a, b, c, d,
                                              0, 0).scale(self.reflen)
        else:
            return matplotlib.transforms.Affine2D().scale(
                (self.figure.dpi / 72.0))

    def draw(self, renderer):
        if self.reflen:
            # Note: only works for aspect ratio 1!
            factor = (self.axes.transData.frozen().to_values()[0] /
                      self.figure.dpi * 72.0 * self.reflen)
            self.set_linewidths(self._linewidths_orig * factor)

        return collections.Collection.draw(self, renderer)


if has3d:
    # 3D is optional
    _sort3d = True

    # Compute the projection of a 3D length into 2D data coordinates
    # for this we use 2 3D half-circles that are projected into 2D
    # (this gives the same length as projecting the full unit sphere)

    _xs = []
    _ys = []
    _zs = []

    for phi in np.linspace(0, pi, 21):
        _xs.append(cos(phi))
        _ys.append(sin(phi))
        _zs.append(0)
        _ys.append(cos(phi))
        _zs.append(sin(phi))
        _xs.append(0)

    _unit_sphere = np.array([_xs, _ys, _zs])

    def projected_length(ax, length):
        xc = sum(ax.get_xlim3d())/2.0
        yc = sum(ax.get_ylim3d())/2.0
        zc = sum(ax.get_zlim3d())/2.0

        xs = _unit_sphere[0] * length + xc
        ys = _unit_sphere[1] * length + yc
        zs = _unit_sphere[2] * length + zc

        xp, yp, _ = mplot3d.proj3d.proj_transform(xs, ys, zs, ax.get_proj())
        xc, yc, _ = mplot3d.proj3d.proj_transform(xc, yc, zc, ax.get_proj())

        coords = np.array([xp, yp]) - np.repeat([[xc], [yc]], len(xs), axis=1)
        return sqrt(np.sum(coords**2, axis=0).max())


    class Line3DCollection(mplot3d.art3d.Line3DCollection):
        def __init__(self, segments, reflen=None, zorder=0,
                     **kwargs):
            mplot3d.art3d.Line3DCollection.__init__(self, segments,
                                                    **kwargs)
            self.reflen = reflen
            self._zorder3d = zorder

        def set_linewidths(self, linewidths):
            self._linewidths_orig = _nparray_if_array(linewidths)

        def do_3d_projection(self, renderer):
            mplot3d.art3d.Line3DCollection.do_3d_projection(self, renderer)
            # The whole 3D ordering is flawed in mplot3d when several
            # collections are added. We just use normal zorder. Note the
            # "-" due to the different logic in the 3d plotting, we still want
            # larger zorder values to be plotted on top of smaller ones.
            return -self._zorder3d

        def draw(self, renderer):
            if self.reflen:
                proj_len = projected_length(self.axes, self.reflen)
                a, _, _, d, _, _ = self.axes.transData.frozen().to_values()
                # Note: unlike in the 2D case, where we can enforce equal
                #       aspect ratio, this (currently) does not work with
                #       3D plots in matplotlib. As an approximation, we
                #       thus scale with the average of the x- and y-axis
                #       transformation.
                factor = proj_len * (a + d) * 0.5 / self.figure.dpi * 72.0
                mplot3d.art3d.Line3DCollection.set_linewidths(self,
                                            self._linewidths_orig * factor)
            else:
                mplot3d.art3d.Line3DCollection.set_linewidths(self,
                                                     self._linewidths_orig)

            mplot3d.art3d.Line3DCollection.draw(self, renderer)


    class Path3DCollection(mplot3d.art3d.Patch3DCollection):
        def __init__(self, paths, sizes, reflen=None, zorder=0, offsets=None,
                     **kwargs):
            paths = [matplotlib.patches.PathPatch(path) for path in paths]

            if offsets is not None:
                mplot3d.art3d.Patch3DCollection.__init__(self, paths,
                                                         offsets=offsets[:,:2],
                                                         **kwargs)
                self.set_3d_properties(zs=offsets[:,2], zdir="z")
            else:
                mplot3d.art3d.Patch3DCollection.__init__(self, paths,
                                                         **kwargs)


            self.reflen = reflen
            self._zorder3d = zorder

            self._paths_orig = np.array(paths, dtype='object')
            self._linewidths_orig = _nparray_if_array(self.get_linewidths())
            self._linewidths_orig2 = self._linewidths_orig
            self._array_orig = _nparray_if_array(self.get_array())
            self._facecolors_orig = _nparray_if_array(self.get_facecolors())
            self._edgecolors_orig = _nparray_if_array(self.get_edgecolors())

            self._orig_transforms = np.array([
                matplotlib.transforms.Affine2D().scale(x)
                for x in sizes], dtype='object')
            self._transforms = self._orig_transforms

        def set_array(self, array):
            self._array_orig = _nparray_if_array(array)
            mplot3d.art3d.Patch3DCollection.set_array(self, array)

        def set_color(self, colors):
            self._facecolors_orig = _nparray_if_array(colors)
            self._edgecolors_orig = self._facecolors_orig
            mplot3d.art3d.Patch3DCollection.set_color(self, colors)

        def set_edgecolors(self, colors):
            colors = matplotlib.colors.colorConverter.to_rgba_array(colors)
            self._edgecolors_orig = _nparray_if_array(colors)
            mplot3d.art3d.Patch3DCollection.set_edgecolors(self, colors)

        def get_transforms(self):
            # this is exact only for an isometric projection, for the
            # perspective projection used in mplot3d it's an approximation
            return self._transforms

        def get_transform(self):
            if self.reflen:
                proj_len = projected_length(self.axes, self.reflen)

                # For the paths, use the data transformation but
                # strip the offset (will be added later with offsets)
                a, b, c, d, e, f = self.axes.transData.frozen().to_values()
                return matplotlib.transforms.Affine2D().from_values(a, b, c, d,
                                                         0, 0).scale(proj_len)
            else:
                return matplotlib.transforms.Affine2D().scale(
                    (self.figure.dpi / 72.0))


        def do_3d_projection(self, renderer):
            xs, ys, zs = self._offsets3d

            # numpy complains about zero-length index arrays
            if len(xs) == 0:
                return -self._zorder3d

            vs = np.empty((len(xs), 3), np.float_)
            vs[:, 0], vs[:, 1], vs[:, 2], _ = \
                mplot3d.proj3d.proj_transform_clip(xs, ys, zs,
                                                   renderer.M)

            if _sort3d:
                indx = vs[:, 2].argsort()[::-1]

                self.set_offsets(vs[indx, :2])

                if len(self._paths_orig) > 1:
                    paths = np.resize(self._paths_orig,
                                      (vs.shape[0], ))
                    self.set_paths(paths[indx])

                if len(self._orig_transforms) > 1:
                    self._transforms = np.resize(self._orig_transforms,
                                             (vs.shape[0], ))
                    self._transforms = self._transforms[indx]

                if (isinstance(self._linewidths_orig, np.ndarray) and
                    len(self._linewidths_orig) > 1):
                    self._linewidths_orig2 =  np.resize(self._linewidths_orig,
                                                        (vs.shape[0], ))
                    self._linewidths_orig2 = self._linewidths_orig2[indx]

                # Note: here array, facecolors and edgecolors are
                #       guaranteed to be 2d numpy arrays or None.
                #       (And array is the same length as the
                #       coordinates)

                if self._array_orig is not None:
                    mplot3d.art3d.Patch3DCollection.set_array(self,
                                                self._array_orig[indx])

                if (self._facecolors_orig is not None and
                    self._facecolors_orig.shape[0] > 1):
                    shape = list(self._facecolors_orig.shape)
                    shape[0] = vs.shape[0]
                    mplot3d.art3d.Patch3DCollection.set_facecolors(self,
                                         np.resize(self._facecolors_orig,
                                                   shape)[indx])

                if (self._edgecolors_orig is not None and
                    self._edgecolors_orig.shape[0] > 1):
                    shape = list(self._edgecolors_orig.shape)
                    shape[0] = vs.shape[0]
                    mplot3d.art3d.Patch3DCollection.set_edgecolors(self,
                                         np.resize(self._edgecolors_orig,
                                                   shape)[indx])
            else:
                self.set_offsets(vs[:, :2])


            # the whole 3D ordering is flawed in mplot3d when several
            # collections are added. We just use normal zorder, but correct
            # by the projected z-coord of the "center of gravity", normalized
            # by the projected z-coord of the world coordinates.
            # In doing so, several Path3DCollections are plotted probably
            # in the right order (it's not exact) if they have the same
            # zorder. Still, smaller and larger integer zorders are plotted
            # below or on top.

            minx, maxx, miny, maxy, minz, maxz = self.axes.get_w_lims()
            corners = np.zeros((8, 3), np.float_)
            corners[[0, 1, 2, 3], 0] = minx
            corners[[4, 5, 6, 7], 0] = maxx
            corners[[0, 1, 4, 5], 0] = miny
            corners[[2, 3, 6, 7], 0] = maxy
            corners[[0, 2, 4, 6], 0] = minz
            corners[[1, 3, 5, 7], 0] = maxz

            cz = np.empty((8,), np.float_)
            _, _, cz[:], _ = mplot3d.proj3d.proj_transform_clip(corners[:, 0],
                                                                corners[:, 1],
                                                                corners[: ,2],
                                                                renderer.M)

            return -self._zorder3d + vs[:, 2].mean()/cz.ptp()


        def draw(self, renderer):
            if self.reflen:
                proj_len = projected_length(self.axes, self.reflen)
                a, _, _, d, _, _ = self.axes.transData.frozen().to_values()
                factor = proj_len * (a + d) * 0.5 / self.figure.dpi * 72.0

                self.set_linewidths(self._linewidths_orig2 * factor)

            mplot3d.art3d.Patch3DCollection.draw(self, renderer)


# matplotlib helper functions.

def set_colors(color, collection, cmap, norm=None):
    """Process a color specification to a format accepted by collections.

    Parameters
    ----------
    color : color specification
    collection : instance of a subclass of `matplotlib.collections.Collection`
        Collection to which the color is added.
    cmap : `matplotlib` color map specification or None
        Color map to be used if colors are specified as floats.
    norm : `matplotlib` color norm
        Norm to be used if colors are specified as floats.
    """

    length = max(len(collection.get_paths()), len(collection.get_offsets()))

    # matplotlib gets confused if dtype='object'
    if (isinstance(color, np.ndarray) and
        color.dtype == np.dtype('object')):
        color = tuple(color)

    if isinstance(collection, mplot3d.art3d.Line3DCollection):
        length = len(collection._segments3d)  # Once again, matplotlib fault!

    if _isarray(color) and len(color) == length:
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


symbol_dict = {'O': 'o',
               's': ('p', 4, 45),
               'S': ('P', 4, 45)}

def get_symbol(symbols):
    """Return the path corresponding to the description in `symbol`
    """
    # figure out if list of symbols or single symbol
    if not hasattr(symbols, '__getitem__'):
        symbols = [symbols]
    elif len(symbols) == 3 and symbols[0] in ('p', 'P'):
        # most likely a polygon specification (at least not a valid
        # other symbol)
        symbols = [symbols]

    symbols = [symbol_dict[symbol] if symbol in symbol_dict else symbol
               for symbol in symbols]

    paths = []
    for symbol in symbols:
        if isinstance(symbol, matplotlib.path.Path):
            return symbol
        elif hasattr(symbol, '__getitem__') and len(symbol) == 3:
            kind, n, angle = symbol

            if kind in ['p', 'P']:
                if kind == 'p':
                    radius = 1.0 / cos(pi / n)
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


def symbols(axes, pos, symbol='o', size=1, reflen=None,
            facecolor='k', edgecolor='k',
            linewidth=None, cmap=None, norm=None, zorder=0,
            **kwargs):
    """
    Add a collection of symbols to an axes instance. Can deal with
    2D and 3D data.

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
        If `reflen` is `None`, the symbol sizes and linewidths are
        given in points (absolute size in the figure space). If
        `reflen` is a number, the symbol sizes and linewidths are
        given in units of `reflen` in data space (i.e. scales with the
        scale of the plot). Defaults to `None`.
    facecolor: color definition, optional
    edgecolor: color definition, optional
        Defines the fill and edge color of the symbol, repsectively.
        Either a single object that is a proper matplotlib color
        definition or a sequence of such objects of appropriate
        length.  Defaults to all black.
    cmap : `matplotlib` color map specification or None
        Color map to be used if colors are specified as floats.
    norm : `matplotlib` color norm
        Norm to be used if colors are specified as floats.
    zorder: int
        Order in which different collections are drawn: larger
        `zorder` means the collection is drawn over collections with
        smaller `zorder` values.
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
    except:
        size = (size, )

    if dim == 2:
        Collection = PathCollection
    else:
        Collection = Path3DCollection

    if len(pos) == 0 or np.all(symbol == 'no symbol') or np.all(size == 0):
        coll = Collection([], sizes=size, reflen=reflen,
                      linewidths=linewidth,
                      offsets=np.empty((0, dim)), transOffset=axes.transData,
                      zorder=zorder)
        coll.update(kwargs)
        if dim == 2:
            axes.add_collection(coll)
        else:
            axes.add_collection3d(coll)
        axes.autoscale_view()
        return coll

    paths = get_symbol(symbol)
    coll = Collection(paths, sizes=size, reflen=reflen,
                      linewidths=linewidth,
                      offsets=pos, transOffset=axes.transData,
                      zorder=zorder)

    set_colors(facecolor, coll, cmap, norm)
    coll.set_edgecolors(edgecolor)
    coll.update(kwargs)

    if dim == 2:
        axes.add_collection(coll)
    else:
        axes.add_collection3d(coll)

    axes.autoscale_view()

    return coll


def lines(axes, pos0, pos1, reflen=None, colors='k', linestyles='solid',
          cmap=None, norm=None, zorder=0, **kwargs):
    """Add a collection of line segments to an axes instance. Can deal with
    2D and 3D data.

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
    cmap : `matplotlib` color map specification or None
        Color map to be used if colors are specified as floats.
    norm : `matplotlib` color norm
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
        axes.autoscale_view()
        return coll

    had_data = axes.has_data()

    segments = np.c_[pos0, pos1].reshape(pos0.shape[0], 2, dim)

    coll = Collection(segments, reflen=reflen, linestyles=linestyles,
                      zorder=zorder)
    set_colors(colors, coll, cmap, norm)
    coll.update(kwargs)

    min_max = lambda a, b: np.array([min(a.min(), b.min()),
                                     max(a.max(), b.max())])
    coord_min_max = [min_max(pos0[:, i], pos1[:, i]) for i in xrange(dim)]

    if dim == 2:
        axes.add_collection(coll)

        corners = ((coord_min_max[0][0], coord_min_max[1][0]),
                   (coord_min_max[0][1], coord_min_max[1][1]))

        axes.update_datalim(corners)
        axes.autoscale_view()
    else:
        axes.add_collection3d(coll)
        axes.auto_scale_xyz(*coord_min_max, had_data=had_data)

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
        args and kwargs passed to `print_figure` of `matplotlib`
    show : bool
        Whether to call `matplotlib.pyplot.show()`.  Only has an effect if the
        output uses pyplot.

    Notes
    -----
    The behavior of this function producing a file is different from that of
    matplotlib in that the `dpi` attribute of the figure is used by defaul
    instead of the matplotlib config setting.
    """
    if not _mpl_enabled:
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
            msg = 'matplotlib.pyplot is unavailable.  Execute `import ' \
            'matplotlib.pyplot` or use a different output mode.'
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
        canvas.print_figure(file, *savefile_opts[0],
                            **savefile_opts[1])
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
        A site is a `builder.Site` instance if the system is not finalized,
        and an integer otherwise.  For system sites `lead_number` is `None` and
        `copy_number` is `0`, for leads both are integers.
    lead_cells : list of slices
        `lead_cells[i]` gives the position of all the coordinates of lead
        `i` within `sites`.

    Notes
    -----
    Leads are only supported if they are of the same type as the original
    system, i.e.  sites of `builder.BuilderLead` leads are returned with an
    unfinalized system, and sites of `system.InfiniteSystem` leads are
    returned with a finalized system.
    """
    lead_cells = []
    if isinstance(sys, builder.Builder):
        sites = [(site, None, 0) for site in sys.sites()]
        for leadnr, lead in enumerate(sys.leads):
            start = len(sites)
            if hasattr(lead, 'builder') and len(lead.interface):
                sites.extend(((site, leadnr, i) for site in
                              lead.builder.sites() for i in
                              xrange(num_lead_cells)))
            lead_cells.append(slice(start, len(sites)))
    elif isinstance(sys, system.FiniteSystem):
        sites = [(i, None, 0) for i in xrange(sys.graph.num_nodes)]
        for leadnr, lead in enumerate(sys.leads):
            start = len(sites)
            # We will only plot leads with a graph and with a symmetry.
            if hasattr(lead, 'graph') and hasattr(lead, 'symmetry') and \
                    len(sys.lead_interfaces[leadnr]):
                sites.extend(((site, leadnr, i) for site in
                              xrange(lead.cell_size) for i in
                              xrange(num_lead_cells)))
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
    # numpy is really slow when making a numpy array from a tinyarray
    # (buffer interface seems very slow). It's much faster to first
    # convert to a tuple and then to convert to numpy array ...

    is_builder = isinstance(sys, builder.Builder)
    num_lead_cells = site_lead_nr[-1][2] + 1
    if is_builder:
        pos = np.array(ta.array([i[0].pos for i in site_lead_nr]))
    else:
        sys_from_lead = lambda lead: (sys if (lead is None)
                                      else sys.leads[lead])
        pos = np.array(ta.array([sys_from_lead(i[1]).pos(i[0])
                                 for i in site_lead_nr]))
    if pos.dtype == object:  # Happens if not all the pos are same length.
        raise ValueError("pos attribute of the sites does not have consistent"
                         " values.")
    dim = pos.shape[1]

    def get_vec_domain(lead_nr):
        if lead_nr is None:
            return np.zeros((dim,)), 0
        if is_builder:
            sym = sys.leads[lead_nr].builder.symmetry
            try:
                site = sys.leads[lead_nr].interface[0]
            except IndexError:
                return (0, 0)
        else:
            sym = sys.leads[lead_nr].symmetry
            try:
                site = sys.site(sys.lead_interfaces[lead_nr][0])
            except IndexError:
                return (0, 0)
        dom = sym.which(site)[0] + 1
        # Conversion to numpy array here useful for efficiency
        vec = np.array(sym.periods)[0]
        return vec, dom
    vecs_doms = dict((i, get_vec_domain(i)) for i in xrange(len(sys.leads)))
    vecs_doms[None] = np.zeros((dim,)), 0
    for k, v in vecs_doms.iteritems():
        vecs_doms[k] = [v[0] * i for i in xrange(v[1], v[1] + num_lead_cells)]
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
        A site is a `builder.Site` instance if the system is not finalized,
        and an integer otherwise.  For system sites `lead_number` is `None` and
        `copy_number` is `0`, for leads both are integers.
    lead_cells : list of slices
        `lead_cells[i]` gives the position of all the coordinates of lead
        `i` within `hoppings`.

    Notes
    -----
    Leads are only supported if they are of the same type as the original
    system, i.e.  hoppings of `builder.BuilderLead` leads are returned with an
    unfinalized system, and hoppings of `system.InfiniteSystem` leads are
    returned with a finalized system.
    """
    hoppings = []
    lead_cells = []
    if isinstance(sys, builder.Builder):
        hoppings.extend(((hop, None, 0) for hop in sys.hoppings()))

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

        for leadnr, lead in enumerate(sys.leads):
            start = len(hoppings)
            if hasattr(lead, 'builder') and len(lead.interface):
                hoppings.extend(((hop, leadnr, i) for hop in
                                lead_hoppings(lead.builder) for i in
                                xrange(num_lead_cells)))
            lead_cells.append(slice(start, len(hoppings)))
    elif isinstance(sys, system.System):
        def ll_hoppings(sys):
            for i in xrange(sys.graph.num_nodes):
                for j in sys.graph.out_neighbors(i):
                    if i < j:
                        yield i, j
        hoppings.extend(((hop, None, 0) for hop in ll_hoppings(sys)))
        for leadnr, lead in enumerate(sys.leads):
            start = len(hoppings)
            # We will only plot leads with a graph and with a symmetry.
            if hasattr(lead, 'graph') and hasattr(lead, 'symmetry') and \
                    len(sys.lead_interfaces[leadnr]):
                hoppings.extend(((hop, leadnr, i) for hop in
                                 ll_hoppings(lead) for i in
                                 xrange(num_lead_cells)))
            lead_cells.append(slice(start, len(hoppings)))
    else:
        raise TypeError('Unrecognized system type.')
    return hoppings, lead_cells


def sys_leads_hopping_pos(sys, hop_lead_nr):
    """Return arrays of coordinates of all hoppings in a system.

    Parameters
    ----------
    sys : `kwant.builder.Builder` or `kwant.system.System` instance
        The system, coordinates of sites of which should be returned.
    hoppings : list of `(hopping, leadnr, copynr)` tuples
        Output of `sys_leads_hoppings` applied to the system.

    Returns
    -------
    coords : (end_site, start_site): tuple of NumPy arrays of floats
        Array of coordinates of the hoppings.  The first half of coordinates
        in each array entry are those of the first site in the hopping, the
        last half are those of the second site.

    Notes
    -----
    This function uses `site.pos` property to get the position of a builder
    site and `sys.pos(sitenr)` for finalized systems.  This function requires
    that all the positions of all the sites have the same dimensionality.
    """
    is_builder = isinstance(sys, builder.Builder)
    if len(hop_lead_nr) == 0:
        return np.empty((0, 3)), np.empty((0, 3))
    num_lead_cells = hop_lead_nr[-1][2] + 1
    if is_builder:
        pos = np.array(ta.array([ta.array(tuple(i[0][0].pos) +
                                          tuple(i[0][1].pos))
                                 for i in hop_lead_nr]))
    else:
        sys_from_lead = lambda lead: (sys if (lead is None)
                                      else sys.leads[lead])
        pos = ta.array([ta.array(tuple(sys_from_lead(i[1]).pos(i[0][0])) +
                                 tuple(sys_from_lead(i[1]).pos(i[0][1])))
                        for i in hop_lead_nr])
        pos = np.array(pos)
    if pos.dtype == object:  # Happens if not all the pos are same length.
        raise ValueError("pos attribute of the sites does not have consistent"
                         " values.")
    dim = pos.shape[1]

    def get_vec_domain(lead_nr):
        if lead_nr is None:
            return np.zeros((dim,)), 0
        if is_builder:
            sym = sys.leads[lead_nr].builder.symmetry
            try:
                site = sys.leads[lead_nr].interface[0]
            except IndexError:
                return (0, 0)
        else:
            sym = sys.leads[lead_nr].symmetry
            try:
                site = sys.site(sys.lead_interfaces[lead_nr][0])
            except IndexError:
                return (0, 0)
        dom = sym.which(site)[0] + 1
        vec = np.array(sym.periods)[0]
        return np.r_[vec, vec], dom

    vecs_doms = dict((i, get_vec_domain(i)) for i in xrange(len(sys.leads)))
    vecs_doms[None] = np.zeros((dim,)), 0
    for k, v in vecs_doms.iteritems():
        vecs_doms[k] = [v[0] * i for i in xrange(v[1], v[1] + num_lead_cells)]
    pos += [vecs_doms[i[1]][i[2]] for i in hop_lead_nr]
    return np.copy(pos[:, : dim / 2]), np.copy(pos[:, dim / 2:])


# Useful plot functions (to be extended).

_defaults = {'site_symbol': {2: 'o', 3: 'o'},
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
         show=True, dpi=None, fig_size=None,
         ax=None):
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

    site_symbol : symbol specification, function, array or `None`
        Symbol used for representing a site in the plot. Can be specified as

        - 'o': circle with radius of 1 unit.
        - 's': square with inner circle radius of 1 unit.
        - `('p', nvert, angle)`: regular polygon with `nvert` vertices,
          rotated by `angle`. `angle` is given in degrees, and ``angle=0``
          corresponds to one edge of the polygon pointing upward. The
          radius of the inner circle is 1 unit.
        - 'no symbol': no symbol is plotted.
        - 'S', `('P', nvert, angle)`: as the lower-case variants described
          above, but with an area equal to a circle of radius 1. (Makes
          the visual size of the symbol equal to the size of a circle with
        - radius 1).
        - matplotlib.path.Path instance.

        Instead of a single symbol, different symbols can be specified
        for different sites by passing a function that returns a valid
        symbol specification for each site, or by passing an array of
        symbols specifications (only for kwant.system.FiniteSystem).
    site_size : number, function, array or `None`
        Relative (linear) size of the site symbol.
    site_color : `matplotlib` color description, function, array or `None`
        A color used for plotting a site in the system. If a colormap is used,
        it should be a function returning single floats or a one-dimensional
        array of floats.
    site_edgecolor : `matplotlib` color description, function, array or `None`
        Color used for plotting the edges of the site symbols. Only
        valid matplotlib color descriptions are allowed (and no
        combination of floats and colormap as for site_color).
    site_lw : number, function, array or `None`
        Linewidth of the site symbol edges.
    hop_color : `matplotlib` color description or a function
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
    lead_color : `matplotlib` color description or `None`
        For the leads, `num_lead_cells` copies of the lead unit cell
        are plotted. They are plotted in color fading from `lead_color`
        to white (alpha values in `lead_color` are supported) when moving
        from the system into the lead. Is also applied to the
        hoppings.
    lead_site_edgecolor : `matplotlib` color description or `None`
        Color of the symbol edges (no fading done).
    lead_site_lw : number or `None`
        Linewidth of the lead symbols.
    lead_hop_lw : number or `None`
        Linewidth of the lead hoppings.
    cmap : `matplotlib` color map or a tuple of two color maps or `None`
        The color map used for sites and optionally hoppings.
    pos_transform : function or `None`
        Transformation to be applied to the site position.
    colorbar : bool
        Whether to show a colorbar if colormap is used
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether `matplotlib.pyplot.show()` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float or `None`
        Number of pixels per inch.  If not set the `matplotlib` default is
        used.
    fig_size : tuple or `None`
        Figure size `(width, height)` in inches.  If not set, the default
        `matplotlib` value is used.
    ax : `matplotlib.axes.Axes` instance or `None`
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ingored.

    Returns
    -------
    result : matplotlib figure or axes instance
        A figure with the output if ``ax==None``, otherwise `ax`.

    Notes
    -----
    - If `None` is passed for a plot property, a default value depending on
      the dimension is chosen. Typically, the default values result in
      acceptable plots.

    - The meaning of "site" depends on whether the system to be plotted is a
      builder or a low level system.  For builders, a site is a
      kwant.builder.Site object.  For low level systems, a site is an integer
      -- the site number.

    - The dimensionality of the plot (2D vs 3D) is inferred from the coordinate
      array.  If there are more than three coordinates, only the first three
      are used.  If there is just one coordinate, the second one is padded with
      zeros.

    - The system is scaled to fit the smaller dimension of the figure, given
      its aspect ratio.

    """

    # Generate data.
    sites, lead_sites_slcs = sys_leads_sites(sys, num_lead_cells)
    n_sys_sites = sum(i[1] is None for i in sites)
    sites_pos = sys_leads_pos(sys, sites)
    hops, lead_hops_slcs = sys_leads_hoppings(sys, num_lead_cells)
    n_sys_hops = sum(i[1] is None for i in hops)
    end_pos, start_pos = sys_leads_hopping_pos(sys, hops)

    # Choose plot type.
    def resize_to_dim(array):
        if array.shape[1] != dim:
            ar = np.zeros((len(array), dim), dtype=float)
            ar[:, : min(dim, array.shape[1])] = \
                array[:, : min(dim, array.shape[1])]
            return ar
        else:
            return array

    dim = 3 if (sites_pos.shape[1] == 3) else 2
    sites_pos = resize_to_dim(sites_pos)
    end_pos = resize_to_dim(end_pos)
    start_pos = resize_to_dim(start_pos)

    # Determine the reference length.
    if unit == 'pt':
        reflen = None
    elif unit == 'nn':
        if n_sys_hops:
            # If hoppings are present use their lengths to determine the
            # minimal one.
            distances = np.sort(np.sqrt(np.sum((end_pos - start_pos)**2,
                                               axis=1)))
        else:
            # If no hoppings are present, use for the same purpose distances
            # from ten randomly selected points to the remaining points in the
            # system.
            points = sites_pos[np.random.randint(len(sites_pos), size=10)].T
            distances = (sites_pos.T.reshape(1, -1, dim) -
                         points.reshape(-1, 1, dim)).reshape(-1, dim)
            distances = np.sort(np.sqrt(np.sum(distances**2, axis=1)))
        # Then check if distances are present that are way shorter than the
        # longest one. Then take first distance longer than these short
        # ones. This heuristic will fail for too large systems, or systems with
        # hoppings that vary by orders and orders of magnitude, but for sane
        # cases it will work.
        long_dist_coord = np.searchsorted(distances, 1e-8 * distances[-1])
        reflen = distances[long_dist_coord]

    else:
        # The last allowed value is float-compatible.
        try:
            reflen = float(unit)
        except:
            raise ValueError('Invalid value of unit argument.')

    # Apply transformations to the data
    if pos_transform is not None:
        sites_pos = np.apply_along_axis(pos_transform, 1, sites_pos)
        end_pos = np.apply_along_axis(pos_transform, 1, end_pos)
        start_pos = np.apply_along_axis(pos_transform, 1, start_pos)

    # make all specs proper: either constant or lists/np.arrays:
    def make_proper_site_spec(spec, fancy_indexing=False):
        if callable(spec):
            spec = [spec(i[0]) for i in sites if i[1] is None]
        if (fancy_indexing and _isarray(spec)
            and not isinstance(spec, np.ndarray)):
            try:
                spec = np.asarray(spec)
            except:
                spec = np.asarray(spec, dtype='object')
        return spec

    def make_proper_hop_spec(spec, fancy_indexing=False):
        if callable(spec):
            spec = [spec(*i[0]) for i in hops if i[1] is None]
        if (fancy_indexing and _isarray(spec)
            and not isinstance(spec, np.ndarray)):
            try:
                spec = np.asarray(spec)
            except:
                spec = np.asarray(spec, dtype='object')
        return spec

    site_symbol = make_proper_site_spec(site_symbol)
    if site_symbol is None: site_symbol = _defaults['site_symbol'][dim]
    # separate different symbols (not done in 3D, the separation
    # would mess up sorting)
    if (_isarray(site_symbol) and dim != 3 and
        (len(site_symbol) != 3 or site_symbol[0] not in ('p', 'P'))):
        symbol_dict = defaultdict(list)
        for i, symbol in enumerate(site_symbol):
            symbol_dict[symbol].append(i)
        symbol_slcs = []
        for symbol, indx in symbol_dict.iteritems():
            symbol_slcs.append((symbol, np.array(indx)))
        fancy_indexing = True
    else:
        symbol_slcs = [(site_symbol, slice(n_sys_sites))]
        fancy_indexing = False

    site_size = make_proper_site_spec(site_size, fancy_indexing)
    site_color = make_proper_site_spec(site_color, fancy_indexing)
    site_edgecolor = make_proper_site_spec(site_edgecolor, fancy_indexing)
    site_lw = make_proper_site_spec(site_lw, fancy_indexing)

    hop_color = make_proper_hop_spec(hop_color)
    hop_lw = make_proper_hop_spec(hop_lw)

    # Choose defaults depending on dimension, if None was given
    if site_size is None: site_size = _defaults['site_size'][dim]
    if site_color is None: site_color = _defaults['site_color'][dim]
    if site_edgecolor is None:
        site_edgecolor = _defaults['site_edgecolor'][dim]
    if site_lw is None: site_lw = _defaults['site_lw'][dim]

    if hop_color is None: hop_color = _defaults['hop_color'][dim]
    if hop_lw is None: hop_lw = _defaults['hop_lw'][dim]

    # if symbols are split up into different collections,
    # the colormapping will fail without normalization
    norm = None
    if len(symbol_slcs) > 1:
        try:
            if site_color.ndim == 1 and len(site_color) == n_sys_sites:
                site_color = np.asarray(site_color, dtype=float)
                norm = matplotlib.colors.Normalize(site_color.min(),
                                                   site_color.max())
        except:
            pass

    # take spec also for lead, if it's not a list/array, default, otherwise
    if lead_site_symbol is None:
        lead_site_symbol = (site_symbol if not _isarray(site_symbol)
                            else _defaults['site_symbol'][dim])
    if lead_site_size is None:
        lead_site_size = (site_size if not _isarray(site_size)
                          else _defaults['site_size'][dim])
    if lead_color is None:
        lead_color = _defaults['lead_color'][dim]
    lead_color = matplotlib.colors.colorConverter.to_rgba(lead_color)

    if lead_site_edgecolor is None:
        lead_site_edgecolor = (site_edgecolor if not _isarray(site_edgecolor)
                               else _defaults['site_edgecolor'][dim])
    if lead_site_lw is None:
        lead_site_lw = (site_lw if not _isarray(site_lw)
                        else _defaults['site_lw'][dim])
    if lead_hop_lw is None:
        lead_hop_lw = (hop_lw if not _isarray(hop_lw)
                       else _defaults['hop_lw'][dim])

    if isinstance(cmap, tuple):
        hop_cmap = cmap[1]
        cmap = cmap[0]
    else:
        hop_cmap = None

    # make a new figure unless axes specified
    if not ax:
        fig = Figure()
        if dpi is not None:
            fig.set_dpi(dpi)
        if fig_size is not None:
            fig.set_figwidth(fig_size[0])
            fig.set_figheight(fig_size[1])

        if dim == 2:
            ax = fig.add_subplot(111, aspect='equal')
            ax.set_xmargin(0.05)
            ax.set_ymargin(0.05)
        else:
            warnings.filterwarnings('ignore', message=r'.*rotation.*')
            ax = fig.add_subplot(111, projection='3d')
            warnings.resetwarnings()
    else:
        fig = None

    # plot system sites and hoppings
    for symbol, slc in symbol_slcs:
        size = site_size[slc] if _isarray(site_size) else site_size
        col = site_color[slc] if _isarray(site_color) else site_color
        edgecol = (site_edgecolor[slc] if _isarray(site_edgecolor)
                   else site_edgecolor)
        lw  = site_lw[slc] if _isarray(site_lw) else site_lw

        symbols(ax, sites_pos[slc], size=size,
                reflen=reflen, symbol=symbol, facecolor=col,
                edgecolor=edgecol, linewidth=lw,
                cmap=cmap, norm=norm, zorder=2)

    end, start = end_pos[: n_sys_hops], start_pos[: n_sys_hops]
    lines(ax, end, start, reflen, hop_color,
          linewidths=hop_lw, zorder=1, cmap=hop_cmap)

    # plot lead sites and hoppings
    norm = matplotlib.colors.Normalize(-0.5, num_lead_cells - 0.5)
    lead_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(None,
        [lead_color, (1, 1, 1, lead_color[3])])

    for sites_slc, hops_slc in zip(lead_sites_slcs, lead_hops_slcs):
        lead_site_colors = np.array([i[2] for i in
                                     sites[sites_slc]], dtype=float)

        # Note: the previous version of the code had in addition this
        # line in the 3D case:
        # lead_site_colors = 1 / np.sqrt(1. + lead_site_colors)
        symbols(ax, sites_pos[sites_slc], size=lead_site_size,
                reflen=reflen, symbol=lead_site_symbol,
                facecolor=lead_site_colors, edgecolor=lead_site_edgecolor,
                linewidth=lead_site_lw,
                cmap=lead_cmap, zorder=2, norm=norm)

        lead_hop_colors = np.array([i[2] for i in
                                    hops[hops_slc]], dtype=float)

        # Note: the previous version of the code had in addition this
        # line in the 3D case:
        # lead_hop_colors = 1 / np.sqrt(1. + lead_hop_colors)
        end, start = end_pos[hops_slc], start_pos[hops_slc]
        lines(ax, end, start, reflen, lead_hop_colors, linewidths=lead_hop_lw,
              cmap=lead_cmap, norm=norm, zorder=1)

    if dim == 3:
        # make axis limits the same in all directions
        # (3D only works decently for equal aspect ratio. Since
        #  this doesn't work out of the box in mplot3d, this is a
        #  workaround)
        min_ = np.min(sites_pos, 0)
        max_ = np.max(sites_pos, 0)
        w = np.max(max_ - min_) / 2
        m = (min_ + max_) / 2
        ax.auto_scale_xyz(*[(i - w, i + w) for i in m], had_data=True)

    if ax.collections[0].get_array() is not None and colorbar:
        fig.colorbar(ax.collections[0])
    if ax.collections[1].get_array() is not None and colorbar:
        fig.colorbar(ax.collections[1])

    return output_fig(fig, file=file, show=show) if fig else ax


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
        Passed to `scipy.interpolate.griddata`: "nearest" (default), "linear",
        or "cubic"
    oversampling : integer, optional
        Number of pixels per reference length.  Defaults to 3.

    Returns
    -------
    array : 2d NumPy array
        The interpolated values.
    min, max : vectors
        The real-space coordinates of the two extreme ([0, 0] and [-1, -1])
        points of `array`.

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

    if a is None:
        points = coords[np.random.randint(len(coords), size=10)]
        a = np.min(tree.query(points, 2)[0][:, 1])
    elif a <= 0:
        raise ValueError("The distance a must be strictly positive.")

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


def map(sys, value, colorbar=True, cmap=None, vmin=None, vmax=None,
         a=None, method='nearest', oversampling=3, num_lead_cells=0,
        file=None, show=True,  dpi=None, fig_size=None):
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
        Whether to show a color bar.  Defaults to `True`.
    cmap : `matplotlib` color map or `None`
        The color map used for sites and optionally hoppings, if `None`,
        `matplotlib` default is used.
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
        Passed to `scipy.interpolate.griddata`: "nearest" (default), "linear",
        or "cubic"
    oversampling : integer, optional
        Number of pixels per reference length.  Defaults to 3.
    num_lead_cells : integer, optional
        number of lead unit cells that should be plotted to indicate
        the position of leads. Defaults to 0.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether `matplotlib.pyplot.show()` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output.

    Notes
    -----
    - When plotting a system on a square lattice and `method` is "nearest", it
      makes sense to set `oversampling` to ``1``.  Then, each site will
      correspond to exactly one pixel.
    """
    sites = sys_leads_sites(sys, 0)[0]
    coords = sys_leads_pos(sys, sites)
    if coords.shape[1] != 2:
        raise ValueError('Only 2D systems can be plotted this way.')
    if callable(value):
        value = [value(site[0]) for site in sites]
    else:
        if not isinstance(sys, system.FiniteSystem):
            raise ValueError('List of values is only allowed as input'
                             'for finalized systems.')
    value = np.array(value)
    img, min, max = mask_interpolate(coords, value, a, method, oversampling)
    border = 0.5 * (max - min) / (np.asarray(img.shape) - 1)
    min -= border
    max += border
    fig = Figure()
    if dpi is not None:
        fig.set_dpi(dpi)
    if fig_size is not None:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])
    ax = fig.add_subplot(111, aspect='equal')

    # Note that we tell imshow to show the array created by mask_interpolate
    # faithfully and not to interpolate by itself another time.
    image = ax.imshow(img.T, extent=(min[0], max[0], min[1], max[1]),
                      origin='lower', interpolation='none', cmap=cmap,
                      vmin=vmin, vmax=vmax)
    if num_lead_cells:
        plot(sys, num_lead_cells, site_symbol='no symbol', hop_lw=0,
             lead_site_symbol='s', lead_site_size=0.501,
             lead_site_lw=0, lead_hop_lw=0, lead_color='black',
             colorbar=False, ax=ax)

    if colorbar:
        fig.colorbar(image)
    return output_fig(fig, file=file, show=show)


def bands(sys, momenta=65, args=(), file=None, show=True, dpi=None,
          fig_size=None):
    """Plot band structure of a translationally invariant 1D system.

    Parameters
    ----------
    sys : kwant.system.InfiniteSystem
        A system bands of which are to be plotted.
    args : tuple, defaults to empty
        Positional arguments to pass to the ``hamiltonian`` method.
    momenta : int or 1D array-like
        Either a number of sampling points on the interval [-pi, pi], or an
        array of points at which the band structure has to be evaluated.
    file : string or file object or `None`
        The output file.  If `None`, output will be shown instead.
    show : bool
        Whether `matplotlib.pyplot.show()` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float
        Number of pixels per inch.  If not set the `matplotlib` default is
        used.
    fig_size : tuple
        Figure size `(width, height)` in inches.  If not set, the default
        `matplotlib` value is used.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output.

    Notes
    -----
    See `physics.Bands` for the calculation of dispersion without plotting.
    """
    momenta = np.array(momenta)
    if momenta.ndim != 1:
        momenta = np.linspace(-np.pi, np.pi, momenta)

    bands = physics.Bands(sys, args=args)
    energies = [bands(k) for k in momenta]

    fig = Figure()
    if dpi is not None:
        fig.set_dpi(dpi)
    if fig_size is not None:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(momenta, energies)
    return output_fig(fig, file=file, show=show)


# TODO (Anton): Fix plotting of parts of the system using color = np.nan.
# Not plotting sites currently works, not plotting hoppings does not.
# TODO (Anton): Allow a more flexible treatment of position than pos_transform
# (an interface for user-defined pos).
