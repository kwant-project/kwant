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

import itertools
import warnings
import numpy as np
from scipy import spatial, interpolate
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
    from matplotlib.cbook import is_string_like, is_sequence_of_strings
    try:
        from mpl_toolkits import mplot3d
    except ImportError:
        warnings.warn("3D plotting not available.", RuntimeWarning)
except ImportError:
    warnings.warn("matplotlib is not available, only iterator-providing"
                  "functions will work.", RuntimeWarning)
    _mpl_enabled = False

from . import system, builder, physics

__all__ = ['plot', 'map', 'bands', 'sys_leads_sites', 'sys_leads_hoppings',
           'sys_leads_pos', 'sys_leads_hopping_pos', 'mask_interpolate']


# matplotlib helper functions.

def set_edge_colors(color, collection, cmap, norm=None):
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
    length = len(collection.get_paths())
    if isinstance(collection, mplot3d.art3d.Line3DCollection):
        length = len(collection._segments3d)  # Once again, matplotlib fault!
    color_is_stringy = is_string_like(color) or is_sequence_of_strings(color)
    if not color_is_stringy:
        color = np.asanyarray(color)
        if color.size == length:
            color = np.ma.ravel(color)
    if color_is_stringy:
        colors = matplotlib.colors.colorConverter.to_rgba_array(color)
    else:
        # The inherent ambiguity is resolved in favor of color
        # mapping, not interpretation as rgb or rgba:
        if color.size == length:
            colors = None  # use cmap, norm after collection is created
        else:
            colors = matplotlib.colors.colorConverter.to_rgba_array(color)
    collection.set_color(colors)
    if colors is None:
        if norm is not None and not isinstance(norm,
                                               matplotlib.colors.Normalize):
            raise ValueError('Illegal value of norm.')
        collection.set_array(np.asarray(color))
        collection.set_cmap(cmap)
        collection.set_norm(norm)


def lines(axes, x0, x1, y0, y1, colors='k', linestyles='solid', cmap=None,
          norm=None, **kwargs):
    """Add a collection of line segments to an axes instance.

    Parameters
    ----------
    axes : matplotlib.axes.Axes instance
        Axes to which the lines have to be added.
    x0 : array_like
        Starting x-coordinates of each line segment
    x1 : array_like
        Ending x-coordinates of each line segment
    y0 : array_like
        Starting y-coordinates of each line segment
    y1 : array_like
        Ending y-coordinates of each line segment
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
    **kwargs : dict
        keyword arguments to pass to `matplotlib.collections.LineCollection`.

    Returns
    -------
    `matplotlib.collections.LineCollection` instance containing all the
    segments that were added.
    """
    coords = (y0, y1, x0, x1)

    if not all(len(coord) == len(y0) for coord in coords):
        raise ValueError('Incompatible lengths of coordinate arrays.')

    if len(x0) == 0:
        coll = collections.LineCollection([], linestyles=linestyles)
        axes.add_collection(coll)
        axes.autoscale_view()
        return coll

    segments = (((i[0], i[1]), (i[2], i[3])) for
                i in itertools.izip(x0, y0, x1, y1))
    coll = collections.LineCollection(segments, linestyles=linestyles)
    set_edge_colors(colors, coll, cmap, norm)
    axes.add_collection(coll)
    coll.update(kwargs)

    minx = min(x0.min(), x1.min())
    maxx = max(x0.max(), x1.max())
    miny = min(y0.min(), y1.min())
    maxy = max(y0.max(), y1.max())

    corners = (minx, miny), (maxx, maxy)

    axes.update_datalim(corners)
    axes.autoscale_view()

    return coll


def lines3d(axes, x0, x1, y0, y1, z0, z1,
            colors='k', linestyles='solid', cmap=None, norm=None, **kwargs):
    """Add a collection of 3D line segments to an Axes3D instance.

    Parameters
    ----------
    axes : matplotlib.axes.Axes instance
        Axes to which the lines have to be added.
    x0 : array_like
        Starting x-coordinates of each line segment
    x1 : array_like
        Ending x-coordinates of each line segment
    y0 : array_like
        Starting y-coordinates of each line segment
    y1 : array_like
        Ending y-coordinates of each line segment
    z0 : array_like
        Starting z-coordinates of each line segment
    z1 : array_like
        Ending z-coordinates of each line segment
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
    **kwargs : dict
        keyword arguments to pass to `matplotlib.collections.LineCollection`.

    Returns
    -------
    `mpl_toolkits.mplot3d.art3d.Line3DCollection` instance containing all the
    segments that were added.
    """
    had_data = axes.has_data()
    coords = (y0, y1, x0, x1, z0, z1)

    if not all(len(coord) == len(y0) for coord in coords):
        raise ValueError('Incompatible lengths of coordinate arrays.')

    if len(x0) == 0:
        coll = mplot3d.art3d.Line3DCollection([], linestyles=linestyles)
        axes.add_collection3d(coll)
        return coll

    segments = [(i[: 3], i[3:]) for
                i in itertools.izip(x0, y0, z0, x1, y1, z1)]
    coll = mplot3d.art3d.Line3DCollection(segments, linestyles=linestyles)
    set_edge_colors(colors, coll, cmap, norm)
    coll.update(kwargs)
    axes.add_collection3d(coll)

    min_max = lambda a, b: np.array(min(a.min(), b.min()),
                                    max(a.max(), b.max()))
    x, y, z = min_max(x0, x1), min_max(y0, y1), min_max(z0, z1)

    axes.auto_scale_xyz(x, y, z, had_data)

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

def sys_leads_sites(sys, n_lead_copies=2):
    """Return all the sites of the system and of the leads as a list.

    Parameters
    ----------
    sys : kwant.builder.Builder or kwant.system.System instance
        The system, sites of which should be returned.
    n_lead_copies : integer
        The number of times lead sites from each lead should be returned.
        This is useful for showing several unit cells of the lead next to the
        system.

    Returns
    -------
    sites : list of (site, lead_number, copy_number) tuples
        A site is a `builder.Site` instance if the system is not finalized,
        and an integer otherwise.  For system sites `lead_number` is `None` and
        `copy_number` is `0`, for leads both are integers.

    Notes
    -----
    Leads are only supported if they are of the same type as the original
    system, i.e.  sites of `builder.BuilderLead` leads are returned with an
    unfinalized system, and sites of `system.InfiniteSystem` leads are
    returned with a finalized system.
    """
    if isinstance(sys, builder.Builder):
        sites = [(site, None, 0) for site in sys.sites()]
        for leadnr, lead in enumerate(sys.leads):
            if hasattr(lead, 'builder') and len(lead.interface):
                sites.extend(((site, leadnr, i) for site in
                              lead.builder.sites() for i in
                              xrange(n_lead_copies)))
    elif isinstance(sys, system.FiniteSystem):
        sites = [(i, None, 0) for i in xrange(sys.graph.num_nodes)]
        for leadnr, lead in enumerate(sys.leads):
            # We will only plot leads with a graph and with a symmetry.
            if hasattr(lead, 'graph') and hasattr(lead, 'symmetry') and \
                    len(sys.lead_interfaces[leadnr]):
                sites.extend(((site, leadnr, i) for site in
                              xrange(lead.slice_size) for i in
                              xrange(n_lead_copies)))
    else:
        raise TypeError('Unrecognized system type.')
    return sites


def sys_leads_pos(sys, site_lead_nr):
    """Return an array of positions of sites in a system.

    Parameters
    ----------
    sys : `kwant.builder.Builder` or `kwant.system.System` instance
        The system, coordinates of sites of which should be returned.
    sites : list of `(site, leadnr, copynr)` tuples
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
    is_builder = isinstance(sys, builder.Builder)
    n_lead_copies = site_lead_nr[-1][2] + 1
    if is_builder:
        pos = np.array([i[0].pos for i in site_lead_nr])
    else:
        sys_from_lead = lambda lead: (sys if (lead is None)
                                      else sys.leads[lead])
        pos = np.array([sys_from_lead(i[1]).pos(i[0]) for i in site_lead_nr])
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
        # TODO (Anton): vec = sym.periods[0] not supported by ta.ndarray
        # Remove conversion to np.ndarray when not necessary anymore.
        vec = np.array(sym.periods)[0]
        return vec, dom
    vecs_doms = dict((i, get_vec_domain(i)) for i in xrange(len(sys.leads)))
    vecs_doms[None] = np.zeros((dim,)), 0
    for k, v in vecs_doms.iteritems():
        vecs_doms[k] = [v[0] * i for i in xrange(v[1], v[1] + n_lead_copies)]
    pos += [vecs_doms[i[1]][i[2]] for i in site_lead_nr]
    return pos


def sys_leads_hoppings(sys, n_lead_copies=2):
    """Return all the hoppings of the system and of the leads as an iterator.

    Parameters
    ----------
    sys : kwant.builder.Builder or kwant.system.System instance
        The system, sites of which should be returned.
    n_lead_copies : integer
        The number of times lead sites from each lead should be returned.
        This is useful for showing several unit cells of the lead next to the
        system.

    Returns
    -------
    hoppings : list of (hopping, lead_number, copy_number) tuples
        A site is a `builder.Site` instance if the system is not finalized,
        and an integer otherwise.  For system sites `lead_number` is `None` and
        `copy_number` is `0`, for leads both are integers.

    Notes
    -----
    Leads are only supported if they are of the same type as the original
    system, i.e.  hoppings of `builder.BuilderLead` leads are returned with an
    unfinalized system, and hoppings of `system.InfiniteSystem` leads are
    returned with a finalized system.
    """
    hoppings = []
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
            if hasattr(lead, 'builder') and len(lead.interface):
                hoppings.extend(((hop, leadnr, i) for hop in
                                lead_hoppings(lead.builder) for i in
                                xrange(n_lead_copies)))
    elif isinstance(sys, system.System):
        def ll_hoppings(sys):
            for i in xrange(sys.graph.num_nodes):
                for j in sys.graph.out_neighbors(i):
                    if i < j:
                        yield i, j
        hoppings.extend(((hop, None, 0) for hop in ll_hoppings(sys)))
        for leadnr, lead in enumerate(sys.leads):
            # We will only plot leads with a graph and with a symmetry.
            if hasattr(lead, 'graph') and hasattr(lead, 'symmetry') and \
                    len(sys.lead_interfaces[leadnr]):
                hoppings.extend(((hop, leadnr, i) for hop in
                                 ll_hoppings(lead) for i in
                                 xrange(n_lead_copies)))
    else:
        raise TypeError('Unrecognized system type.')
    return hoppings


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
    n_lead_copies = hop_lead_nr[-1][2] + 1
    if is_builder:
        pos = np.array([np.r_[i[0][0].pos, i[0][1].pos] for i in hop_lead_nr])
    else:
        sys_from_lead = lambda lead: (sys if (lead is None)
                                      else sys.leads[lead])
        pos = [(sys_from_lead(i[1]).pos(i[0][0]),
                sys_from_lead(i[1]).pos(i[0][1])) for i in hop_lead_nr]
        pos = np.array([np.r_[i[0], i[1]] for i in pos])
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
        # TODO (Anton): vec = sym.periods[0] not supported by ta.ndarray
        # Remove conversion to np.ndarray when not necessary anymore.
        vec = np.array(sym.periods)[0]
        return np.r_[vec, vec], dom

    vecs_doms = dict((i, get_vec_domain(i)) for i in xrange(len(sys.leads)))
    vecs_doms[None] = np.zeros((dim,)), 0
    for k, v in vecs_doms.iteritems():
        vecs_doms[k] = [v[0] * i for i in xrange(v[1], v[1] + n_lead_copies)]
    pos += [vecs_doms[i[1]][i[2]] for i in hop_lead_nr]
    return np.copy(pos[:, : dim / 2]), np.copy(pos[:, dim / 2:])


# Useful plot functions (to be extended).

def plot(sys, n_lead_copies=2, site_color='b', hop_color='b', cmap='gray',
         size=4, thickness=None, pos_transform=None, colorbar=True, file=None,
         show=True, dpi=None, fig_size=None):
    """Plot a system in 2 or 3 dimensions.

    Parameters
    ----------
    sys : kwant.builder.Builder or kwant.system.FiniteSystem
        A system to be plotted.
    n_lead_copies : int
        Number of lead copies to be shown with the system.
    site_color : `matplotlib` color description or a function
        A color used for plotting a site in the system or a function returning
        this color when given a site of the system (ignored for lead sites).
        If a colormap is used, this function should return a single float.
    hop_color : `matplotlib` color description or a function
        Same as site_color, but for hoppings.  A function is passed two sites
        in this case.
    cmap : `matplotlib` color map or a tuple of two color maps or `None`
        The color map used for sites and optionally hoppings.
    size : float or `None`
        Site size in points.  If `None`, `matplotlib` default is used.
    thickness : float or `None`
        Line thickness in points.  If `None`, `matplotlib` default is used.
    pos_transform : function or None
        Transformation to be applied to the site position.
    colorbar : bool
        Whether to show a colorbar if colormap is used
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
    sites = sys_leads_sites(sys, n_lead_copies)
    n_sys_sites = sum(i[1] is None for i in sites)
    sites_pos = sys_leads_pos(sys, sites)
    hops = sys_leads_hoppings(sys, n_lead_copies)
    n_sys_hops = sum(i[1] is None for i in hops)
    end_pos, start_pos = sys_leads_hopping_pos(sys, hops)
    # Apply transformations to the data and generate the colors.
    if pos_transform is not None:
        sites_pos = np.apply_along_axis(pos_transform, 1, sites_pos)
        end_pos = np.apply_along_axis(pos_transform, 1, end_pos)
        start_pos = np.apply_along_axis(pos_transform, 1, start_pos)
    if hasattr(site_color, '__call__'):
        site_color = [site_color(i[0]) for i in sites if i[1] is None]
    if hasattr(hop_color, '__call__'):
        hop_color = [hop_color(*i[0]) for i in hops if i[1] is None]
    # Choose plot type.
    dim = 3 if (sites_pos.shape[1] == 3) else 2
    ar = np.zeros((len(sites_pos), dim))
    ar[:, : min(dim, sites_pos.shape[1])] = sites_pos
    sites_pos = ar
    end_pos.resize(len(end_pos), min(dim, end_pos.shape[1]))
    start_pos.resize(len(start_pos), min(dim, start_pos.shape[1]))
    fig = Figure()
    if dpi is not None:
        fig.set_dpi(dpi)
    if fig_size is not None:
        fig.set_figwidth(fig_size[0])
        fig.set_figheight(fig_size[1])
    if isinstance(cmap, tuple):
        hop_cmap = cmap[1]
        cmap = cmap[0]
    else:
        hop_cmap = None
    if dim == 2:
        ax = fig.add_subplot(111, aspect='equal')
        ax.scatter(*sites_pos[: n_sys_sites].T, c=site_color, cmap=cmap,
                   s=size ** 2, zorder=2)
        end, start = end_pos[: n_sys_hops], start_pos[: n_sys_hops]
        lines(ax, end[:, 0], start[:, 0], end[:, 1], start[:, 1], hop_color,
              linewidths=thickness, zorder=1, cmap=hop_cmap)
        lead_site_colors = np.array([i[2] for i in
                                     sites if i[1] is not None], dtype=float)
        # Avoid the matplotlib autoscale bug (remove when fixed)
        if len(sites_pos) > n_sys_sites:
            ax.scatter(*sites_pos[n_sys_sites:].T, c=lead_site_colors,
                       cmap='gist_yarg_r', s=size ** 2, zorder=2,
                       norm=matplotlib.colors.Normalize(-1, n_lead_copies + 1))
        else:
            ax.add_collection(matplotlib.collections.PathCollection([]))
        lead_hop_colors = np.array([i[2] for i in
                                     hops if i[1] is not None], dtype=float)
        end, start = end_pos[n_sys_hops:], start_pos[n_sys_hops:]
        lines(ax, end[:, 0], start[:, 0], end[:, 1], start[:, 1],
              lead_hop_colors, linewidths=thickness, cmap='gist_yarg_r',
              norm=matplotlib.colors.Normalize(-1, n_lead_copies + 1),
              zorder=1)
    else:
        warnings.filterwarnings('ignore', message=r'.*rotation.*')
        ax = fig.add_subplot(111, projection='3d')
        warnings.resetwarnings()
        ax.scatter(*sites_pos[: n_sys_sites].T, c=site_color, cmap=cmap,
                   s=size ** 2)
        end, start = end_pos[: n_sys_hops], start_pos[: n_sys_hops]
        lines3d(ax, end[:, 0], start[:, 0], end[:, 1], start[:, 1],
              end[:, 2], start[:, 2], hop_color, cmap=hop_cmap,
              linewidths=thickness)
        lead_site_colors = np.array([i[2] for i in
                                     sites if i[1] is not None], dtype=float)
        lead_site_colors = 1 / np.sqrt(1. + lead_site_colors)
        # Avoid the matplotlib autoscale bug (remove when fixed)
        if len(sites_pos) > n_sys_sites:
            ax.scatter(*sites_pos[n_sys_sites:].T, c=lead_site_colors,
                       cmap='gist_yarg_r', s=size ** 2,
                       norm=matplotlib.colors.Normalize(-1, n_lead_copies + 1))
        else:
            col = mplot3d.art3d.Patch3DCollection([])
            col.set_3d_properties([], 'z')
            ax.add_collection3d(col)
        lead_hop_colors = np.array([i[2] for i in
                                     hops if i[1] is not None], dtype=float)
        lead_hop_colors = 1 / np.sqrt(1. + lead_hop_colors)
        end, start = end_pos[n_sys_hops:], start_pos[n_sys_hops:]
        lines3d(ax, end[:, 0], start[:, 0], end[:, 1], start[:, 1],
              end[:, 2], start[:, 2],
              lead_hop_colors, linewidths=thickness, cmap='gist_yarg_r',
              norm=matplotlib.colors.Normalize(-1, n_lead_copies + 1))
        min_ = np.min(sites_pos, 0)
        max_ = np.max(sites_pos, 0)
        w = np.max(max_ - min_) / 2
        m = (min_ + max_) / 2
        ax.auto_scale_xyz(*[(i - w, i + w) for i in m], had_data=True)
    if ax.collections[0].get_array() is not None and colorbar:
        fig.colorbar(ax.collections[0])
    if ax.collections[1].get_array() is not None and colorbar:
        fig.colorbar(ax.collections[1])
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
      makes sense to set `oversampling` to ``1`` to minimize the size of the
      output array.
    """
    # Build the bounding box.
    cmin, cmax = coords.min(0), coords.max(0)

    tree = spatial.cKDTree(coords)

    if a is None:
        points = coords[np.random.randint(len(coords), size=10)]
        a = np.min(tree.query(points, 2)[0][:, 1])
    elif a <= 0:
        raise ValueError("The distance a must be strictly positive.")

    shape = (((cmin - cmax) / a + 1) * oversampling).round()
    delta = 0.5 * (oversampling - 1) * a / oversampling
    cmin -= delta
    cmax += delta
    dims = tuple(slice(cmin[i], cmax[i], 1j * shape[i]) for i in
                 range(len(cmin)))
    grid = tuple(np.ogrid[dims])
    img = interpolate.griddata(coords, values, grid, method)
    mask = np.mgrid[dims].reshape(len(cmin), -1).T
    mask = tree.query(mask, eps=1.)[0] > 1.5 * a

    return np.ma.masked_array(img, mask), cmin, cmax


def map(sys, value, colorbar=True, cmap=None,
         a=None, method='nearest', oversampling=3, file=None, show=True,
         dpi=None, fig_size=None):
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
        Whether to show a color bar.  Defaults to `true`.
    cmap : `matplotlib` color map or `None`
        The color map used for sites and optionally hoppings, if `None`,
        `matplotlib` default is used.
    a : float, optional
        Reference length.  If not given, it is determined as a typical
        nearest neighbor distance.
    method : string, optional
        Passed to `scipy.interpolate.griddata` and to `matplotlib`
        `Axes.imshow.interpolation`: "nearest" (default), "linear", or "cubic".
    oversampling : integer, optional
        Number of pixels per reference length.  Defaults to 3.
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
    - See notes of `~kwant.plotter.show_interpolate`.

    - Matplotlib's interpolation is turned off, if the keyword argument
      `method` is not set or set to the default value "nearest".
    """
    sites = sys_leads_sites(sys, 0)
    coords = sys_leads_pos(sys, sites)
    if coords.shape[1] != 2:
        raise ValueError('Only 2D systems can be plotted this way.')
    if hasattr(value, '__call__'):
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
    if method != 'nearest':
        method = 'bi' + method
    image = ax.imshow(img.T, extent=(min[0], max[0], min[1], max[1]),
                      origin='lower', interpolation=method, cmap=cmap)
    if colorbar:
        fig.colorbar(image)
    return output_fig(fig, file=file, show=show)


def bands(sys, momenta=65, file=None, show=True, dpi=None, fig_size=None):
    """Plot band structure of a translationally invariant 1D system.

    Parameters
    ----------
    sys : kwant.system.InfiniteSystem
        A system bands of which are to be plotted.
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

    bands = physics.Bands(sys)
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
