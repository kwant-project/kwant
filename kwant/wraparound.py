# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import collections
import inspect
import cmath

import tinyarray as ta
import numpy as np
import scipy.linalg
import scipy.spatial

from . import builder, system, plotter
from .linalg import lll
from .builder import herm_conj, HermConjOfFunc
from .lattice import TranslationalSymmetry
from ._common import get_parameters


__all__ = ['wraparound', 'plot_2d_bands']


def _hashable(obj):
    return isinstance(obj, collections.Hashable)


def _memoize(f):
    """Decorator to memoize a function that works even with unhashable args.

    This decorator will even work with functions whose args are not hashable.
    The cache key is made up by the hashable arguments and the ids of the
    non-hashable args.  It is up to the user to make sure that non-hashable
    args do not change during the lifetime of the decorator.

    This decorator will keep reevaluating functions that return None.
    """
    def lookup(*args):
        key = tuple(arg if _hashable(arg) else id(arg) for arg in args)
        result = cache.get(key)
        if result is None:
            cache[key] = result = f(*args)
        return result
    cache = {}
    return lookup


def _modify_signature(func, parameter_names, takes_kwargs):
    """Modify the signature of 'func', and return 'func'.

    Parameters
    ----------
    func : callable
    parameter_names: sequence of str
        Parameter names to put in the signature. These will be added as
        'POSITIONAL_OR_KEYWORD' type parameters.
    takes_kwargs: bool
        If 'True', then a 'kwargs' parameter with type 'VAR_KEYWORD' is added
        to the end of the signature.
    """
    params = [inspect.Parameter(name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
               for name in parameter_names]
    if takes_kwargs:
        params.append(inspect.Parameter('kwargs',
                                        inspect.Parameter.VAR_KEYWORD))

    func.__signature__ = inspect.Signature(params)
    return func


## This wrapper is needed so that finalized systems that
## have been wrapped can be queried for their symmetry, which
## is needed for Brillouin zone calculations (plotting).

class WrappedBuilder(builder.Builder):

    def finalized(self):
        ret = super().finalized()
        ret._momentum_names = self._momentum_names
        ret._wrapped_symmetry = self._wrapped_symmetry
        return ret


def wraparound(builder, keep=None, *, coordinate_names=('x', 'y', 'z')):
    """Replace translational symmetries by momentum parameters.

    A new Builder instance is returned.  By default, each symmetry is replaced
    by one scalar momentum parameter that is appended to the already existing
    arguments of the system.  Optionally, one symmetry may be kept by using the
    `keep` argument. The momentum parameters will have names like 'k_n' where
    the 'n' are specified by 'coordinate_names'.

    Parameters
    ----------
    builder : `~kwant.builder.Builder`
    keep : int, optional
        Which (if any) translational symmetry to keep.
    coordinate_names : sequence of strings, default: ('x', 'y', 'z')
        The names of the coordinates along the symmetry
        directions of 'builder'.

    Notes
    -----
    Wraparound is stop-gap functionality until Kwant 2.x which will include
    support for higher-dimension translational symmetry in the low-level system
    format. It will be deprecated in the 2.0 release of Kwant.
    """

    # In the following 'assert' because 'syst.hamiltonian'
    # should force the function to be called with *either* '*args'
    # or '**kwargs', not both. Also, we use different codepaths for
    # the two cases to avoid a performance drop when using '*args' only.

    @_memoize
    def bind_site(val):
        assert callable(val)

        def f(*args, **kwargs):
            a, *args = args
            assert not (args and kwargs)
            if kwargs:
                if not takes_kwargs:
                    kwargs = {p: kwargs[p] for p in extra_params}
                return val(a, **kwargs)
            else:
                return val(a, *args[:mnp])

        params, defaults, takes_kwargs = get_parameters(val)
        extra_params = params[1:]
        return _modify_signature(f, params + momenta, takes_kwargs)

    @_memoize
    def bind_hopping_as_site(elem, val):

        def f(*args, **kwargs):
            a, *args = args
            assert not (args and kwargs)
            sym_a = sym.act(elem, a)
            k = [kwargs[k] for k in momenta] if kwargs else args[mnp:]
            phase = cmath.exp(1j * ta.dot(elem, k))
            if not callable(val):
                v = val
            elif kwargs:
                if not takes_kwargs:
                    kwargs = {p: kwargs[p] for p in extra_params}
                v = val(a, sym_a, **kwargs)
            else:
                v = val(a, sym_a, *args[:mnp])

            pv = phase * v
            return pv + herm_conj(pv)

        params, takes_kwargs = ['_site0'], False
        if callable(val):
            p, defaults, takes_kwargs = get_parameters(val)
            extra_params = p[2:]  # cut off both site parameters
            params += extra_params

        return _modify_signature(f,  params + momenta, takes_kwargs)

    @_memoize
    def bind_hopping(elem, val):

        def f(*args, **kwargs):
            a, b, *args = args
            assert not (args and kwargs)
            sym_b = sym.act(elem, b)
            k = [kwargs[k] for k in momenta] if kwargs else args[mnp:]
            phase = cmath.exp(1j * ta.dot(elem, k))
            if not callable(val):
                v = val
            elif kwargs:
                if not takes_kwargs:
                    kwargs = {p: kwargs[p] for p in extra_params}
                v = val(a, sym_b, **kwargs)
            else:
                v = val(a, sym_b, *args[:mnp])

            return phase * v

        params, takes_kwargs = ['_site0', '_site1'], False
        if callable(val):
            p, defaults, takes_kwargs = get_parameters(val)
            extra_params = p[2:]  # cut off site parameters
            params += extra_params

        return _modify_signature(f, params + momenta, takes_kwargs)

    @_memoize
    def bind_sum(num_sites, *vals):
        # Inside 'f' we do not have to split off only the used args/kwargs
        # because if 'val' is callable, it is guaranteed to have been wrapped
        # by 'bind_site', 'bind_hopping' or 'bind_hopping_as_site', which do
        # the disambiguation for us.
        def f(*args, **kwargs):
            return sum((val(*args, **kwargs) if callable(val) else val)
                       for val in vals)

        # construct joint signature for all 'vals'.
        parameters, takes_kwargs = collections.OrderedDict(), False
        # first the 'site' parameters
        for s in range(num_sites):
            name = '_site{}'.format(s)
            parameters[name] = inspect.Parameter(
                name, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        # now all the other parameters, except for the momenta
        for val in filter(callable, vals):
            val_params, defaults, val_takes_kwargs = get_parameters(val)
            val_params = val_params[num_sites:]   # remove site parameters
            takes_kwargs = takes_kwargs or val_takes_kwargs
            for p in val_params:
                # Skip parameters that exist in previously added functions,
                # and the momenta, which will be placed at the end.
                if p in parameters or p in momenta:
                    continue
                parameters[p] = inspect.Parameter(
                    p, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        # finally add the momenta.
        for k in momenta:
            parameters[k] = inspect.Parameter(
                k, inspect.Parameter.POSITIONAL_OR_KEYWORD)
        if takes_kwargs:
            parameters['kwargs'] = inspect.Parameter(
                'kwargs', inspect.Parameter.VAR_KEYWORD)
        f.__signature__ = inspect.Signature(parameters.values())

        return f


    if len(builder.symmetry.periods) > len(coordinate_names):
        raise ValueError("All symmetry directions must have a name specified "
                         "in coordinate_names")

    momenta = ['k_{}'.format(n) for n, _ in
               zip(coordinate_names, builder.symmetry.periods)]

    if keep is None:
        ret = WrappedBuilder()
        sym = builder.symmetry
    else:
        periods = list(builder.symmetry.periods)
        ret = WrappedBuilder(TranslationalSymmetry(periods.pop(keep)))
        sym = TranslationalSymmetry(*periods)
        momenta.pop(keep)
    # Store the names of the momentum parameters and the symmetry of the
    # old Builder (this will be needed for band structure plotting)
    ret._momentum_names = momenta
    ret._wrapped_symmetry = builder.symmetry

    # Wrapped around system retains conservation law and chiral symmetry.
    # We use 'bind_site' to add the momenta arguments if required.
    cons = builder.conservation_law
    ret.conservation_law = bind_site(cons) if callable(cons) else cons
    chiral = builder.chiral
    ret.chiral = bind_site(chiral) if callable(chiral) else chiral
    # Set these to zero, as they can only hold @ k=0, and we currently
    # have no mechanism for telling the system about the existence
    # (or not) of a symmetry at different parameter values.
    ret.particle_hole = None
    ret.time_reversal = None

    sites = {}
    hops = collections.defaultdict(list)

    mnp = -len(sym.periods)      # Used by the bound functions above.

    # Store lists of values, so that multiple values can be assigned to the
    # same site or hopping.
    for site, val in builder.site_value_pairs():
        # Every 'site' is in the FD of the original symmetry.
        # Move the sites to the FD of the remaining symmetry, this guarantees that
        # every site in the new system is an image of an original FD site translated
        # purely by the remaining symmetry.
        sites[ret.symmetry.to_fd(site)] = [bind_site(val) if callable(val) else val]

    for hop, val in builder.hopping_value_pairs():
        a, b = hop
        # 'a' is in the FD of original symmetry.
        # Get translation from FD of original symmetry to 'b',
        # this is different from 'b_dom = sym.which(b)'.
        b_dom = builder.symmetry.which(b)
        # Throw away part that is in the remaining translation direction, so we get
        # an element of 'sym' which is being wrapped
        b_dom = ta.array([t for i, t in enumerate(b_dom) if i != keep])
        # Pull back using the remainder, which is purely in the wrapped directions.
        # This guarantees that 'b_wa' is an image of an original FD site translated
        # purely by the remaining symmetry.
        b_wa = sym.act(-b_dom, b)
        # Move the hopping to the FD of the remaining symmetry
        a, b_wa = ret.symmetry.to_fd(a, b_wa)

        if a == b_wa:
            # The hopping gets wrapped-around into an onsite Hamiltonian.
            # Since site `a` already exists in the system, we can simply append.
            sites[a].append(bind_hopping_as_site(b_dom, val))
        else:
            # The hopping remains a hopping.
            if any(b_dom) or callable(val):
                # The hopping got wrapped-around or is a function.
                val = bind_hopping(b_dom, val)

            # Make sure that there is only one entry for each hopping
            # pointing in one direction, modulo the remaining translations.
            b_wa_r, a_r = ret.symmetry.to_fd(b_wa, a)
            if (b_wa_r, a_r) in hops:
                assert (a, b_wa) not in hops
                if callable(val):
                    assert not isinstance(val, HermConjOfFunc)
                    val = HermConjOfFunc(val)
                else:
                    val = herm_conj(val)

                hops[b_wa_r, a_r].append(val)
            else:
                hops[a, b_wa].append(val)

    # Copy stuff into result builder, converting lists of more than one element
    # into summing functions.
    for site, vals in sites.items():
        ret[site] = vals[0] if len(vals) == 1 else bind_sum(1, *vals)

    for hop, vals in hops.items():
        ret[hop] = vals[0] if len(vals) == 1 else bind_sum(2, *vals)

    return ret


def plot_2d_bands(syst, k_x=31, k_y=31, params=None,
                  mask_brillouin_zone=False, extend_bbox=0, file=None,
                  show=True, dpi=None, fig_size=None, ax=None):
    """Plot 2D band structure of a wrapped around system.

    This function is primarily useful for systems that have translational
    symmetry vectors that are non-orthogonal (e.g. graphene). This function
    will properly plot the band structure in an orthonormal basis in k-space,
    as opposed to in the basis of reciprocal lattice vectors (which would
    produce a "skewed" Brillouin zone).

    If your system has orthogonal lattice vectors, you are probably better
    off using `kwant.plotter.spectrum`.

    Parameters
    ----------
    syst : `kwant.system.FiniteSystem`
        A 2D system that was finalized from a Builder produced by
        `kwant.wraparound.wraparound`.  Note that this *must* be a finite
        system; so `kwant.wraparound.wraparound` should have been called with
        ``keep=None``.
    k_x, k_y : int or sequence of float, default: 31
        Either a number of sampling points, or a sequence of points at which
        the band structure is to be evaluated, in units of inverse length.
    params : dict, optional
        Dictionary of parameter names and their values, not including the
        momentum parameters.
    mask_brillouin_zone : bool, default: False
        If True, then the band structure will only be plotted over the first
        Brillouin zone. By default the band structure is plotted over a
        rectangular bounding box that contains the Brillouin zone.
    extend_bbox : float, default: 0
        Amount by which to extend the region over which the band structure is
        plotted, expressed as a proportion of the Brillouin zone bounding box
        length. i.e. ``extend_bbox=0.1`` will extend the region by 10% (in all
        directions).
    file : string or file object, optional
        The output file.  If None, output will be shown instead.
    show : bool, default: False
        Whether ``matplotlib.pyplot.show()`` is to be called, and the output is
        to be shown immediately.  Defaults to `True`.
    dpi : float, optional
        Number of pixels per inch.  If not set the ``matplotlib`` default is
        used.
    fig_size : tuple, optional
        Figure size `(width, height)` in inches.  If not set, the default
        ``matplotlib`` value is used.
    ax : ``matplotlib.axes.Axes`` instance, optional
        If `ax` is not `None`, no new figure is created, but the plot is done
        within the existing Axes `ax`. in this case, `file`, `show`, `dpi`
        and `fig_size` are ignored.

    Returns
    -------
    fig : matplotlib figure
        A figure with the output if `ax` is not set, else None.

    Notes
    -----
    This function produces plots where the units of momentum are inverse
    length. This is contrary to `kwant.plotter.bands`, where the units
    of momentum are inverse lattice constant.

    If the lattice vectors for the symmetry of ``syst`` are not orthogonal,
    then part of the plotted band structure will be outside the first Brillouin
    zone (inside the bounding box of the brillouin zone). Setting
    ``mask_brillouin_zone=True`` will cause the plot to be truncated outside of
    the first Brillouin zone.

    See Also
    --------
    kwant.plotter.spectrum
    """
    if not hasattr(syst, '_wrapped_symmetry'):
        raise TypeError("Expecting a system that was produced by "
                        "'kwant.wraparound.wraparound'.")
    if not isinstance(syst, system.FiniteSystem):
        msg = ("All symmetry directions must be wrapped around: specify "
               "'keep=None' when calling 'kwant.wraparound.wraparound'.")
        raise TypeError(msg)

    params = params or {}
    lat_ndim, space_ndim = syst._wrapped_symmetry.periods.shape

    if lat_ndim != 2:
        raise ValueError("Expected a system with a 2D translational symmetry.")
    if space_ndim != lat_ndim:
        raise ValueError("Lattice dimension must equal realspace dimension.")

    # columns of B are lattice vectors
    B = np.array(syst._wrapped_symmetry.periods).T
    # columns of A are reciprocal lattice vectors
    A = np.linalg.pinv(B).T

    ## calculate the bounding box for the 1st Brillouin zone

    # Get lattice points that neighbor the origin, in basis of lattice vectors
    reduced_vecs, transf = lll.lll(A.T)
    neighbors = ta.dot(lll.voronoi(reduced_vecs), transf)
    # Add the origin to these points.
    klat_points = np.concatenate(([[0] * lat_ndim], neighbors))
    # Transform to cartesian coordinates and rescale.
    # Will be used in 'outside_bz' function, later on.
    klat_points = 2 * np.pi * np.dot(klat_points, A.T)
    # Calculate the Voronoi cell vertices
    vor = scipy.spatial.Voronoi(klat_points)
    around_origin = vor.point_region[0]
    bz_vertices = vor.vertices[vor.regions[around_origin]]
    # extract bounding box
    k_max = np.max(np.abs(bz_vertices), axis=0)

    ## build grid along each axis, if needed
    ks = []
    for k, km in zip((k_x, k_y), k_max):
        k = np.array(k)
        if not k.shape:
            if extend_bbox:
                km += km * extend_bbox
            k = np.linspace(-km, km, k)
        ks.append(k)

    # TODO: It is very inefficient to call 'momentum_to_lattice' once for
    #       each point (for trivial Hamiltonians 60% of the time is spent
    #       doing this). We should instead transform the whole grid in one call.

    def momentum_to_lattice(k):
        k, residuals = scipy.linalg.lstsq(A, k)[:2]
        if np.any(abs(residuals) > 1e-7):
            raise RuntimeError("Requested momentum doesn't correspond"
                               " to any lattice momentum.")
        return k

    def ham(k_x, k_y=None, **params):
        # transform into the basis of reciprocal lattice vectors
        k = momentum_to_lattice([k_x] if k_y is None else [k_x, k_y])
        p = dict(zip(syst._momentum_names, k), **params)
        return syst.hamiltonian_submatrix(params=p, sparse=False)

    def outside_bz(k_x, k_y, **_):
        dm = scipy.spatial.distance_matrix(klat_points, [[k_x, k_y]])
        return np.argmin(dm) != 0  # is origin no closest 'klat_point' to 'k'?

    fig = plotter.spectrum(ham,
                           x=('k_x', ks[0]),
                           y=('k_y', ks[1]) if lat_ndim == 2 else None,
                           params=params,
                           mask=(outside_bz if mask_brillouin_zone else None),
                           file=file, show=show, dpi=dpi,
                           fig_size=fig_size, ax=ax)
    return fig
