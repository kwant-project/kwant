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

from . import builder
from .builder import herm_conj, HermConjOfFunc
from .lattice import TranslationalSymmetry
from ._common import get_parameters


__all__ = ['wraparound']


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

        params, takes_kwargs = get_parameters(val)
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
            p, takes_kwargs = get_parameters(val)
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
            p, takes_kwargs = get_parameters(val)
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
            val_params, val_takes_kwargs = get_parameters(val)
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
        sites[site] = [bind_site(val) if callable(val) else val]

    for hop, val in builder.hopping_value_pairs():
        a, b = hop
        b_dom = sym.which(b)
        b_wa = sym.act(-b_dom, b)

        if a == b_wa:
            # The hopping gets wrapped-around into an onsite Hamiltonian.
            # Since site `a` already exists in the system, we can simply append.
            sites[a].append(bind_hopping_as_site(b_dom, val))
        else:
            # The hopping remains a hopping.
            if b != b_wa or callable(val):
                # The hopping got wrapped-around or is a function.
                val = bind_hopping(b_dom, val)

            # Make sure that there is only one entry for each hopping
            # (pointing in one direction).
            if (b_wa, a) in hops:
                assert (a, b_wa) not in hops
                if callable(val):
                    assert not isinstance(val, HermConjOfFunc)
                    val = HermConjOfFunc(val)
                else:
                    val = herm_conj(val)

                hops[b_wa, a].append(val)
            else:
                hops[a, b_wa].append(val)

    # Copy stuff into result builder, converting lists of more than one element
    # into summing functions.
    for site, vals in sites.items():
        ret[site] = vals[0] if len(vals) == 1 else bind_sum(1, *vals)

    for hop, vals in hops.items():
        ret[hop] = vals[0] if len(vals) == 1 else bind_sum(2, *vals)

    return ret
