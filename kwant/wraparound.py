# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import collections
import cmath
import tinyarray as ta

from .builder import Builder, herm_conj, HermConjOfFunc
from .lattice import TranslationalSymmetry


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


def wraparound(builder, keep=None):
    """Replace translational symmetries by momentum parameters.

    A new Builder instance is returned.  By default, each symmetry is replaced
    by one scalar momentum parameter that is appended to the already existing
    arguments of the system.  Optionally, one symmetry may be kept by using the
    `keep` argument.

    Parameters
    ----------
    builder : `~kwant.builder.Builder`
    keep : int, optional
        Which (if any) translational symmetry to keep.
    """

    @_memoize
    def bind_site(val):
        assert callable(val)
        return lambda a, *args: val(a, *args[:mnp])

    @_memoize
    def bind_hopping_as_site(elem, val):
        def f(a, *args):
            phase = cmath.exp(1j * ta.dot(elem, args[mnp:]))
            v = val(a, sym.act(elem, a), *args[:mnp]) if callable(val) else val
            pv = phase * v
            return pv + herm_conj(pv)
        return f

    @_memoize
    def bind_hopping(elem, val):
        def f(a, b, *args):
            phase = cmath.exp(1j * ta.dot(elem, args[mnp:]))
            v = val(a, sym.act(elem, b), *args[:mnp]) if callable(val) else val
            return phase * v
        return f

    @_memoize
    def bind_sum(*vals):
        return lambda *args: sum((val(*args) if callable(val) else val)
                                 for val in vals)

    if keep is None:
        ret = Builder()
        sym = builder.symmetry
    else:
        periods = list(builder.symmetry.periods)
        ret = Builder(TranslationalSymmetry(periods.pop(keep)))
        sym = TranslationalSymmetry(*periods)

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
        ret[site] = vals[0] if len(vals) == 1 else bind_sum(*vals)

    for hop, vals in hops.items():
        ret[hop] = vals[0] if len(vals) == 1 else bind_sum(*vals)

    return ret
