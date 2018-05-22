# Copyright 2011-2015 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import sys
import numpy as np
import numbers
import inspect
import warnings
import importlib
from contextlib import contextmanager
from collections import namedtuple

__all__ = ['KwantDeprecationWarning', 'UserCodeError']


class KwantDeprecationWarning(Warning):
    """Class of warnings about a deprecated feature of Kwant.

    DeprecationWarning has been made invisible by default in Python 2.7 in order
    to not confuse non-developer users with warnings that are not relevant to
    them.  In the case of Kwant, by far most users are developers, so we feel
    that a KwantDeprecationWarning that is visible by default is useful.
    """
    pass


class UserCodeError(Exception):
    """Class for errors that occur in user-provided code.

    Usually users will define value functions that Kwant calls in order to
    evaluate the Hamiltonian.  If one of these function raises an exception
    then it is caught and this error is raised in its place. This makes it
    clear that the error is from the user's code (and not a bug in Kwant) and
    also makes it possible for any libraries that wrap Kwant to detect when a
    user's function causes an error.
    """
    pass


def interleave(seq):
    """Return an iterator that yields pairs of elements from a sequence.

    If 'seq' has an odd number of elements then the last element is dropped.

    Examples
    --------
    >>> list(interleave(range(4)))
    [(0, 1), (2, 3)]
    >>> list(interleave(range(5))
    [(0, 1), (2, 3)]
    """
    # zip, when given the same iterator twice, turns a sequence into a
    # sequence of pairs.
    iseq = iter(seq)
    return zip(iseq, iseq)


def ensure_isinstance(obj, typ, msg=None):
    if isinstance(obj, typ):
        return
    if msg is None:
        msg = "Expecting an instance of {}.".format(typ.__name__)
    raise TypeError(msg)

def ensure_rng(rng=None):
    """Turn rng into a random number generator instance

    If rng is None, return the RandomState instance used by np.random.
    If rng is an integer, return a new RandomState instance seeded with rng.
    If rng is already a RandomState instance, return it.
    Otherwise raise ValueError.
    """
    if rng is None:
        return np.random.mtrand._rand
    if isinstance(rng, numbers.Integral):
        return np.random.RandomState(rng)
    if all(hasattr(rng, attr) for attr in ('random_sample', 'randn',
                                           'randint', 'choice')):
        return rng
    raise ValueError("Expecting a seed or an object that offers the "
                     "numpy.random.RandomState interface.")


@contextmanager
def reraise_warnings(level=3):
    with warnings.catch_warnings(record=True) as caught_warnings:
        yield
    for warning in caught_warnings:
        warnings.warn(warning.message, stacklevel=level)


_Params = namedtuple('_Params', ('required', 'defaults', 'takes_kwargs'))


def get_parameters(func):
    """Get the names of the parameters to 'func' and whether it takes kwargs.

    Returns
    -------
    required : list
        Names of positional, and keyword only parameters that do not have a
        default value and that appear in the signature of 'func'.
    defaults : list
        Names of parameters that have a default value.
    takes_kwargs : bool
        True if 'func' takes '**kwargs'.
    """
    sig = inspect.signature(func)
    pars = sig.parameters

    # Signature.parameters is an *ordered mapping*
    required_params = [k for (k, v) in pars.items()
                       if v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                                     inspect.Parameter.KEYWORD_ONLY)
                       and v.default is inspect._empty]
    default_params = [k for (k, v) in pars.items()
                      if v.default is not inspect._empty]
    takes_kwargs = any(i.kind is inspect.Parameter.VAR_KEYWORD
                       for i in pars.values())
    return _Params(required_params, default_params, takes_kwargs)


class lazy_import:
    def __init__(self, module, package='kwant', deprecation_warning=False):
        if module.startswith('.') and not package:
            raise ValueError('Cannot import a relative module without a package.')
        self.__module = module
        self.__package = package
        self.__deprecation_warning = deprecation_warning

    def __getattr__(self, name):
        if self.__deprecation_warning:
            msg = ("Accessing {0} without an explicit import is deprecated. "
                   "Instead, explicitly 'import {0}'."
                  ).format('.'.join((self.__package, self.__module)))
            warnings.warn(msg, KwantDeprecationWarning, stacklevel=2)
        relative_module = '.' + self.__module
        mod = importlib.import_module(relative_module, self.__package)
        # Replace this _LazyModuleProxy with an actual module
        package = sys.modules[self.__package]
        setattr(package, self.__module, mod)
        return getattr(mod, name)
