# Copyright 2011-2015 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import subprocess
import os
import numpy as np
import numbers
import inspect

__all__ = ['version', 'KwantDeprecationWarning', 'UserCodeError']

package_root = os.path.dirname(os.path.realpath(__file__))
distr_root = os.path.dirname(package_root)
version_file = '_kwant_version.py'

def get_version_from_git():
    try:
        p = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                             cwd=distr_root,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return
    if p.wait() != 0:
        return
    if not os.path.samefile(p.communicate()[0].decode().rstrip('\n'), distr_root):
        # The top-level directory of the current Git repository is not the same
        # as the root directory of the Kwant distribution: do not extract the
        # version from Git.
        return

    # git describe --first-parent does not take into account tags from branches
    # that were merged-in.
    for opts in [['--first-parent'], []]:
        try:
            p = subprocess.Popen(['git', 'describe', '--long'] + opts,
                                 cwd=distr_root,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return
        if p.wait() == 0:
            break
    else:
        return
    description = p.communicate()[0].decode().strip('v').rstrip('\n')

    release, dev, git = description.rsplit('-', 2)
    version = [release]
    labels = []
    if dev != "0":
        version.append(".dev{}".format(dev))
        labels.append(git)

    try:
        p = subprocess.Popen(['git', 'diff', '--quiet'], cwd=distr_root)
    except OSError:
        labels.append('confused') # This should never happen.
    else:
        if p.wait() == 1:
            labels.append('dirty')

    if labels:
        version.append('+')
        version.append(".".join(labels))

    return "".join(version)



# populate the version_info dictionary with values stored in the version file
version_info = {}
with open(os.path.join(package_root, version_file), 'r') as f:
    exec(f.read(), {}, version_info)
version = version_info['version']
version_is_from_git = (version == "__use_git__")
if version_is_from_git:
    version = get_version_from_git()
    if not version:
        version = "unknown"


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


def get_parameters(func):
    """Get the names of the parameters to 'func' and whether it takes kwargs.

    Returns
    -------
    names : list
        Positional, keyword and keyword only parameter names in the order
        that they appear in the signature of 'func'.
    takes_kwargs : bool
        True if 'func' takes '**kwargs'.
    """
    sig = inspect.signature(func)
    pars = sig.parameters

    # Signature.parameters is an *ordered mapping*
    names = [k for (k, v) in pars.items()
             if v.kind in (inspect.Parameter.POSITIONAL_OR_KEYWORD,
                           inspect.Parameter.KEYWORD_ONLY)]
    takes_kwargs = any(i.kind is inspect.Parameter.VAR_KEYWORD
                       for i in pars.values())
    return names, takes_kwargs
