# Copyright 2011-2015 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import subprocess
import os

__all__ = ['version', 'KwantDeprecationWarning']

distr_root = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))

def get_version_from_git():
    try:
        p = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                             cwd=distr_root,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return
    if p.wait() != 0:
        return
    # TODO: use os.path.samefile once we depend on Python >= 3.3.
    if os.path.normpath(p.communicate()[0].rstrip('\n')) != distr_root:
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
    description = p.communicate()[0].strip('v').rstrip('\n')

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



from _kwant_version import version
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


def ensure_isinstance(obj, typ, msg=None):
    if isinstance(obj, typ):
        return
    if msg is None:
        msg = "Expecting an instance of {}.".format(typ.__name__)
    raise TypeError(msg)
