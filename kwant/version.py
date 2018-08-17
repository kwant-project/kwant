# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

# This module must also work with Python 2.
from __future__ import print_function

import sys
import subprocess
import os

# No public API
__all__ = []

package_root = os.path.dirname(os.path.realpath(__file__))
distr_root = os.path.dirname(package_root)

def ensure_python(required_version=(3, 5)):
    v = sys.version_info
    if v[:3] < required_version:
        error = "This version of Kwant requires Python {} or above.".format(
            ".".join(str(p) for p in required_version))
        if v[0] == 2:
            error += "\nKwant 1.1 is the last version to support Python 2."
        print(error, file=sys.stderr)
        sys.exit(1)


def get_version_from_git():
    try:
        p = subprocess.Popen(['git', 'rev-parse', '--show-toplevel'],
                             cwd=distr_root,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return
    if p.wait() != 0:
        return
    if not os.path.samefile(p.communicate()[0].decode().rstrip('\n'),
                            distr_root):
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


# TODO: change this logic when there is a git pretty-format
#       that gives the same output as 'git describe'.
#       Currently we can only tell the tag the current commit is
#       pointing to, or its hash (with no version info)
#       if it is not tagged.
def get_version_from_git_archive(version_info):
    try:
        refnames = version_info['refnames']
        git_hash = version_info['git_hash']
    except KeyError:
        # These fields are not present if we are running from an sdist.
        # Execution should never reach here, though
        return None

    if git_hash.startswith('$Format') or refnames.startswith('$Format'):
        # variables not expanded during 'git archive'
        return None

    VTAG = 'tag: v'  # Our version tags always start with 'v'
    refs = set(r.strip() for r in refnames.split(","))
    version_tags = set(r[len(VTAG):] for r in refs if r.startswith(VTAG))
    if version_tags:
        release, *_ = sorted(version_tags)  # prefer e.g. "2.0" over "2.0rc1"
        return release
    else:
        return ''.join(('unknown', '+g', git_hash))


def init(version_file='_kwant_version.py'):
    global version, version_is_from_git
    version_info = {}
    with open(os.path.join(package_root, version_file), 'rb') as f:
        exec(f.read(), {}, version_info)
    version = version_info['version']
    version_is_from_git = (version == "__use_git__")
    if version_is_from_git:
        version = get_version_from_git()
        if not version:
            version = get_version_from_git_archive(version_info)
        if not version:
            version = "unknown"

init()
