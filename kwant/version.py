# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import subprocess
import os

__all__ = ['version']


# When changing this function, remember to also change its twin in ../setup.py.
def get_version_from_git():
    kwant_dir = os.path.dirname(os.path.abspath(__file__))
    try:
        p = subprocess.Popen(['git', 'describe'], cwd=kwant_dir,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return

    if p.wait() != 0:
        return
    version = p.communicate()[0].strip()

    if version[0] == 'v':
        version = version[1:]

    try:
        p = subprocess.Popen(['git', 'diff', '--quiet'], cwd=kwant_dir)
    except OSError:
        version += '-confused'  # This should never happen.
    else:
        if p.wait() == 1:
            version += '-dirty'
    return version

version = get_version_from_git()
if version is None:
    try:
        from _static_version import version
    except:
        version = "unknown"
