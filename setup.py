#!/usr/bin/env python

import sys, subprocess, os
from distutils.core import setup
from distutils.extension import Extension
import numpy as np


# This is an exact copy of the function from kwant/version.py.  We can't import
# it here (because kwant is not yet built when this scipt is run), so we just
# include a copy.
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

try:
    from Cython.Distutils import build_ext
except ImportError:
    use_cython = False
else:
    use_cython = True

def get_static_version():
    try:
        with open('kwant/_static_version.py') as f:
            contents = f.read()
            assert contents[:11] == "version = '"
            assert contents[-2:] == "'\n"
            return contents[11:-2]
    except:
        return None

git_version = get_version_from_git()
static_version = get_static_version()
if git_version is not None:
    version = git_version
    if static_version != git_version:
        with open('kwant/_static_version.py', 'w') as f:
            f.write("version = '%s'\n" % version)
elif static_version is not None:
    version = static_version
else:
    version = 'unknown'

# List of tuples (args, keywords) to be passed to Extension, possibly after
# replacing ".pyx" with ".c" if Cython is not to be used.
extensions = [ # (["kwant.graph.scotch", ["kwant/graph/scotch.pyx"]],
               #  {"libraries" : ["scotch", "scotcherr"]}),
               (["kwant.graph.core", ["kwant/graph/core.pyx"]], {}),
               (["kwant.graph.utils", ["kwant/graph/utils.pyx"]], {}),
               (["kwant.graph.slicer", ["kwant/graph/slicer.pyx",
                                        "kwant/graph/c_slicer/partitioner.cc",
                                        "kwant/graph/c_slicer/slicer.cc"]],
                {}),
               (["kwant.linalg.lapack", ["kwant/linalg/lapack.pyx"]],
                {"libraries" : ["lapack", "blas"]}) ]

cmdclass = {}
ext_modules = []
include_dirs = [np.get_include()]

for args, keywords in extensions:
    if not use_cython:
        if 'language' in keywords:
            if keywords['language'] == 'c':
                ext = '.c'
            elif keywords['language'] == 'c++':
                ext = '.cpp'
            else:
                print >>sys.stderr, 'Unknown language'
                exit(1)
        else:
            ext = '.c'
        args[1] = [s.replace('.pyx', ext) for s in args[1]]
    ext_modules.append(Extension(*args, **keywords))
if use_cython:
    cmdclass.update({'build_ext': build_ext})

setup(name='kwant',
      version=version,
      author='A. R. Akhmerov, C. W. Groth, X. Waintal, M. Wimmer',
      author_email='cwg@falma.de',
      description="A package for numerical quantum transport calculations.",
      license="not to be distributed",
      packages=["kwant", "kwant.graph", "kwant.linalg", "kwant.physics",
                "kwant.solvers"],
      cmdclass=cmdclass,
      ext_modules=ext_modules,
      include_dirs = include_dirs)
