#!/usr/bin/env python

import sys, subprocess, os
from distutils.core import setup
from distutils.extension import Extension
import numpy as np

run_cython = '--run-cython' in sys.argv
if run_cython:
    sys.argv.remove('--run-cython')
    from Cython.Distutils import build_ext
    cmdclass = {'build_ext': build_ext}
else:
    cmdclass = {}

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
               (["kwant._system", ["kwant/_system.pyx"]],
                {"include_dirs" : ["kwant/graph"]}),
               (["kwant.graph.core", ["kwant/graph/core.pyx"]],
                {"depends" : ["kwant/graph/core.pxd", "kwant/graph/defs.h",
                              "kwant/graph/defs.pxd"]}),
               (["kwant.graph.utils", ["kwant/graph/utils.pyx"]],
                {"depends" : ["kwant/graph/defs.h", "kwant/graph/defs.pxd",
                              "kwant/graph/core.pxd"]}),
               (["kwant.graph.slicer", ["kwant/graph/slicer.pyx",
                                        "kwant/graph/c_slicer/partitioner.cc",
                                        "kwant/graph/c_slicer/slicer.cc"]],
                {"depends" : ["kwant/graph/defs.h", "kwant/graph/defs.pxd",
                              "kwant/graph/core.pxd",
                              "kwant/graph/c_slicer.pxd",
                              "kwant/graph/c_slicer/bucket_list.h",
                              "kwant/graph/c_slicer/graphwrap.h",
                              "kwant/graph/c_slicer/partitioner.h",
                              "kwant/graph/c_slicer/slicer.h"]}),
               (["kwant.linalg.lapack", ["kwant/linalg/lapack.pyx"]],
                {"libraries" : ["lapack", "blas"],
                 "depends" : ["kwant/linalg/f_lapack.pxd"]}) ]

ext_modules = []
for args, keywords in extensions:
    if not run_cython:
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
        pyx_files = []
        cythonized_files = []
        sources = []
        for f in args[1]:
            if f[-4:] == '.pyx':
                pyx_files.append(f)
                f = f[:-4] + ext
                cythonized_files.append(f)
            sources.append(f)
        args[1] = sources

        try:
            cythonized_oldest = min(os.stat(f).st_mtime
                                    for f in cythonized_files)
        except OSError:
            msg = "{0} is missing. Run `./setup.py --run-cython build'."
            print >>sys.stderr, msg.format(f)
            exit(1)
        for f in pyx_files + keywords.get('depends', []):
            if os.stat(f).st_mtime > cythonized_oldest:
                msg = "{0} has been modified. " \
                "Run `./setup.py --run-cython build'."
                print >>sys.stderr, msg.format(f)
                exit(1)

    ext_modules.append(Extension(*args, **keywords))

include_dirs = [np.get_include()]

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
