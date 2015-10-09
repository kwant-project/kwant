#!/usr/bin/env python

# Copyright 2011-2015 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

from __future__ import print_function

import sys
import re
import os
import glob
import imp
import subprocess
import ConfigParser
from distutils.core import setup, Extension, Command
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsModuleError, \
    CCompilerError
from distutils.command.build import build
from distutils.command.sdist import sdist

import numpy

CONFIG_FILE = 'build.conf'
README_FILE = 'README.rst'
README_END_BEFORE = 'See also in this directory:'
STATIC_VERSION_PATH = ('kwant', '_kwant_version.py')
REQUIRED_CYTHON_VERSION = (0, 22)
NO_CYTHON_OPTION = '--no-cython'
TUT_DIR = 'tutorial'
TUT_GLOB = 'doc/source/tutorial/*.py'
TUT_HIDDEN_PREFIX = '#HIDDEN'

# Let Kwant itself determine its own version.  We cannot simply import kwant, as
# it is not built yet.
_dont_write_bytecode_saved = sys.dont_write_bytecode
sys.dont_write_bytecode = True
try:
    imp.load_source(STATIC_VERSION_PATH[-1].split('.')[0],
                    os.path.join(*STATIC_VERSION_PATH))
except IOError:
    pass
_common = imp.load_source('_common', 'kwant/_common.py')
sys.dont_write_bytecode = _dont_write_bytecode_saved

version = _common.version
version_is_from_git = _common.version_is_from_git

try:
    sys.argv.remove(NO_CYTHON_OPTION)
    cythonize = False
except ValueError:
    cythonize = True

if cythonize:
    try:
        import Cython
    except:
        cython_version = ()
    else:
        match = re.match('([0-9.]*)(.*)', Cython.__version__)
        cython_version = [int(n) for n in match.group(1).split('.')]
        # Decrease version if the version string contains a suffix.
        if match.group(2):
            while cython_version[-1] == 0:
                cython_version.pop()
            cython_version[-1] -= 1
        cython_version = tuple(cython_version)

if cythonize and cython_version:
    from Cython.Distutils import build_ext
else:
    from distutils.command.build_ext import build_ext

distr_root = os.path.dirname(os.path.abspath(__file__))

def banner(title=''):
    starred = title.center(79, '*')
    return '\n' + starred if title else starred

error_msg = """{header}
The compilation of Kwant has failed.  Please examine the error message
above and consult the installation instructions in README.rst.
You might have to customize {{file}}.

Build configuration was:

{{summary}}
{sep}
"""
error_msg = error_msg.format(header=banner(' Error '), sep=banner())

class kwant_build_ext(build_ext):
    def run(self):
        if not config_file_present:
            # Create an empty config file if none is present so that the
            # extensions will not be rebuilt each time.  Only depending on the
            # config file if it is present would make it impossible to detect a
            # necessary rebuild due to a deleted config file.
            with open(CONFIG_FILE, 'w') as f:
                f.write('# Created by setup.py - feel free to modify.\n')

        try:
            build_ext.run(self)
        except (DistutilsError, CCompilerError):
            print(error_msg.format(file=CONFIG_FILE, summary=build_summary),
                  file=sys.stderr)
            raise
        print(banner(' Build summary '))
        print(build_summary)
        print(banner())


class kwant_build_tut(Command):
    description = "build the tutorial scripts"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        if not os.path.exists(TUT_DIR):
            os.mkdir(TUT_DIR)
        for in_fname in glob.glob(TUT_GLOB):
            out_fname = os.path.join(TUT_DIR, os.path.basename(in_fname))
            with open(in_fname) as in_file:
                with open(out_fname, 'w') as out_file:
                    for line in in_file:
                        if not line.startswith(TUT_HIDDEN_PREFIX):
                            out_file.write(line)


# Our version of the "build" command also makes sure the tutorial is made.
# Even though the tutorial is not necessary for installation, and "build" is
# supposed to make everything needed to install, this is a robust way to ensure
# that the tutorial is present.
class kwant_build(build):
    sub_commands = [('build_tut', None)] + build.sub_commands

    def run(self):
        build.run(self)
        write_version(os.path.join(self.build_lib, *STATIC_VERSION_PATH))


class kwant_test(Command):
    description = "build, then run the unit tests"
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        try:
            from nose.core import run
        except ImportError:
            raise DistutilsModuleError('nose <http://nose.readthedocs.org/> '
                                       'is needed to run the tests')
        self.run_command('build')
        major, minor = sys.version_info[:2]
        lib_dir = "build/lib.{0}-{1}.{2}".format(get_platform(), major, minor)
        print()
        if not run(argv=[__file__, '-v', lib_dir]):
            raise DistutilsError('at least one of the tests failed')


def git_lsfiles():
    if not version_is_from_git:
        return

    try:
        p = subprocess.Popen(['git', 'ls-files'], cwd=distr_root,
                             stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        return

    if p.wait() != 0:
        return
    return p.communicate()[0].split('\n')[:-1]


# Make the command "sdist" depend on "build".  This verifies that the
# distribution in the current state actually builds.  It also makes sure that
# the Cython-made C files and the tutorial will be included in the source
# distribution and that they will be up-to-date.
class kwant_sdist(sdist):
    sub_commands = [('build', None)] + sdist.sub_commands

    def run(self):
        names = git_lsfiles()
        trustworthy = True
        if names is None:
            # Check that MANIFEST exists and has not been generated by
            # distutils.
            try:
                with open(distr_root + '/MANIFEST', 'r') as f:
                    line = f.read()
            except IOError:
                print("Error: MANIFEST file is missing and Git is not"
                      " available to regenerate it.", file=sys.stderr)
                exit(1)
            trustworthy = not line.strip().startswith('#')
        else:
            # Generate MANIFEST file.
            with open(distr_root + '/MANIFEST', 'w') as f:
                for name in names:
                    a, sep, b = name.rpartition('/')
                    if b == '.gitignore':
                        continue
                    stem, dot, extension = b.rpartition('.')
                    if extension == 'pyx':
                        f.write(''.join([a, sep, stem, dot, 'c', '\n']))
                    f.write(name + '\n')
                f.write('MANIFEST\n')

        sdist.run(self)

        if names is None:
            print(banner(' Warning '),
"""Git was not available for re-generating the MANIFEST file (the list of file
names to be included in the source distribution).  The old MANIFEST was used.""",
                  banner(),
                  sep='\n', file=sys.stderr)

        if not trustworthy:
            print(banner(' Warning '),
"""The existing MANIFEST file seems to have been generated by distutils (it begins
with a comment).  It may well be incomplete.""",
                  banner(),
                  sep='\n', file=sys.stderr)

    def make_release_tree(self, base_dir, files):
        sdist.make_release_tree(self, base_dir, files)
        write_version(os.path.join(base_dir, *STATIC_VERSION_PATH))


def write_version(fname):
    # This could be a hard link, so try to delete it first.  Is there any way
    # to do this atomically together with opening?
    try:
        os.remove(fname)
    except OSError:
        pass
    with open(fname, 'w') as f:
        f.write("# This file has been created by setup.py.\n")
        f.write("version = '{}'\n".format(version))


def long_description():
    text = []
    try:
        with open(README_FILE) as f:
            for line in f:
                if line.startswith(README_END_BEFORE):
                    break
                text.append(line.rstrip())
            while text[-1] == "":
                text.pop()
    except:
        return ''
    return '\n'.join(text)


def packages():
    return [root.replace('/', '.')
            for root, dnames, fnames in os.walk('kwant')
            if '__init__.py' in fnames or root.endswith('/tests')]


def search_mumps():
    """Return the configuration for MUMPS if it is available in a known way.

    This is known to work with the MUMPS provided by the Debian package
    libmumps-scotch-dev."""

    libs = ['zmumps_scotch', 'mumps_common_scotch', 'pord', 'mpiseq_scotch',
            'gfortran']

    cmd = ['gcc']
    cmd.extend(['-l' + lib for lib in libs])
    cmd.extend(['-o/dev/null', '-xc', '-'])
    try:
        p = subprocess.Popen(cmd, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    except OSError:
        pass
    else:
        p.communicate(input='int main() {}\n')
        if p.wait() == 0:
            return {'libraries': libs}
    return {}


def extensions():
    """Return a list of tuples (args, kwrds) to be passed to
    Extension. possibly after replacing ".pyx" with ".c" if Cython is not to be
    used."""

    global build_summary, config_file_present
    build_summary = []

    #### Add components of Kwant without external compile-time dependencies.
    result = [
        (['kwant._system', ['kwant/_system.pyx']],
         {'include_dirs': ['kwant/graph']}),
        (['kwant.graph.core', ['kwant/graph/core.pyx']],
         {'depends': ['kwant/graph/core.pxd', 'kwant/graph/defs.h',
                      'kwant/graph/defs.pxd']}),
        (['kwant.graph.utils', ['kwant/graph/utils.pyx']],
         {'depends': ['kwant/graph/defs.h', 'kwant/graph/defs.pxd',
                      'kwant/graph/core.pxd']}),
        (['kwant.graph.slicer', ['kwant/graph/slicer.pyx',
                                 'kwant/graph/c_slicer/partitioner.cc',
                                 'kwant/graph/c_slicer/slicer.cc']],
         {'depends': ['kwant/graph/defs.h', 'kwant/graph/defs.pxd',
                      'kwant/graph/core.pxd',
                      'kwant/graph/c_slicer.pxd',
                      'kwant/graph/c_slicer/bucket_list.h',
                      'kwant/graph/c_slicer/graphwrap.h',
                      'kwant/graph/c_slicer/partitioner.h',
                      'kwant/graph/c_slicer/slicer.h']})]

    #### Add components of Kwant with external compile-time dependencies.
    config = ConfigParser.ConfigParser()
    try:
        with open(CONFIG_FILE) as f:
            config.readfp(f)
    except IOError:
        config_file_present = False
    else:
        config_file_present = True

    kwrds_by_section = {}
    for section in config.sections():
        kwrds_by_section[section] = kwrds = {}
        for name, value in config.items(section):
            kwrds[name] = value.split()

    # Setup LAPACK.
    lapack = kwrds_by_section.get('lapack')
    if lapack:
        build_summary.append('User-configured LAPACK and BLAS')
    else:
        lapack = {'libraries': ['lapack', 'blas']}
        build_summary.append('Default LAPACK and BLAS')
    kwrds = lapack.copy()
    kwrds.setdefault('depends', []).extend(
        [CONFIG_FILE, 'kwant/linalg/f_lapack.pxd'])
    result.append((['kwant.linalg.lapack', ['kwant/linalg/lapack.pyx']],
                   kwrds))

    # Setup MUMPS.
    kwrds = kwrds_by_section.get('mumps')
    if kwrds:
        build_summary.append('User-configured MUMPS')
    else:
        kwrds = search_mumps()
        if kwrds:
            build_summary.append('Auto-configured MUMPS')
    if kwrds:
        for name, value in lapack.iteritems():
            kwrds.setdefault(name, []).extend(value)
        kwrds.setdefault('depends', []).extend(
            [CONFIG_FILE, 'kwant/linalg/cmumps.pxd'])
        result.append((['kwant.linalg._mumps', ['kwant/linalg/_mumps.pyx']],
                       kwrds))
    else:
        build_summary.append('No MUMPS support')

    build_summary = '\n'.join(build_summary)
    return result


def complain_cython_unavailable():
    assert not cythonize or cython_version < REQUIRED_CYTHON_VERSION
    if cythonize:
        msg = ("Install Cython {0} or newer so it can be made\n"
               "or use a source distribution of Kwant.")
        ver = '.'.join(str(e) for e in REQUIRED_CYTHON_VERSION)
        print(msg.format(ver), file=sys.stderr)
    else:
        print("Run setup.py without {}.".format(NO_CYTHON_OPTION),
              file=sys.stderr)


def ext_modules(extensions):
    """Prepare the ext_modules argument for distutils' setup."""
    result = []
    problematic_files = []
    for args, kwrds in extensions:
        if not cythonize or cython_version < REQUIRED_CYTHON_VERSION:
            # Cython is not going to be run: replace pyx extension by that of
            # the shipped translated file.
            if 'language' in kwrds:
                if kwrds['language'] == 'c':
                    ext = '.c'
                elif kwrds['language'] == 'c++':
                    ext = '.cpp'
                else:
                    print('Unknown language', file=sys.stderr)
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

            # Complain if cythonized files are older than Cython source files.
            try:
                cythonized_oldest = min(os.stat(f).st_mtime
                                        for f in cythonized_files)
            except OSError:
                print("error: Cython-generated file {} is missing.".format(f),
                      file=sys.stderr)
                complain_cython_unavailable()
                exit(1)
            for f in pyx_files + kwrds.get('depends', []):
                if f == CONFIG_FILE:
                    # The config file is only a dependency for the compilation
                    # of the cythonized file, not for the cythonization.
                    continue
                if os.stat(f).st_mtime > cythonized_oldest:
                    problematic_files.append(f)

        result.append(Extension(*args, **kwrds))

    if problematic_files:
        problematic_files = ", ".join(problematic_files)
        msg = ("Some Cython source files are newer than files that should have\n"
               " been derived from them, but {}.\n"
               "\n"
               "Affected files: {}")
        if cythonize:
            if not cython_version:
                reason = "Cython is not installed"
            else:
                reason = "the installed Cython is too old"
            print(banner(" Error "), msg.format(reason, problematic_files),
                  banner(), sep="\n", file=sys.stderr)
            print()
            complain_cython_unavailable()
            exit(1)
        else:
            reason = "the option --no-cython has been given"
            print(banner(" Warning "), msg.format(reason, problematic_files),
                  banner(), sep='\n', file=sys.stderr)

    return result


def main():
    setup(name='kwant',
          version=version,
          author='C. W. Groth (CEA), M. Wimmer, '
                 'A. R. Akhmerov, X. Waintal (CEA), and others',
          author_email='authors@kwant-project.org',
          description="Package for numerical quantum transport calculations.",
          long_description=long_description(),
          platforms=["Unix", "Linux", "Mac OS-X", "Windows"],
          url="http://kwant-project.org/",
          license="BSD",
          packages=packages(),
          cmdclass={'build': kwant_build,
                    'sdist': kwant_sdist,
                    'build_ext': kwant_build_ext,
                    'build_tut': kwant_build_tut,
                    'test': kwant_test},
          ext_modules=ext_modules(extensions()),
          include_dirs=[numpy.get_include()])

if __name__ == '__main__':
    main()
