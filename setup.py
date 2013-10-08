#!/usr/bin/env python

# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

CONFIG_FILE = 'build.conf'
README_FILE = 'README'
STATIC_VERSION_FILE = 'kwant/_static_version.py'
REQUIRED_CYTHON_VERSION = (0, 17, 1)
NO_CYTHON_OPTION = '--no-cython'
TUT_DIR = 'tutorial'
TUT_GLOB = 'doc/source/tutorial/*.py'
TUT_HIDDEN_PREFIX = '#HIDDEN'

import sys
import os
import glob
import subprocess
import ConfigParser
from distutils.core import setup, Extension, Command
from distutils.util import get_platform
from distutils.errors import DistutilsError, DistutilsModuleError, \
    CCompilerError
from distutils.command.build import build as distutils_build
from distutils.command.sdist import sdist as distutils_sdist
import numpy

try:
    import Cython
except:
    cython_version = ()
else:
    cython_version = tuple(
        int(n) for n in Cython.__version__.split('-')[0].split('.'))

try:
    sys.argv.remove(NO_CYTHON_OPTION)
    cythonize = False
except ValueError:
    cythonize = True

if cythonize and cython_version:
    from Cython.Distutils import build_ext
else:
    from distutils.command.build_ext import build_ext

distr_root = os.path.dirname(os.path.abspath(__file__))


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
            print >>sys.stderr, \
"""{0}
The compilation of Kwant has failed.  Please examine the error message
above and consult the installation instructions in README.
You might have to customize {1}.
{0}
Build configuration was:
{2}
{0}""".format('*' * 70, CONFIG_FILE, build_summary)
            raise
        print '**************** Build summary ****************'
        print build_summary


class build_tut(Command):
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
class kwant_build(distutils_build):
    sub_commands = [('build_tut', None)] + distutils_build.sub_commands


class test(Command):
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
        print '**************** Tests ****************'
        if not run(argv=[__file__, '-v', lib_dir]):
            raise DistutilsError('at least one of the tests failed')


def git_lsfiles():
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
class kwant_sdist(distutils_sdist):
    sub_commands = [('build', None)] + distutils_sdist.sub_commands

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
                print >>sys.stderr, "error: MANIFEST file is missing and " \
                    "Git is not available to regenerate it."
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
                f.write(STATIC_VERSION_FILE + '\n')
                f.write('MANIFEST\n')

        distutils_sdist.run(self)

        if names is None:
            print >>sys.stderr, \
    """**************** Warning ****************
Git was not available for re-generating the MANIFEST file (the list of file
names to be included in the source distribution).  The old MANIFEST was used."""

        if not trustworthy:
            print >>sys.stderr, \
    """**************** Warning ****************
The existing MANIFEST file seems to have been generated by distutils (it begins
with a comment).  It may well be incomplete."""


# This is an exact copy of the function from kwant/version.py.  We can't import
# it here (because Kwant is not yet built when this scipt is run), so we just
# include a copy.
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
            p = subprocess.Popen(['git', 'describe'] + opts, cwd=distr_root,
                                 stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        except OSError:
            return
        if p.wait() == 0:
            break
    else:
        return
    version = p.communicate()[0].rstrip('\n')

    if version[0] == 'v':
        version = version[1:]

    try:
        p = subprocess.Popen(['git', 'diff', '--quiet'], cwd=distr_root)
    except OSError:
        version += '-confused'  # This should never happen.
    else:
        if p.wait() == 1:
            version += '-dirty'
    return version


def get_static_version():
    """Return the version as recorded inside the source code."""
    try:
        with open(STATIC_VERSION_FILE) as f:
            contents = f.read()
            assert contents[:11] == "version = '"
            assert contents[-2:] == "'\n"
            return contents[11:-2]
    except:
        return None


def version():
    """Determine the version of Kwant.  Return it and save it in a file."""
    git_version = get_version_from_git()
    static_version = get_static_version()
    if git_version is not None:
        version = git_version
        if static_version != git_version:
            with open(STATIC_VERSION_FILE, 'w') as f:
                f.write("version = '%s'\n" % version)
    elif static_version is not None:
        version = static_version
    else:
        version = 'unknown'
    return version


def long_description():
    text = []
    try:
        with open(README_FILE) as f:
            for line in f:
                if line == "\n":
                    break
                text.append(line.rstrip())
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
        msg = "Install Cython {0} or newer so it can be made or use a source " \
            "distribution of Kwant."
        ver = '.'.join(str(e) for e in REQUIRED_CYTHON_VERSION)
        print >>sys.stderr, msg.format(ver)
    else:
        print >>sys.stderr, "Run setup.py without", \
            NO_CYTHON_OPTION


def ext_modules(extensions):
    """Prepare the ext_modules argument for distutils' setup."""
    result = []
    for args, kwrds in extensions:
        if not cythonize or cython_version < REQUIRED_CYTHON_VERSION:
            if 'language' in kwrds:
                if kwrds['language'] == 'c':
                    ext = '.c'
                elif kwrds['language'] == 'c++':
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
                print >>sys.stderr, \
                    "error: Cython-generated file {0} is missing.".format(f)
                complain_cython_unavailable()
                exit(1)
            for f in pyx_files + kwrds.get('depends', []):
                if f == CONFIG_FILE:
                    # The config file is only a dependency for the compilation
                    # of the cythonized file, not for the cythonization.
                    continue
                if os.stat(f).st_mtime > cythonized_oldest:
                    msg = "error: {0} is newer than its source file, but "
                    if cythonize and not cython_version:
                        msg += "Cython is not installed."
                    elif cythonize:
                        msg += "the installed Cython is too old."
                    else:
                        msg += "Cython is not to be run."
                    print >>sys.stderr, msg.format(f)
                    complain_cython_unavailable()
                    exit(1)

        result.append(Extension(*args, **kwrds))

    return result


def main():
    setup(name='kwant',
          version=version(),
          author='C. W. Groth (CEA), M. Wimmer, A. R. Akhmerov, X. Waintal (CEA), and others',
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
                    'build_tut': build_tut,
                    'test': test},
          ext_modules=ext_modules(extensions()),
          include_dirs=[numpy.get_include()])

if __name__ == '__main__':
    main()
