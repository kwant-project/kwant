#!/usr/bin/env python3

# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.

from __future__ import print_function

import sys

import os
import importlib
import subprocess
import configparser
import argparse
from pathlib import Path
from ctypes.util import find_library

from setuptools import setup, find_packages, Extension
from setuptools.errors import CCompilerError, BaseError as DistutilsError
from setuptools.command.sdist import sdist as sdist_orig
from setuptools.command.build_ext import build_ext as build_ext_orig
# Packaging is a dependency of setuptools, so we are free to use it.
from packaging.version import Version, parse as parse_version
import setuptools

if parse_version(setuptools.__version__) < Version("63.0"):
    # TODO: remove this once we depend on setuptools >= 63.0
    os.environ['SETUPTOOLS_USE_DISTUTILS'] = 'local'
    from distutils.command.build import build as build_orig
else:
    from setuptools.command.build import build as build_orig


STATIC_VERSION_PATH = 'kwant/_kwant_version.py'

distr_root = Path(__file__).resolve().parent


def configure_extensions(exts, aliases=(), build_summary=None):
    """Modify extension configuration according to the configuration file

    `exts` must be a dict of (name, kwargs) tuples that can be used like this:
    `Extension(name, **kwargs).  This function modifies the kwargs according to
    the configuration file.

    This function removes `--configfile` from `sys.argv`.
    """
    global config_file, config_file_present

    #### Determine the name of the configuration file.
    default_config_file = Path('build.conf')
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument('--configfile', type=argparse.FileType('r'))
    known, unknown = parser.parse_known_args()
    config_file = known.configfile
    sys.argv = [sys.argv[0]] + unknown
    if config_file is None and default_config_file.is_file():
        config_file = default_config_file.open()

    #### Read build configuration file.
    configs = configparser.ConfigParser()
    if (config_file_present := config_file is not None):
        configs.read_file(config_file)

    config_file = config_file.name if config_file is not None else 'build.conf'

    #### Handle section aliases.
    for short, long in aliases:
        if short in configs:
            if long in configs:
                sys.exit(
                    f'Error: both {short} and {long} '
                    f'sections present in {config_file}.'
                )
            configs[long] = configs[short]
            del configs[short]

    #### Apply config from file.  Use [DEFAULT] section for missing sections.
    defaultconfig = configs.defaults()
    for name, kwargs in exts.items():
        config = configs[name] if name in configs else defaultconfig
        for key, value in config.items():

            # Most, but not all, keys are lists of strings
            if key == 'language':
                pass
            elif key == 'optional':
                value = bool(int(value))
            else:
                value = value.split()

            if key == 'define_macros':
                value = [tuple(entry.split('=', maxsplit=1))
                         for entry in value]
                value = [(entry[0], None) if len(entry) == 1 else entry
                         for entry in value]

            if key in kwargs:
                msg = 'Caution: user config in file {} shadows {}.{}.'
                if build_summary is not None:
                    build_summary.append(msg.format(config_file, name, key))
            kwargs[key] = value

        kwargs.setdefault('depends', []).append(config_file)
        if config is not defaultconfig:
            del configs[name]

    if (unknown_sections := ", ".join(configs.sections())):
        sys.exit(f"Unknown sections in {config_file}: {unknown_sections}")

    return exts


def check_versions():
    global version, version_is_from_git

    # Let Kwant itself determine its own version.  We cannot simply import
    # kwant, as it is not built yet.
    spec = importlib.util.spec_from_file_location('version', 'kwant/version.py')
    version_module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(version_module)

    version_module.ensure_python()
    version = version_module.version
    version_is_from_git = version_module.version_is_from_git


def init_cython():
    """Set the global variable `cythonize` (and other related globals).

    The variable `cythonize` can be in three states:

    * If Cython should be run and is ready, it contains the `cythonize()`
      function.

    * If Cython is not to be run, it contains `False`.

    * If Cython should, but cannot be run it contains `None`.  A help message
      on how to solve the problem is stored in `cython_help`.

    This function removes `--cython` from `sys.argv`.
    """
    global cythonize, cython_help

    cython_option = '--cython'
    required_cython_version = Version("3.0")
    try:
        sys.argv.remove(cython_option)
        cythonize = True
    except ValueError:
        cythonize = version_is_from_git

    if cythonize:
        try:
            import Cython
            from Cython.Build import cythonize
        except ImportError:
            cythonize = None
        else:
            if parse_version(Cython.__version__) < required_cython_version:
                cythonize = None

        if cythonize is None:
            cython_help = (f"Install Cython >= {required_cython_version} or use"
                    " a source distribution (tarball) of Kwant.")
    else:
        cython_help = f"Run setup.py with the {cython_option} option to enable Cython."


def banner(title=''):
    starred = title.center(79, '*')
    return '\n' + starred if title else starred


class build_ext(build_ext_orig):
    def run(self):
        if not config_file_present:
            # Create an empty config file if none is present so that the
            # extensions will not be rebuilt each time.  Only depending on the
            # config file if it is present would make it impossible to detect a
            # necessary rebuild due to a deleted config file.
            with open(config_file, 'w') as f:
                f.write('# Build configuration created by setup.py '
                        '- feel free to modify.\n')

        try:
            super().run()
        except (DistutilsError, CCompilerError):
            error_msg = self.__error_msg.format(
                header=banner(' Error '), sep=banner())
            print(error_msg.format(file=config_file, summary=build_summary),
                  file=sys.stderr)
            raise
        print(banner(' Build summary '), *build_summary, sep='\n')
        print(banner())

    __error_msg = """{header}
The compilation of Kwant has failed.  Please examine the error message
above and consult the installation instructions in README.rst.
You might have to customize {{file}}.

Build configuration was:

{{summary}}
{sep}
"""


class build(build_orig):
    def run(self):
        super().run()
        write_version(Path(self.build_lib) / STATIC_VERSION_PATH)


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
    return p.communicate()[0].decode().split('\n')[:-1]


# Make the command "sdist" depend on "build".  This verifies that the
# distribution in the current state actually builds.  It also makes sure that
# the Cython-made C files will be up-to-date and included in the source.
class sdist(sdist_orig):
    sub_commands = [('build', None)] + sdist_orig.sub_commands

    def run(self):
        """Create MANIFEST.in from git if possible, otherwise check that
        MANIFEST.in is present.

        Right now (2015) generating MANIFEST.in seems to be the only way to
        include files in the source distribution that setuptools does not think
        should be there.  Setting include_package_data to True makes setuptools
        include *.pyx and other source files in the binary distribution.
        """
        manifest_in_file = 'MANIFEST.in'
        manifest = distr_root / manifest_in_file
        names = git_lsfiles()
        if names is None:
            if not (manifest.is_file() and os.access(manifest, os.R_OK)):
                sys.exit(f"Error: {manifest_in_file}"
                         "file is missing and Git is not available"
                         " to regenerate it.")
        else:
            with open(manifest, 'w') as f:
                for name in names:
                    a, sep, b = name.rpartition('/')
                    if b == '.gitignore':
                        continue
                    stem, dot, extension = b.rpartition('.')
                    f.write('include {}'.format(name))
                    if extension == 'pyx':
                        f.write(''.join([' ', a, sep, stem, dot, 'c']))
                    f.write('\n')

        super().run()

        if names is None:
            msg = ("Git was not available to generate the list of files to be "
                   "included in the\nsource distribution. The old {} was used.")
            msg = msg.format(manifest_in_file)
            print(banner(' Caution '), msg, banner(), sep='\n', file=sys.stderr)

    def make_release_tree(self, base_dir, files):
        super().make_release_tree(base_dir, files)
        write_version(Path(base_dir) / STATIC_VERSION_PATH)


def write_version(path):
    # This could be a hard link, so try to delete it first.  Is there any way
    # to do this atomically together with opening?
    path.unlink(missing_ok=True)
    path.write_text(
        f"# This file has been created by setup.py.\n{version = }\n"
    )


def long_description():
    source = Path('README.rst')
    if not source.is_file():
        return ''
    text = source.read_text()
    return text[:text.find('See also in this directory:')]


def search_mumps():
    """Return the configuration for MUMPS if it is available in a known way.

    This is known to work with the MUMPS provided by the Debian package
    libmumps-scotch-dev and the MUMPS binaries in the conda-forge channel."""
    lib_sets = [
        # Debian
        ['zmumps_scotch', 'mumps_common_scotch', 'mpiseq_scotch',
         'pord', 'gfortran'],
        # Conda (via conda-forge).
        ['zmumps_seq', 'mumps_common_seq'],
    ]
    return next(
        (libs for libs in lib_sets if all(find_library(lib) for lib in libs)),
        []
    )


def configure_special_extensions(exts, build_summary):
    #### Special config for MUMPS.
    mumps = exts['kwant.linalg._mumps']
    if 'libraries' in mumps:
        build_summary.append('User-configured MUMPS')
    else:
        mumps_libs = search_mumps()
        if mumps_libs:
            mumps['libraries'] = mumps_libs
            build_summary.append('Auto-configured MUMPS')
        else:
            mumps = None
            del exts['kwant.linalg._mumps']
            build_summary.append('No MUMPS support')

    return exts


def maybe_cythonize(exts):
    """Prepare a list of `Extension` instances, ready for `setup()`.

    The argument `exts` must be a mapping of names to kwargs to be passed
    on to `Extension`.

    If Cython is to be run, create the extensions and calls `cythonize()` on
    them.  If Cython is not to be run, replace .pyx file with .c or .cpp,
    check timestamps, and create the extensions.
    """
    if cythonize:
        return cythonize([Extension(name, **kwargs)
                          for name, kwargs in exts.items()],
                         language_level=3,
                         compiler_directives={'linetrace': True})

    # Cython is not going to be run: replace pyx extension by that of
    # the shipped translated file.

    result = []
    problematic_files = []
    for name, kwargs in exts.items():
        language = kwargs.get('language')
        if language is None:
            ext = '.c'
        elif language == 'c':
            ext = '.c'
        elif language == 'c++':
            ext = '.cpp'
        else:
            sys.exit(f"Unknown language: {language}")

        pyx_files = []
        cythonized_files = []
        sources = []
        for f in kwargs['sources']:
            if f.endswith('.pyx'):
                pyx_files.append(f)
                f = f.rstrip('.pyx') + ext
                cythonized_files.append(f)
            sources.append(f)
        kwargs['sources'] = sources

        # Complain if cythonized files are older than Cython source files.
        try:
            cythonized_oldest = min(os.stat(f).st_mtime
                                    for f in cythonized_files)
        except OSError:
            sys.exit("\n".join([
                banner(" Error "),
                f"Cython-generated file {f} is missing.",
                "",
                cython_help,
                banner(),
            ]))

        for f in pyx_files + kwargs.get('depends', []):
            if f == config_file:
                # The config file is only a dependency for the compilation
                # of the cythonized file, not for the cythonization.
                continue
            if os.stat(f).st_mtime > cythonized_oldest:
                problematic_files.append(f)

        result.append(Extension(name, **kwargs))

    if problematic_files:
        msg = ("Some Cython source files are newer than files that have "
               "been derived from them:\n{}")
        msg = msg.format(", ".join(problematic_files))

        # Cython should be run but won't.  Signal an error if this is because
        # Cython *cannot* be run, warn otherwise.
        error = cythonize is None
        if cythonize is False:
            dontworry = ('(Do not worry about this if you are building Kwant '
                         'from unmodified sources,\n'
                         'e.g. with "pip install".)\n\n')
            msg = dontworry + msg

        print(banner(" Error " if error else " Caution "), msg, "",
              cython_help, banner(), sep="\n", file=sys.stderr)
        if error:
            sys.exit(1)

    return result


def maybe_add_numpy_include(exts):
    # Add NumPy header path to include_dirs of all the extensions.
    try:
        import numpy
    except ImportError:
        print(banner(' Caution '), 'NumPy header directory cannot be determined'
              ' ("import numpy" failed).', banner(), sep='\n', file=sys.stderr)
    else:
        numpy_include = numpy.get_include()
        for ext in exts.values():
            ext.setdefault('include_dirs', []).append(numpy_include)
    return exts


def main():
    check_versions()

    exts = dict([
        ('kwant._system',
         dict(sources=['kwant/_system.pyx'],
              include_dirs=['kwant/graph'])),
        ('kwant.operator',
         dict(sources=['kwant/operator.pyx'],
              include_dirs=['kwant/graph'])),
        ('kwant.graph.core',
         dict(sources=['kwant/graph/core.pyx'],
              depends=['kwant/graph/core.pxd', 'kwant/graph/defs.h',
                       'kwant/graph/defs.pxd'])),
        ('kwant.graph.dijkstra',
         dict(sources=['kwant/graph/dijkstra.pyx'])),
        ('kwant.linalg.lapack',
         dict(sources=['kwant/linalg/lapack.pyx'])),
        ('kwant.linalg._mumps',
         dict(sources=['kwant/linalg/_mumps.pyx'],
              depends=['kwant/linalg/cmumps.pxd']))])
    # Stop relying on numpy deprecated API.
    for ext in exts.values():
        ext['define_macros'] = [
            ('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')
        ]

    aliases = [('mumps', 'kwant.linalg._mumps')]

    init_cython()

    global build_summary
    build_summary = []
    exts = configure_extensions(exts, aliases, build_summary)
    exts = configure_special_extensions(exts, build_summary)
    exts = maybe_add_numpy_include(exts)
    exts = maybe_cythonize(exts)

    classifiers = """\
        Development Status :: 5 - Production/Stable
        Intended Audience :: Science/Research
        Intended Audience :: Developers
        Programming Language :: Python :: 3 :: Only
        Topic :: Software Development
        Topic :: Scientific/Engineering
        Operating System :: POSIX
        Operating System :: Unix
        Operating System :: MacOS :: MacOS X
        Operating System :: Microsoft :: Windows"""

    packages = find_packages('.')
    setup(name='kwant',
          version=version,
          author='C. W. Groth (CEA), M. Wimmer, '
                 'A. R. Akhmerov, X. Waintal (CEA), and others',
          author_email='authors@kwant-project.org',
          description=("Package for numerical quantum transport calculations "
                       "(Python 3 version)"),
          long_description=long_description(),
          platforms=["Unix", "Linux", "Mac OS-X", "Windows"],
          url="https://kwant-project.org/",
          license="BSD",
          packages=packages,
          package_data={p: ['*.pxd', '*.h'] for p in packages},
          cmdclass={'build': build,
                    'sdist': sdist,
                    'build_ext': build_ext},
          ext_modules=exts,
          install_requires=['numpy >= 1.18.0', 'scipy >= 1.3.0',
                            'tinyarray >= 1.2.2'],
          extras_require={
              # The oldest versions between: Debian stable, Ubuntu LTS
              'plotting': 'matplotlib >= 3.2.2',
              'continuum': 'sympy >= 1.5.1',
              # qsymm is only packaged on PyPI
              'qsymm': 'qsymm >= 1.3.0',
          },
          python_requires='>=3.8',
          classifiers=[c.strip() for c in classifiers.split('\n')])

if __name__ == '__main__':
    main()
