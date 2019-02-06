# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.
"""Pytest plugin to ignore packages that have uninstalled dependencies.

This ignores packages on test collection, which is required when the
tests reside in a package that itself requires the dependency to be
installed.
"""

import importlib


# map from subpackage to sequence of dependency module names
subpackage_dependencies = {
    'kwant/continuum': ['sympy'],
    'kwant/tests/test_qsymm': ['qsymm', 'sympy'],
}


# map from subpackage to sequence of dependency modules that are not installed
dependencies_not_installed = {}
for package, dependencies in subpackage_dependencies.items():
    not_installed = []
    for dep in dependencies:
        try:
            importlib.import_module(dep)
        except ImportError:
            not_installed.append(dep)
    if len(not_installed) != 0:
        dependencies_not_installed[package] = not_installed


def pytest_ignore_collect(path, config):
    for subpackage, not_installed in dependencies_not_installed.items():
        if subpackage in path.strpath:
            print('ignoring {} because the following dependencies are not '
                  'installed: {}'.format(subpackage, ', '.join(not_installed)))
            return True
