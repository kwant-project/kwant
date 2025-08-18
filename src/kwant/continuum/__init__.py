# Copyright 2011-2017 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.

try:
    from .discretizer import discretize, discretize_symbolic, build_discretized
    from ._common import sympify, lambdify
    from ._common import momentum_operators, position_operators
    from .landau_levels import to_landau_basis, discretize_landau, LandauLattice
except ImportError:
    # Optional dependency (sympy) is not available. Make the package importable
    # so test collection can run (tests use pytest.importorskip to skip when
    # sympy is missing). Accessing names will raise an informative ImportError.
    __all__ = []

    def __getattr__(name):
        # Defer the import error until attribute access so pytest.importorskip
        # in test modules can execute during collection.
        raise ImportError(
            "'kwant.continuum' is not available because one or more of its "
            "dependencies (for example 'sympy') is not installed.")

    def __dir__():
        return []
else:
    __all__ = ['discretize', 'discretize_symbolic', 'build_discretized',
               'to_landau_basis', 'discretize_landau', 'LandauLattice',
               'sympify', 'lambdify', 'momentum_operators', 'position_operators']
