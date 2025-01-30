#!/usr/bin/env python3

# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.

from setuptools import setup, Extension
import os


def add_numpy_include(exts):
    # Add NumPy header path to include_dirs of all the extensions.
    import numpy

    numpy_include = numpy.get_include()
    for ext in exts.values():
        ext.setdefault("include_dirs", []).append(numpy_include)
    return exts


exts = {
    "kwant._system": dict(
        sources=["src/kwant/_system.pyx"], include_dirs=["src/kwant/graph"]
    ),
    "kwant.operator": dict(
        sources=["src/kwant/operator.pyx"], include_dirs=["src/kwant/graph"]
    ),
    "kwant.graph.core": dict(
        sources=["src/kwant/graph/core.pyx"],
        depends=[
            "src/kwant/graph/core.pxd",
            "src/kwant/graph/defs.h",
            "src/kwant/graph/defs.pxd",
        ],
    ),
    "kwant.graph.dijkstra": dict(sources=["src/kwant/graph/dijkstra.pyx"]),
    "kwant.linalg.lapack": dict(sources=["src/kwant/linalg/lapack.pyx"]),
}

if "MUMPS_LIBS" in os.environ:
    exts["kwant.linalg._mumps"] = dict(
        sources=["src/kwant/linalg/_mumps.pyx"],
        depends=["src/kwant/linalg/cmumps.pxd"],
        libraries=os.environ["MUMPS_LIBS"].split(","),
    )

# Stop relying on numpy deprecated API.
for ext in exts.values():
    ext["define_macros"] = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]


setup(
    ext_modules=[
        Extension(name, **kwargs) for name, kwargs in add_numpy_include(exts).items()
    ],
)
