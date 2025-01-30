#!/usr/bin/env python3

# Copyright 2011-2018 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# https://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# https://kwant-project.org/authors.

import os

from setuptools import setup, Extension
import numpy

exts = [
    Extension(
        "kwant._system", ["src/kwant/_system.pyx"], include_dirs=["src/kwant/graph"]
    ),
    Extension(
        "kwant.operator", ["src/kwant/operator.pyx"], include_dirs=["src/kwant/graph"]
    ),
    Extension(
        "kwant.graph.core",
        ["src/kwant/graph/core.pyx"],
        depends=[
            "src/kwant/graph/core.pxd",
            "src/kwant/graph/defs.h",
            "src/kwant/graph/defs.pxd",
        ],
    ),
    Extension("kwant.graph.dijkstra", ["src/kwant/graph/dijkstra.pyx"]),
    Extension("kwant.linalg.lapack", ["src/kwant/linalg/lapack.pyx"]),
]

if "MUMPS_LIBS" in os.environ:
    exts.append(
        Extension(
            "kwant.linalg._mumps",
            ["src/kwant/linalg/_mumps.pyx"],
            depends=["src/kwant/linalg/cmumps.pxd"],
            libraries=os.environ["MUMPS_LIBS"].split(","),
        )
    )

for ext in exts:
    ext.define_macros = [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]
    ext.include_dirs.append(numpy.get_include())

setup(ext_modules=exts)
