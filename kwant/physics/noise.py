# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np
from ..solvers.common import SMatrix

__all__ = ['two_terminal_shotnoise']


def two_terminal_shotnoise(smatrix):
    r"""Compute the shot-noise in a two-terminal setup.

    In a two terminal system the shot noise is given by `tr((1 - t*t^\dagger) *
    t*t^\dagger)`.

    Parameters
    ----------
    smatrix : `~kwant.solvers.common.SMatrix` instance
        A two terminal scattering matrix.

    Returns
    -------
    noise : float
        Shot noise measured in noise quanta `2 e^3 |V| / pi hbar`.
    """

    if not isinstance(smatrix, SMatrix):
        raise NotImplementedError("Noise expressions in terms of Green's "
                                  "functions are not implemented.")

    if len(smatrix.lead_info) != 2:
        raise ValueError("Only works for two-terminal systems!")

    t = smatrix.submatrix(smatrix.out_leads[0], smatrix.in_leads[0])
    ttdag = np.dot(t, t.conj().T)
    return np.trace(ttdag - np.dot(ttdag, ttdag)).real


# A general multi-terminal routine for noise would need to also have the
# voltages at various leads as input.  (See
# http://arxiv.org/abs/cond-mat/9910158) It could still be based on
# smatrix._a_ttdagger_a_inv, i.e. be also valid also for self-energy leads,
# provided that only true transmission blocks are used.  As long as nobody needs
# it though, it does make little sense to make such a routine.
