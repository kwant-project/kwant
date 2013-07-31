# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import numpy as np

def two_terminal_shotnoise(smatrix):
    """Compute the shot-noise in a two-terminal setup.
    [Given by tr((1 - t*t^\dagger) * t*t^\dagger)]."""

    if len(smatrix.lead_info) != 2:
        raise ValueError("Only works for two-terminal systems!")

    if 1 in smatrix.out_leads and 0 in smatrix.in_leads:
        ttdag = smatrix._a_ttdagger_a_inv(1, 0)
    if 0 in smatrix.out_leads and 1 in smatrix.in_leads:
        ttdag = smatrix._a_ttdagger_a_inv(0, 1)
    else:
        raise ValueError("Need S-matrix block for transmission!")

    ttdag -= np.dot(ttdag, ttdag)
    return np.trace(ttdag).real


# A general multi-terminal routine for noise would need to also have the
# voltages at various leads as input.  (See
# http://arxiv.org/abs/cond-mat/9910158) It could still be based on
# smatrix._a_ttdagger_a_inv, i.e. be also valid also for self-energy leads,
# provided that only true transmission blocks are used As long as nobody needs
# it though, it does make little sense to make such a routine.
