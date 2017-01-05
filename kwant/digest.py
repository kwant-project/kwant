# Copyright 2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

"""Random-access random numbers

This module provides routines that given some input compute a "random" output
that depends on the input in a (cryptographically) intractable way.

This turns out to be very useful when one needs to map some irregular objects to
random numbers in a deterministic and reproducible way.

Internally, the md5 hash algorithm is used.  The randomness thus generated is
good enough to pass the "dieharder" battery of tests: see the function `test` of
this module.
"""

from math import pi, log, sqrt, cos
from hashlib import md5
from struct import unpack

__all__ = ['uniform', 'gauss', 'test']


TWOPI = 2 * pi
BPF = 53                        # Number of bits in a Python float.
BPF_MASK = 2**53 - 1
RECIP_BPF = 2**-BPF


def str_to_bytes(s):
    """Return bytes if the input is a string, else return the object as-is."""
    try:
        return s.encode('utf8')
    except AttributeError:
        return s

def uniform2(input, salt=''):
    """Return two independent [0,1)-distributed numbers."""
    input = str_to_bytes(input)
    salt = str_to_bytes(salt)
    input = memoryview(input).tobytes() + salt
    a, b = unpack('qq', md5(input).digest())
    a &= BPF_MASK
    b &= BPF_MASK
    return a * RECIP_BPF, b * RECIP_BPF


def uniform(input, salt=''):
    """md5-hash `input` and `salt` and map the result to the [0,1) interval.

    `input` must be some object that supports the buffer protocol (i.e. a string
    or a numpy/tinyarray array).  `salt` must be a string or a bytes object.
    """
    return uniform2(input, salt)[0]


def gauss(input, salt=''):
    """md5-hash `input` and `salt` and return the result as a standard normal
    distributed variable.

    `input` must be some object that supports the buffer protocol (i.e. a string
    or a numpy/tinyarray array).  `salt` must be a string or a bytes object.
    """
    # This uses the Box-Muller transform.  Only one of the two results is
    # computed.
    a, b = uniform2(input, salt)
    return cos(a * TWOPI) * sqrt(-2.0 * log(1.0 - b))


def test(n=20000):  # skip coverage
    """Test the generator with the dieharder suite generating n**2 samples.

    Executing this function may take a very long time.
    """
    import os
    import tempfile
    import subprocess
    from tinyarray import array
    from struct import pack

    f = tempfile.NamedTemporaryFile(delete=False)
    try:
        for x in range(n):
            for y in range(n):
                a = array((x, y))
                i = int(2**32 * uniform(a))
                f.write(pack('I', i))
        f.close()
        subprocess.call(['dieharder', '-a', '-g', '201', '-f', f.name])
    finally:
        os.remove(f.name)
