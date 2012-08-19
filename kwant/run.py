"""Support for running functions from the command line"""

from __future__ import division
import os
import struct
import sys
import numpy
import scipy
from .version import version
__all__ = ['randomize', 'exec_argv']

numpy_version = numpy.version.version
if not numpy.version.release:
    numpy_version += '-non-release'

scipy_version = scipy.version.version
if not scipy.version.release:
    scipy_version += '-non-release'


def randomize():
    """Seed numpy's random generator according to RNG_SEED environment
    variable.

    If RNG_SEED is undefined or has the value "random", the seed is read from
    /dev/urandom.  Otherwise, the value of RNG_SEED (which may be the decimal
    representation of an 8-byte signed integer) is used as seed.

    """
    try:
        seed = os.environ['RNG_SEED']
    except KeyError:
        seed = 'random'
    if seed == 'random':
        f = open('/dev/urandom')
        seed = struct.unpack('Q', f.read(8))[0]
        f.close()
    else:
        seed = int(seed)

    # numpy.random.seed only uses the lower 32 bits of the seed argument, so we
    # split our 64 bit seed into two 32 bit numbers.
    assert seed >= 0 and seed < 1 << 64
    seed_lo = int((seed & 0xffffffff) - (1 << 31))
    seed_hi = int((seed >> 32) - (1 << 31))
    numpy.random.seed((seed_lo, seed_hi))

    return seed


def exec_argv(vars):
    """Execute command line arguments as python statements.

    First, the versions of kwant, scipy and numpy are reported on stdout.
    numpy's random number generator is initialized by `run.randomize()` and the
    seed reported.

    Then each command line argument is executed as a python statement within
    the environment specified by `vars`.  Most of the time `vars` should be set
    to the return value of `globals()`.

    """

    if len(sys.argv) == 1:
        help('__main__')
        return

    seed = randomize()
    print '#### kwant %s, scipy %s, numpy %s' % \
        (version, scipy_version, numpy_version)
    print "#### numpy random seed: %d" % seed
    for statement in sys.argv[1:]:
        print "## %s" % statement
        exec statement in vars
