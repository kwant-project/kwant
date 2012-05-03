import tempfile, os
from nose.tools import assert_raises
import numpy as np
import kwant
from kwant import plotter

lat = kwant.lattice.Square()

def make_ribbon(width, dir, E, t):
    b = kwant.Builder(kwant.TranslationalSymmetry([(dir, 0)]))

    # Add sites to the builder.
    for y in xrange(width):
        b[lat(0, y)] = E

    # Add hoppings to the builder.
    for y in xrange(width):
        b[lat(0, y), lat(1, y)] = t
        if y+1 < width:
            b[lat(0, y), lat(0, y+1)] = t

    return b


def make_rectangle(length, width, E, t):
    b = kwant.Builder()

    # Add sites to the builder.
    for x in xrange(length):
        for y in xrange(width):
            b[lat(x, y)] = E

    # Add hoppings to the builder.
    for x in xrange(length):
        for y in xrange(width):
            if x+1 < length:
                b[lat(x, y), lat(x+1, y)] = t
            if y+1 < width:
                b[lat(x, y), lat(x, y+1)] = t

    return b

def test_plot():
    E = 4.0
    t = -1.0
    length = 5
    width = 5

    b = make_rectangle(length, width, E, t)
    b.attach_lead(make_ribbon(width, -1, E, t))
    b.attach_lead(make_ribbon(width, 1, E, t))

    directory = tempfile.mkdtemp()
    filename = os.path.join(directory, "test.pdf")

    kwant.plot(b.finalized(), filename=filename,
               symbols=plotter.Circle(r=0.25, fcol=plotter.red),
               lines=plotter.Line(lw=0.1, lcol=plotter.red),
               lead_symbols=plotter.Circle(r=0.25, fcol=plotter.black),
               lead_lines=plotter.Line(lw=0.1, lcol=plotter.black),
               lead_fading=[0, 0.2, 0.4, 0.6, 0.8])

    os.unlink(filename)
    os.rmdir(directory)

def test_non_2d_fails():
    directory = tempfile.mkdtemp()
    filename = os.path.join(directory, "test.pdf")

    for d in [1, 2, 3, 15]:
        b = kwant.Builder()
        lat = kwant.make_lattice(np.identity(d))
        site = kwant.builder.Site(lat, (0,) * d)
        b[site] = 0
        if d == 2:
            kwant.plot(b, filename=filename)
            plotter.interpolate(b, b)
        else:
            assert_raises(ValueError, kwant.plot, b)
            assert_raises(ValueError, plotter.interpolate, b, b)

    os.unlink(filename)
    os.rmdir(directory)
