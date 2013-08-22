# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import tempfile
import warnings
import nose
import kwant
from kwant import plotter
if plotter.mpl_enabled:
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot


def test_importable_without_matplotlib():
    prefix, sep, suffix = plotter.__file__.rpartition('.')
    if suffix == 'pyc':
        suffix = 'py'
    assert suffix == 'py'
    fname = sep.join((prefix, suffix))
    with open(fname) as f:
        code = f.read()
    code = code.replace('from . import', 'from kwant import')
    code = code.replace('matplotlib', 'totalblimp')
    exec code


def sys_2d(W=3, r1=3, r2=8):
    a = 1
    t = 1.0
    lat = kwant.lattice.square(a)
    sys = kwant.Builder()

    def ring(pos):
        (x, y) = pos
        rsq = x ** 2 + y ** 2
        return r1 ** 2 < rsq < r2 ** 2

    sys[lat.shape(ring, (0, r1 + 1))] = 4 * t
    sys[lat.neighbors()] = -t
    sym_lead0 = kwant.TranslationalSymmetry(lat.vec((-1, 0)))
    lead0 = kwant.Builder(sym_lead0)
    lead2 = kwant.Builder(sym_lead0)

    lead_shape = lambda pos: (-W / 2 < pos[1] < W / 2)

    lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
    lead2[lat.shape(lead_shape, (0, 0))] = 4 * t
    sys.attach_lead(lead2)
    lead0[lat.neighbors()] = - t
    lead1 = lead0.reversed()
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)
    return sys


def sys_3d(W=3, r1=2, r2=4, a=1, t=1.0):
    lat = kwant.lattice.general(((a, 0, 0), (0, a, 0), (0, 0, a)))
    sys = kwant.Builder()

    def ring(pos):
        (x, y, z) = pos
        rsq = x ** 2 + y ** 2
        return (r1 ** 2 < rsq < r2 ** 2) and abs(z) < 2
    sys[lat.shape(ring, (0, -r2 + 1, 0))] = 4 * t
    sys[lat.neighbors()] = - t
    sym_lead0 = kwant.TranslationalSymmetry(lat.vec((-1, 0, 0)))
    lead0 = kwant.Builder(sym_lead0)

    lead_shape = lambda pos: (-W / 2 < pos[1] < W / 2) and abs(pos[2]) < 2

    lead0[lat.shape(lead_shape, (0, 0, 0))] = 4 * t
    lead0[lat.neighbors()] = - t
    lead1 = lead0.reversed()
    sys.attach_lead(lead0)
    sys.attach_lead(lead1)
    return sys


def test_plot():
    plot = plotter.plot
    if not plotter.mpl_enabled:
        raise nose.SkipTest
    sys2d = sys_2d()
    sys3d = sys_3d()
    color_opts = ['k', (lambda site: site.tag[0]),
                  lambda site: (abs(site.tag[0] / 100),
                                abs(site.tag[1] / 100), 0)]
    for color in color_opts:
        for sys in (sys2d, sys3d):
            fig = plot(sys, site_color=color, cmap='binary', show=False)
            if color != 'k' and isinstance(color(iter(sys2d.sites()).next()),
                                           float):
                assert fig.axes[0].collections[0].get_array() is not None
            assert len(fig.axes[0].collections) == (8 if sys is sys2d else 6)
    color_opts = ['k', (lambda site, site2: site.tag[0]),
                  lambda site, site2: (abs(site.tag[0] / 100),
                                       abs(site.tag[1] / 100), 0)]
    for color in color_opts:
        for sys in (sys2d, sys3d):
            fig = plot(sys2d, hop_color=color, cmap='binary', show=False,
                       fig_size=(2, 10), dpi=30)
            if color != 'k' and isinstance(color(iter(sys2d.sites()).next(),
                                                      None), float):
                assert fig.axes[0].collections[1].get_array() is not None

    assert isinstance(plot(sys3d, show=False).axes[0], mplot3d.axes3d.Axes3D)

    sys2d.leads = []
    plot(sys2d, show=False)
    del sys2d[list(sys2d.hoppings())]
    plot(sys2d, show=False)
    with tempfile.TemporaryFile('w+b') as output:
        plot(sys3d, file=output)
        warnings.simplefilter('ignore')
        plot(sys2d.finalized(), file=output)
        warnings.simplefilter('once')


def test_map():
    sys = sys_2d()
    with tempfile.TemporaryFile('w+b') as output:
        plotter.map(sys, lambda site: site.tag[0], file=output,
                          method='linear', a=4, oversampling=4, cmap='flag')
        warnings.simplefilter('ignore')
        plotter.map(sys.finalized(), xrange(len(sys.sites())),
                          file=output)
        warnings.simplefilter('once')
        nose.tools.assert_raises(ValueError, plotter.map,
                                 sys, xrange(len(sys.sites())), file=output)
