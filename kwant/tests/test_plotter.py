# Copyright 2011-2013 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

import tempfile
import warnings
import numpy as np
import kwant
from kwant import plotter
import pytest

if plotter.mpl_enabled:
    from mpl_toolkits import mplot3d
    from matplotlib import pyplot  # pragma: no flakes


def test_importable_without_matplotlib():
    prefix, sep, suffix = plotter.__file__.rpartition('.')
    if suffix in ['pyc', 'pyo']:
        suffix = 'py'
    assert suffix == 'py'
    fname = sep.join((prefix, suffix))
    with open(fname) as f:
        code = f.read()
    code = code.replace('from . import', 'from kwant import')
    code = code.replace('matplotlib', 'totalblimp')

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        exec(code)               # Trigger the warning.
        assert len(w) == 1
        assert issubclass(w[0].category, RuntimeWarning)
        assert "only iterator-providing functions" in str(w[0].message)


def syst_2d(W=3, r1=3, r2=8):
    a = 1
    t = 1.0
    lat = kwant.lattice.square(a)
    syst = kwant.Builder()

    def ring(pos):
        (x, y) = pos
        rsq = x ** 2 + y ** 2
        return r1 ** 2 < rsq < r2 ** 2

    syst[lat.shape(ring, (0, r1 + 1))] = 4 * t
    syst[lat.neighbors()] = -t
    sym_lead0 = kwant.TranslationalSymmetry(lat.vec((-1, 0)))
    lead0 = kwant.Builder(sym_lead0)
    lead2 = kwant.Builder(sym_lead0)

    lead_shape = lambda pos: (-W / 2 < pos[1] < W / 2)

    lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
    lead2[lat.shape(lead_shape, (0, 0))] = 4 * t
    syst.attach_lead(lead2)
    lead0[lat.neighbors()] = - t
    lead1 = lead0.reversed()
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    return syst


def syst_3d(W=3, r1=2, r2=4, a=1, t=1.0):
    lat = kwant.lattice.general(((a, 0, 0), (0, a, 0), (0, 0, a)))
    syst = kwant.Builder()

    def ring(pos):
        (x, y, z) = pos
        rsq = x ** 2 + y ** 2
        return (r1 ** 2 < rsq < r2 ** 2) and abs(z) < 2
    syst[lat.shape(ring, (0, -r2 + 1, 0))] = 4 * t
    syst[lat.neighbors()] = - t
    sym_lead0 = kwant.TranslationalSymmetry(lat.vec((-1, 0, 0)))
    lead0 = kwant.Builder(sym_lead0)

    lead_shape = lambda pos: (-W / 2 < pos[1] < W / 2) and abs(pos[2]) < 2

    lead0[lat.shape(lead_shape, (0, 0, 0))] = 4 * t
    lead0[lat.neighbors()] = - t
    lead1 = lead0.reversed()
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)
    return syst


@pytest.mark.skipif(not plotter.mpl_enabled, reason="No matplotlib available.")
def test_plot():
    plot = plotter.plot
    syst2d = syst_2d()
    syst3d = syst_3d()
    color_opts = ['k', (lambda site: site.tag[0]),
                  lambda site: (abs(site.tag[0] / 100),
                                abs(site.tag[1] / 100), 0)]
    with tempfile.TemporaryFile('w+b') as out:
        for color in color_opts:
            for syst in (syst2d, syst3d):
                fig = plot(syst, site_color=color, cmap='binary', file=out)
                if (color != 'k' and
                    isinstance(color(next(iter(syst2d.sites()))), float)):
                    assert fig.axes[0].collections[0].get_array() is not None
                assert len(fig.axes[0].collections) == (8 if syst is syst2d else
                                                        6)
        color_opts = ['k', (lambda site, site2: site.tag[0]),
                      lambda site, site2: (abs(site.tag[0] / 100),
                                           abs(site.tag[1] / 100), 0)]
        for color in color_opts:
            for syst in (syst2d, syst3d):
                fig = plot(syst2d, hop_color=color, cmap='binary', file=out,
                           fig_size=(2, 10), dpi=30)
                if color != 'k' and isinstance(color(next(iter(syst2d.sites())),
                                                          None), float):
                    assert fig.axes[0].collections[1].get_array() is not None

        assert isinstance(plot(syst3d, file=out).axes[0], mplot3d.axes3d.Axes3D)

        syst2d.leads = []
        plot(syst2d, file=out)
        del syst2d[list(syst2d.hoppings())]
        plot(syst2d, file=out)

        plot(syst3d, file=out)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plot(syst2d.finalized(), file=out)

def good_transform(pos):
    x, y = pos
    return y, x

def bad_transform(pos):
    x, y = pos
    return x, y, 0

@pytest.mark.skipif(not plotter.mpl_enabled, reason="No matplotlib available.")
def test_map():
    syst = syst_2d()
    with tempfile.TemporaryFile('w+b') as out:
        plotter.map(syst, lambda site: site.tag[0], pos_transform=good_transform,
                    file=out, method='linear', a=4, oversampling=4, cmap='flag')
        pytest.raises(ValueError, plotter.map, syst,
                      lambda site: site.tag[0],
                      pos_transform=bad_transform, file=out)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plotter.map(syst.finalized(), range(len(syst.sites())),
                        file=out)
        pytest.raises(ValueError, plotter.map, syst,
                      range(len(syst.sites())), file=out)


def test_mask_interpolate():
    # A coordinate array with coordinates of two points almost coinciding.
    coords = np.array([[0, 0], [1e-7, 1e-7], [1, 1], [1, 0]])

    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        plotter.mask_interpolate(coords, np.ones(len(coords)), a=1)
        assert len(w) == 1
        assert issubclass(w[-1].category, RuntimeWarning)
        assert "coinciding" in str(w[-1].message)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pytest.raises(ValueError, plotter.mask_interpolate,
                      coords, np.ones(len(coords)))
        pytest.raises(ValueError, plotter.mask_interpolate,
                      coords, np.ones(2 * len(coords)))
