"""The script generating Kwant logo. In addition to Kwant it also needs Python
image library (PIL)."""

import Image
import ImageFont
import ImageDraw
import matplotlib
import numpy as np
import scipy
import kwant

def main():
    def bbox(array):
        x, y = np.where(array)
        return np.min(x), np.max(x), np.min(y), np.max(y)

    # Prepare an image.
    x = 500
    y = 160
    im = Image.new('L', (x, y), 255)
    draw = ImageDraw.Draw(im)

    # Select a font for the logo and make an image of the logo.  We use a font
    # available in Debian/Ubuntu, but it can also be downloaded e.g. at
    # http://www.fonts2u.com/free-monospaced-bold.font
    fontfile = "/usr/share/fonts/truetype/freefont/FreeMonoBold.ttf"
    font = ImageFont.truetype(fontfile, 150)
    draw.text((10, 10), "kwant", font=font)

    dy = 3
    dx1 = 5
    dx2 = 3
    mu_system = 3.8

    # The the coordinates of text.
    textpos = (1. - np.array(im.getdata()) / 255.).reshape(y, x)

    # Cut away empty space around the letters.
    xmin, xmax, ymin, ymax = bbox(textpos)
    textpos = textpos[(xmin - 1) : (xmax + dx2)][:, (ymin - dy) : (ymax + dy)]
    xmin, xmax, ymin, ymax = bbox(textpos)

    # Add an underscore that touches the lettes.
    geometry = np.copy(textpos)
    geometry[(xmax - dx1) : (xmax + dx2)][:, (ymin - dy) : (ymax + dy)] = 1

    # Find x-coordinates separating the letters.
    nonempty = np.apply_along_axis(np.sum, 0, textpos) > 0
    borders = np.where(np.diff(nonempty))[0]
    letters = borders.reshape(-1, 2)
    gaps = borders[1:-1].reshape(-1, 2)

    # Construct the system, and calculate LDOS.
    sys = kwant.Builder()
    lat = kwant.lattice.square()
    sys[(lat(*coord) for coord in np.argwhere(geometry))] = mu_system
    sys[lat.neighbors()] = -1
    lead = kwant.Builder(kwant.TranslationalSymmetry((1, 0)))
    for y1 in range(ymin - dy, ymax + dy):
        lead[lat(0, y1)] = mu_system
    lead[lat.neighbors()[0]] = -3
    sys.attach_lead(lead)
    sys = sys.finalized()
    ldos = kwant.solvers.default.ldos(sys, energy=0)

    # Due to the letters having different overall thickness, the LDOS is larger
    # in some letters, which makes them have visually different colors. We
    # adjust this by normalizing each letter to its maximum.
    def normalize_data(data):
        sums = []
        for letter in letters:
            letter_data = data[:, slice(*letter)]
            letter_data = letter_data[np.nonzero(letter_data)]
            sums.append(np.max(letter_data))
        weights = np.zeros(data.shape[1])
        for i, letter in enumerate(letters):
            weights[slice(*letter)] = 1/sums[i]
        for i, gap in enumerate(gaps):
            weights[slice(*gap)] = np.linspace(1 / sums[i], 1 / sums[i+1],
                                               gap[1] - gap[0])
        new_data = data * weights.reshape(1, -1)
        new_data /= np.max(new_data)
        return new_data

    # Here we apply a nonlinear transformation to LDOS to ensure that the
    # result is not too empty or not too dark.
    out = np.zeros(textpos.shape)
    for i, rho in enumerate(ldos**.2):
        x1, y1 = sys.site(i).tag
        out[x1, y1] = rho
    out = normalize_data(out)

    # We use the original text data as a transparency mask for anti-aliasing.
    out = matplotlib.cm.PuBu(out, bytes=True)
    out[:, :, 3] = 255 * geometry
    scipy.misc.imsave('logo.png', out)

if __name__ == '__main__':
    main()
