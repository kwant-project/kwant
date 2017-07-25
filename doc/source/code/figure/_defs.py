################################################################
# Make matplotlib work without X11
################################################################
import matplotlib
matplotlib.use('Agg')

################################################################
# Prepend Kwant's build directory to sys.path
################################################################
import sys
from distutils.util import get_platform
sys.path.insert(0, "../../../../build/lib.{0}-{1}.{2}".format(
        get_platform(), *sys.version_info[:2]))

################################################################
# Define constants for plotting
################################################################
pt_to_in = 1. / 72.

# Default width of figures in pts
figwidth_pt = 600
figwidth_in = figwidth_pt * pt_to_in

# Width for smaller figures
figwidth_small_pt = 400
figwidth_small_in = figwidth_small_pt * pt_to_in

# Sizes for matplotlib figures
mpl_width_in = figwidth_pt * pt_to_in
mpl_label_size = None  # font sizes in points
mpl_tick_size = None

# dpi for conversion from inches
dpi = 90
