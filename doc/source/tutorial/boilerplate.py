import matplotlib
import matplotlib.pyplot
from matplotlib_inline.backend_inline import set_matplotlib_formats

matplotlib.rcParams['figure.figsize'] = matplotlib.pyplot.figaspect(1) * 2
set_matplotlib_formats('svg')
