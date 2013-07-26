from math import sqrt
import scipy.sparse.linalg as sla
import matplotlib.pyplot
import kwant

sys = kwant.Builder()
lat = kwant.lattice.general([(1,0,0), (0,1,0), (0,0,1)])

t = 1.0
R = 10

sys[(lat(x,y,z) for x in xrange(-R-1, R+1)
     for y in xrange(-R-1, R+1) for z in xrange(R+1)
     if sqrt(x**2 + y**2 + z**2) < R + 0.01)] = 4 * t
sys[(lat(x,y,z) for x in xrange(-2*R, 2*R + 1)
     for y in xrange(-R, R+1) for z in xrange(-R, 0))] = 4 * t
sys[lat.neighbors()] = -t
sys = sys.finalized()
kwant.plot(sys)

ham = sys.hamiltonian_submatrix(sparse=True)
wf = abs(sla.eigs(ham, sigma=0.2, k=6)[1][:,0])**2
size = 3 * wf / wf.max()
kwant.plot(sys, site_size=size, site_color=(0,0,1,0.05), site_lw=0)
