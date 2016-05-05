from math import sqrt
import scipy.sparse.linalg as sla
from matplotlib import pyplot
import kwant

syst = kwant.Builder()
lat = kwant.lattice.general([(1,0,0), (0,1,0), (0,0,1)])

t = 1.0
R = 10

syst[(lat(x,y,z) for x in range(-R-1, R+1)
     for y in range(-R-1, R+1) for z in range(R+1)
     if sqrt(x**2 + y**2 + z**2) < R + 0.01)] = 4 * t
syst[(lat(x,y,z) for x in range(-2*R, 2*R + 1)
     for y in range(-R, R+1) for z in range(-R, 0))] = 4 * t
syst[lat.neighbors()] = -t
syst = syst.finalized()
kwant.plot(syst)

ham = syst.hamiltonian_submatrix(sparse=True)
wf = abs(sla.eigs(ham, sigma=0.2, k=6)[1][:,0])**2
size = 3 * wf / wf.max()
kwant.plot(syst, site_size=size, site_color=(0,0,1,0.05), site_lw=0)
