# Frequently Asked Questions
# ==========================

import kwant
import numpy as np
import tinyarray
import matplotlib
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = (3.5, 3.5)


######## What is a Site? ##############

#HIDDEN_BEGIN_site
a = 1
lat = kwant.lattice.square(a)
syst = kwant.Builder()

syst[lat(1, 0)] = 4
syst[lat(1, 1)] = 4

kwant.plot(syst)
#HIDDEN_END_site



################# What is a family? a tag? a lattice? ##################

#HIDDEN_BEGIN_lattice
# 2 Monatomic lattices
primitive_vectors = [(1, 0), (0, 1)]
lat1 = kwant.lattice.Monatomic(primitive_vectors, offset=(0, 0))  # equivalent to kwant.lattice.square()
lat2 = kwant.lattice.Monatomic(primitive_vectors, offset=(0.5, 0.5))

# 1 Polyatomic lattice containing two sublattices
lat3 = kwant.lattice.Polyatomic([(1, 0) , (0, 1)], [(0, 0) , (0.5, 0.5)])
subA, subB = lat3.sublattices


syst = kwant.Builder()

syst[lat1(0, 0)] = 4  # syst[subA(0, 0)] = 4
syst[lat2(0, 0)] = 4  # syst[subB(0, 0)] = 4

kwant.plot(syst)
#HIDDEN_END_lattice


########## What is a hopping? #######################

a = 1
lat = kwant.lattice.square(a)
syst = kwant.Builder()

syst[lat(1, 0)] = 4
syst[lat(1, 1)] = 4
#HIDDEN_BEGIN_hopping
syst[(lat(1, 0), lat(1, 1))] = -1
#HIDDEN_END_hopping

kwant.plot(syst)

########### How to make a hole in a system? ###################"

#HIDDEN_BEGIN_hole
# Define the lattice and the (empty) system
a = 2
lat = kwant.lattice.cubic(a)
syst = kwant.Builder()

L = 10
W = 10
H = 2

# Add sites to the system in a cuboid

syst[(lat(i, j, k) for i in range(L) for j in range(W) for k in range(H))] = 4
kwant.plot(syst)

# Delete sites to create a hole

def in_hole(site):
    x, y, z = site.pos / a - (L/2, W/2, H/2)  # position relative to centre
    return abs(x) < L / 4 and abs(y) < W / 4

for site in filter(in_hole, list(syst.sites())):
    del syst[site]

kwant.plot(syst)
#HIDDEN_END_hole


################ How can we get access to the sites of our system? ####################

builder = kwant.Builder()
lat = kwant.lattice.square()
builder[(lat(i, j) for i in range(3) for j in range(3))] = 4
#HIDDEN_BEGIN_sites1
# Before finalizing the system

sites = list(builder.sites())  # sites() doe *not* return a list

#HIDDEN_END_sites1
#HIDDEN_BEGIN_sites2
# After finalizing the system
syst = builder.finalized()
sites = syst.sites  # syst.sites is an actual list
#HIDDEN_END_sites2
#HIDDEN_BEGIN_sites3
i = syst.id_by_site[lat(0, 2)]  # we want the id of the site lat(0, 2)
#HIDDEN_END_sites3


################ How to plot a polyatomic lattice with different colors? ##############"

#HIDDEN_BEGIN_colors1
lat = kwant.lattice.kagome()
syst = kwant.Builder()

a, b, c = lat.sublattices  # The kagome lattice has 3 sublattices
#HIDDEN_END_colors1

#HIDDEN_BEGIN_colors2
# Plot sites from different families in different colors

def plot_system(syst):

    def family_color(site):
        if site.family == a:
            return 'red'
        if site.family == b:
            return 'green'
        else:
            return 'blue'

    def hopping_lw(site1, site2):
        return 0.1 if site1.family == site2.family else 0.1

    kwant.plot(syst, site_lw=0.1, site_color=family_color, hop_lw=hopping_lw)


## Adding sites and hoppings
for i in range(4):
    for j in range (4):
        syst[a(i, j)] = 4  # red
        syst[b(i, j)] = 4  # green
        syst[c(i, j)] = 4  # blue

syst[lat.neighbors()] = -1

## Plotting the system
plot_system(syst)

#HIDDEN_END_colors2


############### How to create all hoppings in a given direction using Hoppingkind? ################

# Monatomic lattice

#HIDDEN_BEGIN_direction1

# Create hopping between neighbors with HoppingKind
a = 1
syst = kwant.Builder()
lat = kwant.lattice.square(a)
syst[ (lat(i, j) for i in range(5) for j in range(5)) ] = 4

syst[kwant.builder.HoppingKind((1, 0), lat)] = -1
kwant.plot(syst)
#HIDDEN_END_direction1

# Polyatomic lattice

lat = kwant.lattice.kagome()
syst = kwant.Builder()

a, b, c = lat.sublattices  # The kagome lattice has 3 sublattices


def plot_system(syst):

    def family_color(site):
        if site.family == a:
            return 'blue'
        if site.family == b:
            return 'red'
        else:
            return 'green'

    kwant.plot(syst, site_size=0.15, site_lw=0.05, site_color=family_color)


for i in range(4):
    for j in range (4):
        syst[a(i, j)] = 4  # red
        syst[b(i, j)] = 4  # green
        syst[c(i, j)] = 4  # blue


#HIDDEN_BEGIN_direction2
# equivalent to syst[kwant.builder.HoppingKind((0, 1), b)] = -1
syst[kwant.builder.HoppingKind((0, 1), b, b)] = -1
#HIDDEN_END_direction2
plot_system(syst)
# Delete the hoppings previously created
del syst[kwant.builder.HoppingKind((0, 1), b, b)]
#HIDDEN_BEGIN_direction3
syst[kwant.builder.HoppingKind((0, 0), a, b)] = -1
syst[kwant.builder.HoppingKind((0, 0), a, c)] = -1
syst[kwant.builder.HoppingKind((0, 0), c, b)] = -1
#HIDDEN_END_direction3
plot_system(syst)


########## How to create the hoppings between adjacent sites? ################

# Monatomic lattice

#HIDDEN_BEGIN_adjacent1

# Create hoppings with lat.neighbors()
syst = kwant.Builder()
lat = kwant.lattice.square()
syst[(lat(i, j) for i in range(3) for j in range(3))] = 4

syst[lat.neighbors()] = -1  # Equivalent to lat.neighbors(1)
kwant.plot(syst)

del syst[lat.neighbors()]  # Delete all nearest-neighbor hoppings
syst[lat.neighbors(2)] = -1

kwant.plot(syst)
#HIDDEN_END_adjacent1

# Polyatomic lattice

#HIDDEN_BEGIN_FAQ6

# Hoppings using .neighbors()
#HIDDEN_BEGIN_adjacent2
# Create the system
lat = kwant.lattice.kagome()
syst = kwant.Builder()
a, b, c = lat.sublattices  # The kagome lattice has 3 sublattices

for i in range(4):
    for j in range (4):
        syst[a(i, j)] = 4  # red
        syst[b(i, j)] = 4  # green
        syst[c(i, j)] = 4  # blue

syst[lat.neighbors()] = -1
#HIDDEN_END_adjacent2
plot_system(syst)
del syst[lat.neighbors()]  # Delete the hoppings previously created
#HIDDEN_BEGIN_adjacent3
syst[a.neighbors()] = -1
#HIDDEN_END_adjacent3
plot_system(syst)
del syst[a.neighbors()]  # Delete the hoppings previously created


syst[lat.neighbors(2)] = -1
plot_system(syst)
del syst[lat.neighbors(2)]


##### How to create a lead with a lattice different from the scattering region? ##########

# Plot sites from different families in different colors

def plot_system(syst):

    def family_color(site):
        if site.family == subA:
            return 'blue'
        if site.family == subB:
            return 'yellow'
        else:
            return 'green'

    kwant.plot(syst, site_lw=0.1, site_color=family_color)


#HIDDEN_BEGIN_different_lattice1
# Define the scattering Region
L = 5
W = 5

lat = kwant.lattice.honeycomb()
subA, subB = lat.sublattices

syst = kwant.Builder()
syst[(subA(i, j) for i in range(L) for j in range(W))] = 4
syst[(subB(i, j) for i in range(L) for j in range(W))] = 4
syst[lat.neighbors()] = -1
#HIDDEN_END_different_lattice1
plot_system(syst)

#HIDDEN_BEGIN_different_lattice2
# Create a lead
lat_lead = kwant.lattice.square()
sym_lead1 = kwant.TranslationalSymmetry((0, 1))

lead1 = kwant.Builder(sym_lead1)
lead1[(lat_lead(i, 0) for i in range(2, 7))] = 4
lead1[lat_lead.neighbors()] = -1
#HIDDEN_END_different_lattice2
plot_system(lead1)

#HIDDEN_BEGIN_different_lattice3
syst[(lat_lead(i, 5) for i in range(2, 7))] = 4
syst[lat_lead.neighbors()] = -1

# Manually attach sites from graphene to square lattice
syst[((lat_lead(i+2, 5), subB(i, 4)) for i in range(5))] = -1
#HIDDEN_END_different_lattice3
plot_system(syst)

#HIDDEN_BEGIN_different_lattice4
syst.attach_lead(lead1)
#HIDDEN_END_different_lattice4
plot_system(syst)


############# How to cut a finite system out of a system with translationnal symmetries? ###########

#HIDDEN_BEGIN_fill1
# Create 3d model.
cubic = kwant.lattice.cubic()
sym_3d = kwant.TranslationalSymmetry([1, 0, 0], [0, 1, 0], [0, 0, 1])
model = kwant.Builder(sym_3d)
model[cubic(0, 0, 0)] = 4
model[cubic.neighbors()] = -1
#HIDDEN_END_fill1

#HIDDEN_BEGIN_fill2
# Build scattering region (white).
def cuboid_shape(site):
    x, y, z = abs(site.pos)
    return x < 4 and y < 10 and z < 3

cuboid = kwant.Builder()
cuboid.fill(model, cuboid_shape, (0, 0, 0));
#HIDDEN_END_fill2
kwant.plot(cuboid);

#HIDDEN_BEGIN_fill3
# Build electrode (black).
def electrode_shape(site):
    x, y, z = site.pos - (0, 5, 2)
    return y**2 + z**2 < 2.3**2

electrode = kwant.Builder(kwant.TranslationalSymmetry([1, 0, 0]))
electrode.fill(model, electrode_shape, (0, 5, 2))  # lead

# Scattering region
cuboid.fill(electrode, lambda s: abs(s.pos[0]) < 7, (0, 5, 4))

cuboid.attach_lead(electrode)
#HIDDEN_END_fill3
kwant.plot(cuboid);


###### How does Kwant order the propagating modes of a lead? ######

#HIDDEN_BEGIN_pm
lat = kwant.lattice.square()

lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
lead[(lat(0, i) for i in range(3))] = 4
lead[lat.neighbors()] = -1

flead = lead.finalized()

E = 2.5
prop_modes, _ = flead.modes(energy=E)
#HIDDEN_END_pm

def plot_and_label_modes(lead, E):
    # Plot the different modes
    pmodes, _ = lead.modes(energy=E)
    kwant.plotter.bands(lead, show=False)
    for i, k in enumerate(pmodes.momenta):
        plt.plot(k, E, 'ko')
        plt.annotate(str(i), xy=(k, E), xytext=(-5, 8),
                     textcoords='offset points',
                     bbox=dict(boxstyle='round,pad=0.1',fc='white', alpha=0.7))
    plt.plot([-3, 3], [E, E], 'r--')
    plt.ylim(E-1, E+1)
    plt.xlim(-2, 2)
    plt.xlabel("momentum")
    plt.ylabel("energy")
    plt.show()

plot_and_label_modes(flead, E)
plt.clf()

# More involved example

s0 = np.eye(2)
sz = np.array([[1, 0], [0, -1]])

lead2 = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))

lead2[(lat(0, i) for i in range(2))] = np.diag([1.8, -1])
lead2[lat.neighbors()] = -1 * sz

flead2 = lead2.finalized()

plot_and_label_modes(flead2, 1)
plt.clf()


###### How does Kwant order components of an individual wavefunction? ######

def circle(R):
    return lambda r: np.linalg.norm(r) < R


def make_system(lat):
    norbs = lat.norbs
    syst = kwant.Builder()
    syst[lat.shape(circle(3), (0, 0))] = 4 * np.eye(norbs)
    syst[lat.neighbors()] = -1 * np.eye(norbs)

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[(lat(0, i) for i in range(-1, 2))] = 4 * np.eye(norbs)
    lead[lat.neighbors()] = -1 * np.eye(norbs)

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst.finalized()


#HIDDEN_BEGIN_ord1
lat = kwant.lattice.square(norbs=1)
syst = make_system(lat)
scattering_states = kwant.wave_function(syst, energy=1)
wf = scattering_states(0)[0]  # scattering state from lead 0 incoming in mode 0

idx = syst.id_by_site[lat(0, 0)]  # look up index of site

print('wavefunction on lat(0, 0): ', wf[idx])
#HIDDEN_END_ord1

#HIDDEN_BEGIN_ord2
lat = kwant.lattice.square(norbs=2)
syst = make_system(lat)
scattering_states = kwant.wave_function(syst, energy=1)
wf = scattering_states(0)[0]  # scattering state from lead 0 incoming in mode 0

idx = syst.id_by_site[lat(0, 0)]  # look up index of site

# Group consecutive degrees of freedom from 'wf' together; these correspond
# to degrees of freedom on the same site.
wf = wf.reshape(-1, 2)

print('wavefunction on lat(0, 0): ', wf[idx])
#HIDDEN_END_ord2
