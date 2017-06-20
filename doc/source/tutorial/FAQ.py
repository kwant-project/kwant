
from cmath import exp
import numpy as np
import kwant
import matplotlib.pyplot
import matplotlib as mpl
from matplotlib import pyplot as plt
matplotlib.rcParams['figure.figsize'] = (3.5,3.5)
import tinyarray


######## What is a Site? ##############

#HIDDEN_BEGIN_FAQ122
a = 1
lat = kwant.lattice.square(a)
syst = kwant.Builder()

syst[lat(1,0)] = 4
syst[lat(1,1)] = 4

kwant.plot(syst)
#HIDDEN_END_FAQ122



################# What is a family? a tag? a lattice? ##################

#HIDDEN_BEGIN_FAQ123
a = 1
## 2 Monatomic lattices
lat1 = kwant.lattice.Monatomic( [(1,0) , (0,1)], offset = (0,0)) ## equivalent to kwant.lattice.square()
lat2 = kwant.lattice.Monatomic( [(1,0) , (0,1)], offset = (0.5,0.5))

## 1 Polyatomic lattice containing two sublattices
lat3 = kwant.lattice.Polyatomic([(1,0) , (0,1)], [(0,0) , (0.5,0.5)])
subA, subB = lat3.sublattices


syst = kwant.Builder()

syst[lat1(0,0)] = 4 ## syst[subA(0,0)] = 4
syst[lat2(0,0)] = 4 ## syst[subB(0,0)] = 4

kwant.plot(syst)
#HIDDEN_END_FAQ123



########## What is a hopping? #######################

a = 1
lat = kwant.lattice.square(a)
syst = kwant.Builder()

syst[lat(1,0)] = 4
syst[lat(1,1)] = 4
#HIDDEN_BEGIN_FAQ124
syst[(lat(1,0) , lat(1,1))] = -1
#HIDDEN_END_FAQ124

kwant.plot(syst)



########### How to make a hole in a system? ###################"

#HIDDEN_BEGIN_FAQ2

## Definition of the lattice and the system:
a = 2
lat = kwant.lattice.cubic(a)
syst = kwant.Builder()

L = 10
W = 10
H = 2

## Adding sites to the system:

syst[ (lat(i,j,k) for i in range(L) for j in range(W) for k in range(H)) ] = 4
kwant.plot(syst)


## Deleting sites to create a hole

for i in range(L):
    for j in range(W):
        for k in range(H):
            x, y, z = lat(i,j,k).pos
            if ((L-2)*a/4 <= x <= 3*L*a/4) and (W*a/4 <= y <= 3*(W-2)*a/4) and (0 <= z <= (H-1)*a):
                del syst[lat(i,j,k)]
kwant.plot(syst)
#HIDDEN_END_FAQ2



################ How can we get access to the sites of our system? ####################

builder = kwant.Builder()
lat = kwant.lattice.square()
builder[(lat(i,j) for i in range(3) for j in range(3))] = 4
#HIDDEN_BEGIN_FAQ3
## Before finalizing the system

sites = []
for site in builder.sites():
    sites.append(site) ## here we choose to add the sites to a list

#HIDDEN_END_FAQ3
#HIDDEN_BEGIN_FAQ7
## After finalizing the system

syst = builder.finalized()
i = syst.id_by_site[lat(0,2)] ## we want the id of the site lat(0,2)
 # syst.sites[i].tag  Returns the tag of lat(0,2)
 # syst.sites[i].pos  Returns the pos of lat(0,2)
#HIDDEN_END_FAQ7



################ How to plot a polyatomic lattice with different colors? ##############"

#HIDDEN_BEGIN_FAQ8
lat = kwant.lattice.kagome()
syst = kwant.Builder()

a , b, c = lat.sublattices ## The kagome lattice has 3 sublattices
#HIDDEN_END_FAQ8

#HIDDEN_BEGIN_FAQ9
## Plot sites from different families in different colors

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
        syst[a(i,j)] = 4 ## red
        syst[b(i,j)] = 4 ## green
        syst[c(i,j)] = 4 ## blue

syst[lat.neighbors()] = -1

## Plotting the system
plot_system(syst)

#HIDDEN_END_FAQ9



############### How to create every hoppings in a given direction using Hoppingkind? ################

## Monatomic lattice

#HIDDEN_BEGIN_FAQ4

## Creation of hopping between neighbors with HoppingKind
a = 1
syst = kwant.Builder()
lat = kwant.lattice.square(a)
syst[ (lat(i,j) for i in range(5) for j in range(5)) ] = 4

syst[kwant.builder.HoppingKind((1,0), lat)] = -1
kwant.plot(syst)
#HIDDEN_END_FAQ4

## Polyatomic lattice

lat = kwant.lattice.kagome()
syst = kwant.Builder()

a , b, c = lat.sublattices ## The kagome lattice has 3 sublattices

def plot_system(syst):

    def family_color(site):
        if site.family == a:
            return 'blue'
        if site.family == b:
            return 'red'
        else:
            return 'green'


    kwant.plot(syst, site_size= 0.15 , site_lw=0.05, site_color=family_color)

for i in range(4):
    for j in range (4):
        syst[a(i,j)] = 4 ## red
        syst[b(i,j)] = 4 ## green
        syst[c(i,j)] = 4 ## blue



#HIDDEN_BEGIN_FAQ13
syst[kwant.builder.HoppingKind((0,1), b, b)] = -1 # equivalent to syst[kwant.builder.HoppingKind((0,1), b)] = -1
#HIDDEN_END_FAQ13
plot_system(syst)
del syst[kwant.builder.HoppingKind((0,1), b, b)] ## delete the hoppings previously created
#HIDDEN_BEGIN_FAQ14
syst[kwant.builder.HoppingKind((0,0), a, b)] = -1
syst[kwant.builder.HoppingKind((0,0), a, c)] = -1
syst[kwant.builder.HoppingKind((0,0), c, b)] = -1
#HIDDEN_END_FAQ14
plot_system(syst)



########## How to create the hoppings between adjacent sites? ################

## Monatomic lattice

#HIDDEN_BEGIN_FAQ5

## Creation of hoppings with lat.neighbors()
syst = kwant.Builder()
lat = kwant.lattice.square()
syst[ (lat(i,j) for i in range(3) for j in range(3)) ] = 4

syst[lat.neighbors()] = -1 ## equivalent to lat.neighbors(1)
kwant.plot(syst)

del syst[lat.neighbors()] ## deletes every hoppings previously created to add new one
syst[lat.neighbors(2)] = -1

kwant.plot(syst)
#HIDDEN_END_FAQ5

## Polyatomic lattice

#HIDDEN_BEGIN_FAQ6

## Hoppings using .neighbors()
#HIDDEN_BEGIN_FAQ10
## Creation of the system
lat = kwant.lattice.kagome()
syst = kwant.Builder()
a , b, c = lat.sublattices ## The kagome lattice has 3 sublattices
for i in range(4):
    for j in range (4):
        syst[a(i,j)] = 4 ## red
        syst[b(i,j)] = 4 ## green
        syst[c(i,j)] = 4 ## blue

## .neighbors()
syst[lat.neighbors()] = -1
#HIDDEN_END_FAQ10
plot_system(syst)
del syst[lat.neighbors()] ## delete the hoppings previously created
#HIDDEN_BEGIN_FAQ11
syst[a.neighbors()] = -1
#HIDDEN_END_FAQ11
plot_system(syst)
del syst[a.neighbors()] ## deletes every hoppings previously created to add new one


#HIDDEN_BEGIN_FAQ12A
syst[lat.neighbors(2)] = -1
#HIDDEN_END_FAQ12A
plot_system(syst)
del syst[lat.neighbors(2)]



##### How to create a lead with a lattice different from the scattering region? ##########


#HIDDEN_BEGIN_FAQAA
## Plot sites from different families in different colors

def plot_system(syst):

    def family_color(site):
        if site.family == subA:
            return 'blue'
        if site.family == subB:
            return 'yellow'
        else:
            return 'green'


    kwant.plot(syst, site_lw=0.1, site_color=family_color)

#HIDDEN_END_FAQAA
#HIDDEN_BEGIN_FAQAB
## Defining the scattering Region
a = 2
lat = kwant.lattice.honeycomb(a)
syst = kwant.Builder()

L = 5
W = 5

subA, subB = lat.sublattices

## Adding sites to the system:

syst[ (subA(i,j) for i in range(L) for j in range(W)) ] = 4
syst[ (subB(i,j) for i in range(L) for j in range(W)) ] = 4
syst[lat.neighbors()] = -1
plot_system(syst)
#HIDDEN_END_FAQAB
#HIDDEN_BEGIN_FAQAC
## We manually add sites of the same lead lattice


lat2 = kwant.lattice.square(a)

def shapetop(pos):
    x, y = pos
    return ( 4 <= x <= 12 ) and ( 8 < y <= 10 )

def shapebot(pos):
    x, y = pos
    return ( 0 <= x <= 8 ) and ( -2 <= y < 0 )


syst[lat2.shape(shapetop, (4,10))] = 4
syst[lat2.shape(shapebot, (0,-2))] = 4

syst[lat2.neighbors()] = -1
syst[((lat2(i,-1), subA(i,0)) for i in range(5))] = -1
syst[((lat2(i+2,5), subB(i,4)) for i in range(5))] = -1
plot_system(syst)
#HIDDEN_END_FAQAC
#HIDDEN_BEGIN_FAQAD
## Creation of the top lead

lat_lead = kwant.lattice.square(a)
sym_lead1 = kwant.TranslationalSymmetry((0,a))
lead1 = kwant.Builder(sym_lead1)


def lead_shape_top(pos): ## Shape of the lead
    (x, y) = pos
    return (4 <= x <= 12)


lead1[lat_lead.shape(lead_shape_top, (4,12))] = 4
lead1[lat_lead.neighbors()] = -1

syst.attach_lead(lead1)
plot_system(syst)
#HIDDEN_END_FAQAD
#HIDDEN_BEGIN_FAQAE
## Creation of the bottom lead

sym_lead2 = kwant.TranslationalSymmetry((0,-a))
lead2 = kwant.Builder(sym_lead2)

def lead_shape_bot(pos): ## Shape of the lead
    (x, y) = pos
    return (0 <= x <= 8)

lead2[lat_lead.shape(lead_shape_bot, (0,-4))] = 4
lead2[lat_lead.neighbors()] = -1

syst.attach_lead(lead2)

plot_system(syst)
#HIDDEN_END_FAQAE



############# How to cut a finite system out of a system with translationnal symmetries? ###########

#HIDDEN_BEGIN_FAQccc
# Create  3d model.
cubic = kwant.lattice.cubic()
sym_3d = kwant.TranslationalSymmetry([1, 0, 0], [0, 1, 0], [0, 0, 1])
model = kwant.Builder(sym_3d)
model[cubic(0, 0, 0)] = 4
model[cubic.neighbors()] = -1
#HIDDEN_END_FAQccc

#HIDDEN_BEGIN_FAQddd
# Build scattering region (white).
def cuboid_shape(site):
    x, y, z = abs(site.pos)
    return x < 4 and y < 10 and z < 3

cuboid = kwant.Builder()
cuboid.fill(model, cuboid_shape, (0, 0, 0));

kwant.plot(cuboid);
#HIDDEN_END_FAQddd

#HIDDEN_BEGIN_FAQeee
# Build electrode (black).
def electrode_shape(site):
    x, y, z = site.pos - (0, 5, 2)
    return y**2 + z**2 < 2.3**2

electrode = kwant.Builder(kwant.TranslationalSymmetry([1, 0, 0]))
electrode.fill(model, electrode_shape, (0, 5, 2)); ## lead

cuboid.fill(electrode, lambda s: abs(s.pos[0]) < 7, (0, 5, 4)); ## scattering region

cuboid.attach_lead(electrode)
kwant.plot(cuboid);
#HIDDEN_END_FAQeee


###### How to extract the wavefunction informations on a specific site? ###############

#HIDDEN_BEGIN_FAQAF
## Creation of the system
a = 2
lat = kwant.lattice.square(a)
syst = kwant.Builder()

syst[((lat(i,j) for i in range(5) for j in range(3)))] = 4
syst[lat.neighbors()] = -1
kwant.plot(syst)

sym_lead = kwant.TranslationalSymmetry((-a,0))
lead = kwant.Builder(sym_lead)
lead[(lat(0,i) for i in range(3))] = 4
lead[lat.neighbors()] = -1
syst.attach_lead(lead)
syst.attach_lead(lead.reversed())
kwant.plot(syst)

fsyst = syst.finalized()

## Plot the different modes
Ef = 3.0
lead = lead.finalized()
kwant.plotter.bands(lead, show=False)
kwant.plotter.bands
plt.plot([-3,3],[Ef,Ef], 'r--')
plt.xlabel("momentum [(lattice constant)^-1]")
plt.ylabel("energy [t]")
plt.show()
#HIDDEN_END_FAQAF
plt.clf()

#HIDDEN_BEGIN_FAQAG
wf = kwant.wave_function(fsyst, Ef)
wf_left_lead = wf(0) ## Wave function for the first lead (left)
#HIDDEN_END_FAQAG


#HIDDEN_BEGIN_FAQAH

wf_left_0 = wf_left_lead[0] ## We choose the the mode with the highest k (mode in blue)


tag = lat.closest((6,2)) ## returns the tag  of the site from lat based on the position

i = fsyst.id_by_site[lat(*tag)] ## Returns the index in the low_level system based on the tag

wf_site = wf_left_0[i] ## Returns the wave function on the site

#HIDDEN_END_FAQAH


#HIDDEN_BEGIN_FAQAO
tau_x = tinyarray.array([[0, 1], [1, 0]])
tau_y = tinyarray.array([[0, -1j], [1j, 0]])
tau_z = tinyarray.array([[1, 0], [0, -1]])

lat = kwant.lattice.square(norbs=2)


	### Creation of the Builder ###

def make_system(a=1, W=10, L=10, barrier=1.5, barrierpos=(3, 4),
                mu=0.4, Delta=0.1, Deltapos=4, t=1.0, phs=True):
    # Start with an empty tight-binding system. On each site, there
    # are now electron and hole orbitals, so we must specify the
    # number of orbitals per site. The orbital structure is the same
    # as in the Hamiltonian.
    syst = kwant.Builder()

    #### Define the scattering region. ####
    # The superconducting order parameter couples electron and hole orbitals
    # on each site, and hence enters as an onsite potential.
    # The pairing is only included beyond the point 'Deltapos' in the scattering region.
    syst[(lat(x, y) for x in range(Deltapos) for y in range(W))] = (4 * t - mu) * tau_z
    syst[(lat(x, y) for x in range(Deltapos, L) for y in range(W))] = (4 * t - mu) * tau_z + Delta * tau_x

    # The tunnel barrier
    syst[(lat(x, y) for x in range(barrierpos[0], barrierpos[1])
         for y in range(W))] = (4 * t + barrier - mu) * tau_z

    # Hoppings
    syst[lat.neighbors()] = -t * tau_z
    #### Define the leads. ####
    # Left lead - normal, so the order parameter is zero.
    sym_left = kwant.TranslationalSymmetry((-a, 0))
    # Specify the conservation law used to treat electrons and holes separately.
    # We only do this in the left lead, where the pairing is zero.
    lead0 = kwant.Builder(sym_left, conservation_law=-tau_z, particle_hole=tau_y)
    lead0[(lat(0, j) for j in range(W))] = (4 * t - mu) * tau_z
    lead0[lat.neighbors()] = -t * tau_z
    # Right lead - superconducting, so the order parameter is included.
    sym_right = kwant.TranslationalSymmetry((a, 0))
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(0, j) for j in range(W))] = (4 * t - mu) * tau_z + Delta * tau_x
    lead1[lat.neighbors()] = -t * tau_z

    #### Attach the leads and return the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)

    return syst


syst = make_system(W=10)

    # Check that the system looks as intended.
kwant.plot(syst)

    # Finalize the system.
syst = syst.finalized()

	#Plot the mods of the lead
lead = syst.leads[0]

Ef=0.5
kwant.plotter.bands(lead, show=False)
kwant.plotter.bands
plt.plot([-3,3],[Ef,Ef], 'r--')
plt.xlabel("momentum [(lattice constant)^-1]")
plt.ylabel("energy [t]")
plt.show()

plt.clf()

wf = kwant.wave_function(syst, Ef)
wf_left_lead = wf(0) # Wave function for the first lead (left)



#HIDDEN_END_FAQAO


#HIDDEN_BEGIN_FAQAP
nb_degrees = 2 ## 2 degrees of freedom

wf_left_0 = wf_left_lead[0] ## We choose the the mode with the highest k (mode in blue)


tag = lat.closest((6,2)) ## returns the tag  of the site from lat based on the position

i = syst.id_by_site[lat(*tag)] ## Returns the index in the low_level system based on the tag

tab_wf = []

for k in range(nb_degrees) : ## loop over the number of degrees of freedom
	tab_wf.append(wf_left_0[nb_degrees * i +k]) # different states of the given site


#HIDDEN_END_FAQAP
