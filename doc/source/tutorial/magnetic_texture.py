# Tutorial 2.7. Spin textures
# ===========================
#
# Physics background
# ------------------
#  - Spin textures
#  - Skyrmions
#
# Kwant features highlighted
# --------------------------
#  - operators
#  - plotting vector fields

from math import sin, cos, tanh, pi
import itertools
import numpy as np
import tinyarray as ta
import matplotlib.pyplot as plt

import kwant

sigma_0 = ta.array([[1, 0], [0, 1]])
sigma_x = ta.array([[0, 1], [1, 0]])
sigma_y = ta.array([[0, -1j], [1j, 0]])
sigma_z = ta.array([[1, 0], [0, -1]])

# vector of Pauli matrices σ_αiβ where greek
# letters denote spinor indices
sigma = np.rollaxis(np.array([sigma_x, sigma_y, sigma_z]), 1)

#HIDDEN_BEGIN_model
def field_direction(pos, r0, delta):
    x, y = pos
    r = np.linalg.norm(pos)
    r_tilde = (r - r0) / delta
    theta = (tanh(r_tilde) - 1) * (pi / 2)

    if r == 0:
        m_i = [0, 0, 1]
    else:
        m_i = [
            (x / r) * sin(theta),
            (y / r) * sin(theta),
            cos(theta),
        ]

    return np.array(m_i)


def scattering_onsite(site, r0, delta, J):
    m_i = field_direction(site.pos, r0, delta)
    return J * np.dot(m_i, sigma)


def lead_onsite(site, J):
    return J * sigma_z
#HIDDEN_END_model


#HIDDEN_BEGIN_syst
lat = kwant.lattice.square(norbs=2)

def make_system(L=80):

    syst = kwant.Builder()

    def square(pos):
        return all(-L/2 < p < L/2 for p in pos)

    syst[lat.shape(square, (0, 0))] = scattering_onsite
    syst[lat.neighbors()] = -sigma_0

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)),
                         conservation_law=-sigma_z)

    lead[lat.shape(square, (0, 0))] = lead_onsite
    lead[lat.neighbors()] = -sigma_0

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst
#HIDDEN_END_syst


def plot_vector_field(syst, params):
    xmin, ymin = min(s.tag for s in syst.sites)
    xmax, ymax = max(s.tag for s in syst.sites)
    x, y = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))

    m_i = [field_direction(p, **params) for p in zip(x.flat, y.flat)]
    m_i = np.reshape(m_i, x.shape + (3,))
    m_i = np.rollaxis(m_i, 2, 0)

    fig, ax = plt.subplots(1, 1)
    im = ax.quiver(x, y, *m_i, pivot='mid', scale=75)
    fig.colorbar(im)
    plt.show()


def plot_densities(syst, densities):
    fig, axes = plt.subplots(1, len(densities))
    for ax, (title, rho) in zip(axes, densities):
        kwant.plotter.map(syst, rho, ax=ax, a=4)
        ax.set_title(title)
    plt.show()


def plot_currents(syst, currents):
    fig, axes = plt.subplots(1, len(currents))
    if not hasattr(axes, '__len__'):
        axes = (axes,)
    for ax, (title, current) in zip(axes, currents):
        kwant.plotter.current(syst, current, ax=ax, colorbar=False)
        ax.set_title(title)
    plt.show()


def main():
    syst = make_system().finalized()

#HIDDEN_BEGIN_wavefunction
    params = dict(r0=20, delta=10, J=1)
    wf = kwant.wave_function(syst, energy=-1, params=params)
    psi = wf(0)[0]
#HIDDEN_END_wavefunction

    plot_vector_field(syst, dict(r0=20, delta=10))

#HIDDEN_BEGIN_ldos
    # even (odd) indices correspond to spin up (down)
    up, down = psi[::2], psi[1::2]
    density = np.abs(up)**2 + np.abs(down)**2
#HIDDEN_END_ldos

#HIDDEN_BEGIN_lsdz
    # spin down components have a minus sign
    spin_z = np.abs(up)**2 - np.abs(down)**2
#HIDDEN_END_lsdz

#HIDDEN_BEGIN_lsdy
    # spin down components have a minus sign
    spin_y = 1j * (down.conjugate() * up - up.conjugate() * down)
#HIDDEN_END_lsdy

#HIDDEN_BEGIN_lden
    rho = kwant.operator.Density(syst)
    rho_sz = kwant.operator.Density(syst, sigma_z)
    rho_sy = kwant.operator.Density(syst, sigma_y)

    # calculate the expectation values of the operators with 'psi'
    density = rho(psi)
    spin_z = rho_sz(psi)
    spin_y = rho_sy(psi)
#HIDDEN_END_lden

    plot_densities(syst, [
        ('$σ_0$', density),
        ('$σ_z$', spin_z),
        ('$σ_y$', spin_y),
    ])

#HIDDEN_BEGIN_current
    J_0 = kwant.operator.Current(syst)
    J_z = kwant.operator.Current(syst, sigma_z)
    J_y = kwant.operator.Current(syst, sigma_y)

    # calculate the expectation values of the operators with 'psi'
    current = J_0(psi)
    spin_z_current = J_z(psi)
    spin_y_current = J_y(psi)
#HIDDEN_END_current

    plot_currents(syst, [
        ('$J_{σ_0}$', current),
        ('$J_{σ_z}$', spin_z_current),
        ('$J_{σ_y}$', spin_y_current),
    ])

#HIDDEN_BEGIN_following
    def following_m_i(site, r0, delta):
        m_i = field_direction(site.pos, r0, delta)
        return np.dot(m_i, sigma)

    J_m = kwant.operator.Current(syst, following_m_i)

    # evaluate the operator
    m_current = J_m(psi, params=dict(r0=25, delta=10))
#HIDDEN_END_following

    plot_currents(syst, [
        (r'$J_{\mathbf{m}_i}$', m_current),
        ('$J_{σ_z}$', spin_z_current),
    ])


#HIDDEN_BEGIN_density_cut
    def circle(site):
        return np.linalg.norm(site.pos) < 20

    rho_circle = kwant.operator.Density(syst, where=circle, sum=True)

    all_states = np.vstack((wf(0), wf(1)))
    dos_in_circle = sum(rho_circle(p) for p in all_states) / (2 * pi)
    print('density of states in circle:', dos_in_circle)
#HIDDEN_END_density_cut

#HIDDEN_BEGIN_current_cut
    def left_cut(site_to, site_from):
        return site_from.pos[0] <= -39 and site_to.pos[0] > -39

    def right_cut(site_to, site_from):
        return site_from.pos[0] < 39 and site_to.pos[0] >= 39

    J_left = kwant.operator.Current(syst, where=left_cut, sum=True)
    J_right = kwant.operator.Current(syst, where=right_cut, sum=True)

    Jz_left = kwant.operator.Current(syst, sigma_z, where=left_cut, sum=True)
    Jz_right = kwant.operator.Current(syst, sigma_z, where=right_cut, sum=True)

    print('J_left:', J_left(psi), ' J_right:', J_right(psi))
    print('Jz_left:', Jz_left(psi), ' Jz_right:', Jz_right(psi))
#HIDDEN_END_current_cut

#HIDDEN_BEGIN_bind
    J_m = kwant.operator.Current(syst, following_m_i)
    J_z = kwant.operator.Current(syst, sigma_z)

    J_m_bound = J_m.bind(params=dict(r0=25, delta=10, J=1))
    J_z_bound = J_z.bind(params=dict(r0=25, delta=10, J=1))

    # Sum current local from all scattering states on the left at energy=-1
    wf_left = wf(0)
    J_m_left = sum(J_m_bound(p) for p in wf_left)
    J_z_left = sum(J_z_bound(p) for p in wf_left)
#HIDDEN_END_bind

    plot_currents(syst, [
        (r'$J_{\mathbf{m}_i}$ (from left)', J_m_left),
        (r'$J_{σ_z}$ (from left)', J_z_left),
    ])


if __name__ == '__main__':
    main()
