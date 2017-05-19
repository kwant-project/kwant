# Tutorial 2.8. Calculating spectral density with the Kernel Polynomial Method
# ============================================================================
#
# Physics background
# ------------------
#  - Chebyshev polynomials, random trace approximation, spectral densities.
#
# Kwant features highlighted
# --------------------------
#  - kpm module,kwant operators.

import scipy

# For plotting
from matplotlib import pyplot as plt

#HIDDEN_BEGIN_sys1
# necessary imports
import kwant
import numpy as np


# define the system
def make_syst(r=30, t=-1, a=1):
    syst = kwant.Builder()
    lat = kwant.lattice.honeycomb(a, norbs=1)

    def circle(pos):
        x, y = pos
        return x ** 2 + y ** 2 < r ** 2

    syst[lat.shape(circle, (0, 0))] = 0.
    syst[lat.neighbors()] = t
    syst.eradicate_dangling()

    return syst
#HIDDEN_END_sys1


# Plot several density of states curves on the same axes.
def plot_dos(labels_to_data):
    for label, (x, y) in labels_to_data:
        plt.plot(x, y, label=label, linewidth=2)
    plt.legend(loc=2, framealpha=0.5)
    plt.xlabel("energy [t]")
    plt.ylabel("DoS [a.u.]")
    plt.show()
    plt.clf()


def site_size_conversion(densities):
    return 3 * np.abs(densities) / max(densities)


# Plot several local density of states maps in different subplots
def plot_ldos(fsyst, axes, titles_to_data, file_name=None):
    for ax, (title, ldos) in zip(axes, titles_to_data):
        site_size = site_size_conversion(ldos)  # convert LDoS to sizes
        kwant.plot(fsyst, site_size=site_size, site_color=(0, 0, 1, 0.3), ax=ax)
        ax.set_title(title)
        ax.set(adjustable='box-forced', aspect='equal')
    plt.show()
    plt.clf()


def simple_dos_example():
#HIDDEN_BEGIN_kpm1
    fsyst = make_syst().finalized()

    spectrum = kwant.kpm.SpectralDensity(fsyst)
#HIDDEN_END_kpm1

#HIDDEN_BEGIN_kpm2
    energies, densities = spectrum()
#HIDDEN_END_kpm2

#HIDDEN_BEGIN_kpm3
    energy_subset = np.linspace(0, 2)
    density_subset = spectrum(energy_subset)
#HIDDEN_END_kpm3

    plot_dos([
        ('densities', (energies, densities)),
        ('density subset', (energy_subset, density_subset)),
    ])


def dos_integrating_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)

#HIDDEN_BEGIN_int1
    print('identity resolution:', spectrum.integrate())
#HIDDEN_END_int1

#HIDDEN_BEGIN_int2
    # Fermi energy 0.1 and temperature 0.2
    fermi = lambda E: 1 / (np.exp((E - 0.1) / 0.2) + 1)

    print('number of filled states:', spectrum.integrate(fermi))
#HIDDEN_END_int2


def increasing_accuracy_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    original_dos = spectrum()  # get unaltered DoS

#HIDDEN_BEGIN_acc1
    spectrum.add_moments(energy_resolution=0.03)
#HIDDEN_END_acc1

    increased_resolution_dos = spectrum()

    plot_dos([
        ('density', original_dos),
        ('higher energy resolution', increased_resolution_dos),
    ])

#HIDDEN_BEGIN_acc2
    spectrum.add_moments(100)
    spectrum.add_vectors(5)
#HIDDEN_END_acc2

    increased_moments_dos = spectrum()

    plot_dos([
        ('density', original_dos),
        ('higher number of moments', increased_moments_dos),
    ])


def operator_example(fsyst):
#HIDDEN_BEGIN_op1
    # identity matrix
    matrix_op = scipy.sparse.eye(len(fsyst.sites))
    matrix_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=matrix_op)
#HIDDEN_END_op1

#HIDDEN_BEGIN_op2
    # 'sum=True' means we sum over all the sites
    kwant_op = kwant.operator.Density(fsyst, sum=True)
    operator_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)
#HIDDEN_END_op2

    plot_dos([
        ('identity matrix', matrix_spectrum()),
        ('kwant.operator.Density', operator_spectrum()),
    ])


def ldos_example(fsyst):
#HIDDEN_BEGIN_op3
    # 'sum=False' is the default, but we include it explicitly here for clarity.
    kwant_op = kwant.operator.Density(fsyst, sum=False)
    local_dos = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)
#HIDDEN_END_op3

#HIDDEN_BEGIN_op4
    zero_energy_ldos = local_dos(energy=0)
    finite_energy_ldos = local_dos(energy=1)

    _, axes = plt.subplots(1, 2, figsize=(12, 7))
    plot_ldos(fsyst, axes,[
        ('energy = 0', zero_energy_ldos),
        ('energy = 1', finite_energy_ldos),
    ])
#HIDDEN_END_op4


def vector_factory_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
#HIDDEN_BEGIN_fact1
    # construct vectors with n random elements -1 or +1.
    def binary_vectors(n):
        return np.rint(np.random.random_sample(n)) * 2 - 1

    custom_factory = kwant.kpm.SpectralDensity(fsyst,
                                               vector_factory=binary_vectors)
#HIDDEN_END_fact1
    plot_dos([
        ('default vector factory', spectrum()),
        ('binary vector factory', custom_factory()),
    ])


def bilinear_map_operator_example(fsyst):
#HIDDEN_BEGIN_blm
    rho = kwant.operator.Density(fsyst, sum=True)

    # sesquilinear map that does the same thing as `rho`
    def rho_alt(bra, ket):
        return np.vdot(bra, ket)

    rho_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho)
    rho_alt_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho_alt)
#HIDDEN_END_blm

    plot_dos([
        ('kwant.operator.Density', rho_spectrum()),
        ('bilinear operator', rho_alt_spectrum()),
    ])


def main():
    simple_dos_example()

    fsyst = make_syst().finalized()

    dos_integrating_example(fsyst)
    increasing_accuracy_example(fsyst)
    operator_example(fsyst)
    ldos_example(fsyst)
    vector_factory_example(fsyst)
    bilinear_map_operator_example(fsyst)


# Call the main function if the script gets executed (as opposed to imported).
# See <http://docs.python.org/library/__main__.html>.
if __name__ == '__main__':
    main()
