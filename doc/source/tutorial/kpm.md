Calculating spectral density with the kernel polynomial method
==============================================================

We have already seen in the \"[closed-systems]{role="ref"}\" tutorial
that we can use Kwant simply to build Hamiltonians, which we can then
directly diagonalize using routines from Scipy.

This already allows us to treat systems with a few thousand sites
without too many problems. For larger systems one is often not so
interested in the exact eigenenergies and eigenstates, but more in the
*density of states*.

The kernel polynomial method (KPM), is an algorithm to obtain a
polynomial expansion of the density of states. It can also be used to
calculate the spectral density of arbitrary operators. Kwant has an
implementation of the KPM method that is based on the algorithms
presented in Ref.[^1].

Roughly speaking, KPM approximates the density of states (or any other
spectral density) by expanding the action of the Hamiltonian (and
operator of interest) on a (small) set of *random vectors* as a sum of
Chebyshev polynomials up to some order, and then averaging. The accuracy
of the method can be tuned by modifying the order of the Chebyshev
expansion and the number of random vectors. See notes on
[accuracy](..%20specialnote::%20Performance%20and%20accuracy) below for
details.

::: {.seealso}
The complete source code of this example can be found in
[kernel\_polynomial\_method.py \</code/download/kernel\_polynomial\_method.py\>]{role="download"}

The KPM method is especially well suited for large systems, and in the
case when one is not interested in individual eigenvalues, but rather in
obtaining an approximate spectral density.

The accuracy in the energy resolution is dominated by the number of
moments. The lowest accuracy is at the center of the spectrum, while
slightly higher accuracy is obtained at the edges of the spectrum. If we
use the KPM method (with the Jackson kernel, see Ref.[^2]) to describe a
delta peak at the center of the spectrum, we will obtain a function
similar to a Gaussian of width $σ=πa/N$, where $N$ is the number of
moments, and $a$ is the width of the spectrum.

On the other hand, the random vectors will *explore* the range of the
spectrum, and as the system gets bigger, the number of random vectors
that are necessary to sample the whole spectrum reduces. Thus, a small
number of random vectors is in general enough, and increasing this
number will not result in a visible improvement of the approximation.
:::

Introduction
------------

Our aim is to use the kernel polynomial method to obtain the spectral
density $ρ_A(E)$, as a function of the energy $E$, of some Hilbert space
operator $A$. We define

$$ρ_A(E) = ρ(E) A(E),$$

where $A(E)$ is the expectation value of $A$ for all the eigenstates of
the Hamiltonian with energy $E$, and the density of states is

$$ρ(E) = \sum_{k=0}^{D-1} δ(E-E_k),$$

$D$ being the Hilbert space dimension, and $E_k$ the eigenvalues.

In the special case when $A$ is the identity, then $ρ_A(E)$ is simply
$ρ(E)$, the density of states.

Calculating the density of states
---------------------------------

In the following example, we will use the KPM implementation in Kwant to
obtain the density of states of a graphene disk.

We start by importing kwant and defining our system.

```python
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

```

After making a system we can then create a
`~kwant.kpm.SpectralDensity` object that represents the density
of states for this system.

```python



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
    fsyst = make_syst().finalized()

    spectrum = kwant.kpm.SpectralDensity(fsyst)

```

The `~kwant.kpm.SpectralDensity` can then be called like a
function to obtain a sequence of energies in the spectrum of the
Hamiltonian, and the corresponding density of states at these energies.

```python


    energies, densities = spectrum()

```

When called with no arguments, an optimal set of energies is chosen
(these are not evenly distributed over the spectrum, see Ref.[^3] for
details), however it is also possible to provide an explicit sequence of
energies at which to evaluate the density of states.

```python


    energy_subset = np.linspace(0, 2)
    density_subset = spectrum(energy_subset)

```

![image](/code/figure/kpm_dos.*)

In addition to being called like functions,
`~kwant.kpm.SpectralDensity` objects also have a method
`~kwant.kpm.SpectralDensity.integrate` which can be used to
integrate the density of states against some distribution function over
the whole spectrum. If no distribution function is specified, then the
uniform distribution is used:

```python


    plot_dos([
        ('densities', (energies, densities)),
        ('density subset', (energy_subset, density_subset)),
    ])


def dos_integrating_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)

    print('identity resolution:', spectrum.integrate())

```

::: {.literalinclude}
/code/figure/kpm\_normalization.txt
:::

We see that the integral of the density of states is normalized to the
total number of available states in the system. If we wish to calculate,
say, the number of states populated in equilibrium, then we should
integrate with respect to a Fermi-Dirac distribution:

```python


    # Fermi energy 0.1 and temperature 0.2
    fermi = lambda E: 1 / (np.exp((E - 0.1) / 0.2) + 1)

    print('number of filled states:', spectrum.integrate(fermi))

```

::: {.literalinclude}
/code/figure/kpm\_total\_states.txt
:::

::: {.specialnote}
Stability and performance: spectral bounds

The KPM method internally rescales the spectrum of the Hamiltonian to
the interval ``(-1, 1)`` (see Ref[^4] for details), which requires
calculating the boundaries of the spectrum (using
``scipy.sparse.linalg.eigsh``). This can be very costly for large systems,
so it is possible to pass this explicitly as via the ``bounds`` parameter
when instantiating the `~kwant.kpm.SpectralDensity` (see the
class documentation for details).

Additionally, `~kwant.kpm.SpectralDensity` accepts a parameter
``epsilon``, which ensures that the rescaled Hamiltonian (used
internally), always has a spectrum strictly contained in the interval
``(-1, 1)``. If bounds are not provided then the tolerance on the bounds
calculated with ``scipy.sparse.linalg.eigsh`` is set to ``epsilon/2``.
:::

Increasing the accuracy of the approximation
--------------------------------------------

`~kwant.kpm.SpectralDensity` has two methods for increasing the
accuracy of the method, each of which offers different levels of control
over what exactly is changed.

The simplest way to obtain a more accurate solution is to use the
``add_moments`` method:

```python



def increasing_accuracy_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    original_dos = spectrum()  # get unaltered DoS

    spectrum.add_moments(energy_resolution=0.03)

```

This will update the number of calculated moments and also the default
number of sampling points such that the maximum distance between
successive energy points is ``energy_resolution`` (see notes on
[accuracy](..%20specialnote::%20Performance%20and%20accuracy)).

![image](/code/figure/kpm_dos_acc.*)

Alternatively, you can directly increase the number of moments with
``add_moments``, or the number of random vectors with ``add_vectors``.

```python


    increased_resolution_dos = spectrum()

    plot_dos([
        ('density', original_dos),
        ('higher energy resolution', increased_resolution_dos),
    ])

    spectrum.add_moments(100)
    spectrum.add_vectors(5)

```

![image](/code/figure/kpm_dos_r.*)

Calculating the spectral density of an operator {#operator_spectral_density}
-----------------------------------------------

Above, we saw how to calculate the density of states by creating a
`~kwant.kpm.SpectralDensity` and passing it a finalized Kwant
system. When instantiating a `~kwant.kpm.SpectralDensity` we may
optionally supply an operator in addition to the system. In this case it
is the spectral density of the given operator that is calculated.

`~kwant.kpm.SpectralDensity` accepts the operators in a few
formats:

-   *explicit matrices* (numpy array of scipy sparse matrices will work)
-   *operators* from `kwant.operator`

If an explicit matrix is provided then it must have the same shape as
the system Hamiltonian.

```python


    increased_moments_dos = spectrum()

    plot_dos([
        ('density', original_dos),
        ('higher number of moments', increased_moments_dos),
    ])


def operator_example(fsyst):
    # identity matrix
    matrix_op = scipy.sparse.eye(len(fsyst.sites))
    matrix_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=matrix_op)

```

Or, to do the same calculation using `kwant.operator.Density`:

```python


    # 'sum=True' means we sum over all the sites
    kwant_op = kwant.operator.Density(fsyst, sum=True)
    operator_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)

```

Using operators from `kwant.operator` allows us to calculate
quantities such as the *local* density of states by telling the operator
not to sum over all the sites of the system:

```python


    plot_dos([
        ('identity matrix', matrix_spectrum()),
        ('kwant.operator.Density', operator_spectrum()),
    ])


def ldos_example(fsyst):
    # 'sum=False' is the default, but we include it explicitly here for clarity.
    kwant_op = kwant.operator.Density(fsyst, sum=False)
    local_dos = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)

```

`~kwant.kpm.SpectralDensity` will properly handle this vector
output, which allows us to plot the local density of states at different
point in the spectrum:

```python


    zero_energy_ldos = local_dos(energy=0)
    finite_energy_ldos = local_dos(energy=1)

    _, axes = plt.subplots(1, 2, figsize=(12, 7))
    plot_ldos(fsyst, axes,[
        ('energy = 0', zero_energy_ldos),
        ('energy = 1', finite_energy_ldos),
    ])

```

![image](/code/figure/kpm_ldos.*)

This nicely illustrates the edge states of the graphene dot at zero
energy, and the bulk states at higher energy.

Advanced topics
---------------

### Custom distributions for random vectors

By default `~kwant.kpm.SpectralDensity` will use random vectors
whose components are unit complex numbers with phases drawn from a
uniform distribution. There are several reasons why you may wish to make
a different choice of distribution for your random vectors, for example
to enforce certain symmetries or to only use real-valued vectors.

To change how the random vectors are generated, you need only specify a
function that takes the dimension of the Hilbert space as a single
parameter, and which returns a vector in that Hilbert space:

```python



def vector_factory_example(fsyst):
    spectrum = kwant.kpm.SpectralDensity(fsyst)
    # construct vectors with n random elements -1 or +1.
    def binary_vectors(n):
        return np.rint(np.random.random_sample(n)) * 2 - 1

    custom_factory = kwant.kpm.SpectralDensity(fsyst,
                                               vector_factory=binary_vectors)

```

### Reproducible calculations

Because KPM internally uses random vectors, running the same calculation
twice will not give bit-for-bit the same result. However, similarly to
the funcions in `~kwant.rmt`, the random number generator can be
directly manipulated by passing a value to the ``rng`` parameter of
`~kwant.kpm.SpectralDensity`. ``rng`` can itself be a random number
generator, or it may simply be a seed to pass to the numpy random number
generator (that is used internally by default).

### Defining operators as sesquilinear maps

[Above](#operator_spectral_density), we showed how
`~kwant.kpm.SpectralDensity` can calculate the spectral density
of operators, and how we can define operators by using
`kwant.operator`. If you need even more flexibility,
`~kwant.kpm.SpectralDensity` will also accept a *function* as its
``operator`` parameter. This function must itself take two parameters,
``(bra, ket)`` and must return either a scalar or a one-dimensional array.
In order to be meaningful the function must be a sesquilinear map, i.e.
antilinear in its first argument, and linear in its second argument.
Below, we compare two methods for computing the local density of states,
one using `kwant.operator.Density`, and the other using a custom
function.

```python

    plot_dos([
        ('default vector factory', spectrum()),
        ('binary vector factory', custom_factory()),
    ])


def bilinear_map_operator_example(fsyst):
    rho = kwant.operator.Density(fsyst, sum=True)

    # sesquilinear map that does the same thing as `rho`
    def rho_alt(bra, ket):
        return np.vdot(bra, ket)

    rho_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho)
    rho_alt_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho_alt)


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

```

**References**

[^1]: [Rev. Mod. Phys., Vol. 78, No. 1
    (2006)](https://arxiv.org/abs/cond-mat/0504627).

[^2]: [Rev. Mod. Phys., Vol. 78, No. 1
    (2006)](https://arxiv.org/abs/cond-mat/0504627).

[^3]: [Rev. Mod. Phys., Vol. 78, No. 1
    (2006)](https://arxiv.org/abs/cond-mat/0504627).

[^4]: [Rev. Mod. Phys., Vol. 78, No. 1
    (2006)](https://arxiv.org/abs/cond-mat/0504627).
