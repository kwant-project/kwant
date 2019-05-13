##############################################################
Calculating spectral density with the kernel polynomial method
##############################################################

We have already seen in the ":ref:`closed-systems`" tutorial that we can use
Kwant simply to build Hamiltonians, which we can then directly diagonalize
using routines from Scipy.

This already allows us to treat systems with a few thousand sites without too
many problems.  For larger systems one is often not so interested in the exact
eigenenergies and eigenstates, but more in the *density of states*.

The kernel polynomial method (KPM), is an algorithm to obtain a polynomial
expansion of the density of states. It can also be used to calculate the
spectral density of arbitrary operators.  Kwant has an implementation of the
KPM method `kwant.kpm`, that is based on the algorithms presented in Ref. [1]_.

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download:script:`kernel_polynomial_method`

.. jupyter-kernel::
    :id: kernel_polynomial_method

.. jupyter-execute::
    :hide-code:

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
    from matplotlib import pyplot

.. jupyter-execute:: boilerplate.py
    :hide-code:

Introduction
************

Our aim is to use the kernel polynomial method to obtain the spectral density
:math:`ρ_A(E)`, as a function of the energy :math:`E`, of some Hilbert space
operator :math:`A`.  We define

.. math::

    ρ_A(E) = ρ(E) A(E),

where :math:`A(E)` is the expectation value of :math:`A` for all the
eigenstates of the Hamiltonian with energy :math:`E`,  and the density of
states is

.. math::

  ρ(E) = \sum_{k=0}^{D-1} δ(E-E_k) = \mathrm{Tr}\left(\delta(E-H)\right),

where :math:`H` is the Hamiltonian of the system, :math:`D` the
Hilbert space dimension, and :math:`E_k` the eigenvalues.

In the special case when :math:`A` is the identity, then :math:`ρ_A(E)` is
simply :math:`ρ(E)`, the density of states.


Calculating the density of states
*********************************

Roughly speaking, KPM approximates the density of states, or any other spectral
density, by expanding the action of the Hamiltonian and operator of interest
on a small set of *random vectors* (or *local vectors* for local density of
states), as a sum of Chebyshev polynomials up to some order, and then averaging.
The accuracy of the method can be tuned by modifying the order of the Chebyshev
expansion and the number of vectors. See notes on accuracy_ below for details.

.. _accuracy:
.. specialnote:: Performance and accuracy

    The KPM method is especially well suited for large systems, and in the
    case when one is not interested in individual eigenvalues, but rather
    in obtaining an approximate spectral density.

    The accuracy in the energy resolution is dominated by the number of
    moments. The lowest accuracy is at the center of the spectrum, while
    slightly higher accuracy is obtained at the edges of the spectrum.
    If we use the KPM method (with the Jackson kernel, see Ref. [1]_) to
    describe a delta peak at the center of the spectrum, we will obtain a
    function similar to a Gaussian of width :math:`σ=πa/N`, where
    :math:`N` is the number of moments, and :math:`a` is the width of the
    spectrum.

    On the other hand, the random vectors will *explore* the range of the
    spectrum, and as the system gets bigger, the number of random vectors
    that are necessary to sample the whole spectrum reduces. Thus, a small
    number of random vectors is in general enough, and increasing this number
    will not result in a visible improvement of the approximation.

The global *spectral density* :math:`ρ_A(E)` is approximated by the stochastic
trace, the average expectation value of random vectors :math:`r`

.. math::

    ρ_A(E) = \mathrm{Tr}\left(A\delta(E-H)\right) \sim \frac{1}{R}
    \sum_r \langle r \lvert A \delta(E-H) \rvert r \rangle,

while the *local spectral density* for a site :math:`i` is

.. math::

    ρ^i_A(E) = \langle i \lvert A \delta(E-H) \rvert i \rangle,

which is an exact expression.


Global spectral densities using random vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

In the following example, we will use the KPM implementation in Kwant
to obtain the (global) density of states of a graphene disk.

We start by importing kwant and defining our system.

.. jupyter-execute::

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

.. jupyter-execute::
    :hide-code:

    ## common plotting routines ##

    # Plot several density of states curves on the same axes.
    def plot_dos(labels_to_data):
        pyplot.figure()
        for label, (x, y) in labels_to_data:
            pyplot.plot(x, y.real, label=label, linewidth=2)
        pyplot.legend(loc=2, framealpha=0.5)
        pyplot.xlabel("energy [t]")
        pyplot.ylabel("DoS [a.u.]")
        pyplot.show()


    # Plot fill density of states plus curves on the same axes.
    def plot_dos_and_curves(dos, labels_to_data):
        pyplot.figure()
        pyplot.fill_between(dos[0], dos[1], label="DoS [a.u.]",
                         alpha=0.5, color='gray')
        for label, (x, y) in labels_to_data:
            pyplot.plot(x, y, label=label, linewidth=2)
        pyplot.legend(loc=2, framealpha=0.5)
        pyplot.xlabel("energy [t]")
        pyplot.ylabel("$σ [e^2/h]$")
        pyplot.show()


    def site_size_conversion(densities):
        return 3 * np.abs(densities) / max(densities)


    # Plot several local density of states maps in different subplots
    def plot_ldos(syst, densities):
        fig, axes = pyplot.subplots(1, len(densities), figsize=(7*len(densities), 7))
        for ax, (title, rho) in zip(axes, densities):
            kwant.plotter.density(syst, rho.real, ax=ax)
            ax.set_title(title)
            ax.set(adjustable='box', aspect='equal')
        pyplot.show()

After making a system we can then create a `~kwant.kpm.SpectralDensity`
object that represents the density of states for this system.

.. jupyter-execute::

    fsyst = make_syst().finalized()

    spectrum = kwant.kpm.SpectralDensity(fsyst)

The `~kwant.kpm.SpectralDensity` can then be called like a function to obtain a
sequence of energies in the spectrum of the Hamiltonian, and the corresponding
density of states at these energies.

.. jupyter-execute::

    energies, densities = spectrum()

When called with no arguments, an optimal set of energies is chosen (these are
not evenly distributed over the spectrum, see Ref. [1]_ for details), however
it is also possible to provide an explicit sequence of energies at which to
evaluate the density of states.

.. jupyter-execute::

    energy_subset = np.linspace(0, 2)
    density_subset = spectrum(energy_subset)

.. jupyter-execute::
    :hide-code:

    plot_dos([
        ('densities', (energies, densities)),
        ('density subset', (energy_subset, density_subset)),
    ])

In addition to being called like functions, `~kwant.kpm.SpectralDensity`
objects also have a method `~kwant.kpm.SpectralDensity.integrate` which can be
used to integrate the density of states against some distribution function over
the whole spectrum. If no distribution function is specified, then the uniform
distribution is used:

.. jupyter-execute::

    print('identity resolution:', spectrum.integrate())

We see that the integral of the density of states is normalized to the total
number of available states in the system. If we wish to calculate, say, the
number of states populated in equilibrium, then we should integrate with
respect to a Fermi-Dirac distribution:

.. jupyter-execute::

    # Fermi energy 0.1 and temperature 0.2
    fermi = lambda E: 1 / (np.exp((E - 0.1) / 0.2) + 1)

    print('number of filled states:', spectrum.integrate(fermi))

.. specialnote:: Stability and performance: spectral bounds

    The KPM method internally rescales the spectrum of the Hamiltonian to the
    interval ``(-1, 1)`` (see Ref. [1]_ for details), which requires calculating
    the boundaries of the spectrum (using ``scipy.sparse.linalg.eigsh``). This
    can be very costly for large systems, so it is possible to pass this
    explicitly as via the ``bounds`` parameter when instantiating the
    `~kwant.kpm.SpectralDensity` (see the class documentation for details).

    Additionally, `~kwant.kpm.SpectralDensity` accepts a parameter ``epsilon``,
    which ensures that the rescaled Hamiltonian (used internally), always has a
    spectrum strictly contained in the interval ``(-1, 1)``. If bounds are not
    provided then the tolerance on the bounds calculated with
    ``scipy.sparse.linalg.eigsh`` is set to ``epsilon/2``.


Local spectral densities using local vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

The *local density of states* can be obtained without
using random vectors, and using local vectors instead. This approach is best
when we want to estimate the local density on a small number of sites of
the system. The accuracy of this approach depends only on the number of
moments, but the computational cost increases linearly with the number of
sites sampled.

To output local densities for each local vector, and not the average,
we set the parameter ``mean=False``, and the local vectors will be created
with the `~kwant.kpm.LocalVectors` generator (see advanced_topics_ for details).

The spectral density can be restricted to the expectation value inside
a region of the system by passing a ``where`` function or list of sites
to the `~kwant.kpm.RandomVectors` or `~kwant.kpm.LocalVectors` generators.

In the following example, we compute the local density of states at the center
of the graphene disk, and we add a staggering potential between the two
sublattices.

.. jupyter-execute::

    def make_syst_staggered(r=30, t=-1, a=1, m=0.1):
        syst = kwant.Builder()
        lat = kwant.lattice.honeycomb(a, norbs=1)

        def circle(pos):
            x, y = pos
            return x ** 2 + y ** 2 < r ** 2

        syst[lat.a.shape(circle, (0, 0))] = m
        syst[lat.b.shape(circle, (0, 0))] = -m
        syst[lat.neighbors()] = t
        syst.eradicate_dangling()

        return syst

Next, we choose one site of each sublattice "A" and "B",

.. jupyter-execute::

    fsyst_staggered = make_syst_staggered().finalized()
    # find 'A' and 'B' sites in the unit cell at the center of the disk
    center_tag = np.array([0, 0])
    where = lambda s: s.tag == center_tag
    # make local vectors
    vector_factory = kwant.kpm.LocalVectors(fsyst_staggered, where)

and plot their respective local density of states.

.. jupyter-execute::

    # 'num_vectors' can be unspecified when using 'LocalVectors'
    local_dos = kwant.kpm.SpectralDensity(fsyst_staggered, num_vectors=None,
                                          vector_factory=vector_factory,
                                          mean=False)
    energies, densities = local_dos()

.. jupyter-execute::
    :hide-code:

    plot_dos([
        ('A sublattice', (energies, densities[:, 0])),
        ('B sublattice', (energies, densities[:, 1])),
    ])

Note that there is no noise comming from the random vectors.


Increasing the accuracy of the approximation
********************************************

`~kwant.kpm.SpectralDensity` has two methods for increasing the accuracy
of the method, each of which offers different levels of control over what
exactly is changed.

The simplest way to obtain a more accurate solution is to use the
``add_moments`` method:

.. jupyter-execute::
    :hide-code:

    spectrum = kwant.kpm.SpectralDensity(fsyst)
    original_dos = spectrum()

.. jupyter-execute::

   spectrum.add_moments(energy_resolution=0.03)

This will update the number of calculated moments and also the default
number of sampling points such that the maximum distance between successive
energy points is ``energy_resolution`` (see notes on accuracy_).

Alternatively, you can directly increase the number of moments
with ``add_moments``, or the number of random vectors with ``add_vectors``.
The added vectors will be generated with the ``vector_factory``.

.. jupyter-execute::

    spectrum.add_moments(100)
    spectrum.add_vectors(5)

.. jupyter-execute::
    :hide-code:

    increased_moments_dos = spectrum()
    plot_dos([
        ('density', original_dos),
        ('higher number of moments', increased_moments_dos),
    ])

.. _operator_spectral_density:


Calculating the spectral density of an operator
***********************************************

Above, we saw how to calculate the density of states by creating a
`~kwant.kpm.SpectralDensity` and passing it a finalized Kwant system.
When instantiating a `~kwant.kpm.SpectralDensity` we may optionally
supply an operator in addition to the system. In this case it is
the spectral density of the given operator that is calculated.

`~kwant.kpm.SpectralDensity` accepts the operators in a few formats:

* *explicit matrices* (numpy array or scipy sparse matrices will work)
* *operators* from `kwant.operator`

If an explicit matrix is provided then it must have the same
shape as the system Hamiltonian.

.. jupyter-execute::

    # identity matrix
    matrix_op = scipy.sparse.eye(len(fsyst.sites))
    matrix_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=matrix_op)

Or, to do the same calculation using `kwant.operator.Density`:

.. jupyter-execute::

    # 'sum=True' means we sum over all the sites
    kwant_op = kwant.operator.Density(fsyst, sum=True)
    operator_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)

Spectral density with random vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

Using operators from `kwant.operator` allows us to calculate quantities
such as the *local* density of states by telling the operator not to
sum over all the sites of the system:

.. jupyter-execute::

    # 'sum=False' is the default, but we include it explicitly here for clarity.
    kwant_op = kwant.operator.Density(fsyst, sum=False)
    local_dos = kwant.kpm.SpectralDensity(fsyst, operator=kwant_op)

`~kwant.kpm.SpectralDensity` will properly handle this vector output,
and will average the local density obtained with random vectors.

The accuracy of this approximation depends on the number of random vectors used.
This allows us to plot an approximate local density of states at different
points in the spectrum:

.. jupyter-execute::

    zero_energy_ldos = local_dos(energy=0)
    finite_energy_ldos = local_dos(energy=1)

.. jupyter-execute::
    :hide-code:

    plot_ldos(fsyst, [
        ('energy = 0', zero_energy_ldos),
        ('energy = 1', finite_energy_ldos)
    ])

Calculating Kubo conductivity
*****************************

The Kubo conductivity can be calculated for a closed system with two
KPM expansions. In `~kwant.kpm.Conductivity` we implemented the
Kubo-Bastin formula of the conductivity and any temperature (see Ref. [2]_).
With the help of `~kwant.kpm.Conductivity`,
we can calculate any element of the conductivity tensor :math:`σ_{αβ}`,
that relates the applied electric field to the expected current.

.. math::

    j_α = σ_{α, β} E_β

In the following example, we will calculate the longitudinal
conductivity :math:`σ_{xx}` and the Hall conductivity
:math:`σ_{xy}`, for the Haldane model. This model is the first
and one of the most simple ones for a topological insulator.

The Haldane model consist of a honeycomb lattice, similar to graphene,
with nearest neigbours hoppings. To turn it into a topological
insulator we add imaginary second nearest neigbours hoppings, where
the sign changes for each sublattice.

.. jupyter-execute::

    def make_syst_topo(r=30, a=1, t=1, t2=0.5):
        syst = kwant.Builder()
        lat = kwant.lattice.honeycomb(a, norbs=1, name=['a', 'b'])

        def circle(pos):
            x, y = pos
            return x ** 2 + y ** 2 < r ** 2

        syst[lat.shape(circle, (0, 0))] = 0.
        syst[lat.neighbors()] = t
        # add second neighbours hoppings
        syst[lat.a.neighbors()] = 1j * t2
        syst[lat.b.neighbors()] = -1j * t2
        syst.eradicate_dangling()

        return lat, syst.finalized()

To calculate the bulk conductivity, we will select sites in the unit cell
in the middle of the sample, and create a vector factory that outputs local
vectors

.. jupyter-execute::

    # construct the Haldane model
    lat, fsyst_topo = make_syst_topo()
    # find 'A' and 'B' sites in the unit cell at the center of the disk
    where = lambda s: np.linalg.norm(s.pos) < 1

    # component 'xx'
    s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
    cond_xx = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='x', mean=True,
                                     num_vectors=None, vector_factory=s_factory)
    # component 'xy'
    s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
    cond_xy = kwant.kpm.conductivity(fsyst_topo, alpha='x', beta='y', mean=True,
                                     num_vectors=None, vector_factory=s_factory)

    energies = cond_xx.energies
    cond_array_xx = np.array([cond_xx(e, temperature=0.01) for e in energies])
    cond_array_xy = np.array([cond_xy(e, temperature=0.01) for e in energies])

    # area of the unit cell per site
    area_per_site = np.abs(np.cross(*lat.prim_vecs)) / len(lat.sublattices)
    cond_array_xx /= area_per_site
    cond_array_xy /= area_per_site

Note that the Kubo conductivity must be normalized with the area covered
by the vectors. In this case, each local vector represents a site, and
covers an area of half a unit cell, while the sum covers one unit cell.
It is possible to use random vectors to get an average expectation
value of the conductivity over large parts of the system. In this
case, the area that normalizes the result, is the area covered by
the random vectors.

.. jupyter-execute::
    :hide-code:


    s_factory = kwant.kpm.LocalVectors(fsyst_topo, where)
    spectrum = kwant.kpm.SpectralDensity(fsyst_topo, num_vectors=None,
                                         vector_factory=s_factory)

    plot_dos_and_curves(
    (spectrum.energies, spectrum.densities * 8),
    [
        (r'Longitudinal conductivity $\sigma_{xx} / 4$',
         (energies, cond_array_xx / 4)),
        (r'Hall conductivity $\sigma_{xy}$',
         (energies, cond_array_xy))],
    )

.. _advanced_topics:


Advanced topics
***************

Custom distributions of vectors
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
By default `~kwant.kpm.SpectralDensity` will use random vectors
whose components are unit complex numbers with phases drawn
from a uniform distribution. The generator is accesible through
`~kwant.kpm.RandomVectors`.

For large systems, one will generally resort to random vectors to sample the
Hilbert space of the system. There are several reasons why you may
wish to make a different choice of distribution for your random vectors,
for example to enforce certain symmetries or to only use real-valued vectors.

To change how the random vectors are generated, you need only specify a
function that takes the dimension of the Hilbert space as a single parameter,
and which returns a vector in that Hilbert space:

.. jupyter-execute::

    # construct a generator of vectors with n random elements -1 or +1.
    n = fsyst.hamiltonian_submatrix(sparse=True).shape[0]
    def binary_vectors():
        while True:
           yield np.rint(np.random.random_sample(n)) * 2 - 1

    custom_factory = kwant.kpm.SpectralDensity(fsyst,
                                               vector_factory=binary_vectors())

Aditionally, a `~kwant.kpm.LocalVectors` generator is also available, that
returns local vectors that correspond to the sites passed. Note that
the vectors generated match the sites in order, and will be exhausted
after all vectors are drawn from the factory.

Both `~kwant.kpm.RandomVectors` and `~kwant.kpm.LocalVectors` take the
argument ``where``, that restricts the non zero values of the vectors
generated to the sites in ``where``.


Reproducible calculations
^^^^^^^^^^^^^^^^^^^^^^^^^
Because KPM internally uses random vectors, running the same calculation
twice will not give bit-for-bit the same result. However, similarly to
the funcions in `~kwant.rmt`, the random number generator can be directly
manipulated by passing a value to the ``rng`` parameter of
`~kwant.kpm.SpectralDensity`. ``rng`` can itself be a random number generator,
or it may simply be a seed to pass to the numpy random number generator
(that is used internally by default).

Defining operators as sesquilinear maps
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
`Above`__, we showed how `~kwant.kpm.SpectralDensity` can calculate the
spectral density of operators, and how we can define operators by using
`kwant.operator`.  If you need even more flexibility,
`~kwant.kpm.SpectralDensity` will also accept a *function* as its ``operator``
parameter. This function must itself take two parameters, ``(bra, ket)`` and
must return either a scalar or a one-dimensional array. In order to be
meaningful the function must be a sesquilinear map, i.e. antilinear in its
first argument, and linear in its second argument. Below, we compare two
methods for computing the local density of states, one using
`kwant.operator.Density`, and the other using a custom function.

.. jupyter-execute::

    rho = kwant.operator.Density(fsyst, sum=True)

    # sesquilinear map that does the same thing as `rho`
    def rho_alt(bra, ket):
        return np.vdot(bra, ket)

    rho_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho)
    rho_alt_spectrum = kwant.kpm.SpectralDensity(fsyst, operator=rho_alt)

__ operator_spectral_density_


.. rubric:: References

.. [1] `Rev. Mod. Phys., Vol. 78, No. 1 (2006)
    <https://arxiv.org/abs/cond-mat/0504627>`_.
.. [2] `Phys. Rev. Lett. 114, 116602 (2015)
    <https://arxiv.org/abs/1410.8140>`_.
