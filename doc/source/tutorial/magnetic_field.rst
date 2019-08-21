Adding magnetic field
---------------------

Computing Landau levels in a harmonic oscillator basis
......................................................

.. seealso::
    You can execute the code examples live in your browser by
    activating thebelab:

    .. thebe-button:: Activate Thebelab

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download:script:`landau-levels`

.. jupyter-kernel::
    :id: landau-levels

When electrons move in an external magnetic field, their motion perpendicular
to the field direction is quantized into discrete Landau levels. Kwant implements
an efficient scheme for computing the Landau levels of arbitrary continuum
Hamiltonians. The general scheme revolves around rewriting the Hamiltonian in terms
of a basis of harmonic oscillator states [#]_, and is commonly illustrated in textbooks
for quadratic Hamiltonians.

To demonstrate the general scheme, let us consider a magnetic field oriented along
the :math:`z` direction :math:`\vec{B} = (0, 0, B)`, such that electron motion
in the :math:`xy` plane is Landau quantized. The magnetic field enters the Hamiltonian
through the kinetic momentum

.. math:: \hbar \vec{k} = - i \hbar \nabla + e\vec{A}(x, y).

In the symmetric gauge :math:`\vec{A}(x, y) = (B/2)[-y, x, 0]`, we introduce ladder
operators with the substitution

.. math::

    k_x = \frac{1}{\sqrt{2} l_B} (a + a^\dagger), \quad \quad
    k_y = \frac{i}{\sqrt{2} l_B} (a - a^\dagger),

with the magnetic length :math:`l_B = \sqrt{\hbar/eB}`. The ladder operators obey the
commutation relation

.. math:: [a, a^\dagger] = 1,

and define a quantum harmonic oscillator. We can thus write any electron continuum
Hamiltonian in terms of :math:`a` and :math:`a^\dagger`. Such a Hamiltonian has a
simple matrix representation in the eigenbasis of the number operator :math:`a^\dagger a`.
The eigenstates satisfy :math:`a^\dagger a | n \rangle = n | n \rangle` with the integer
Landau level index :math:`n \geq 0`, and in coordinate representation are proportional to

.. math::

    \psi_n (x, y) = \left( \frac{\partial}{ \partial w} - \frac{w^*}{4 l_B^2} \right)
    w^n e^{-|w|^2/4l_B^2},

with :math:`w = x + i y`. The matrix elements of the ladder operators are

.. math::

    \langle n | a | m \rangle = \sqrt{m}~\delta_{n, m-1}, \quad \quad
    \langle n | a^\dagger | m \rangle = \sqrt{m + 1}~\delta_{n, m+1}.

Truncating the basis to the first :math:`N` Landau levels allows us to approximate
the Hamiltonian as a discrete, finite matrix.

We can now formulate the algorithm that Kwant uses to compute Landau levels.

    1. We take a generic continuum Hamiltonian, written in terms of the kinetic
    momentum :math:`\vec{k}`. The Hamiltonian must be translationally
    invariant along the directions perpendicular to the field direction.

    2. We substitute the momenta perpendicular to the magnetic field with the ladder
    operators :math:`a` and :math:`a^\dagger`.

    3. We construct a `kwant.builder.Builder` using a special lattice which includes
    the Landau level index :math:`n` as a degree of freedom on each site. The directions
    normal to the field direction are not included in the builder, because they are
    encoded in the Landau level index.

This procedure is automated with `kwant.continuum.discretize_landau`.

As an example, let us take the Bernevig-Hughes-Zhang model that we first considered in the
discretizer tutorial ":ref:`discretize-bhz-model`":

.. math::

    C + M σ_0 \otimes σ_z + F(k_x^2 + k_y^2) σ_0 \otimes σ_z + D(k_x^2 + k_y^2) σ_0 \otimes σ_0
    + A k_x σ_z \otimes σ_x + A k_y σ_0 \otimes σ_y.

We can discretize this Hamiltonian in a basis of Landau levels as follows

.. jupyter-execute::

    import numpy as np
    import scipy.linalg
    from matplotlib import pyplot

    import kwant
    import kwant.continuum

.. jupyter-execute:: boilerplate.py
    :hide-code:

.. jupyter-execute::

    hamiltonian = """
       + C * identity(4) + M * kron(sigma_0, sigma_z)
       - F * (k_x**2 + k_y**2) * kron(sigma_0, sigma_z)
       - D * (k_x**2 + k_y**2) * kron(sigma_0, sigma_0)
       + A * k_x * kron(sigma_z, sigma_x)
       - A * k_y * kron(sigma_0, sigma_y)
    """

    syst = kwant.continuum.discretize_landau(hamiltonian, N=10)
    syst = syst.finalized()

We can then plot the spectrum of the system as a function of magnetic field, and
observe a crossing of Landau levels at finite magnetic field near zero energy,
characteristic of a quantum spin Hall insulator with band inversion.

.. jupyter-execute::

    params = dict(A=3.645, F =-68.6, D=-51.2, M=-0.01, C=0)
    b_values = np.linspace(0.0001, 0.0004, 200)

    fig = kwant.plotter.spectrum(syst, ('B', b_values), params=params, show=False)
    pyplot.ylim(-0.1, 0.2);


Comparing with tight-binding
============================
In the limit where fewer than one quantum of flux is threaded through a plaquette of
the discretization lattice we can compare the discretization in Landau levels with
a discretization in realspace.

.. jupyter-execute::

    lat = kwant.lattice.square()
    syst = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))

    def peierls(to_site, from_site, B):
        y = from_site.tag[1]
        return -1 * np.exp(-1j * B * y)

    syst[(lat(0, j) for j in range(-19, 20))] = 4
    syst[lat.neighbors()] = -1
    syst[kwant.HoppingKind((1, 0), lat)] = peierls
    syst = syst.finalized()

    landau_syst = kwant.continuum.discretize_landau("k_x**2 + k_y**2", N=5)
    landau_syst = landau_syst.finalized()

Here we plot the dispersion relation for the discretized ribbon and compare it
with the Landau levels shown as dashed lines.

.. jupyter-execute::

    fig, ax = pyplot.subplots(1, 1)
    ax.set_xlabel("momentum")
    ax.set_ylabel("energy")
    ax.set_ylim(0, 1)

    params = dict(B=0.1)

    kwant.plotter.bands(syst, ax=ax, params=params)

    h = landau_syst.hamiltonian_submatrix(params=params)
    for ev in scipy.linalg.eigvalsh(h):
      ax.axhline(ev, linestyle='--')

The dispersion and the Landau levels diverge with increasing energy, because the real space
discretization of the ribbon gives a worse approximation to the dispersion at higher energies.


Discretizing 3D models
======================
Although the preceding examples have only included the plane perpendicular to the
magnetic field, the Landau level quantization also works if the direction
parallel to the field is included. In fact, although the system must be
translationally invariant in the plane perpendicular to the field, the system may
be arbitrary along the parallel direction. For example, it is therefore possible to
model a heterostructure and/or set up a scattering problem along the field direction.

Let's say that we wish to to model a heterostructure with a varying potential
:math:`V` along the direction of a magnetic field, :math:`z`, that includes
Zeeman splitting and Rashba spin-orbit coupling:

.. math::

    \frac{\hbar^2}{2m}\sigma_0(k_x^2 + k_y^2 + k_z^2)
    + V(z)\sigma_0
    + \frac{\mu_B B}{2}\sigma_z
    + \hbar\alpha(\sigma_x k_y - \sigma_y k_x).

We can discretize this Hamiltonian in a basis of Landau levels as before:

.. jupyter-execute::

    continuum_hamiltonian = """
        (k_x**2 + k_y**2 + k_z**2) * sigma_0
        + V(z) * sigma_0
        + mu * B * sigma_z / 2
        + alpha * (sigma_x * k_y - sigma_y * k_x)
    """

    template = kwant.continuum.discretize_landau(continuum_hamiltonian, N=10)

This creates a system with a single translational symmetry, along
the :math:`z` direction, which we can use as a template
to construct our heterostructure:

.. jupyter-execute::

    def hetero_structure(site):
        z, = site.pos
        return 0 <= z < 10

    def hetero_potential(z):
        if z < 2:
          return 0
        elif z < 7:
          return 0.5
        else:
          return 0.7

    syst = kwant.Builder()
    syst.fill(template, hetero_structure, (0,))

    syst = syst.finalized()

    params = dict(
        B=0.5,
        mu=0.2,
        alpha=0.4,
        V=hetero_potential,
    )

    syst.hamiltonian_submatrix(params=params);


.. rubric:: Footnotes

.. [#] `Wikipedia <https://en.wikipedia.org/wiki/Landau_quantization>`_ has
    a nice introduction to Landau quantization.
