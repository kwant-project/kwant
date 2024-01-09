Superconductors: orbital degrees of freedom, conservation laws and symmetries
-----------------------------------------------------------------------------

.. seealso::
    You can execute the code examples live in your browser by
    activating thebelab:

    .. thebe-button:: Activate Thebelab

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download:script:`superconductor`

.. jupyter-kernel::
    :id: superconductor

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.6. "Superconductors": orbitals, conservation laws and symmetries
    # ===========================================================================
    #
    # Physics background
    # ------------------
    # - conductance of a NS-junction (Andreev reflection, superconducting gap)
    #
    # Kwant features highlighted
    # --------------------------
    # - Implementing electron and hole ("orbital") degrees of freedom
    #   using conservation laws.
    # - Use of discrete symmetries to relate scattering states.

    import kwant

    from matplotlib import pyplot

    import tinyarray
    import numpy as np

    tau_x = tinyarray.array([[0, 1], [1, 0]])
    tau_y = tinyarray.array([[0, -1j], [1j, 0]])
    tau_z = tinyarray.array([[1, 0], [0, -1]])

.. jupyter-execute:: boilerplate.py
    :hide-code:

This example deals with superconductivity on the level of the
Bogoliubov-de Gennes (BdG) equation. In this framework, the Hamiltonian
is given as

.. math::

    H = \begin{pmatrix}
            H_0 - \mu      & \Delta \\
            \Delta^\dagger & \mu - \mathcal{T} H_0 \mathcal{T}^{-1}
        \end{pmatrix}

where :math:`H_0` is the Hamiltonian of the system without
superconductivity, :math:`\mu` the chemical potential, :math:`\Delta`
the superconducting order parameter, and :math:`\mathcal{T}`
the time-reversal operator. The BdG Hamiltonian introduces
electron and hole degrees of freedom (an artificial doubling -
be aware of the fact that electron and hole excitations
are related!), which we will need to include in our model with Kwant.

For this we restrict ourselves to a simple spinless system without
magnetic field, so that :math:`\Delta` is just a number (which we
choose real), and :math:`\mathcal{T}H_0\mathcal{T}^{-1}=H_0^*=H_0`.
Furthermore, note that the Hamiltonian has particle-hole symmetry
:math:`\mathcal{P}`, i. e. :math:`\mathcal{P}H\mathcal{P}^{-1}=-H`.

Care must be taken when transport calculations are done with
the BdG equation. Electrons and holes carry charge with
opposite sign, such that it is necessary to separate the electron
and hole degrees of freedom in the scattering matrix.
In particular, the conductance of a N-S-junction is given as

.. math::

    G = \frac{e^2}{h} (N - R_\text{ee} + R_\text{he})\,,

where :math:`N` is the number of electron channels in the normal lead, and
:math:`R_\text{ee}` the total probability of reflection from electrons
to electrons in the normal lead, and :math:`R_\text{eh}` the total
probability of reflection from electrons to holes in the normal
lead. Fortunately, in Kwant it is straightforward
to partition the scattering matrix in these two degrees
of freedom.

Let us consider a system that consists of a normal lead on the left,
a superconductor on the right, and a tunnel barrier in between:

.. image:: /figure/superconductor_transport_sketch.*

We implement the BdG Hamiltonian in Kwant using a 2x2 matrix structure
for all Hamiltonian matrix elements, as we did
previously in the :ref:`spin example <tutorial_spinorbit>`.
We start by declaring some parameters that will be used in the following code:

.. jupyter-execute::

    a = 1
    W, L = 10, 10
    barrier = 1.5
    barrierpos = (3, 4)
    mu = 0.4
    Delta = 0.1
    Deltapos = 4
    t = 1.0

and we declare the square lattice and construct the scattering region with the following:

.. jupyter-execute::

    # Start with an empty tight-binding system. On each site, there
    # are now electron and hole orbitals, so we must specify the
    # number of orbitals per site. The orbital structure is the same
    # as in the Hamiltonian.
    lat = kwant.lattice.square(norbs=2)
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

Note the new argument ``norbs`` to `~kwant.lattice.square`. This is
the number of orbitals per site in the discretized BdG Hamiltonian - of course,
``norbs = 2``, since each site has one electron orbital and one hole orbital.
It is necessary to specify ``norbs`` here, such that we may later separate the
scattering matrix into electrons and holes. Aside from this, creating the system
is syntactically equivalent to :ref:`spin example <tutorial_spinorbit>`.
The only difference is that the Pauli matrices now act in electron-hole space.
Note that the tunnel barrier is added by overwriting previously set
on-site matrix elements.

The superconducting order parameter is nonzero only in a part of the
scattering region - the part to the right of the tunnel barrier. Thus,
the scattering region is split into a superconducting part (the right
side of it), and a normal part where the pairing is zero (the left side
of it). The next step towards computing conductance is to attach leads.
Let's attach two leads: a normal one to the left end, and a superconducting
one to the right end. Starting with the left lead, we have:

.. jupyter-execute::

    #### Define the leads. ####
    # Left lead - normal, so the order parameter is zero.
    sym_left = kwant.TranslationalSymmetry((-a, 0))
    # Specify the conservation law used to treat electrons and holes separately.
    # We only do this in the left lead, where the pairing is zero.
    lead0 = kwant.Builder(sym_left, conservation_law=-tau_z, particle_hole=tau_y)
    lead0[(lat(0, j) for j in range(W))] = (4 * t - mu) * tau_z
    lead0[lat.neighbors()] = -t * tau_z

Note the two new new arguments in `~kwant.builder.Builder`, ``conservation_law``
and ``particle_hole``. For the purpose of computing conductance, ``conservation_law``
is the essential one, as it allows us to separate the electron and hole degrees of
freedom. Note that it is not necessary to specify ``particle_hole``
in `~kwant.builder.Builder` to correctly compute the conductance in this example.
We will discuss the argument ``particle_hole`` later on. First, let us
discuss ``conservation_law`` in more detail.

Observe that electrons and holes are uncoupled in the left (normal) lead, since
the superconducting order parameter that couples them is zero.
Consequently, we may view the electron and hole degrees of freedom as being
conserved, and may therefore separate them in the Hamiltonian.

In more technical terms, the conservation law implies that the Hamiltonian
can be block diagonalized into uncoupled electron and hole blocks. Since
the blocks are uncoupled, we can construct scattering states in each block
independently. Of course, any scattering state from the electron (hole) block
is entirely electron (hole) like. As a result, the scattering matrix separates
into blocks that describe the scattering between different types of carriers,
such as electron to electron, hole to electron, et cetera.

As we saw above, conservation laws in Kwant are specified with the
``conservation_law`` argument in `~kwant.builder.Builder`.
Specifically, ``conservation_law`` is a matrix that acts on a single *site*
and it must in addition have integer eigenvalues.
Of course, it must also commute with the onsite Hamiltonian and hoppings
to adjacent sites. Internally, Kwant then uses the eigenvectors of the
conservation law to block diagonalize the Hamiltonian. Here, we've specified
the conservation law :math:`-\sigma_z`, such that the eigenvectors with
eigenvalues :math:`-1` and :math:`1` pick out the electron and hole
blocks, respectively. Internally in Kwant, the blocks are stored in the order
of ascending eigenvalues of the conservation law.

In order to move on with the conductance calculation, let's attach the second
lead to the right side of the scattering region:

.. jupyter-execute::

    # Right lead - superconducting, so the order parameter is included.
    sym_right = kwant.TranslationalSymmetry((a, 0))
    lead1 = kwant.Builder(sym_right)
    lead1[(lat(0, j) for j in range(W))] = (4 * t - mu) * tau_z + Delta * tau_x
    lead1[lat.neighbors()] = -t * tau_z

    #### Attach the leads and finalize the system. ####
    syst.attach_lead(lead0)
    syst.attach_lead(lead1)

    syst = syst.finalized()

The second (right) lead is superconducting, such that the electron and hole
blocks are coupled. Of course, this means that we can not separate them into
uncoupled blocks as we did before, and therefore no conservation law is specified.

Kwant is now aware of the block structure of the Hamiltonian in the left lead.
This means that we can extract transmission and reflection amplitudes not only
into the left lead, but also between different conservation law blocks in
the left lead. Generally if leads :math:`i` and :math:`j` both have a conservation
law specified, ``smatrix.transmission((i, a), (j, b))`` gives us
the scattering probability of carriers from block :math:`b` of lead :math:`j`, to
block :math:`a` of lead :math:`i`. In our example, reflection from electrons to
electrons in the left lead is thus ``smatrix.transmission((0, 0), (0, 0))`` (Don't get
confused by the fact that it says ``transmission`` -- transmission
into the same lead is reflection), and reflection from electrons to holes
is ``smatrix.transmission((0, 1), (0, 0))``:

.. jupyter-execute::

    def plot_conductance(syst, energies):
        # Compute conductance
        data = []
        for energy in energies:
            smatrix = kwant.smatrix(syst, energy)
            # Conductance is N - R_ee + R_he
            data.append(smatrix.submatrix((0, 0), (0, 0)).shape[0] -
                        smatrix.transmission((0, 0), (0, 0)) +
                        smatrix.transmission((0, 1), (0, 0)))

        pyplot.figure()
        pyplot.plot(energies, data)
        pyplot.xlabel("energy [t]")
        pyplot.ylabel("conductance [e^2/h]")
        pyplot.show()

Note that ``smatrix.submatrix((0, 0), (0, 0))`` returns the block concerning
reflection of electrons to electrons, and from its size we can extract the number of modes
:math:`N`.

For the default parameters, we obtain the following conductance:

.. jupyter-execute::
    :hide-code:

    plot_conductance(syst, energies=[0.002 * i for i in range(-10, 100)])

We a see a conductance that is proportional to the square of the tunneling
probability within the gap, and proportional to the tunneling probability
above the gap. At the gap edge, we observe a resonant Andreev reflection.

Remember that when we defined `~kwant.builder.Builder` for the left lead above,
we not only declared an electron-hole conservation law, but also that the Hamiltonian
has the particle-hole symmetry :math:`\mathcal{P} = \sigma_y` which anticommutes
with the Hamiltonian, using the argument ``particle_hole``.
In Kwant, whenever one or more of the fundamental discrete symmetries
(time-reversal, particle-hole and chiral) are present in a lead Hamiltonian,
they can be declared in `~kwant.builder.Builder`. Kwant then automatically uses
them to construct scattering states that obey the specified symmetries. In this
example, we have a discrete symmetry declared in addition to a conservation law.
For any two conservation law blocks that are transformed to each other by the
discrete symmetry, Kwant then automatically computes the scattering states of one
block by applying the symmetry operator to the scattering states of the other.

Now, :math:`\mathcal{P}` relates electrons and holes
at *opposite* energies. However, a scattering problem is always solved at a
fixed energy, so generally :math:`\mathcal{P}` does not give a relation between
the electron and hole blocks. The exception is of course at zero energy, in which
case particle-hole symmetry transforms between the electron and hole blocks, resulting
in a symmetric scattering matrix. We can check the symmetry explicitly with

.. jupyter-execute::

    def check_PHS(syst):
        # Scattering matrix
        s = kwant.smatrix(syst, energy=0)
        # Electron to electron block
        s_ee = s.submatrix((0,0), (0,0))
        # Hole to hole block
        s_hh = s.submatrix((0,1), (0,1))
        print('s_ee: \n', np.round(s_ee, 3))
        print('s_hh: \n', np.round(s_hh[::-1, ::-1], 3))
        print('s_ee - s_hh^*: \n',
              np.round(s_ee - s_hh[::-1, ::-1].conj(), 3), '\n')
        # Electron to hole block
        s_he = s.submatrix((0,1), (0,0))
        # Hole to electron block
        s_eh = s.submatrix((0,0), (0,1))
        print('s_he: \n', np.round(s_he, 3))
        print('s_eh: \n', np.round(s_eh[::-1, ::-1], 3))
        print('s_he + s_eh^*: \n',
              np.round(s_he + s_eh[::-1, ::-1].conj(), 3))

which yields the output

.. jupyter-execute::
    :hide-code:

    check_PHS(syst)

Note that :math:`\mathcal{P}` flips the sign of momentum, and for the parameters
we consider here, there are two electron and two hole modes active at zero energy.
We thus reorder the matrix elements of the scattering matrix blocks above,
to ensure that the same matrix elements in the electron and hole blocks relate
scattering states and their particle hole partners.

.. specialnote:: Technical details

    - If you are only interested in particle (thermal) currents you do not need
      to separate the electron and hole degrees of freedom. Still, separating them
      using a conservation law makes the lead calculation in the solving
      phase more efficient.
