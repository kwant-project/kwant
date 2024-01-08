More interesting systems: spin, potential, shape
------------------------------------------------

Each of the following three examples highlights different ways to go beyond the
very simple examples of the previous section.

.. _tutorial_spinorbit:

Matrix structure of on-site and hopping elements
................................................

.. seealso::
    You can execute the code examples live in your browser by
    activating thebelab:

    .. thebe-button:: Activate Thebelab

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download-script:`spin_orbit`

.. jupyter-kernel::
    :id: spin_orbit

We begin by extending the simple 2DEG-Hamiltonian by a Rashba spin-orbit
coupling and a Zeeman splitting due to an external magnetic field:

.. math::

    H = \frac{-\hbar^2}{2 m} (\partial_x^2+\partial_y^2) -
      i \alpha (\partial_x \sigma_y - \partial_y \sigma_x) +
      E_\text{Z} \sigma_z +  V(y)

Here :math:`\sigma_{x,y,z}` denote the Pauli matrices.

It turns out that this well studied Rashba-Hamiltonian has some peculiar
properties in (ballistic) nanowires: It was first predicted theoretically
in `Phys. Rev. Lett. 90, 256601 (2003)
<https://doi.org/10.1103/PhysRevLett.90.256601>`_ that such a system should
exhibit non-monotonic conductance steps due to a spin-orbit gap. Only
very recently, this non-monotonic behavior has been supposedly
observed in experiment: `Nature Physics 6, 336 (2010)
<https://doi.org/10.1038/nphys1626>`_. Here
we will show that a very simple extension of our previous examples will
exactly show this behavior (Note though that no care was taken to choose
realistic parameters).

The tight-binding model corresponding to the Rashba-Hamiltonian naturally
exhibits a 2x2-matrix structure of onsite energies and hoppings.  In order to
use matrices in our program, we import the Tinyarray package.  (`NumPy
<https://numpy.org/>`_ would work as well, but Tinyarray is much faster
for small arrays.)

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.3.1. Matrix structure of on-site and hopping elements
    # ================================================================
    #
    # Physics background
    # ------------------
    #  Gaps in quantum wires with spin-orbit coupling and Zeeman splititng,
    #  as theoretically predicted in
    #   https://doi.org/10.1103/PhysRevLett.90.256601
    #  and (supposedly) experimentally oberved in
    #   https://doi.org/10.1038/nphys1626
    #
    # Kwant features highlighted
    # --------------------------
    #  - Numpy matrices as values in Builder

    import kwant

    # For plotting
    from matplotlib import pyplot

.. jupyter-execute:: boilerplate.py
    :hide-code:

.. jupyter-execute::

    # For matrix support
    import tinyarray

For convenience, we define the Pauli-matrices first (with :math:`\sigma_0` the
unit matrix):

.. jupyter-execute::

    # define Pauli-matrices for convenience
    sigma_0 = tinyarray.array([[1, 0], [0, 1]])
    sigma_x = tinyarray.array([[0, 1], [1, 0]])
    sigma_y = tinyarray.array([[0, -1j], [1j, 0]])
    sigma_z = tinyarray.array([[1, 0], [0, -1]])

and we also define some other parameters useful for constructing our system:

.. jupyter-execute::

    t = 1.0
    alpha = 0.5
    e_z = 0.08
    W, L = 10, 30

Previously, we used numbers as the values of our matrix elements.
However, `~kwant.builder.Builder` also accepts matrices as values, and
we can simply write:

.. jupyter-execute::
    :hide-code:

    lat = kwant.lattice.square(norbs=2)
    syst = kwant.Builder()

.. jupyter-execute::

    #### Define the scattering region. ####
    syst[(lat(x, y) for x in range(L) for y in range(W))] = \
        4 * t * sigma_0 + e_z * sigma_z
    # hoppings in x-direction
    syst[kwant.builder.HoppingKind((1, 0), lat, lat)] = \
        -t * sigma_0 + 1j * alpha * sigma_y / 2
    # hoppings in y-directions
    syst[kwant.builder.HoppingKind((0, 1), lat, lat)] = \
        -t * sigma_0 - 1j * alpha * sigma_x / 2

Note that we specify ``norbs=2`` when creating the lattice, as each site
has 2 degrees of freedom associated with it, giving us 2x2 matrices as
onsite/hopping terms.
Note that the Zeeman energy adds to the onsite term, whereas the Rashba
spin-orbit term adds to the hoppings (due to the derivative operator).
Furthermore, the hoppings in x and y-direction have a different matrix
structure. We now cannot use ``lat.neighbors()`` to add all the hoppings at
once, since we now have to distinguish x and y-direction. Because of that, we
have to explicitly specify the hoppings in the form expected by
`~kwant.builder.HoppingKind`:

- A tuple with relative lattice indices.  For example, `(1, 0)` means
  hopping from `(i, j)` to `(i+1, j)`, whereas `(1, 1)` would
  mean hopping to `(i+1, j+1)`.
- The target lattice (where to hop to)
- The source lattice (where the hopping originates)

Since we are only dealing with a single lattice here, source and target
lattice are identical, but still must be specified  (for an example
with hopping between different (sub)lattices, see :ref:`tutorial-graphene`).

Again, it is enough to specify one direction of the hopping (i.e.
when specifying `(1, 0)` it is not necessary to specify `(-1, 0)`),
`~kwant.builder.Builder` assures hermiticity.

The leads also allow for a matrix structure,


.. jupyter-execute::
    :hide-code:

    #### Define the left lead. ####
    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))

.. jupyter-execute::

    lead[(lat(0, j) for j in range(W))] = 4 * t * sigma_0 + e_z * sigma_z
    # hoppings in x-direction
    lead[kwant.builder.HoppingKind((1, 0), lat, lat)] = \
        -t * sigma_0 + 1j * alpha * sigma_y / 2
    # hoppings in y-directions
    lead[kwant.builder.HoppingKind((0, 1), lat, lat)] = \
        -t * sigma_0 - 1j * alpha * sigma_x / 2

.. jupyter-execute::
    :hide-code:

    #### Attach the leads and finalize the system. ####
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())
    syst = syst.finalized()

The remainder of the code is unchanged, and as a result we should obtain
the following, clearly non-monotonic conductance steps:

.. jupyter-execute::
    :hide-code:

    # Compute conductance
    energies=[0.01 * i - 0.3 for i in range(100)]
    data = []
    for energy in energies:
        smatrix = kwant.smatrix(syst, energy)
        data.append(smatrix.transmission(1, 0))

    pyplot.figure()
    pyplot.plot(energies, data)
    pyplot.xlabel("energy [t]")
    pyplot.ylabel("conductance [e^2/h]")
    pyplot.show()

.. specialnote:: Technical details

  - The Tinyarray package, one of the dependencies of Kwant, implements
    efficient small arrays.  It is used internally in Kwant for storing small
    vectors and matrices.  For performance, it is preferable to define small
    arrays that are going to be used with Kwant using Tinyarray.  However,
    NumPy would work as well::

        import numpy
        sigma_0 = numpy.array([[1, 0], [0, 1]])
        sigma_x = numpy.array([[0, 1], [1, 0]])
        sigma_y = numpy.array([[0, -1j], [1j, 0]])
        sigma_z = numpy.array([[1, 0], [0, -1]])

    Tinyarray arrays behave for most purposes like NumPy arrays except that
    they are immutable: they cannot be changed once created.  This is important
    for Kwant: it allows them to be used directly as dictionary keys.

  - It should be emphasized that the relative hopping used for
    `~kwant.builder.HoppingKind` is given in terms of
    lattice indices, i.e. relative to the Bravais lattice vectors.
    For a square lattice, the Bravais lattice vectors are simply
    `(a,0)` and `(0,a)`, and hence the mapping from
    lattice indices `(i,j)` to real space and back is trivial.
    This becomes more involved in more complicated lattices, where
    the real-space directions corresponding to, for example, `(1,0)`
    and `(0,1)` need not be orthogonal any more
    (see :ref:`tutorial-graphene`).


Spatially dependent values through functions
............................................

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download-script:`quantum_well`

.. jupyter-kernel::
    :id: quantum_well

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.3.2. Spatially dependent values through functions
    # ============================================================
    #
    # Physics background
    # ------------------
    #  transmission through a quantum well
    #
    # Kwant features highlighted
    # --------------------------
    #  - Functions as values in Builder

    import kwant

    # For plotting
    from matplotlib import pyplot

.. jupyter-execute:: boilerplate.py
    :hide-code:

Up to now, all examples had position-independent matrix-elements
(and thus translational invariance along the wire, which
was the origin of the conductance steps). Now, we consider the
case of a position-dependent potential:

.. math::

    H = \frac{\hbar^2}{2 m} (\partial_x^2+\partial_y^2) + V(x, y)

The position-dependent potential enters in the onsite energies. One
possibility would be to again set the onsite matrix elements of each
lattice point individually (as in :ref:`tutorial_quantum_wire`). However,
changing the potential then implies the need to build up the system again.

Instead, we use a python *function* to define the onsite energies. We
define the potential profile of a quantum well as:

.. jupyter-execute::

    W, L, L_well = 10, 30, 10

    def potential(site, pot):
        (x, y) = site.pos
        if (L - L_well) / 2 < x < (L + L_well) / 2:
            return pot
        else:
            return 0

This function takes two arguments: the first of type `~kwant.builder.Site`,
from which you can get the real-space coordinates using ``site.pos``, and the
value of the potential as the second.  Note that in `potential` we can access
variables `L` and `L_well` that are defined globally.

Kwant now allows us to pass a function as a value to
`~kwant.builder.Builder`:

.. jupyter-execute::

    a = 1
    t = 1.0

    def onsite(site, pot):
        return 4 * t + potential(site, pot)

    lat = kwant.lattice.square(a, norbs=1)
    syst = kwant.Builder()

    syst[(lat(x, y) for x in range(L) for y in range(W))] = onsite
    syst[lat.neighbors()] = -t

.. jupyter-execute::
    :hide-code:

    #### Define and attach the leads. ####
    lead = kwant.Builder(kwant.TranslationalSymmetry((-a, 0)))
    lead[(lat(0, j) for j in range(W))] = 4 * t
    lead[lat.neighbors()] = -t
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    syst = syst.finalized()

For each lattice point, the corresponding site is then passed as the
first argument to the function `onsite`. The values of any additional
parameters, which can be used to alter the Hamiltonian matrix elements
at a later stage, are specified later during the call to `smatrix`.
Note that we had to define `onsite`, as it is
not possible to mix values and functions as in ``syst[...] = 4 * t +
potential``.

For the leads, we just use constant values as before. If we passed a
function also for the leads (which is perfectly allowed), this
function would need to be compatible with the translational symmetry
of the lead -- this should be kept in mind.

Finally, we compute the transmission probability:

.. jupyter-execute::

    def plot_conductance(syst, energy, welldepths):

        # Compute conductance
        data = []
        for welldepth in welldepths:
            smatrix = kwant.smatrix(syst, energy, params=dict(pot=-welldepth))
            data.append(smatrix.transmission(1, 0))

        pyplot.figure()
        pyplot.plot(welldepths, data)
        pyplot.xlabel("well depth [t]")
        pyplot.ylabel("conductance [e^2/h]")
        pyplot.show()

``kwant.smatrix`` allows us to specify a dictionary, `params`, that contains
the additional arguments required by the Hamiltonian matrix elements.
In this example we are able to solve the system for different depths
of the potential well by passing the potential value (remember above
we defined our `onsite` function that takes a parameter named `pot`).
We obtain the result:

.. jupyter-execute::
    :hide-code:

    plot_conductance(syst, energy=0.2,
                     welldepths=[0.01 * i for i in range(100)])

Starting from no potential (well depth = 0), we observe the typical
oscillatory transmission behavior through resonances in the quantum well.

.. warning::

    If functions are used to set values inside a lead, then they must satisfy
    the same symmetry as the lead does.  There is (currently) no check and
    wrong results will be the consequence of a misbehaving function.

.. specialnote:: Technical details

  - Functions can also be used for hoppings. In this case, they take
    two `~kwant.builder.Site`'s as arguments and then an arbitrary number
    of additional arguments.

  - Apart from the real-space position `pos`, `~kwant.builder.Site` has also an
    attribute `tag` containing the lattice indices of the site.

.. _tutorial-abring:

Nontrivial shapes
.................

.. seealso::
    The complete source code of this example can be found in
    :jupyter-download-script:`ab_ring`

.. jupyter-kernel::
    :id: ab_ring

.. jupyter-execute::
    :hide-code:

    # Tutorial 2.3.3. Nontrivial shapes
    # =================================
    #
    # Physics background
    # ------------------
    #  Flux-dependent transmission through a quantum ring
    #
    # Kwant features highlighted
    # --------------------------
    #  - More complex shapes with lattices
    #  - Allows for discussion of subtleties of `attach_lead` (not in the
    #    example, but in the tutorial main text)
    #  - Modifcations of hoppings/sites after they have been added

    from cmath import exp
    from math import pi

    import kwant

    # For plotting
    from matplotlib import pyplot

.. jupyter-execute:: boilerplate.py
    :hide-code:

Up to now, we only dealt with simple wire geometries. Now we turn to the case
of a more complex geometry, namely transport through a quantum ring
that is pierced by a magnetic flux :math:`\Phi`:

.. image:: /figure/ab_ring_sketch.*

For a flux line, it is possible to choose a gauge such that a
charged particle acquires a phase :math:`e\Phi/h` whenever it
crosses the branch cut originating from the flux line (branch
cut shown as red dashed line) [#]_. There are more symmetric gauges, but
this one is most convenient to implement numerically.

Defining such a complex structure adding individual lattice sites
is possible, but cumbersome. Fortunately, there is a more convenient solution:
First, define a boolean function defining the desired shape, i.e. a function
that returns ``True`` whenever a point is inside the shape, and
``False`` otherwise:

.. jupyter-execute::

    r1, r2 = 10, 20

    def ring(pos):
        (x, y) = pos
        rsq = x ** 2 + y ** 2
        return (r1 ** 2 < rsq < r2 ** 2)

Note that this function takes a real-space position as argument (not a
`~kwant.builder.Site`).

We can now simply add all of the lattice points inside this shape at
once, using the function `~kwant.lattice.Square.shape`
provided by the lattice:

.. jupyter-execute::

    a = 1
    t = 1.0

    lat = kwant.lattice.square(a, norbs=1)
    syst = kwant.Builder()

    syst[lat.shape(ring, (0, r1 + 1))] = 4 * t
    syst[lat.neighbors()] = -t

Here, ``lat.shape`` takes as a second parameter a (real-space) point that is
inside the desired shape. The hoppings can still be added using
``lat.neighbors()`` as before.

Up to now, the system contains constant hoppings and onsite energies,
and we still need to include the phase shift due to the magnetic flux.
This is done by **overwriting** the values of hoppings in x-direction
along the branch cut in the lower arm of the ring. For this we select
all hoppings in x-direction that are of the form `(lat(1, j), lat(0, j))`
with ``j<0``:

.. jupyter-execute::

    # In order to introduce a flux through the ring, we introduce a phase on
    # the hoppings on the line cut through one of the arms.  Since we want to
    # change the flux without modifying the Builder instance repeatedly, we
    # define the modified hoppings as a function that takes the flux as its
    # parameter phi.
    def hopping_phase(site1, site2, phi):
        return -t * exp(1j * phi)

    def crosses_branchcut(hop):
        ix0, iy0 = hop[0].tag

        # builder.HoppingKind with the argument (1, 0) below
        # returns hoppings ordered as ((i+1, j), (i, j))
        return iy0 < 0 and ix0 == 1  # ix1 == 0 then implied

    # Modify only those hopings in x-direction that cross the branch cut
    def hops_across_cut(syst):
        for hop in kwant.builder.HoppingKind((1, 0), lat, lat)(syst):
            if crosses_branchcut(hop):
                yield hop

    syst[hops_across_cut] = hopping_phase

Here, `crosses_branchcut` is a boolean function that returns ``True`` for
the desired hoppings. We then use again a generator (this time with
an ``if``-conditional) to set the value of all hoppings across
the branch cut to `fluxphase`. The rationale
behind using a function instead of a constant value for the hopping
is again that we want to vary the flux through the ring, without
constantly rebuilding the system -- instead the flux is governed
by the parameter `phi`.

For the leads, we can also use the ``lat.shape``-functionality:

.. jupyter-execute::

    #### Define the leads. ####
    W = 10

    sym_lead = kwant.TranslationalSymmetry((-a, 0))
    lead = kwant.Builder(sym_lead)


    def lead_shape(pos):
        (x, y) = pos
        return (-W / 2 < y < W / 2)

    lead[lat.shape(lead_shape, (0, 0))] = 4 * t
    lead[lat.neighbors()] = -t

Here, the shape must be compatible with the translational symmetry
of the lead ``sym_lead``. In particular, this means that it should extend to
infinity along the translational symmetry direction (note how there is
no restriction on ``x`` in ``lead_shape``) [#]_.

Attaching the leads is done as before:

.. jupyter-execute::
    :hide-output:

    #### Attach the leads ####
    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

In fact, attaching leads seems not so simple any more for the current
structure with a scattering region very much different from the lead
shapes. However, the choice of unit cell together with the
translational vector allows to place the lead unambiguously in real space --
the unit cell is repeated infinitely many times in the direction and
opposite to the direction of the translational vector.
Kwant examines the lead starting from infinity and traces it
back (going opposite to the direction of the translational vector)
until it intersects the scattering region. At this intersection,
the lead is attached:

.. image:: /figure/ab_ring_sketch2.*

After the lead has been attached, the system should look like this:

.. jupyter-execute::
    :hide-code:

    kwant.plot(syst);

The computation of the conductance goes in the same fashion as before.
Finally you should get the following result:


.. jupyter-execute::
    :hide-code:

    def plot_conductance(syst, energy, fluxes):
        # compute conductance

        normalized_fluxes = [flux / (2 * pi) for flux in fluxes]
        data = []
        for flux in fluxes:
            smatrix = kwant.smatrix(syst, energy, params=dict(phi=flux))
            data.append(smatrix.transmission(1, 0))

        pyplot.figure()
        pyplot.plot(normalized_fluxes, data)
        pyplot.xlabel("flux [flux quantum]")
        pyplot.ylabel("conductance [e^2/h]")
        pyplot.show()

    # We should see a conductance that is periodic with the flux quantum
    plot_conductance(syst.finalized(), energy=0.15,
                     fluxes=[0.01 * i * 3 * 2 * pi for i in range(100)])

where one can observe the conductance oscillations with the
period of one flux quantum.

.. specialnote:: Technical details

  - Leads have to have proper periodicity. Furthermore, the Kwant
    format requires the hopping from the leads to the scattering
    region to be identical to the hoppings between unit cells in
    the lead. `~kwant.builder.Builder.attach_lead` takes care of
    all these details for you! In fact, it even adds points to
    the scattering region, if proper attaching requires this. This
    becomes more apparent if we attach the leads a bit further away
    from the central axis o the ring, as was done in this example:

    .. jupyter-kernel::
        :id: ab_ring_note1

    .. jupyter-execute::
        :hide-code:

        import kwant
        from matplotlib import pyplot

    .. jupyter-execute:: boilerplate.py
        :hide-code:

    .. jupyter-execute::
        :hide-code:

        a = 1
        t = 1.0
        W = 10
        r1, r2 = 10, 20

        lat = kwant.lattice.square(norbs=1)
        syst = kwant.Builder()
        def ring(pos):
            (x, y) = pos
            rsq = x**2 + y**2
            return ( r1**2 < rsq < r2**2)
        syst[lat.shape(ring, (0, 11))] = 4 * t
        syst[lat.neighbors()] = -t
        sym_lead0 = kwant.TranslationalSymmetry((-a, 0))
        lead0 = kwant.Builder(sym_lead0)
        def lead_shape(pos):
            (x, y) = pos
            return (-1 < x < 1) and ( 0.5 * W < y < 1.5 * W )
        lead0[lat.shape(lead_shape, (0, W))] = 4 * t
        lead0[lat.neighbors()] = -t
        lead1 = lead0.reversed()
        syst.attach_lead(lead0)
        syst.attach_lead(lead1)

        kwant.plot(syst);


  - Per default, `~kwant.builder.Builder.attach_lead` attaches
    the lead to the "outside" of the structure, by tracing the
    lead backwards, coming from infinity.

    One can also attach the lead to the inside of the structure,
    by providing an alternative starting point from where
    the lead is traced back::

        syst.attach_lead(lead1, lat(0, 0))

    starts the trace-back in the middle of the ring, resulting
    in the lead being attached to the inner circle:

    .. jupyter-kernel::
        :id: ab_ring_note2

    .. jupyter-execute::
        :hide-code:

        import kwant
        from matplotlib import pyplot

    .. jupyter-execute:: boilerplate.py
        :hide-code:

    .. jupyter-execute::
        :hide-code:

        a = 1
        t = 1.0
        W = 10
        r1, r2 = 10, 20

        lat = kwant.lattice.square(a, norbs=1)
        syst = kwant.Builder()
        def ring(pos):
            (x, y) = pos
            rsq = x**2 + y**2
            return ( r1**2 < rsq < r2**2)
        syst[lat.shape(ring, (0, 11))] = 4 * t
        syst[lat.neighbors()] = -t
        sym_lead0 = kwant.TranslationalSymmetry((-a, 0))
        lead0 = kwant.Builder(sym_lead0)
        def lead_shape(pos):
            (x, y) = pos
            return (-1 < x < 1) and ( -W/2 < y < W/2  )
        lead0[lat.shape(lead_shape, (0, 0))] = 4 * t
        lead0[lat.neighbors()] = -t
        lead1 = lead0.reversed()
        syst.attach_lead(lead0)
        syst.attach_lead(lead1, lat(0, 0))

        kwant.plot(syst);

    Note that here the lead is treated as if it would pass over
    the other arm of the ring, without intersecting it.

.. rubric:: Footnotes

.. [#] The corresponding vector potential is :math:`A_x(x,y)=\Phi \delta(x)
       \Theta(-y)` which yields the correct magnetic field :math:`B(x,y)=\Phi
       \delta(x)\delta(y)`.
.. [#] Despite the "infinite" shape, the unit cell will still be finite; the
       `~kwant.lattice.TranslationalSymmetry` takes care of that.
