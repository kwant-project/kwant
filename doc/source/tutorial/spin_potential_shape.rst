More interesting systems: spin, potential, shape
------------------------------------------------

Each of the following three examples highlights different ways to go beyond the
very simple examples of the previous section.

.. _tutorial_spinorbit:

Matrix structure of on-site and hopping elements
................................................

.. seealso::
    The complete source code of this example can be found in
    :download:`spin_orbit.py </code/download/spin_orbit.py>`

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
<http://prl.aps.org/abstract/PRL/v90/i25/e256601>`_ that such a system should
exhibit non-monotonic conductance steps due to a spin-orbit gap. Only
very recently, this non-monotonic behavior has been supposedly
observed in experiment: `Nature Physics 6, 336 (2010)
<http://www.nature.com/nphys/journal/v6/n5/abs/nphys1626.html>`_. Here
we will show that a very simple extension of our previous examples will
exactly show this behavior (Note though that no care was taken to choose
realistic parameters).

The tight-binding model corresponding to the Rashba-Hamiltonian naturally
exhibits a 2x2-matrix structure of onsite energies and hoppings.  In order to
use matrices in our program, we import the Tinyarray package.  (`NumPy
<http://www.numpy.org/>`_ would work as well, but Tinyarray is much faster
for small arrays.)

.. literalinclude:: /code/include/spin_orbit.py
    :start-after: #HIDDEN_BEGIN_xumz
    :end-before: #HIDDEN_END_xumz

For convenience, we define the Pauli-matrices first (with :math:`\sigma_0` the
unit matrix):

.. literalinclude:: /code/include/spin_orbit.py
    :start-after: #HIDDEN_BEGIN_hwbt
    :end-before: #HIDDEN_END_hwbt

Previously, we used numbers as the values of our matrix elements.
However, `~kwant.builder.Builder` also accepts matrices as values, and
we can simply write:

.. literalinclude:: /code/include/spin_orbit.py
    :start-after: #HIDDEN_BEGIN_uxrm
    :end-before: #HIDDEN_END_uxrm

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

.. literalinclude:: /code/include/spin_orbit.py
    :start-after: #HIDDEN_BEGIN_yliu
    :end-before: #HIDDEN_END_yliu

The remainder of the code is unchanged, and as a result we should obtain
the following, clearly non-monotonic conductance steps:

.. image:: /code/figure/spin_orbit_result.*

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
    :download:`quantum_well.py </code/download/quantum_well.py>`

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

.. literalinclude:: /code/include/quantum_well.py
    :start-after: #HIDDEN_BEGIN_ehso
    :end-before: #HIDDEN_END_ehso

This function takes two arguments: the first of type `~kwant.builder.Site`,
from which you can get the real-space coordinates using ``site.pos``, and the
value of the potential as the second.  Note that in `potential` we can access
variables of the surrounding function: `L` and `L_well` are taken from the
namespace of `make_system`.

Kwant now allows us to pass a function as a value to
`~kwant.builder.Builder`:

.. literalinclude:: /code/include/quantum_well.py
    :start-after: #HIDDEN_BEGIN_coid
    :end-before: #HIDDEN_END_coid

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

.. literalinclude:: /code/include/quantum_well.py
    :start-after: #HIDDEN_BEGIN_sqvr
    :end-before: #HIDDEN_END_sqvr

``kwant.smatrix`` allows us to specify a list, `args`, that will be passed as
additional arguments to the functions that provide the Hamiltonian matrix
elements.  In this example we are able to solve the system for different depths
of the potential well by passing the potential value. We obtain the result:

.. image:: /code/figure/quantum_well_result.*

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
    :download:`ab_ring.py </code/download/ab_ring.py>`

Up to now, we only dealt with simple wire geometries. Now we turn to the case
of a more complex geometry, namely transport through a quantum ring
that is pierced by a magnetic flux :math:`\Phi`:

.. image:: /code/figure/ab_ring_sketch.*

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

.. literalinclude:: /code/include/ab_ring.py
    :start-after: #HIDDEN_BEGIN_eusz
    :end-before: #HIDDEN_END_eusz

Note that this function takes a real-space position as argument (not a
`~kwant.builder.Site`).

We can now simply add all of the lattice points inside this shape at
once, using the function `~kwant.lattice.Square.shape`
provided by the lattice:

.. literalinclude:: /code/include/ab_ring.py
    :start-after: #HIDDEN_BEGIN_lcak
    :end-before: #HIDDEN_END_lcak

Here, ``lat.shape`` takes as a second parameter a (real-space) point that is
inside the desired shape. The hoppings can still be added using
``lat.neighbors()`` as before.

Up to now, the system contains constant hoppings and onsite energies,
and we still need to include the phase shift due to the magnetic flux.
This is done by **overwriting** the values of hoppings in x-direction
along the branch cut in the lower arm of the ring. For this we select
all hoppings in x-direction that are of the form `(lat(1, j), lat(0, j))`
with ``j<0``:

.. literalinclude:: /code/include/ab_ring.py
    :start-after: #HIDDEN_BEGIN_lvkt
    :end-before: #HIDDEN_END_lvkt

Here, `crosses_branchcut` is a boolean function that returns ``True`` for
the desired hoppings. We then use again a generator (this time with
an ``if``-conditional) to set the value of all hoppings across
the branch cut to `fluxphase`. The rationale
behind using a function instead of a constant value for the hopping
is again that we want to vary the flux through the ring, without
constantly rebuilding the system -- instead the flux is governed
by the parameter `phi`.

For the leads, we can also use the ``lat.shape``-functionality:

.. literalinclude:: /code/include/ab_ring.py
    :start-after: #HIDDEN_BEGIN_qwgr
    :end-before: #HIDDEN_END_qwgr

Here, the shape must be compatible with the translational symmetry
of the lead ``sym_lead``. In particular, this means that it should extend to
infinity along the translational symmetry direction (note how there is
no restriction on ``x`` in ``lead_shape``) [#]_.

Attaching the leads is done as before:

.. literalinclude:: /code/include/ab_ring.py
    :start-after: #HIDDEN_BEGIN_skbz
    :end-before: #HIDDEN_END_skbz

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

.. image:: /code/figure/ab_ring_sketch2.*

After the lead has been attached, the system should look like this:

.. image:: /code/figure/ab_ring_syst.*

The computation of the conductance goes in the same fashion as before.
Finally you should get the following result:

.. image:: /code/figure/ab_ring_result.*

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

    .. image:: /code/figure/ab_ring_note1.*

  - Per default, `~kwant.builder.Builder.attach_lead` attaches
    the lead to the "outside" of the structure, by tracing the
    lead backwards, coming from infinity.

    One can also attach the lead to the inside of the structure,
    by providing an alternative starting point from where
    the lead is traced back::

        syst.attach_lead(lead1, lat(0, 0))

    starts the trace-back in the middle of the ring, resulting
    in the lead being attached to the inner circle:

    .. image:: /code/figure/ab_ring_note2.*

    Note that here the lead is treated as if it would pass over
    the other arm of the ring, without intersecting it.

.. rubric:: Footnotes

.. [#] The corresponding vector potential is :math:`A_x(x,y)=\Phi \delta(x)
       \Theta(-y)` which yields the correct magnetic field :math:`B(x,y)=\Phi
       \delta(x)\delta(y)`.
.. [#] Despite the "infinite" shape, the unit cell will still be finite; the
       `~kwant.lattice.TranslationalSymmetry` takes care of that.
