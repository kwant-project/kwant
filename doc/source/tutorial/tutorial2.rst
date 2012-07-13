Adding more structure to the problem
------------------------------------

.. _tutorial_spinorbit:

Matrix structure of on-site and hopping elements
................................................

In the next examples, we will extend the previous examples and add more
structure. We begin by extending the simple 2DEG-Hamiltonian by
a Rashba spin-orbit coupling and a Zeeman splitting due to
an external magnetic field:

.. math::

    H = \frac{\hbar^2}{2 m} (\partial_x^2+\partial_y^2) -
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

The tight-binding model corresponding to the Rashba-Hamiltonian
naturally exhibits a 2x2-matrix structure of onsite energies and hoppings.
In order to deal with matrices in python, kwant uses the `numpy package
<numpy.scipy.org>`_. In order to use matrices in our program, we thus also
have to import that package:

.. literalinclude:: ../../../examples/tutorial2a.py
    :lines: 19

For convenience, we define the Pauli-matrices first (with `sigma_0` the
unit matrix):

.. literalinclude:: ../../../examples/tutorial2a.py
    :lines: 22-25

Previously, we used numbers as the values of our matrix elements.
However, `~kwant.builder.Builder` also accepts matrices as values, and
we can simply write:

.. literalinclude:: ../../../examples/tutorial2a.py
    :lines: 37-44

Note that the Zeeman energy adds to the onsite term, whereas the Rashba
spin-orbit term adds to the hoppings (due to the derivative operator).
Furthermore, the hoppings in x and y-direction have a different matrix
structure. We still use `~kwant.builder.Builder.possible_hoppings`
to add all the hoppings at once, but we now have to distinguish
x and y-direction. Because of that, we have to explicitly specify
the hoppings in the form expected by
`~kwant.builder.Builder.possible_hoppings`:

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

.. literalinclude:: ../../../examples/tutorial2a.py
    :lines: 52-58

The remainder of the code is unchanged, and as a result we should obtain
the following, clearly non-monotonic conductance steps:

.. image:: ../images/tutorial2a_result.*

.. seealso::
     The full source code can be found in
     :download:`example/tutorial2a.py <../../../examples/tutorial2a.py>`

.. specialnote:: Technical details

  - It should be emphasized that the relative hopping used for
    `~kwant.builder.Builder.possible_hoppings` is given in terms of
    lattice indices, i.e. relative to the Bravais lattice vectors.
    For a square lattice, the Bravais lattice vectors are simply
    :math:`(a,0)` and :math:`(0,a)`, and hence the mapping from
    lattice indices `(i,j)` to realspace and back is trivial.
    This becomes more involved in more complicated lattices, where
    the realspace directions corresponding to, for example, `(1,0)`
    and `(0,1)` need not be orthogonal any more
    (see :ref:`tutorial-graphene`).


Spatially dependent values through functions
............................................

Up to now, all examples had position-independent matrix-elements
(and thus translational invariance along the wire, which
was the origin of the conductance steps). Now, we consider the
case of a position-dependent potential:

.. math::

    H = \frac{\hbar^2}{2 m} (\partial_x^2+\partial_y^2) + V(x, y)

The position-dependent potential enters in the onsite energies.
One possibility would be to again set the onsite matrix elements
of each lattice point individually (as in tutorial1a.py). However,
changing the potential then implies the need to build up the system again.

Instead, we use a python *function* to define the onsite energies. We
define the potential profile of a quantum well as:

.. literalinclude:: ../../../examples/tutorial2b.py
    :lines: 16-18, 22, 28-34

This function takes one argument which is of type
`~kwant.builder.Site`, from which you can get the realspace
coordinates using ``site.pos``. Note that we use several global
variables to define the behavior of `potential()`: `L` and `L_well`
are variables taken from the namespace of `make_system`, the variable `pot`
is taken from the global module namespace. By this one can change the
behavior of `potential()` at another place, for example by setting
`pot` to a different value. We will use this later to compute
the transmission as a function of well depth.

kwant now allows us to pass a function as a value to
`~kwant.builder.Builder`:

.. literalinclude:: ../../../examples/tutorial2b.py
    :lines: 36-41

For each lattice point, the corresponding site is then passed to the
function `onsite()`. Note that we had to define `onsite()`, as it is
not possible to mix values and functions as in ``sys[...] = 4 * t +
potential``.

For the leads, we just use constant values as before. If we passed a
function also for the leads (which is perfectly allowed), this
function would need to be compatible with the translational symmetry
of the lead -- this should be kept in mind.

Finally, we compute the transmission probability:

.. literalinclude:: ../../../examples/tutorial2b.py
    :lines: 65, 68-77

Since we change the value of the global variable `pot` to vary the
well depth, python requires us to write ``global pot`` to `enable
access to it
<http://docs.python.org/faq/programming.html#what-are-the-rules-for-local-and-global-variables-in-python>`_.
Subsequent calls to :func:`kwant.solve <kwant.solvers.sparse.solve>`
then will use the updated value of pot, and we get the result:

.. image:: ../images/tutorial2b_result.*

Starting from no potential (well depth = 0), we observe the typical
oscillatory transmission behavior through resonances in the quantum well.

.. seealso::
     The full source code can be found in
     :download:`example/tutorial2b.py <../../../examples/tutorial2b.py>`

.. warning::

    If functions are used to set values inside a lead, then they must satisfy
    the same symmetry as the lead does.  There is (currently) no check and
    wrong results will be the consequence of a misbehaving function.

.. specialnote:: Technical details

  - Functions can also be used for hoppings. In this case, they take
    two `~kwant.builder.Site`'s as arguments.

  - In example/tutorial2b.py, line 16

    .. literalinclude:: ../../../examples/tutorial2b.py
        :lines: 16

    is not really necessary. If this line was left out, the
    global variable `pot` would in fact be created by the
    first assignment in `plot_conductance()`.

  - Apart from the realspace position `pos`, `~kwant.builder.Site`
    has also an attribute `tag` containing the lattice indices
    of the site.

  - Since we use a global variable to change the value of the
    potential, let us briefly reflect on how python determines
    which variable to use.

    In our example, the function `potential()` uses the variable
    `pot` which is not defined in the function itself. In this case,
    python looks for the variable in the enclosing scopes, i.e.
    inside the functions/modules/scripts that enclose the
    corresponding piece of code. For example, in

    >>> def f():
    ...     def g():
    ...         print string
    ...     return g
    ...
    >>> g = f()
    >>> string = "global"
    >>> g()
    global

    function `g()` defined inside `f()` uses the global variable
    `string` (which was actually created only *after* the
    definition of `g()`!). Note that this only works as long as
    one only reads `string`; if `g()` was to write to string,
    it would need to add ``global string`` to `g()`, as we
    did in `plot_conductance()`.

    Things change if the function `f()` also contains a variable
    of the same name:

    >>> def f():
    ...     def g():
    ...         print string
    ...     string = "local"
    ...     return g
    ...
    >>> g = f()
    >>> g()
    local
    >>> string = "global"
    >>> g()
    local

    In this case, `g()` always uses the local variable inside `f()`
    (unless we would add ``global string`` in `g()`).

  - `~kwant.builder.Builder` in fact accepts not only functions but any python
    object which is callable.  We can take advantage of the fact that instances
    of python classes with a `__call__` method can be called just as if they
    were functions::

        class Well:
            def __init__(self, a, b=0):
                self.a = a
                self.b = b

            def __call__(self, site):
                x, y = site.pos
                return self.a * (x**2 + y**2) + b

        well = Well(3, 4)

        sys[...] = well

        well.a = ...

    This approach allows to avoid the use of global variables.  Parameters can
    be changed inside the object.

.. _tutorial-abring:

Nontrivial shapes
.................

Up to now, we only dealt with simple wire geometries. Now we turn to the case
of a more complex geometry, namely transport through a quantum ring
that is pierced by a magnetic flux :math:`\Phi`:

.. image:: ../images/tutorial2c_sketch.*

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

.. literalinclude:: ../../../examples/tutorial2c.py
    :lines: 20, 24-27, 30-33

Note that this function takes a realspace position as argument (not a
`~kwant.builder.Site`).

We can now simply add all of the lattice points inside this shape at
once, using the function `~kwant.lattice.Square.shape`
provided by the lattice:

.. literalinclude:: ../../../examples/tutorial2c.py
    :lines: 36-38

Here, ``lat.shape()`` takes as a second parameter a (realspace) point
that is inside the desired shape. The hoppings can still be added
using `~kwant.builder.Builder.possible_hoppings` as before.

Up to now, the system contains constant hoppings and onsite energies,
and we still need to include the phase shift due to the magnetic flux.
This is done by **overwriting** the values of hoppings in x-direction
along the branch cut in the lower arm of the ring. For this we select
all hoppings in x-direction that are of the form `((1, j), (0, j))`
with ``j<0``:

.. literalinclude:: ../../../examples/tutorial2c.py
    :lines: 46-58

Here, `crosses_branchcut` is a boolean function that returns ``True`` for
the desired hoppings. We then use again a generator (this time with
an ``if``-conditional) to set the value of all hoppings across
the branch cut to `fluxphase`. The rationale
behind using a function instead of a constant value for the hopping
is again that we want to vary the flux through the ring, without
constantly rebuilding the system -- instead the flux is governed
by the global variable `phi`.

For the leads, we can also use the ``lat.shape()``-functionality:

.. literalinclude:: ../../../examples/tutorial2c.py
    :lines: 62-71

Here, the shape must cover *at least* one unit cell of the lead
(it does not hurt if it covers more unit cells).

Attaching the leads is done as before:

.. literalinclude:: ../../../examples/tutorial2c.py
    :lines: 78-79

In fact, attaching leads seems not so simple any more for the current
structure with a scattering region very much different from the lead
shapes. However, the choice of unit cell together with the
translational vector allows to place the lead unambiguously in realspace --
the unit cell is repeated infinitely many times in the direction and
opposite to the direction of the translational vector.
kwant examines the lead starting from infinity and traces it
back (going opposite to the direction of the translational vector)
until it intersects the scattering region. At this intersection,
the lead is attached:

.. image:: ../images/tutorial2c_sketch2.*

After the lead has been attached, the system should look like this:

.. image:: ../images/tutorial2c_sys.*

The computation of the conductance goes in the same fashion as before.
Finally you should get the following result:

.. image:: ../images/tutorial2c_result.*

where one can observe the conductance oscillations with the
period of one flux quantum.

.. seealso::
     The full source code can be found in
     :download:`example/tutorial2c.py <../../../examples/tutorial2c.py>`

.. specialnote:: Technical details

  - Note that in this example, we did not need to set
    ``sys.default_site_group = lat``. All lattice points were
    added using functionality from ``lat`` and thus were
    proper sites already.

  - Leads have to have proper periodicity. Furthermore, the kwant
    format requires the hopping from the leads to the scattering
    region to be identical to the hoppings between unit cells in
    the lead. `~kwant.builder.Builder.attach_lead` takes care of
    all these details for you! In fact, it even adds points to
    the scattering region, if proper attaching requires this. This
    becomes more apparent if we attach the leads a bit further away
    from the central axis o the ring, as was done in this example:

    .. image:: ../images/tutorial2c_note1.*

  - Per default, `~kwant.builder.Builder.attach_lead` attaches
    the lead to the "outside" of the structure, by tracing the
    lead backwards, coming from infinity.

    One can also attach the lead to the inside of the structure,
    by providing an alternative starting point from where
    the lead is traced back::

        sys.attach_lead(lead1, lat(0, 0))

    starts the trace-back in the middle of the ring, resulting
    in the lead being attached to the inner circle:

    .. image:: ../images/tutorial2c_note2.*

    Note that here the lead is treated as if it would pass over
    the other arm of the ring, without intersecting it.

.. rubric:: Footnotes

.. [#] The corresponding vector potential is :math:`A_x(x,y)=\Phi \delta(x)
       \Theta(-y)` which yields the correct magnetic field :math:`B(x,y)=\Phi
       \delta(x)\delta(y)`.
