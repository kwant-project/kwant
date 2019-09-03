Frequently asked questions
--------------------------
This FAQ complements the regular Kwant tutorials and thus does not cover
questions that are discussed there.  The `Kwant paper
<https://downloads.kwant-project.org/doc/kwant-paper.pdf>`_ also digs deeper
into Kwant's structure.

.. seealso::
    You can execute the code examples live in your browser by
    activating thebelab:

    .. thebe-button:: Activate Thebelab

.. jupyter-execute::
    :hide-code:

    import kwant
    import numpy as np
    import tinyarray
    import matplotlib
    from matplotlib import pyplot as plt

.. jupyter-execute:: boilerplate.py
    :hide-code:

What is a system, and what is a builder?
========================================
A Kwant system represents a particular tight-binding model. It contains a graph
whose edges and vertices are assigned values, and that corresponds to the
Hamiltonian matrix of the model being simulated.

In Kwant the creation of the system is separated from its use in numerical
calculations. First an instance of the `~kwant.builder.Builder` class is used
to construct the model, then the `~kwant.builder.Builder.finalize` method is
called, which produces a so-called low-level `~kwant.system.System` that can be
used by Kwant's solvers.

The interface of builders mimics Python mappings (e.g. dictionaries).  The
familiar square-bracket syntax allows to set, get and delete items that
correspond to elements of the system graph, e.g. ``syst[key] = value``.  An
item consists of a key and an associated value.  Keys are `sites <What is a
site?_>`_ and `hoppings <What is a hopping?_>`_.  Values can be numbers, arrays
of numbers, or functions that return numbers or arrays.

Finalizing a builder returns a copy of the system with the graph *structure*
frozen.  (This can be equivalently seen as freezing the system geometry or the
sparsity structure of the Hamiltonian.)  The associated *values* are taken over
verbatim.  Note that finalizing does not freeze the Hamiltonian matrix: only
its structure is fixed, values that are functions may depend on an arbitrary
number of parameters.

In the documentation and in mailing list discussions, the general term
"system" can refer either to a ``Builder`` or to a low-level
``System``, and the context will determine which specific class is being
referred to. The terms "builder" and "low-level system" (or "finalized system")
refer respectively to ``Builder`` and ``System``.


What is a site?
===============
Kwant is a tool for working with tight-binding models, which can be viewed as a
graph composed of edges and vertices.  Sites are Kwant’s labels for the
vertices.  Sites have two attributes: a *family* and a *tag*.  The
combination of family and tag uniquely defines a site.

For example let us create an empty tight binding system and add two sites:

.. jupyter-execute::

    a = 1
    lat = kwant.lattice.square(a, norbs=1)
    syst = kwant.Builder()

    syst[lat(1, 0)] = 4
    syst[lat(1, 1)] = 4

    kwant.plot(syst);

In the above snippet we added 2 sites: ``lat(1, 0)`` and ``lat(0, 1)``. Both
of these sites belong to the same family, ``lat``, but have different tags:
``(1, 0)`` and ``(0, 1)`` respectively.

Both sites were given the value 4 which means that the above system corresponds
to the Hamiltonian matrix

.. math::
    H = \left(
    \begin{array}{cc}
    4 & 0 \\
    0 & 4
    \end{array}
    \right).


What is a hopping?
==================
A hopping is simply a tuple of two of sites, which defines an edge of the graph
that makes up a tight-binding model.  Other sequences of sites that are not
tuples, for example lists, are not treated as hoppings.

Starting from the example code from `What is a site?`_, we can add a hopping
to our system in the following way:

.. jupyter-execute::

    syst[(lat(1, 0), lat(1, 1))] = 1j

    kwant.plot(syst);

Visually, a hopping is represented as a line that joins two sites.

The Hamiltonian matrix is now

.. math::
    H = \left(
    \begin{array}{cc}
    4 & i \\
    -i & 4
    \end{array}
    \right).

Note how adding ``(site_a, site_b)`` to a system and assigning it a value
``v``, implicitly adds the hopping ``(site_b, site_a)`` with the Hermitian
conjugate of ``v`` as value.


What is a site family, and what is a tag?
=========================================
A site family groups related sites together, and a tag serves as a unique
identifier for a site within a given family.

In the previous example we saw a family that was suggestively called ``lat``,
which had sites whose tags were pairs of integers. In this specific example
the site family also happens to be a regular Bravais lattice, and the tags take
on the meaning of lattice coordinates for a site on this lattice.

The concept of families and tags is, however, more general. For example, one
could implement a mesh that can be locally refined in certain areas, by having
a family where sites belong to a `quadtree
<https://en.wikipedia.org/wiki/Quadtree>`_, or an amorphous blob where sites
are tagged by letters of the alphabet.


What is a lattice?
==================
Kwant allows to define and use Bravais lattices for dealing with collections of
regularly placed sites. They know about things like what sites are
neighbors, or what sites belong to a given region of real space.
`~kwant.lattice.Monatomic` lattices have a single site in their basis, while
`~kwant.lattice.Polyatomic` lattices have more than one site in their basis.

Monatomic lattices in Kwant *are also site families*, with sites that are
tagged by tuples of integers: the site's coordinates in the basis of
primitive vectors of the lattice. Polyatomic lattices, however, are *not*
site families, since lattice coordinates are not enough information to uniquely
identify a site if there is more than one site in the basis. Polyatomic
lattices do, however, have an attribute ``sublattices`` that is a list of
monatomic lattices that together make up the whole polyatomic lattice.

Let's create two monatomic lattices (``lat_a`` and ``lat_b``).  ``(1, 0)``
and ``(0, 1)`` will be the primitive vectors and ``(0, 0)`` and ``(0.5, 0.5)``
the origins of the two lattices:

.. jupyter-execute::

    # Two monatomic lattices
    primitive_vectors = [(1, 0), (0, 1)]
    lat_a = kwant.lattice.Monatomic(primitive_vectors, offset=(0, 0), norbs=1)
    lat_b = kwant.lattice.Monatomic(primitive_vectors, offset=(0.5, 0.5), norbs=1)
    # lat1 is equivalent to kwant.lattice.square(norbs=1)

    syst = kwant.Builder()

    syst[lat_a(0, 0)] = 4
    syst[lat_b(0, 0)] = 4

    kwant.plot(syst);

We can also create a ``Polyatomic`` lattice with the same primitive vectors and
two sites in the basis:

.. jupyter-execute::

    # One polyatomic lattice containing two sublattices
    lat = kwant.lattice.Polyatomic([(1, 0), (0, 1)], [(0, 0), (0.5, 0.5)], norbs=1)
    sub_a, sub_b = lat.sublattices

The two sublattices ``sub_a`` and ``sub_b`` are nothing else than ``Monatomic``
instances, and are equivalent to ``lat_a`` and ``lat_b`` that we created
previously.  The advantage of the second approach is that there is now a
``Polyatomic`` object that is aware of both of its sublattices, and we can do
things like calculate neighboring sites, even between sublattices, which would
not be possible with the two separate ``Monatomic`` lattices.

The `kwant.lattice` module also defines several convenience functions, such as
`~kwant.lattice.square` and `~kwant.lattice.honeycomb`, for creating lattices
of common types, without having to explicitly specify all of the lattice
vectors and basis vectors.


When plotting, how to color the different sublattices differently?
==================================================================
In the following example we shall use a kagome lattice, which has three sublattices.

.. jupyter-execute::

    lat = kwant.lattice.kagome(norbs=1)
    syst = kwant.Builder()

    a, b, c = lat.sublattices  # The kagome lattice has 3 sublattices

As we can see below, we create a new plotting function that assigns a color for each family, and a different size for the hoppings depending on the family of the two sites. Finally we add sites and hoppings to our system and plot it with the new function.

.. jupyter-execute::

    # Plot sites from different families in different colors
    def family_color(site):
        if site.family == a:
            return 'red'
        if site.family == b:
            return 'green'
        else:
            return 'blue'

    def plot_system(syst):
        kwant.plot(syst, site_lw=0.1, site_color=family_color)

    ## Add sites and hoppings.
    for i in range(4):
        for j in range (4):
            syst[a(i, j)] = 4
            syst[b(i, j)] = 4
            syst[c(i, j)] = 4

    syst[lat.neighbors()] = -1

    ## Plot the system.
    plot_system(syst)


How to create many similar hoppings in one go?
==============================================
This can be achieved with an instance of the class `kwant.builder.HoppingKind`.
In fact, sites and hoppings are not the only possible keys when assigning
values to a `~kwant.builder.Builder`.  There exists a mechanism to
`~kwant.builder.Builder.expand` more general keys into these simple keys.

A ``HoppingKind``, the most comonly used general key, is a way of specifying
all hoppings of a particular "kind", between two site families. For example
``HoppingKind((1, 0), lat_a, lat_b)`` represents all hoppings of the form
``(lat_a(x + (1, 0)), lat_b(x))``, where ``x`` is a tag (here, a pair of
integers).

The following example shows how this can be used:

.. jupyter-execute::

    # Create hopping between neighbors with HoppingKind
    a = 1
    syst = kwant.Builder()
    lat = kwant.lattice.square(a, norbs=1)
    syst[ (lat(i, j) for i in range(5) for j in range(5)) ] = 4

    syst[kwant.builder.HoppingKind((1, 0), lat)] = -1
    kwant.plot(syst);

Note that ``HoppingKind`` only works with site families so you cannot use
them directly with ``Polyatomic`` lattices; you have to explicitly specify
the sublattices when creating a ``HoppingKind``:

.. jupyter-execute::
    :hide-code:

    lat = kwant.lattice.kagome(norbs=1)
    syst = kwant.Builder()

    a, b, c = lat.sublattices  # The kagome lattice has 3 sublattices

    for i in range(4):
        for j in range (4):
            syst[a(i, j)] = 4
            syst[b(i, j)] = 4
            syst[c(i, j)] = 4

.. jupyter-execute::

    # equivalent to syst[kwant.builder.HoppingKind((0, 1), b)] = -1
    syst[kwant.builder.HoppingKind((0, 1), b, b)] = -1

Here, we want the hoppings between the sites from sublattice b with a direction of (0,1) in the lattice coordinates.

.. jupyter-execute::
    :hide-code:

    plot_system(syst)

.. jupyter-execute::
    :hide-code:

    # Delete the hoppings previously created
    del syst[kwant.builder.HoppingKind((0, 1), b, b)]

.. jupyter-execute::

    syst[kwant.builder.HoppingKind((0, 0), a, b)] = -1
    syst[kwant.builder.HoppingKind((0, 0), a, c)] = -1
    syst[kwant.builder.HoppingKind((0, 0), c, b)] = -1

Here, we create hoppings between the sites of the same lattice coordinates but from different families.

.. jupyter-execute::

    plot_system(syst)


How to set the hoppings between adjacent sites?
===============================================
``Polyatomic`` and ``Monatomic`` lattices have a method `~kwant.lattice.Polyatomic.neighbors`
that returns a list of ``HoppingKind`` instances that connect sites with their
(n-nearest) neighors:

.. jupyter-execute::

    # Create hoppings with lat.neighbors()
    syst = kwant.Builder()
    lat = kwant.lattice.square(norbs=1)
    syst[(lat(i, j) for i in range(3) for j in range(3))] = 4

    syst[lat.neighbors()] = -1  # Equivalent to lat.neighbors(1)
    kwant.plot(syst);

.. jupyter-execute::

    del syst[lat.neighbors()]  # Delete all nearest-neighbor hoppings
    syst[lat.neighbors(2)] = -1

    kwant.plot(syst);

As we can see in the figures above, ``lat.neighbors()`` returns the hoppings between the first nearest neighbors and ``lat.neighbors(2)`` returns the hoppings between the second nearest neighbors.

When using a ``Polyatomic`` lattice ``neighbors()`` knows about the different
sublattices:

.. jupyter-execute::

    # Create the system
    lat = kwant.lattice.kagome(norbs=1)
    syst = kwant.Builder()
    a, b, c = lat.sublattices  # The kagome lattice has 3 sublattices

    for i in range(4):
        for j in range (4):
            syst[a(i, j)] = 4  # red
            syst[b(i, j)] = 4  # green
            syst[c(i, j)] = 4  # blue

    syst[lat.neighbors()] = -1

    plot_system(syst)

However, if we use the ``neighbors()`` method of a single sublattice, we will
only get the neighbors *on that sublattice*:

.. jupyter-execute::

    del syst[lat.neighbors()]  # Delete the hoppings previously created
    syst[a.neighbors()] = -1
    plot_system(syst)

Note in the above that there are *only* hoppings between the red sites. This
is an artifact of the visualisation: the blue and green sites just happen to lie
in the path of the hoppings, but are not connected by them.


How to make a hole in a system?
===============================
To make a hole in the system, use ``del syst[site]``, just like with any other
mapping. In the following example we remove all sites inside some "hole"
region:

.. jupyter-execute::

    # Define the lattice and the (empty) system
    a = 2
    lat = kwant.lattice.cubic(a, norbs=1)
    syst = kwant.Builder()

    L = 10
    W = 10
    H = 2

    # Add sites to the system in a cuboid

    syst[(lat(i, j, k) for i in range(L) for j in range(W) for k in range(H))] = 4
    kwant.plot(syst);

.. jupyter-execute::

    # Delete sites to create a hole

    def in_hole(site):
        x, y, z = site.pos / a - (L/2, W/2, H/2)  # position relative to centre
        return abs(x) < L / 4 and abs(y) < W / 4

    for site in filter(in_hole, list(syst.sites())):
        del syst[site]

    kwant.plot(syst);


``del syst[site]`` also works after hoppings have been added to the system.
If a site is deleted, then all the hoppings to/from that site are also deleted.


How to access a system's sites?
===============================
The ways of accessing system sites is slightly different depending on whether
we are talking about a ``Builder`` or ``System`` (see `What is a system, and
what is a builder?`_ if you do not know the difference).

We can access the sites of a ``Builder`` by using its `~kwant.builder.Builder.sites` method:

.. jupyter-execute::
    :hide-code:

    builder = kwant.Builder()
    lat = kwant.lattice.square(norbs=1)
    builder[(lat(i, j) for i in range(3) for j in range(3))] = 4

.. jupyter-execute::

    # Before finalizing the system

    sites = list(builder.sites())  # sites() doe *not* return a list

The ``sites()`` method returns an *iterator* over the system sites, and in the
above example we create a list from the contents of this iterator, which
contains all the sites. At this stage the ordering of sites is not fixed, so if
you add more sites to the ``Builder`` and call ``sites()`` again, the sites may
well be returned in a different order.

After finalization, when we are dealing with a ``System``, the sites themselves
are stored in a list, which can be accessed via the ``sites`` attribute:

.. jupyter-execute::

    # After finalizing the system
    syst = builder.finalized()
    sites = syst.sites  # syst.sites is an actual list

The order of sites in a ``System`` is fixed, and also defines the ordering of
the system Hamiltonian, system wavefunctions etc. (see `How does Kwant order components of an individual wavefunction?`_ for details).

``System`` also contains the inverse mapping, ``id_by_site`` which gives us
the index of a given site within the system:

.. jupyter-execute::

    i = syst.id_by_site[lat(0, 2)]  # we want the id of the site lat(0, 2)


How to use different lattices for the scattering region and a lead?
===================================================================
Let us take the example of a system containing sites from a honeycomb lattice,
which we want to connect to leads that contain sites from a square lattice.

First we construct the central system:

.. jupyter-execute::

    # Define the scattering Region
    L = 5
    W = 5

    lat = kwant.lattice.honeycomb(norbs=1)
    subA, subB = lat.sublattices

    syst = kwant.Builder()
    syst[(subA(i, j) for i in range(L) for j in range(W))] = 4
    syst[(subB(i, j) for i in range(L) for j in range(W))] = 4
    syst[lat.neighbors()] = -1

    kwant.plot(syst);

and the lead:

.. jupyter-execute::

    # Create a lead
    lat_lead = kwant.lattice.square(norbs=1)
    sym_lead1 = kwant.TranslationalSymmetry((0, 1))

    lead1 = kwant.Builder(sym_lead1)
    lead1[(lat_lead(i, 0) for i in range(2, 7))] = 4
    lead1[lat_lead.neighbors()] = -1

    kwant.plot(syst);

We cannot simply use `~kwant.builder.Builder.attach_lead` to attach this lead to the
system with the honeycomb lattice because Kwant does not know how sites from
these two lattices should be connected.

We must first add a layer of sites from the square lattice to the system and manually
add the hoppings from these sites to the sites from the honeycomb lattice:

.. jupyter-execute::

    syst[(lat_lead(i, 5) for i in range(2, 7))] = 4
    syst[lat_lead.neighbors()] = -1

    # Manually attach sites from graphene to square lattice
    syst[((lat_lead(i+2, 5), subB(i, 4)) for i in range(5))] = -1

    kwant.plot(syst);

``attach_lead()`` will now be able to attach the lead:

.. jupyter-execute::

    syst.attach_lead(lead1)
    kwant.plot(syst);


How to cut a finite system out of a system with translational symmetries?
=========================================================================
This can be achieved using the `~kwant.builder.Builder.fill` method to fill a
``Builder`` with a ``Builder`` with higher symmetry.

When using the ``fill()`` method, we need two systems: the template and the
target. The template is a ``Builder`` with some translational symmetry that
will be repeated in the desired shape to create the final system.

For example, say we want to create a simple model on a cubic lattice:

.. jupyter-execute::

    # Create 3d model.
    cubic = kwant.lattice.cubic(norbs=1)
    sym_3d = kwant.TranslationalSymmetry([1, 0, 0], [0, 1, 0], [0, 0, 1])
    model = kwant.Builder(sym_3d)
    model[cubic(0, 0, 0)] = 4
    model[cubic.neighbors()] = -1

We have now created our "template" ``Builder`` which has 3 translational
symmetries. Next we will fill a system with no translational symmetries with
sites and hoppings from the template inside a cuboid:

.. jupyter-execute::

    # Build scattering region (white).
    def cuboid_shape(site):
        x, y, z = abs(site.pos)
        return x < 4 and y < 10 and z < 3

    cuboid = kwant.Builder()
    cuboid.fill(model, cuboid_shape, (0, 0, 0));
    kwant.plot(cuboid);

We can then use the original template to create a lead, which has 1 translational
symmetry. We can then use this lead as a template to fill another section of
the system with a cylinder of sites and hoppings:

.. jupyter-execute::

    # Build electrode (black).
    def electrode_shape(site):
        x, y, z = site.pos - (0, 5, 2)
        return y**2 + z**2 < 2.3**2

    electrode = kwant.Builder(kwant.TranslationalSymmetry([1, 0, 0]))
    electrode.fill(model, electrode_shape, (0, 5, 2))  # lead

    # Scattering region
    cuboid.fill(electrode, lambda s: abs(s.pos[0]) < 7, (0, 5, 4))

    cuboid.attach_lead(electrode)

    kwant.plot(cuboid);

How does Kwant order the propagating modes of a lead?
=====================================================
A very useful feature of kwant is to calculate the transverse wavefunctions of
propagating modes in a system with 1 translational symmetry.  This can be
achieved with the `~kwant.system.InfiniteSystem.modes` method, which returns a
pair of objects, the first of which contains the propagating modes of the
system in a `~kwant.physics.PropagatingModes` object:

.. jupyter-execute::

    lat = kwant.lattice.square(norbs=1)

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
    lead[(lat(0, i) for i in range(3))] = 4
    lead[lat.neighbors()] = -1

    flead = lead.finalized()

    E = 2.5
    prop_modes, _ = flead.modes(energy=E)

``PropagatingModes`` contains the wavefunctions, velocities and momenta of the
modes at the requested energy (2.5 in this example).  In order to understand
the order in which these quantities are returned it is often useful to look at
the a section of the band structure for the system in question:

.. jupyter-execute::
    :hide-code:

    def plot_and_label_modes(lead, E):
        # Plot the different modes
        pmodes, _ = lead.modes(energy=E)
        kwant.plotter.bands(lead, show=False)
        for i, k in enumerate(pmodes.momenta):
            plt.plot(k, E, 'ko')
            plt.annotate(str(i), xy=(k, E), xytext=(-5, 8),
                         textcoords='offset points',
                         bbox=dict(boxstyle='round,pad=0.1',fc='white', alpha=0.7))
        plt.plot([-3, 3], [E, E], 'r--')
        plt.ylim(E-1, E+1)
        plt.xlim(-2, 2)
        plt.xlabel("momentum")
        plt.ylabel("energy")
        plt.show()

    plot_and_label_modes(flead, E)

On the above band structure we have labelled the 4 modes in the order
that they appear in the output of ``modes()`` at energy 2.5. Note that
the modes are sorted in the following way:

    + First all the modes with negative velocity, then all the modes with
      positive velocity
    + Negative velocity modes are ordered by *increasing* momentum
    + Positive velocity modes are ordered by *decreasing* momentum

For more complicated systems and band structures this can lead to some
unintuitive orderings:

.. jupyter-execute::
    :hide-code:

    lat = kwant.lattice.square(norbs=2)

    s0 = np.eye(2)
    sz = np.array([[1, 0], [0, -1]])

    lead2 = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))

    lead2[(lat(0, i) for i in range(2))] = np.diag([1.8, -1])
    lead2[lat.neighbors()] = -1 * sz

    flead2 = lead2.finalized()

    plot_and_label_modes(flead2, 1)


How does Kwant order scattering states?
=======================================
Scattering states calculated using `~kwant.solvers.default.wave_function` are returned in the
same order as the "incoming" modes of `~kwant.system.InfiniteSystem.modes`.
Kwant considers that the translational symmetry of a lead points "towards
infinity" (*not* towards the system) which means that the incoming modes are
those that have *negative* velocities.

This means that for a lead attached on the left of a scattering region (with
symmetry vector :math:`(-1, 0)`, for example), the
positive :math:`k` direction (when inspecting the lead's band structure) actually
corresponds to the *negative* :math:`x` direction.


How does Kwant order components of an individual wavefunction?
==============================================================
In `How to access a system's sites?`_ we saw that the sites of a
finalized system are available as a list through the ``sites`` attribute, and
that one can look up the index of a site with the ``id_by_site`` attribute.

When all the site families present in a system have only 1 degree of freedom
per site (i.e.  the all the onsites are scalars) then the index into a
wavefunction defined over the system is exactly the site index:

.. jupyter-execute::
    :hide-code:

    def circle(R):
        return lambda r: np.linalg.norm(r) < R


    def make_system(lat):
        norbs = lat.norbs
        syst = kwant.Builder()
        syst[lat.shape(circle(3), (0, 0))] = 4 * np.eye(norbs)
        syst[lat.neighbors()] = -1 * np.eye(norbs)

        lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)))
        lead[(lat(0, i) for i in range(-1, 2))] = 4 * np.eye(norbs)
        lead[lat.neighbors()] = -1 * np.eye(norbs)

        syst.attach_lead(lead)
        syst.attach_lead(lead.reversed())

        return syst.finalized()

.. jupyter-execute::

    lat = kwant.lattice.square(norbs=1)
    syst = make_system(lat)
    scattering_states = kwant.wave_function(syst, energy=1)
    wf = scattering_states(0)[0]  # scattering state from lead 0 incoming in mode 0

    idx = syst.id_by_site[lat(0, 0)]  # look up index of site

    print('wavefunction on lat(0, 0): ', wf[idx])

We see that the wavefunction on a single site is a single complex number, as
expected.

If a site family have more than 1 degree of freedom per site (e.g. spin or
particle-hole) then Kwant places degrees of freedom on the same site adjacent
to one another.  In the case where all site families in the system have the
*same* number of degrees of freedom, we can then simply *reshape* the
wavefunction into a matrix, where the row number indexes the site, and the
column number indexes the degree of freedom on that site:

.. jupyter-execute::

    lat = kwant.lattice.square(norbs=2)
    syst = make_system(lat)
    scattering_states = kwant.wave_function(syst, energy=1)
    wf = scattering_states(0)[0]  # scattering state from lead 0 incoming in mode 0

    idx = syst.id_by_site[lat(0, 0)]  # look up index of site

    # Group consecutive degrees of freedom from 'wf' together; these correspond
    # to degrees of freedom on the same site.
    wf = wf.reshape(-1, 2)

    print('wavefunction on lat(0, 0): ', wf[idx])

We see that the wavefunction on a single site is a *vector* of 2 complex numbers,
as we expect.

If there are different site families present in the system that have *different*
numbers of orbitals per site, then the situation becomes much more involved,
because we cannot simply "reshape" the wavefunction like we did in the
preceding example.
