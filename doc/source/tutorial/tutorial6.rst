Plotting Kwant systems and data in various styles
-------------------------------------------------

The plotting functionality of Kwant has been used extensively (through
`~kwant.plotter.plot` and `~kwant.plotter.map`) in the previous tutorials. In
addition to this basic use, `~kwant.plotter.plot` offers many options to change
the plotting style extensively. It is the goal of this tutorial to show how
these options can be used to achieve various very different objectives.

2D example: graphene quantum dot
................................

We begin by first considering a circular graphene quantum dot (similar to what
has been used in parts of the tutorial :ref:`tutorial-graphene`.)  In contrast
to previous examples, we will also use hoppings beyond next-nearest neighbors:

.. literalinclude:: plot_graphene.py
    :start-after: #HIDDEN_BEGIN_makesys
    :end-before: #HIDDEN_END_makesys

Note that adding hoppings hoppings to the `n`-th nearest neighbors can be
simply done by passing `n` as an argument to
`~kwant.lattice.Polyatomic.neighbors`. Also note that we use the method
`~kwant.builder.Builder.eradicate_dangling` to get rid of single atoms sticking
out of the shape. It is necessary to do so *before* adding the
next-nearest-neighbor hopping [#]_.

Of course, the system can be plotted simply with default settings:

.. literalinclude:: plot_graphene.py
    :start-after: #HIDDEN_BEGIN_plotsys1
    :end-before: #HIDDEN_END_plotsys1

However, due to the richer structure of the lattice, this results in a rather
busy plot:

.. image:: ../images/plot_graphene_sys1.*

A much clearer plot can be obtained by using different colors for both
sublattices, and by having different line widths for different hoppings.  This
can be achieved by passing a function to the arguments of
`~kwant.plotter.plot`, instead of a constant. For properties of sites, this
must be a function taking one site as argument, for hoppings a function taking
the start end end site of hopping as arguments:

.. literalinclude:: plot_graphene.py
    :start-after: #HIDDEN_BEGIN_plotsys2
    :end-before: #HIDDEN_END_plotsys2

Note that since we are using an unfinalized Builder, a `site` is really an
instance of `~kwant.builder.Site`. With these adjustments we arrive at a plot
that carries the same information, but is much easier to interpret:

.. image:: ../images/plot_graphene_sys2.*

Apart from plotting the *system* itself, `~kwant.plotter.plot` can also be used
to plot *data* living on the system.

As an example, we now compute the eigenstates of the graphene quantum dot and
intend to plot the wave function probability in the quantum dot. For aesthetic
reasons (the wave functions look a bit nicer), we restrict ourselves to
nearest-neighbor hopping.  Computing the wave functions is done in the usual
way (note that for a large-scale system, one would probably want to use sparse
linear algebra):

.. literalinclude:: plot_graphene.py
    :start-after: #HIDDEN_BEGIN_plotdata1
    :end-before: #HIDDEN_END_plotdata1

In most cases, to plot the wave function probability, one wouldn't use
`~kwant.plotter.plot`, but rather `~kwant.plotter.map`. Here, we plot the
`n`-th wave function using it:

.. literalinclude:: plot_graphene.py
    :start-after: #HIDDEN_BEGIN_plotdata2
    :end-before: #HIDDEN_END_plotdata2

This results in a standard pseudocolor plot, showing in this case (``n=225``) a
graphene edge state, i.e. a wave function mostly localized at the zigzag edges
of the quantum dot.

.. image:: ../images/plot_graphene_data1.*

However although in general preferable, `~kwant.plotter.map` has a few
deficiencies for this small system: For example, there are a few distortions at
the edge of the dot. (This cannot be avoided in the type of interpolation used
in `~kwant.plotter.map`). However, we can also use `~kwant.plotter.plot` to
achieve a similar, but smoother result.

For this note that `~kwant.plotter.plot` can also take an array of floats (or
function returning floats) as value for the `site_color` argument (the same
holds for the hoppings). Via the colormap specified in `cmap` these are mapped
to color, just as `~kwant.plotter.map` does! In addition, we can also change
the symbol shape depending on the sublattice. With a triangle pointing up and
down on the respective sublattice, the symbols used by plot fill the space
completely:

.. literalinclude:: plot_graphene.py
    :start-after: #HIDDEN_BEGIN_plotdata3
    :end-before: #HIDDEN_END_plotdata3

Note that with ``hop_lw=0`` we deactivate plotting the hoppings (that would not
serve any purpose here). Moreover, ``site_size=0.5`` guarantees that the two
different types of triangles touch precisely: By default, `~kwant.plotter.plot`
takes all sizes in units of the nearest-neighbor spacing. ``site_size=0.5``
thus means half the distance between neighboring sites (and for the triangles
this is interpreted as the radius of the inner circle).

Finally, note that since we are dealing with a finalized system now, a site `i`
is represented by an integer. In order to obtain the original
`~kwant.builder.Site`, ``sys.site(i)`` can be used.

With this we arrive at

.. image:: ../images/plot_graphene_data2.*

with the same information as `~kwant.plotter.map`, but with a cleaner look.

The way how data is presented of course influences what features of the data
are best visible in a given plot. With `~kwant.plotter.plot` one can easily go
beyond pseudocolor-like plots. For example, we can represent the wave function
probability using the symbols itself:

.. literalinclude:: plot_graphene.py
    :start-after: #HIDDEN_BEGIN_plotdata4
    :end-before: #HIDDEN_END_plotdata4

Here, we choose the symbol size proportional to the wave function probability,
while the site color is transparent to also allow for overlapping symbols to be
visible. The hoppings are also plotted in order to show the underlying lattice.

With this, we arrive at

.. image:: ../images/plot_graphene_data3.*

which shows the edge state nature of the wave function most clearly.

.. seealso::
    The full source code can be found in
    :download:`tutorial/plot_graphene.py <../../../tutorial/plot_graphene.py>`

.. rubric:: Footnotes

.. [#] A dangling site is defined as having only one hopping connecting it to
       the rest. With next-nearest-neighbor hopping also all sites that are
       dangling with only nearest-neighbor hopping have more than one hopping.

3D example: zincblende structure
................................

Zincblende is a very common crystal structure of semiconductors. It is a
face-centered cubic crystal with two inequivalent atoms in the unit cell
(i.e. two different types of atoms, unlike diamond which has the same crystal
structure, but two equivalent atoms per unit cell).

It is very easily generated in Kwant with `kwant.lattice.general`:

.. literalinclude:: plot_zincblende.py
    :start-after: #HIDDEN_BEGIN_zincblende1
    :end-before: #HIDDEN_END_zincblende1

Note how we keep references to the two different sublattices for later use.

A three-dimensional structure is created as easily as in two dimensions, by
using the `~kwant.lattice.PolyatomicLattice.shape`-functionality:

.. literalinclude:: plot_zincblende.py
    :start-after: #HIDDEN_BEGIN_zincblende2
    :end-before: #HIDDEN_END_zincblende2

We restrict ourselves here to a simple cuboid, and do not bother to add real
values for onsite and hopping energies, but only the placeholder ``None`` (in a
real calculation, several atomic orbitals would have to be considered).

`~kwant.plotter.plot` can plot 3D systems just as easily as its two-dimensional
counterparts:

.. literalinclude:: plot_zincblende.py
    :start-after: #HIDDEN_BEGIN_plot1
    :end-before: #HIDDEN_END_plot1

resulting in

.. image:: ../images/plot_zincblende_sys1.*

You might notice that the standard options for plotting are quite different in
3D than in 2D. For example, by default hoppings are not printed, but sites are
instead represented by little "balls" touching each other (which is achieved by
a default ``site_size=0.5``). In fact, this style of plotting 3D shows quite
decently the overall geometry of the system.

When plotting into a window, the 3D plots can also be rotated and scaled
arbitrarily, allowing for a good inspection of the geometry from all sides.

.. note::

    Interactive 3D plots usually do not have the proper aspect ratio, but are a
    bit squashed. This is due to bugs in matplotlib's 3D plotting module that
    does not properly honor the corresponding arguments. By resizing the plot
    window however one can manually adjust the aspect ratio.

Also for 3D it is possible to customize the plot. For example, we
can explicitly plot the hoppings as lines, and color sites differently
depending on the sublattice:

.. literalinclude:: plot_zincblende.py
    :start-after: #HIDDEN_BEGIN_plot2
    :end-before: #HIDDEN_END_plot2

which results in a 3D plot that allows to interactively (when plotted
in a window) explore the crystal structure:

.. image:: ../images/plot_zincblende_sys2.*

Hence, a few lines of code using Kwant allow to explore all the different
crystal lattices out there!

.. note::

    - The 3D plots are in fact only *fake* 3D. For example, sites will always
      be plotted above hoppings (this is due to the limitations of matplotlib's
      3d module)
    - Plotting hoppings in 3D is inherently much slower than plotting sites.
      Hence, this is not done by default.

.. seealso::
    The full source code can be found in :download:`tutorial/plot_zincblende.py
    <../../../tutorial/plot_zincblende.py>`
