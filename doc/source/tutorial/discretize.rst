Discretizing continuous Hamiltonians
------------------------------------

Introduction
............

In ":ref:`tutorial_discretization_schrodinger`" we have learnt that Kwant works
with tight-binding Hamiltonians. Often, however, one will start with a
continuum model and will subsequently need to discretize it to arrive at a
tight-binding model.
Although discretizing a Hamiltonian is usually a simple
process, it is tedious and repetitive. The situation is further exacerbated
when one introduces additional on-site degrees of freedom, and tracking all
the necessary terms becomes a chore.
The `~kwant.continuum` sub-package aims to be a solution to this problem.
It is a collection of tools for working with
continuum models and for discretizing them into tight-binding models.

.. seealso::
    The complete source code of this tutorial can be found in
    :download:`discretize.py </code/download/discretize.py>`


.. _tutorial_discretizer_introduction:

Discretizing by hand
....................

As an example, let us consider the following continuum Schrödinger equation
for a semiconducting heterostructure (using the effective mass approximation):

.. math::

    \left( k_x \frac{\hbar^2}{2 m(x)} k_x \right) \psi(x) = E \, \psi(x).

Replacing the momenta by their corresponding differential operators

.. math::
    k_\alpha = -i \partial_\alpha,

for :math:`\alpha = x, y` or :math:`z`, and discretizing on a regular lattice of
points with spacing :math:`a`, we obtain the tight-binding model

.. math::

    H = - \frac{1}{a^2} \sum_i A\left(x+\frac{a}{2}\right)
            \big(\ket{i}\bra{i+1} + h.c.\big)
        + \frac{1}{a^2} \sum_i
            \left( A\left(x+\frac{a}{2}\right) + A\left(x-\frac{a}{2}\right)\right)
            \ket{i} \bra{i},

with :math:`A(x) = \frac{\hbar^2}{2 m(x)}`.

Using `~kwant.continuum.discretize` to obtain a template
........................................................

The function `kwant.continuum.discretize` takes a symbolic Hamiltonian and
turns it into a `~kwant.builder.Builder` instance with appropriate spatial
symmetry that serves as a template.
(We will see how to use the template to build systems with a particular
shape later).

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_symbolic_discretization
    :end-before: #HIDDEN_END_symbolic_discretization

It is worth noting that ``discretize`` treats ``k_x`` and ``x`` as
non-commuting operators, and so their order is preserved during the
discretization process.

The builder produced by ``discretize`` may be printed to show the source code of its onsite and hopping functions (this is a special feature of builders returned by ``discretize``):

.. literalinclude:: /code/figure/discretizer_intro_verbose.txt

.. specialnote:: Technical details

    - ``kwant.continuum`` uses ``sympy`` internally to handle symbolic
      expressions. Strings are converted using `kwant.continuum.sympify`,
      which essentially applies some Kwant-specific rules (such as treating
      ``k_x`` and ``x`` as non-commutative) before calling ``sympy.sympify``

    - The builder returned by ``discretize`` will have an N-D
      translational symmetry, where ``N`` is the number of dimensions that were
      discretized. This is the case, even if there are expressions in the input
      (e.g. ``V(x, y)``) which in principle *may not* have this symmetry.  When
      using the returned builder directly, or when using it as a template to
      construct systems with different/lower symmetry, it is important to
      ensure that any functional parameters passed to the system respect the
      symmetry of the system. Kwant provides no consistency check for this.

    - The discretization process consists of taking input
      :math:`H(k_x, k_y, k_z)`, multiplying it from the right by
      :math:`\psi(x, y, z)` and iteratively applying a second-order accurate
      central derivative approximation for every
      :math:`k_\alpha=-i\partial_\alpha`:

      .. math::
         \partial_\alpha \psi(\alpha) =
            \frac{1}{a} \left( \psi\left(\alpha + \frac{a}{2}\right)
                              -\psi\left(\alpha - \frac{a}{2}\right)\right).

      This process is done separately for every summand in Hamiltonian.
      Once all symbols denoting operators are applied internal algorithm is
      calculating ``gcd`` for hoppings coming from each summand in order to
      find best possible approximation. Please see source code for details.

    - Instead of using ``discretize`` one can use
      `~kwant.continuum.discretize_symbolic` to obtain symbolic output.
      When working interactively in `Jupyter notebooks <https://jupyter.org/>`_
      it can be useful to use this to see a symbolic representation of
      the discretized Hamiltonian. This works best when combined with ``sympy``
      `Pretty Printing <http://docs.sympy.org/latest/tutorial/printing.html#setting-up-pretty-printing>`_.

    - The symbolic result of discretization obtained with
      ``discretize_symbolic`` can be converted into a
      builder using `~kwant.continuum.build_discretized`.
      This can be useful if one wants to alter the tight-binding Hamiltonian
      before building the system.


Building a Kwant system from the template
.........................................

Let us now use the output of ``discretize`` as a template to
build a system and plot some of its energy eigenstate. For this example the
Hamiltonian will be

.. math::

    H = k_x^2 + k_y^2 + V(x, y),

where :math:`V(x, y)` is some arbitrary potential.

First, use ``discretize`` to obtain a
builder that we will use as a template:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_template
    :end-before: #HIDDEN_END_template

We now use this system with the `~kwant.builder.Builder.fill`
method of `~kwant.builder.Builder` to construct the system we
want to investigate:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_fill
    :end-before: #HIDDEN_END_fill

After finalizing this system, we can plot one of the system's
energy eigenstates:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_plot_eigenstate
    :end-before: #HIDDEN_END_plot_eigenstate

.. image:: /code/figure/discretizer_gs.*

Note in the above that we provided the function ``V`` to
``syst.hamiltonian_submatrix`` using ``params=dict(V=potential)``, rather than
via ``args``.

In addition, the function passed as ``V`` expects two input parameters ``x``
and ``y``, the same as in the initial continuum Hamiltonian.


Models with more structure: Bernevig-Hughes-Zhang
.................................................

When working with multi-band systems, like the Bernevig-Hughes-Zhang (BHZ)
model [1]_ [2]_, one can provide matrix input to `~kwant.continuum.discretize`
using ``identity`` and ``kron``. For example, the definition of the BHZ model can be
written succinctly as:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_define_qsh
    :end-before: #HIDDEN_END_define_qsh

We can then make a ribbon out of this template system:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_define_qsh_build
    :end-before: #HIDDEN_END_define_qsh_build

and plot its dispersion using `kwant.plotter.bands`:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_plot_qsh_band
    :end-before: #HIDDEN_END_plot_qsh_band

.. image:: /code/figure/discretizer_qsh_band.*

In the above we see the edge states of the quantum spin Hall effect, which
we can visualize using `kwant.plotter.map`:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_plot_qsh_wf
    :end-before: #HIDDEN_END_plot_qsh_wf

.. image:: /code/figure/discretizer_qsh_wf.*


Limitations of discretization
.............................

It is important to remember that the discretization of a continuum
model is an *approximation* that is only valid in the low-energy
limit. For example, the quadratic continuum Hamiltonian

.. math::

    H_\textrm{continuous}(k_x) = \frac{\hbar^2}{2m}k_x^2


and its discretized approximation

.. math::

    H_\textrm{tight-binding}(k_x) = 2t \big(1 - \cos(k_x a)\big),


where :math:`t=\frac{\hbar^2}{2ma^2}`, are only valid in the limit
:math:`E < t`. The grid spacing :math:`a` must be chosen according
to how high in energy you need your tight-binding model to be valid.

It is possible to set :math:`a` through the ``grid_spacing`` parameter
to `~kwant.continuum.discretize`, as we will illustrate in the following
example. Let us start from the continuum Hamiltonian

.. math::

  H(k) = k_x^2 \mathbb{1}_{2\times2} + α k_x \sigma_y.

We start by defining this model as a string and setting the value of the
:math:`α` parameter:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_ls_def
    :end-before: #HIDDEN_END_ls_def

Now we can use `kwant.continuum.lambdify` to obtain a function that computes
:math:`H(k)`:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_ls_hk_cont
    :end-before: #HIDDEN_END_ls_hk_cont

We can also construct a discretized approximation using
`kwant.continuum.discretize`, in a similar manner to previous examples:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_ls_hk_tb
    :end-before: #HIDDEN_END_ls_hk_tb

Below we can see the continuum and tight-binding dispersions for two
different values of the discretization grid spacing :math:`a`:

.. image:: /code/figure/discretizer_lattice_spacing.*


We clearly see that the smaller grid spacing is, the better we approximate
the original continuous dispersion. It is also worth remembering that the
Brillouin zone also scales with grid spacing: :math:`[-\frac{\pi}{a},
\frac{\pi}{a}]`.


Advanced topics
...............

The input to `kwant.continuum.discretize` and `kwant.continuum.lambdify` can be
not only a ``string``, as we saw above, but also a ``sympy`` expression or
a ``sympy`` matrix.
This functionality will probably be mostly useful to people who
are already experienced with ``sympy``.


It is possible to use ``identity`` (for identity matrix), ``kron`` (for Kronecker product), as well as Pauli matrices ``sigma_0``,
``sigma_x``, ``sigma_y``, ``sigma_z`` in the input to
`~kwant.continuum.lambdify` and `~kwant.continuum.discretize`, in order to simplify
expressions involving matrices. Matrices can also be provided explicitly using
square ``[]`` brackets. For example, all following expressions are equivalent:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_subs_1
    :end-before: #HIDDEN_END_subs_1

.. literalinclude:: /code/figure/discretizer_subs_1.txt

We can use the ``locals`` keyword parameter to substitute expressions
and numerical values:

.. literalinclude:: /code/include/discretize.py
    :start-after: #HIDDEN_BEGIN_subs_2
    :end-before: #HIDDEN_END_subs_2

.. literalinclude:: /code/figure/discretizer_subs_2.txt

Symbolic expressions obtained in this way can be directly passed to all
``discretizer`` functions.

.. specialnote:: Technical details

  Because of the way that ``sympy`` handles commutation relations all symbols
  representing position and momentum operators are set to be non commutative.
  This means that the order of momentum and position operators in the input
  expression is preserved.  Note that it is not possible to define individual
  commutation relations within ``sympy``, even expressions such :math:`x k_y x`
  will not be simplified, even though mathematically :math:`[x, k_y] = 0`.


.. rubric:: References

.. [1] `Science, 314, 1757 (2006)
    <https://arxiv.org/abs/cond-mat/0611399>`_.

.. [2] `Phys. Rev. B 82, 045122 (2010)
    <https://arxiv.org/abs/1005.1682>`_.
