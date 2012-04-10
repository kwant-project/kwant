.. _tutorial-superconductor:

Superconductors: orbital vs lattice degrees of freedom
------------------------------------------------------

This example deals with superconductivity on the level of the
Bogoliubov-de Gennes (BdG) equation. In this framework, the Hamiltonian
is given as

.. math::

    H = \begin{pmatrix} H_0 - \mu& \Delta\\ \Delta^\dagger&\mu-\mathcal{T}H\mathcal{T}^{-1}\end{pmatrix}

where :math:`H_0` is the Hamiltonian of the system without
superconductivity, :math:`\mu` the chemical potential, :math:`\Delta`
the superconducting order parameter, and :math:`\mathcal{T}`
the time-reversal operator. The BdG Hamiltonian introduces
electron and hole degrees of freedom (an artificial doubling -
be aware of the fact that electron and hole excitations
are related!), which we now implement in `kwant`.

For this we restrict ourselves to a simple spin-less system without
magnetic field, so that :math:`\Delta` is just a number (which we
choose real), and :math:`\mathcal{T}H\mathcal{T}^{-1}=H_0^*=H_0`.

"Orbital description": Using matrices
.....................................

We begin by computing the band structure of a superconducting wire.
The most natural way to implement the BdG Hamiltonian is by using a
2x2 matrix structure for all Hamiltonian matrix elements:

.. literalinclude:: ../../../examples/tutorial5a.py
    :lines: 21-45

As you see, the example is syntactically equivalent to our
:ref:`spin example <tutorial_spinorbit>`, the only difference
is now that the Pauli matrices act in electron-hole space.

Computing the band structure then yields the result

.. image:: ../images/tutorial5a_result.*

We clearly observe the superconducting gap in the spectrum. That was easy,
he?

.. seealso::
    The full source code can be found in
    :download:`examples/tutorial5a.py <../../../examples/tutorial5a.py>`


"Lattice description": Using different lattices
...............................................

While it seems most natural to implement the BdG Hamiltonian
using a 2x2 matrix structure for the matrix elements of the Hamiltonian,
we run into a problem when we want to compute electronic transport in
a system consisting of a normal and a superconducting lead:
Since electrons and holes carry charge with opposite sign, we need to
separate electron and hole degrees of freedom in the scattering matrix.
In particular, the conductance of a N-S-junction is given as

.. math::

    G = \frac{e^2}{h} (N - R_\text{ee} + R_\text{he})\,,

where :math:`N` is the number of channels in the normal lead, and
:math:`R_\text{ee}` the total probability of reflection from electrons
to electrons in the normal lead, and :math:`R_\text{eh}` the total
probability of reflection from electrons to holes in the normal
lead. However, the current version of kwant does not allow for an easy
and elegant partitioning of the scattering matrix in these two degrees
of freedom (well, there is one since v0.1.3, see the technical notes
below).

In the following, we will circumvent this problem by introducing
separate "leads" for electrons and holes, making use of different
lattices. The system we consider consists of a normal lead on the left,
a superconductor on the right, and a tunnel barrier inbetween:

.. image:: ../images/tutorial5b_sketch.*

As already mentioned above, we begin by introducing two different
square lattices representing electron and hole degrees of freedom:

.. literalinclude:: ../../../examples/tutorial5b.py
    :lines: 18-19,17,23-24

Any diagonal entry (kinetic energy, potentials, ...) in the BdG
Hamiltonian then corresponds to on-site energies or hoppings within
the *same* lattice, whereas any off-diagonal entry (essentially, the
superconducting order parameter :math:`\Delta`) corresponds
to a hopping between *different* lattices:

.. literalinclude:: ../../../examples/tutorial5b.py
    :lines: 25-46

Note that the tunnel barrier is added by overwriting previously set
on-site matrix elements.

Note further, that in the code above, the superconducting order
parameter is nonzero only in a part of the scattering region.
Consequently, we have added hoppings between electron and hole
lattices only in this region, they remain uncoupled in the normal
part. We use this fact to attach purely electron and hole leads
(comprised of only electron *or* hole lattices) to the
system:

.. literalinclude:: ../../../examples/tutorial5b.py
    :lines: 49-65

This separation into two different leads allows us then later to compute the
reflection probablities between electrons and holes explicitely.

On the superconducting side, we cannot do this separation, and can
only define a single lead coupling electrons and holes:

.. literalinclude:: ../../../examples/tutorial5b.py
    :lines: 70-80

We now have on the left side two leads that are sitting in the same
spatial position, but in different lattice spaces. This ensures that
we can still attach all leads as before:

.. literalinclude:: ../../../examples/tutorial5b.py
    :lines: 83-87

When computing the conductance, we can now extract reflection from
electrons to electrons as ``smatrix.transmission(0, 0)`` (Don't get
confused by the fact that it says ``transmission`` -- transmission
into the same lead is reflection), and reflection from electrons to holes
as ``smatrix.transmission(1, 0)``, by virtue of our electron and hole leads:

.. literalinclude:: ../../../examples/tutorial5b.py
    :lines: 89-97

Note that ``smatrix.submatrix(0,0)`` returns the block concerning reflection
within (electron) lead 0, and from its size we can extract the number of modes
:math:`N`.

Finally, for the default parameters, we obtain the following result:

.. image:: ../images/tutorial5b_result.*

We a see a conductance that is proportional to the square of the tunneling
probability within the gap, and proportional to the tunneling probability
above the gap. At the gap edge, we observe a resonant Andreev reflection.

.. seealso::
    The full source code can be found in
    :download:`examples/tutorial5b.py <../../../examples/tutorial5b.py>`

.. specialnote:: Technical details

    - If you are only interested in particle (thermal) currents you do not need
      to define separate electron and hole leads. In this case, you do not
      need to distinguish them. Still, separating the leads into electron
      and hole degrees of freedom makes the lead calculation in the solving
      phase more efficient.

    - It is in fact possible to separate electron and hole degrees of
      freedom in the scattering matrix, even if one uses matrices for
      these degrees of freedom. In the solve step,
      `~kwant.solvers.sparse.solve` returns an array containing the
      transverse wave functions of the lead modes, if
      ``return_modes=True``. By inspecting the wave functions,
      electron and hole wave functions can be distinguished (they only
      have entries in either the electron part *or* the hole part. If
      you encounter modes with entries in both parts, you hit a very
      unlikely situation in which the standard procedure to compute
      the modes gave you a superposition of electron and hole
      modes. That is still OK for computing particle current, but not
      for electrical current).
