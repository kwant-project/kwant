Computing local quantities: densities and currents
==================================================

In the previous tutorials we have mainly concentrated on calculating
*global* properties such as conductance and band structures. Often,
however, insight can be gained from calculating *locally-defined*
quantities, that is, quantities defined over individual sites or
hoppings in your system. In the [closed-systems]{role="ref"} tutorial we
saw how we could visualize the density associated with the eigenstates
of a system using `kwant.plotter.map`.

In this tutorial we will see how we can calculate more general
quantities than simple densities by studying spin transport in a system
with a magnetic texture.

::: {.seealso}
The complete source code of this example can be found in
[magnetic\_texture.py \</code/download/magnetic\_texture.py\>]{role="download"}
:::

Introduction
------------

Our starting point will be the following spinful tight-binding model on
a square lattice:

$$H = - \sum_{⟨ij⟩}\sum_{α} |iα⟩⟨jα|
+ J \sum_{i}\sum_{αβ} \mathbf{m}_i⋅ \mathbf{σ}_{αβ} |iα⟩⟨iβ|,$$

where latin indices run over sites, and greek indices run over spin. We
can identify the first term as a nearest-neighbor hopping between
like-spins, and the second as a term that couples spins on the same
site. The second term acts like a magnetic field of strength $J$ that
varies from site to site and that, on site $i$, points in the direction
of the unit vector $\mathbf{m}_i$. $\mathbf{σ}_{αβ}$ is a vector of
Pauli matrices. We shall take the following form for $\mathbf{m}_i$:

$$\begin{aligned}
\mathbf{m}_i &=\ \left(
\frac{x_i}{x_i^2 + y_i^2} \sin θ_i,\
\frac{y_i}{x_i^2 + y_i^2} \sin θ_i,\
\cos θ_i \right)^T,
\\
θ_i &=\ \frac{π}{2} (\tanh \frac{r_i - r_0}{δ} - 1),
\end{aligned}$$

where $x_i$ and $y_i$ are the $x$ and $y$ coordinates of site $i$, and
$r_i = \sqrt{x_i^2 + y_i^2}$.

To define this model in Kwant we start as usual by defining functions
that depend on the model parameters:

```python
# Tutorial 2.7. Spin textures
# ===========================
#
# Physics background
# ------------------
#  - Spin textures
#  - Skyrmions
#
# Kwant features highlighted
# --------------------------
#  - operators
#  - plotting vector fields

from math import sin, cos, tanh, pi
import itertools
import numpy as np
import tinyarray as ta
import matplotlib.pyplot as plt

import kwant

sigma_0 = ta.array([[1, 0], [0, 1]])
sigma_x = ta.array([[0, 1], [1, 0]])
sigma_y = ta.array([[0, -1j], [1j, 0]])
sigma_z = ta.array([[1, 0], [0, -1]])

# vector of Pauli matrices σ_αiβ where greek
# letters denote spinor indices
sigma = np.rollaxis(np.array([sigma_x, sigma_y, sigma_z]), 1)

def field_direction(pos, r0, delta):
    x, y = pos
    r = np.linalg.norm(pos)
    r_tilde = (r - r0) / delta
    theta = (tanh(r_tilde) - 1) * (pi / 2)

    if r == 0:
        m_i = [0, 0, -1]
    else:
        m_i = [
            (x / r) * sin(theta),
            (y / r) * sin(theta),
            cos(theta),
        ]

    return np.array(m_i)


def scattering_onsite(site, r0, delta, J):
    m_i = field_direction(site.pos, r0, delta)
    return J * np.dot(m_i, sigma)


def lead_onsite(site, J):
    return J * sigma_z

```

and define our system as a square shape on a square lattice with two
orbitals per site, with leads attached on the left and right:

```python



lat = kwant.lattice.square(norbs=2)

def make_system(L=80):

    syst = kwant.Builder()

    def square(pos):
        return all(-L/2 < p < L/2 for p in pos)

    syst[lat.shape(square, (0, 0))] = scattering_onsite
    syst[lat.neighbors()] = -sigma_0

    lead = kwant.Builder(kwant.TranslationalSymmetry((-1, 0)),
                         conservation_law=-sigma_z)

    lead[lat.shape(square, (0, 0))] = lead_onsite
    lead[lat.neighbors()] = -sigma_0

    syst.attach_lead(lead)
    syst.attach_lead(lead.reversed())

    return syst

```

Below is a plot of a projection of $\mathbf{m}_i$ onto the x-y plane
inside the scattering region. The z component is shown by the color
scale:

![image](/code/figure/mag_field_direction.*)

We will now be interested in analyzing the form of the scattering states
that originate from the left lead:

```python



def plot_vector_field(syst, params):
    xmin, ymin = min(s.tag for s in syst.sites)
    xmax, ymax = max(s.tag for s in syst.sites)
    x, y = np.meshgrid(np.arange(xmin, xmax+1), np.arange(ymin, ymax+1))

    m_i = [field_direction(p, **params) for p in zip(x.flat, y.flat)]
    m_i = np.reshape(m_i, x.shape + (3,))
    m_i = np.rollaxis(m_i, 2, 0)

    fig, ax = plt.subplots(1, 1)
    im = ax.quiver(x, y, *m_i, pivot='mid', scale=75)
    fig.colorbar(im)
    plt.show()


def plot_densities(syst, densities):
    fig, axes = plt.subplots(1, len(densities))
    for ax, (title, rho) in zip(axes, densities):
        kwant.plotter.map(syst, rho, ax=ax, a=4)
        ax.set_title(title)
    plt.show()


def plot_currents(syst, currents):
    fig, axes = plt.subplots(1, len(currents))
    if not hasattr(axes, '__len__'):
        axes = (axes,)
    for ax, (title, current) in zip(axes, currents):
        kwant.plotter.current(syst, current, ax=ax, colorbar=False)
        ax.set_title(title)
    plt.show()


def main():
    syst = make_system().finalized()

    params = dict(r0=20, delta=10, J=1)
    wf = kwant.wave_function(syst, energy=-1, params=params)
    psi = wf(0)[0]

```

Local densities
---------------

If we were simulating a spinless system with only a single degree of
freedom, then calculating the density on each site would be as simple as
calculating the absolute square of the wavefunction like:

    density = np.abs(psi)**2

When there are multiple degrees of freedom per site, however, one has to
be more careful. In the present case with two (spin) degrees of freedom
per site one could calculate the per-site density like:

```python


    plot_vector_field(syst, dict(r0=20, delta=10))

    # even (odd) indices correspond to spin up (down)
    up, down = psi[::2], psi[1::2]
    density = np.abs(up)**2 + np.abs(down)**2

```

With more than one degree of freedom per site we have more freedom as to
what local quantities we can meaningfully compute. For example, we may
wish to calculate the local z-projected spin density. We could calculate
this in the following way:

```python


    # spin down components have a minus sign
    spin_z = np.abs(up)**2 - np.abs(down)**2

```

If we wanted instead to calculate the local y-projected spin density, we
would need to use an even more complicated expression:

```python


    # spin down components have a minus sign
    spin_y = 1j * (down.conjugate() * up - up.conjugate() * down)

```

The `kwant.operator` module aims to alleviate somewhat this
tedious book-keeping by providing a simple interface for defining
operators that act on wavefunctions. To calculate the above quantities
we would use the `~kwant.operator.Density` operator like so:

```python


    rho = kwant.operator.Density(syst)
    rho_sz = kwant.operator.Density(syst, sigma_z)
    rho_sy = kwant.operator.Density(syst, sigma_y)

    # calculate the expectation values of the operators with 'psi'
    density = rho(psi)
    spin_z = rho_sz(psi)
    spin_y = rho_sy(psi)

```

`~kwant.operator.Density` takes a `~kwant.system.System`
as its first parameter as well as (optionally) a square matrix that
defines the quantity that you wish to calculate per site. When an
instance of a `~kwant.operator.Density` is then evaluated with a
wavefunction, the quantity

$$ρ_i = \mathbf{ψ}^†_i \mathbf{M} \mathbf{ψ}_i$$

is calculated for each site $i$, where $\mathbf{ψ}_{i}$ is a vector
consisting of the wavefunction components on that site and $\mathbf{M}$
is the square matrix referred to previously.

Below we can see colorplots of the above-calculated quantities. The
array that is returned by evaluating a `~kwant.operator.Density`
can be used directly with `kwant.plotter.density`:

![image](/code/figure/spin_densities.*)

::: {.specialnote}
Technical Details

Although we refer loosely to \"densities\" and \"operators\" above, a
`~kwant.operator.Density` actually represents a *collection* of
linear operators. This can be made clear by rewriting the above
definition of $ρ_i$ in the following way:

$$ρ_i = \sum_{αβ} ψ^*_{α} \mathcal{M}_{iαβ} ψ_{β}$$

where greek indices run over the degrees of freedom in the Hilbert space
of the scattering region and latin indices run over sites. We can this
identify $\mathcal{M}_{iαβ}$ as the components of a rank-3 tensor and
can represent them as a \"vector of matrices\":

$$\begin{aligned}
\mathcal{M} = \left[
\left(\begin{matrix}
\mathbf{M} & 0 & … \\
0 & 0 & … \\
⋮ & ⋮ & ⋱
\end{matrix}\right)
,\
\left(\begin{matrix}
0 & 0 & … \\
0 & \mathbf{M} & … \\
⋮ & ⋮ & ⋱
\end{matrix}\right)
, … \right]
\end{aligned}$$

where $\mathbf{M}$ is defined as in the main text, and the $0$ are zero
matrices of the same shape as $\mathbf{M}$.
:::

Local currents
--------------

`kwant.operator` also has a class `~kwant.operator.Current`
for calculating local currents, analogously to the local \"densities\"
described above. If one has defined a density via a matrix $\mathbf{M}$
and the above equation, then one can define a local current flowing from
site $b$ to site $a$:

$$J_{ab} = i \left(
\mathbf{ψ}^†_b (\mathbf{H}_{ab})^† \mathbf{M} \mathbf{ψ}_a
- \mathbf{ψ}^†_a \mathbf{M} \mathbf{H}_{ab} \mathbf{ψ}_b
\right),$$

where $\mathbf{H}_{ab}$ is the hopping matrix from site $b$ to site $a$.
For example, to calculate the local current and spin current:

```python


    plot_densities(syst, [
        ('$σ_0$', density),
        ('$σ_z$', spin_z),
        ('$σ_y$', spin_y),
    ])

    J_0 = kwant.operator.Current(syst)
    J_z = kwant.operator.Current(syst, sigma_z)
    J_y = kwant.operator.Current(syst, sigma_y)

    # calculate the expectation values of the operators with 'psi'
    current = J_0(psi)
    spin_z_current = J_z(psi)
    spin_y_current = J_y(psi)

```

Evaluating a `~kwant.operator.Current` operator on a wavefunction
returns a 1D array of values that can be directly used with
`kwant.plotter.current`:

![image](/code/figure/spin_currents.*)

::: {.note}
::: {.admonition-title}
Note
:::

Evaluating a `~kwant.operator.Current` operator on a wavefunction
returns a 1D array of the same length as the number of hoppings in the
system, ordered in the same way as the edges in the system's graph.
:::

::: {.specialnote}
Technical Details

Similarly to how we saw in the previous section that
`~kwant.operator.Density` can be thought of as a collection of
operators, `~kwant.operator.Current` can be defined in a similar
way. Starting from the definition of a \"density\":

$$ρ_a = \sum_{αβ} ψ^*_{α} \mathcal{M}_{aαβ} ψ_{β},$$

we can define *currents* $J_{ab}$ via the continuity equation:

$$\frac{∂ρ_a}{∂t} - \sum_{b} J_{ab} = 0$$

where the sum runs over sites $b$ neigboring site $a$. Plugging in the
definition for $ρ_a$, along with the Schrödinger equation and the
assumption that $\mathcal{M}$ is time independent, gives:

$$J_{ab} = \sum_{αβ}
ψ^*_α \left(i \sum_{γ}
\mathcal{H}^*_{abγα} \mathcal{M}_{aγβ}
- \mathcal{M}_{aαγ} \mathcal{H}_{abγβ}
\right)  ψ_β,$$

where latin indices run over sites and greek indices run over the
Hilbert space degrees of freedom, and

$$\begin{aligned}
\mathcal{H}_{ab} = \left(\begin{matrix}
⋱ & ⋮ & ⋮ & ⋮ & ⋰ \\
⋯ & ⋱ & 0 & \mathbf{H}_{ab} & ⋯ \\
⋯ & 0 & ⋱ & 0 & ⋯ \\
⋯ & 0 & 0 & ⋱ & ⋯ \\
⋰ & ⋮ & ⋮ & ⋮ & ⋱
\end{matrix}\right).
\end{aligned}$$

i.e. $\mathcal{H}_{ab}$ is a matrix that is zero everywhere except on
elements connecting *from* site $b$ *to* site $a$, where it is equal to
the hopping matrix $\mathbf{H}_{ab}$ between these two sites.

This allows us to identify the rank-4 quantity

$$\mathcal{J}_{abαβ} = i \sum_{γ}
\mathcal{H}^*_{abγα} \mathcal{M}_{aγβ}
- \mathcal{M}_{aαγ} \mathcal{H}_{abγβ}$$

as the local current between connected sites.

The diagonal part of this quantity, $\mathcal{J}_{aa}$, represents the
extent to which the density defined by $\mathcal{M}_a$ is not conserved
on site $a$. It can be calculated using `~kwant.operator.Source`,
rather than `~kwant.operator.Current`, which only computes the
off-diagonal part.
:::

Spatially varying operators
---------------------------

The above examples are reasonably simple in the sense that the
book-keeping required to manually calculate the various densities and
currents is still manageable. Now we shall look at the case where we
wish to calculate some projected spin currents, but where the spin
projection axis varies from place to place. More specifically, we want
to visualize the spin current along the direction of $\mathbf{m}_i$,
which changes continuously over the whole scattering region.

Doing this is as simple as passing a *function* when instantiating the
`~kwant.operator.Current`, instead of a constant matrix:

```python


    plot_currents(syst, [
        ('$J_{σ_0}$', current),
        ('$J_{σ_z}$', spin_z_current),
        ('$J_{σ_y}$', spin_y_current),
    ])

    def following_m_i(site, r0, delta):
        m_i = field_direction(site.pos, r0, delta)
        return np.dot(m_i, sigma)

    J_m = kwant.operator.Current(syst, following_m_i)

    # evaluate the operator
    m_current = J_m(psi, params=dict(r0=25, delta=10))

```

The function must take a `~kwant.builder.Site` as its first
parameter, and may optionally take other parameters (i.e. it must have
the same signature as a Hamiltonian onsite function), and must return
the square matrix that defines the operator we wish to calculate.

::: {.note}
::: {.admonition-title}
Note
:::

In the above example we had to pass the extra parameters needed by the
``following_operator`` function via the ``param`` keyword argument. In
general you must pass all the parameters needed by the Hamiltonian via
``params`` (as you would when calling
`~kwant.solvers.default.smatrix` or
`~kwant.solvers.default.wave\_function`). In the previous
examples, however, we used the fact that the system hoppings do not
depend on any parameters (these are the only Hamiltonian elements
required to calculate currents) to avoid passing the system parameters
for the sake of brevity.
:::

Using this we can see that the spin current is essentially oriented
along the direction of $m_i$ in the present regime where the onsite term
in the Hamiltonian is dominant:

![image](/code/figure/spin_current_comparison.*)

::: {.note}
::: {.admonition-title}
Note
:::

Although this example used exclusively `~kwant.operator.Current`,
you can do the same with `~kwant.operator.Density`.
:::

Defining operators over parts of a system
-----------------------------------------

Another useful feature of `kwant.operator` is the ability to
calculate operators over selected parts of a system. For example, we may
wish to calculate the total density of states in a certain part of the
system, or the current flowing through a cut in the system. We can do
this selection when creating the operator by using the keyword parameter
``where``.

### Density of states in a circle

To calculate the density of states inside a circle of radius 20 we can
simply do:

```python


    plot_currents(syst, [
        (r'$J_{\mathbf{m}_i}$', m_current),
        ('$J_{σ_z}$', spin_z_current),
    ])


    def circle(site):
        return np.linalg.norm(site.pos) < 20

    rho_circle = kwant.operator.Density(syst, where=circle, sum=True)

    all_states = np.vstack((wf(0), wf(1)))
    dos_in_circle = sum(rho_circle(p) for p in all_states) / (2 * pi)
    print('density of states in circle:', dos_in_circle)

```

::: {.literalinclude}
/code/figure/circle\_dos.txt
:::

note that we also provide ``sum=True``, which means that evaluating the
operator on a wavefunction will produce a single scalar. This is
semantically equivalent to providing ``sum=False`` (the default) and
running ``numpy.sum`` on the output.

### Current flowing through a cut

Below we calculate the probability current and z-projected spin current
near the interfaces with the left and right leads.

```python


    def left_cut(site_to, site_from):
        return site_from.pos[0] <= -39 and site_to.pos[0] > -39

    def right_cut(site_to, site_from):
        return site_from.pos[0] < 39 and site_to.pos[0] >= 39

    J_left = kwant.operator.Current(syst, where=left_cut, sum=True)
    J_right = kwant.operator.Current(syst, where=right_cut, sum=True)

    Jz_left = kwant.operator.Current(syst, sigma_z, where=left_cut, sum=True)
    Jz_right = kwant.operator.Current(syst, sigma_z, where=right_cut, sum=True)

    print('J_left:', J_left(psi), ' J_right:', J_right(psi))
    print('Jz_left:', Jz_left(psi), ' Jz_right:', Jz_right(psi))

```

::: {.literalinclude}
/code/figure/current\_cut.txt
:::

We see that the probability current is conserved across the scattering
region, but the z-projected spin current is not due to the fact that the
Hamiltonian does not commute with $σ_z$ everywhere in the scattering
region.

::: {.note}
::: {.admonition-title}
Note
:::

``where`` can also be provided as a sequence of
`~kwant.builder.Site` or a sequence of hoppings (i.e. pairs of
`~kwant.builder.Site`), rather than a function.
:::

Advanced Topics
---------------

### Using ``bind`` for speed

In most of the above examples we only used each operator *once* after
creating it. Often one will want to evaluate an operator with many
different wavefunctions, for example with all scattering wavefunctions
at a certain energy, but with the *same set of parameters*. In such
cases it is best to tell the operator to pre-compute the onsite matrices
and any necessary Hamiltonian elements using the given set of
parameters, so that this work is not duplicated every time the operator
is evaluated.

This can be achieved with `~kwant.operator.Current.bind`:

::: {.warning}
::: {.admonition-title}
Warning
:::

Take care that you do not use an operator that was bound to a particular
set of parameters with wavefunctions calculated with a *different* set
of parameters. This will almost certainly give incorrect results.
:::

```python


    J_m = kwant.operator.Current(syst, following_m_i)
    J_z = kwant.operator.Current(syst, sigma_z)

    J_m_bound = J_m.bind(params=dict(r0=25, delta=10, J=1))
    J_z_bound = J_z.bind(params=dict(r0=25, delta=10, J=1))

    # Sum current local from all scattering states on the left at energy=-1
    wf_left = wf(0)
    J_m_left = sum(J_m_bound(p) for p in wf_left)
    J_z_left = sum(J_z_bound(p) for p in wf_left)


    plot_currents(syst, [
        (r'$J_{\mathbf{m}_i}$ (from left)', J_m_left),
        (r'$J_{σ_z}$ (from left)', J_z_left),
    ])


if __name__ == '__main__':
    main()

```

![image](/code/figure/bound_current.*)
