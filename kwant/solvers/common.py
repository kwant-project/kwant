# Copyright 2011-2014 Kwant authors.
#
# This file is part of Kwant.  It is subject to the license terms in the file
# LICENSE.rst found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of Kwant authors can be found in
# the file AUTHORS.rst at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['SparseSolver', 'SMatrix', 'GreensFunction']

from collections import namedtuple
from itertools import product
import abc
import numpy as np
import scipy.sparse as sp
from .._common import ensure_isinstance
from .. import system

# Currently, scipy.sparse does not support matrices with one dimension being
# zero: http://projects.scipy.org/scipy/ticket/1602 We use NumPy dense matrices
# as a replacement.

# TODO: Once this issue is fixed, code for the special cases can be removed
# from _make_linear_sys, _solve_linear_sys and possibly other places marked by
# the line "See comment about zero-shaped sparse matrices at the top of
# common.py".

LinearSys = namedtuple('LinearSys', ['lhs', 'rhs', 'indices', 'num_orb'])


class SparseSolver(object):
    """Solver class for computing physical quantities based on solving
    a liner system of equations.

    `SparseSolver` is an abstract base class.  It cannot be instantiated as it
    does not specify the actual linear solve step.  In order to be directly
    usable, a derived class must implement the methods

    - `_factorize` and `_solve_linear_sys`, that solve a linear system of
      equations

    and the following properties:

    - `lhsformat`, `rhsformat`: Sparse matrix format of left and right hand
       sides of the linear system, respectively.  Must be one of {'coo', 'csc',
       'csr'}.

    - `nrhs`: Number of right hand side vectors that should be solved at once
      in a call to `solve_linear_sys`, when the full solution is computed (i.e.
      kept_vars covers all entries in the solution). This should be not too big
      too avoid excessive memory usage, but for some solvers not too small for
      performance reasons.
    """
    __metaclass__ = abc.ABCMeta

    @abc.abstractmethod
    def _factorized(self, a):
        """
        Return a preprocessed version of a matrix for the use with
        `solve_linear_sys`.

        Parameters
        ----------
        a : a scipy.sparse.coo_matrix sparse matrix.

        Returns
        -------
        factorized_a : object
            factorized lhs to be used with `_solve_linear_sys`.
        """
        pass

    @abc.abstractmethod
    def _solve_linear_sys(self, factorized_a, b, kept_vars):
        """
        Solve the linar system `a x = b`, returning the part of
        the result indicated in kept_vars.

        Parameters
        ----------
        factorized : object
            The result of calling `_factorized` for the matrix a.
        b : sparse matrix
            The right hand side. Its format must match `rhsformat`.
        kept_vars : slice object or sequence of integers
            A sequence of numbers of variables to keep in the solution

        Returns
        -------
        output : NumPy matrix
            Solution to the system of equations.
        """
        pass

    def _make_linear_sys(self, sys, in_leads, energy=0, args=(),
                         check_hermiticity=True, realspace=False):
        """Make a sparse linear system of equations defining a scattering
        problem.

        Parameters
        ----------
        sys : kwant.system.FiniteSystem
            Low level system, containing the leads and the Hamiltonian of a
            scattering region.
        in_leads : sequence of integers
            Numbers of leads in which current or wave function is injected.
        energy : number
            Excitation energy at which to solve the scattering problem.
        args : tuple, defaults to empty
            Positional arguments to pass to the ``hamiltonian`` method.
        check_hermiticity : bool
            Check if Hamiltonian matrices are in fact Hermitian.
            Enables deduction of missing transmission coefficients.
        realspace : bool
            Calculate Green's function between the outermost lead
            sites, instead of lead modes. This is almost always
            more computationally expensive and less stable.

        Returns
        -------
        (lhs, rhs, indices, num_orb) : LinearSys
            `lhs` is a scipy.sparse.csc_matrix, containing the left hand side
            of the system of equations.  `rhs` is a list of matrices with the
            right hand side, with each matrix corresponding to one lead
            mentioned in `in_leads`. `indices` is a list of arrays of variables
            in the system of equations corresponding to the the outgoing modes
            in each lead, or the indices of variables, on which a lead defined
            via self-energy adds the self-energy. Finally `num_orb` is the
            total number of degrees of freedom in the scattering region.

        lead_info : list of objects
            Contains one entry for each lead.  If `realspace=False`, this is an
            instance of `~kwant.physics.PropagatingModes` with a corresponding
            format, otherwise the lead self-energy matrix.

        Notes
        -----
        All the leads should implement a method `modes` if `realspace=False`
        and a method `selfenergy`.

        The system of equations that is created will be described in detail
        elsewhere.
        """
        ensure_isinstance(sys, system.System)

        splhsmat = getattr(sp, self.lhsformat + '_matrix')
        sprhsmat = getattr(sp, self.rhsformat + '_matrix')

        if not sys.lead_interfaces:
            raise ValueError('System contains no leads.')
        lhs, norb = sys.hamiltonian_submatrix(args, sparse=True,
                                              return_norb=True)[:2]
        lhs = getattr(lhs, 'to' + self.lhsformat)()
        lhs = lhs - energy * sp.identity(lhs.shape[0], format=self.lhsformat)
        num_orb = lhs.shape[0]

        if check_hermiticity and len(lhs.data):
            rtol = 1e-13
            atol = 1e-300
            tol = rtol * np.max(np.abs(lhs.data)) + atol
            if np.any(np.abs((lhs - lhs.T.conj()).data) > tol):
                raise ValueError('System Hamiltonian is not Hermitian. '
                                 'Use option `check_hermiticity=False` '
                                 'if this is intentional.')

        offsets = np.empty(norb.shape[0] + 1, int)
        offsets[0] = 0
        offsets[1 :] = np.cumsum(norb)

        # Process the leads, generate the eigenvector matrices and lambda
        # vectors. Then create blocks of the linear system and add them
        # step by step.
        indices = []
        rhs = []
        lead_info = []
        for leadnum, interface in enumerate(sys.lead_interfaces):
            lead = sys.leads[leadnum]
            if not realspace:
                prop, stab = lead.modes(energy, args)
                lead_info.append(prop)
                u = stab.vecs
                ulinv = stab.vecslmbdainv
                nprop = stab.nmodes
                svd_v = stab.sqrt_hop

                if len(u) == 0:
                    rhs.append(None)
                    continue

                indices.append(np.arange(lhs.shape[0], lhs.shape[0] + nprop))

                u_out, ulinv_out = u[:, nprop:], ulinv[:, nprop:]
                u_in, ulinv_in = u[:, :nprop], ulinv[:, :nprop]

                # Construct a matrix of 1's that translates the
                # inter-cell hopping to a proper hopping
                # from the system to the lead.
                iface_orbs = np.r_[tuple(slice(offsets[i], offsets[i + 1])
                                        for i in interface)]

                n_lead_orbs = (svd_v.shape[0] if svd_v is not None
                               else u_out.shape[0])
                if n_lead_orbs != len(iface_orbs):
                    msg = ('Lead {0} has hopping with dimensions '
                           'incompatible with its interface dimension.')
                    raise ValueError(msg.format(leadnum))

                coords = np.r_[[np.arange(len(iface_orbs))], [iface_orbs]]
                transf = sp.csc_matrix((np.ones(len(iface_orbs)), coords),
                                       shape=(iface_orbs.size, lhs.shape[0]))

                if svd_v is not None:
                    v_sp = sp.csc_matrix(svd_v.T.conj()) * transf
                    vdaguout_sp = (transf.T *
                                   sp.csc_matrix(np.dot(svd_v, u_out)))
                    lead_mat = - ulinv_out
                else:
                    v_sp = transf
                    vdaguout_sp = transf.T * sp.csc_matrix(u_out)
                    lead_mat = - ulinv_out

                lhs = sp.bmat([[lhs, vdaguout_sp], [v_sp, lead_mat]],
                              format=self.lhsformat)

                if leadnum in in_leads and nprop > 0:
                    if svd_v is not None:
                        vdaguin_sp = transf.T * sp.csc_matrix(
                            -np.dot(svd_v, u_in))
                    else:
                        vdaguin_sp = transf.T * sp.csc_matrix(-u_in)

                    # defer formation of the real matrix until the proper
                    # system size is known
                    rhs.append((vdaguin_sp, ulinv_in))
                else:
                    rhs.append(None)
            else:
                sigma = lead.selfenergy(energy, args)
                lead_info.append(sigma)
                coords = np.r_[tuple(slice(offsets[i], offsets[i + 1])
                                      for i in interface)]

                if sigma.shape != 2 * coords.shape:
                    msg = ('Self-energy dimension for lead {0} does not '
                           'match the total number of orbitals of the '
                           'sites for which it is defined.')
                    raise ValueError(msg.format(leadnum))

                y, x = np.meshgrid(coords, coords)
                sig_sparse = splhsmat((sigma.flat, [x.flat, y.flat]),
                                      lhs.shape)
                lhs = lhs + sig_sparse  # __iadd__ is not implemented in v0.7
                indices.append(coords)
                if leadnum in in_leads:
                    # defer formation of true rhs until the proper system
                    # size is known
                    rhs.append((coords,))

        # Resize the right-hand sides to be compatible with the full lhs
        for i, mats in enumerate(rhs):
            if isinstance(mats, tuple):
                if len(mats) == 1:
                    # self-energy lead
                    l = mats[0].shape[0]
                    rhs[i] = sprhsmat((-np.ones(l), [mats[0], np.arange(l)]),
                                      shape=(lhs.shape[0], l))
                else:
                    # lead with modes
                    zero_rows = (lhs.shape[0] - mats[0].shape[0] -
                                 mats[1].shape[0])

                    if zero_rows:
                        zero_mat = sprhsmat((zero_rows, mats[0].shape[1]))
                        bmat = [[mats[0]], [mats[1]], [zero_mat]]
                    else:
                        bmat = [[mats[0]], [mats[1]]]

                    rhs[i] = sp.bmat(bmat, format=self.rhsformat)
            elif mats is None:
                # A lead with no rhs.
                rhs[i] = np.zeros((lhs.shape[0], 0))
            else:
                raise RuntimeError('Unknown right-hand side format')

        return LinearSys(lhs, rhs, indices, num_orb), lead_info

    def smatrix(self, sys, energy=0, args=(),
                out_leads=None, in_leads=None, check_hermiticity=True):
        """
        Compute the scattering matrix of a system.

        Parameters
        ----------
        sys : `kwant.system.FiniteSystem`
            Low level system, containing the leads and the Hamiltonian of a
            scattering region.
        energy : number
            Excitation energy at which to solve the scattering problem.
        args : tuple, defaults to empty
            Positional arguments to pass to the ``hamiltonian`` method.
        out_leads : sequence of integers or ``None``
            Numbers of leads where current or wave function is extracted.  None
            is interpreted as all leads. Default is ``None`` and means "all
            leads".
        in_leads : sequence of integers or ``None``
            Numbers of leads in which current or wave function is injected.
            None is interpreted as all leads. Default is ``None`` and means
            "all leads".
        check_hermiticity : ``bool``
            Check if the Hamiltonian matrices are Hermitian.
            Enables deduction of missing transmission coefficients.

        Returns
        -------
        output : `~kwant.solvers.common.SMatrix`
            See the notes below and `~kwant.solvers.common.SMatrix`
            documentation.

        Notes
        -----
        This function can be used to calculate the conductance and other
        transport properties of a system.  See the documentation for its output
        type, `~kwant.solvers.common.SMatrix`.

        The returned object contains the scattering matrix elements from the
        `in_leads` to the `out_leads` as well as information about the lead
        modes.

        Both `in_leads` and `out_leads` must be sorted and may only contain
        unique entries.
        """
        ensure_isinstance(sys, system.System)

        n = len(sys.lead_interfaces)
        if in_leads is None:
            in_leads = range(n)
        else:
            in_leads = list(in_leads)
        if out_leads is None:
            out_leads = range(n)
        else:
            out_leads = list(out_leads)
        if (np.any(np.diff(in_leads) <= 0) or np.any(np.diff(out_leads) <= 0)):
            raise ValueError("Lead lists must be sorted and "
                             "with unique entries.")
        if len(in_leads) == 0 or len(out_leads) == 0:
            raise ValueError("No output is requested.")

        linsys, lead_info = self._make_linear_sys(sys, in_leads, energy, args,
                                                  check_hermiticity, False)

        kept_vars = np.concatenate([coords for i, coords in
                                    enumerate(linsys.indices) if i in
                                    out_leads])

        # Do not perform factorization if no calculation is to be done.
        len_rhs = sum(i.shape[1] for i in linsys.rhs)
        len_kv = len(kept_vars)
        if not(len_rhs and len_kv):
            return SMatrix(np.zeros((len_kv, len_rhs)), lead_info,
                           out_leads, in_leads, check_hermiticity)

        # See comment about zero-shaped sparse matrices at the top of common.py.
        rhs = sp.bmat([[i for i in linsys.rhs if i.shape[1]]],
                      format=self.rhsformat)
        flhs = self._factorized(linsys.lhs)
        data = self._solve_linear_sys(flhs, rhs, kept_vars)

        return SMatrix(data, lead_info, out_leads, in_leads, check_hermiticity)

    def greens_function(self, sys, energy=0, args=(),
                        out_leads=None, in_leads=None, check_hermiticity=True):
        """
        Compute the retarded Green's function of the system between its leads.

        Parameters
        ----------
        sys : `kwant.system.FiniteSystem`
            Low level system, containing the leads and the Hamiltonian of a
            scattering region.
        energy : number
            Excitation energy at which to solve the scattering problem.
        args : tuple, defaults to empty
            Positional arguments to pass to the ``hamiltonian`` method.
        out_leads : sequence of integers or ``None``
            Numbers of leads where current or wave function is extracted.  None
            is interpreted as all leads. Default is ``None`` and means "all
            leads".
        in_leads : sequence of integers or ``None``
            Numbers of leads in which current or wave function is injected.
            None is interpreted as all leads. Default is ``None`` and means
            "all leads".
        check_hermiticity : ``bool``
            Check if the Hamiltonian matrices are Hermitian.
            Enables deduction of missing transmission coefficients.

        Returns
        -------
        output : `~kwant.solvers.common.GreensFunction`
            See the notes below and `~kwant.solvers.common.GreensFunction`
            documentation.

        Notes
        -----
        This function can be used to calculate the conductance and other
        transport properties of a system.  It is often slower and less stable
        than the scattering matrix-based calculation executed by
        `~kwant.smatrix`, and is currently provided mostly for testing
        purposes and compatibility with RGF code.

        It returns an object encapsulating the Green's function elements
        between the system sites interfacing the leads in `in_leads` and those
        interfacing the leads in `out_leads`.  The returned object also
        contains a list with self-energies of the leads.

        Both `in_leads` and `out_leads` must be sorted and may only contain
        unique entries.
        """
        ensure_isinstance(sys, system.System)

        n = len(sys.lead_interfaces)
        if in_leads is None:
            in_leads = range(n)
        else:
            in_leads = list(in_leads)
        if out_leads is None:
            out_leads = range(n)
        else:
            out_leads = list(out_leads)
        if (np.any(np.diff(in_leads) <= 0) or np.any(np.diff(out_leads) <= 0)):
            raise ValueError("Lead lists must be sorted and "
                             "with unique entries.")
        if len(in_leads) == 0 or len(out_leads) == 0:
            raise ValueError("No output is requested.")

        linsys, lead_info = self._make_linear_sys(sys, in_leads, energy, args,
                                                  check_hermiticity, True)

        kept_vars = np.concatenate([coords for i, coords in
                                    enumerate(linsys.indices) if i in
                                    out_leads])

        # Do not perform factorization if no calculation is to be done.
        len_rhs = sum(i.shape[1] for i in linsys.rhs)
        len_kv = len(kept_vars)
        if not(len_rhs and len_kv):
            return GreensFunction(np.zeros((len_kv, len_rhs)), lead_info,
                                  out_leads, in_leads, check_hermiticity)

        # See comment about zero-shaped sparse matrices at the top of common.py.
        rhs = sp.bmat([[i for i in linsys.rhs if i.shape[1]]],
                      format=self.rhsformat)
        flhs = self._factorized(linsys.lhs)
        data = self._solve_linear_sys(flhs, rhs, kept_vars)

        return GreensFunction(data, lead_info, out_leads, in_leads,
                              check_hermiticity)

    def ldos(self, sys, energy=0, args=(), check_hermiticity=True):
        """
        Calculate the local density of states of a system at a given energy.

        Parameters
        ----------
        sys : `kwant.system.FiniteSystem`
            Low level system, containing the leads and the Hamiltonian of the
            scattering region.
        energy : number
            Excitation energy at which to solve the scattering problem.
        args : tuple of arguments, or empty tuple
            Positional arguments to pass to the function(s) which
            evaluate the hamiltonian matrix elements
        check_hermiticity : ``bool``
            Check if the Hamiltonian matrices are Hermitian.

        Returns
        -------
        ldos : a NumPy array
            Local density of states at each orbital of the system.
        """
        ensure_isinstance(sys, system.System)
        if not check_hermiticity:
            raise NotImplementedError("ldos for non-Hermitian Hamiltonians "
                                      "is not implemented yet.")

        for lead in sys.leads:
            if not hasattr(lead, 'modes') and hasattr(lead, 'selfenergy'):
                # TODO: fix this
                raise NotImplementedError("ldos for leads with only "
                                          "self-energy is not implemented yet.")

        linsys = self._make_linear_sys(sys, xrange(len(sys.leads)), energy,
                                       args, check_hermiticity)[0]

        ldos = np.zeros(linsys.num_orb, float)

        # Do not perform factorization if no further calculation is needed.
        if not sum(i.shape[1] for i in linsys.rhs):
            return ldos

        factored = self._factorized(linsys.lhs)

        # See comment about zero-shaped sparse matrices at the top of common.py.
        rhs = sp.bmat([[i for i in linsys.rhs if i.shape[1]]],
                      format=self.rhsformat)
        for j in xrange(0, rhs.shape[1], self.nrhs):
            jend = min(j + self.nrhs, rhs.shape[1])
            psi = self._solve_linear_sys(factored, rhs[:, j:jend],
                                         slice(linsys.num_orb))
            ldos += np.sum(np.square(abs(psi)), axis=1)

        return ldos * (0.5 / np.pi)

    def wave_function(self, sys, energy=0, args=(), check_hermiticity=True):
        """
        Return a callable object for the computation of the wave function
        inside the scattering region.

        Parameters
        ----------
        sys : `kwant.system.FiniteSystem`
            The low level system for which the wave functions are to be
            calculated.
        args : tuple of arguments, or empty tuple
            Positional arguments to pass to the function(s) which
            evaluate the hamiltonian matrix elements
        check_hermiticity : ``bool``
            Check if the Hamiltonian matrices are Hermitian.

        Notes
        -----

        The returned object can be itself called like a function.  Given a lead
        number, it returns a 2d NumPy array that contains the wave function
        within the scattering region due to each incoming mode of the given
        lead.  Index 0 is the mode number, index 1 is the orbital number.  The
        modes appear in the same order as incoming modes in
        `kwant.physics.modes`.

        Examples
        --------
        >>> wf = kwant.solvers.default.wave_function(some_sys, some_energy)
        >>> wfs_of_lead_2 = wf(2)

        """
        return WaveFunction(self, sys, energy, args, check_hermiticity)


class WaveFunction(object):
    def __init__(self, solver, sys, energy, args, check_hermiticity):
        ensure_isinstance(sys, system.System)
        for lead in sys.leads:
            if not hasattr(lead, 'modes'):
                # TODO: figure out what to do with self-energies.
                msg = ('Wave functions for leads with only self-energy'
                       ' are not available yet.')
                raise NotImplementedError(msg)
        linsys = solver._make_linear_sys(sys, xrange(len(sys.leads)), energy,
                                         args, check_hermiticity)[0]
        self.solve = solver._solve_linear_sys
        self.rhs = linsys.rhs
        self.factorized_h = solver._factorized(linsys.lhs)
        self.num_orb = linsys.num_orb

    def __call__(self, lead):
        result = self.solve(self.factorized_h, self.rhs[lead],
                            slice(self.num_orb))
        return result.transpose()


class BlockResult(object):
    """
    ABC for a linear system solution with variable grouping.
    """
    __metaclass__ = abc.ABCMeta

    def __init__(self, data, lead_info, out_leads, in_leads, sizes,
                 current_conserving=False):
        self.data = data
        self.lead_info = lead_info
        self.out_leads = out_leads = list(out_leads)
        self.in_leads = in_leads = list(in_leads)
        self.sizes = sizes = np.array(sizes)
        self.current_conserving = current_conserving
        self.in_offsets = np.zeros(len(in_leads) + 1, int)
        self.in_offsets[1 :] = np.cumsum(self.sizes[in_leads])
        self.out_offsets = np.zeros(len(self.out_leads) + 1, int)
        self.out_offsets[1 :] = np.cumsum(self.sizes[out_leads])

    def block_coords(self, lead_out, lead_in):
        """
        Return slices corresponding to the block from lead_in to lead_out.
        """
        return self.out_block_coords(lead_out), self.in_block_coords(lead_in)

    def out_block_coords(self, lead_out):
        """Return a slice with the rows in the block corresponding to lead_out.
        """
        lead_out = self.out_leads.index(lead_out)
        return slice(self.out_offsets[lead_out],
                     self.out_offsets[lead_out + 1])

    def in_block_coords(self, lead_in):
        """
        Return a slice with the columns in the block corresponding to lead_in.
        """
        lead_in = self.in_leads.index(lead_in)
        return slice(self.in_offsets[lead_in],
                     self.in_offsets[lead_in + 1])

    def submatrix(self, lead_out, lead_in):
        """Return the matrix elements from lead_in to lead_out."""
        return self.data[self.block_coords(lead_out, lead_in)]

    @abc.abstractmethod
    def num_propagating(self, lead):
        """Return the number of propagating modes in the lead."""
        pass

    @abc.abstractmethod
    def _transmission(self, lead_out, lead_in):
        """Return transmission from lead_in to lead_out.

        The calculation is expected to be done directly from the corresponding
        matrix elements.
        """
        pass

    def transmission(self, lead_out, lead_in):
        """Return transmission from lead_in to lead_out.

        If the option ``current_conserving`` has been enabled for this object,
        this method will deduce missing transmission values whenever possible.

        Current conservation is enabled by default for objects returned by
        `~kwant.solvers.default.smatrix` and
        `~kwant.solvers.default.greens_function` whenever the Hamiltonian has
        been verified to be Hermitian (option ``check_hermiticity``, enabled by
        default).

        """
        chosen = [lead_out, lead_in]
        present = self.out_leads, self.in_leads
        # OK means: chosen (outgoing, incoming) lead is present.
        ok = [int(c in p) for c, p in zip(chosen, present)]
        if all(ok):
            return self._transmission(lead_out, lead_in)

        if self.current_conserving:
            all_but_one = len(self.lead_info) - 1
            s_t = self._transmission
            if sum(ok) == 1:
                # Calculate the transmission element by summing over a single
                # row or column.
                sum_dim, ok_dim = ok
                if len(present[sum_dim]) == all_but_one:
                    return (self.num_propagating(chosen[ok_dim]) - sum(
                        s_t(*chosen) for chosen[sum_dim] in present[sum_dim]))
            elif all(len(p) == all_but_one for p in present):
                # Calculate the transmission element at the center of a single
                # missing row and a single missing column.
                return (sum(s_t(o, i) - self.num_propagating(o) if o == i else 0
                            for o, i in product(*present)))

        raise ValueError("Insufficient matrix elements to compute "
                         "transmission({0}, {1}).".format(*chosen))

    def conductance_matrix(self):
        """Return the conductance matrix.

        This is the matrix :math:`C` such that :math:`I = CV` where :math:`I`
        and :math:`V` are, respectively, the vectors of currents and voltages
        for each lead.

        This matrix is useful for calculating non-local resistances.  See
        Section 2.4 of the book by S. Datta.
        """
        n = len(self.lead_info)
        rn = xrange(n)
        result = np.array([[-self.transmission(i, j) if i != j else 0
                            for j in rn] for i in rn])
        # Set the diagonal elements such that the sums of columns are zero.
        result.flat[::n+1] = -result.sum(axis=0)
        return result


class SMatrix(BlockResult):
    """A scattering matrix.

    Transport properties can be easily accessed using the
    `~SMatrix.transmission` method (don't be fooled by the name,
    it can also compute reflection, which is just transmission from one
    lead back into the same lead.)

    `SMatrix` however also allows for a more direct access to the result: The
    data stored in `SMatrix` is a scattering matrix with respect to lead modes
    and these modes themselves. The details of this data can be directly
    accessed through the instance variables `data` and `lead_info`. Subblocks
    of data corresponding to particular leads are conveniently obtained by
    `~SMatrix.submatrix`.

    Attributes
    ----------
    data : NumPy array
        a matrix containing all the requested matrix elements of the scattering
        matrix.
    lead_info : list of data
        a list containing `kwant.physics.PropagatingModes` for each lead.
    out_leads, in_leads : sequence of integers
        indices of the leads where current is extracted (out) or injected
        (in). Only those are listed for which SMatrix contains the
        calculated result.
    """

    def __init__(self, data, lead_info, out_leads, in_leads,
                 current_conserving=False):
        sizes = [len(i.momenta) // 2 for i in lead_info]
        super(SMatrix, self).__init__(data, lead_info, out_leads, in_leads,
                                      sizes, current_conserving)

    def num_propagating(self, lead):
        """Return the number of propagating modes in the lead."""
        return self.sizes[lead]

    def _transmission(self, lead_out, lead_in):
        return np.linalg.norm(self.submatrix(lead_out, lead_in)) ** 2

    def __repr__(self):
        return ("SMatrix(data=%r, lead_info=%r, out_leads=%r, in_leads=%r)" %
                (self.data, self.lead_info, self.out_leads, self.in_leads))


class GreensFunction(BlockResult):
    """
    Retarded Green's function.

    Transport properties can be easily accessed using the
    `~GreensFunction.transmission` method (don't be fooled by the name, it can
    also compute reflection, which is just transmission from one lead back into
    the same lead).

    `GreensFunction` however also allows for a more direct access to the
    result: The data stored in `GreensFunction` is the real-space Green's
    function. The details of this data can be directly accessed through the
    instance variables `data` and `lead_info`. Subblocks of data corresponding
    to particular leads are conveniently obtained by
    `~GreensFunction.submatrix`.

    Attributes
    ----------
    data : NumPy array
        a matrix containing all the requested matrix elements of Green's
        function.
    lead_info : list of matrices
        a list with self-energies of each lead.
    out_leads, in_leads : sequence of integers
        indices of the leads where current is extracted (out) or injected
        (in). Only those are listed for which SMatrix contains the
        calculated result.
    """

    def __init__(self, data, lead_info, out_leads, in_leads,
                 current_conserving=False):
        sizes = [i.shape[0] for i in lead_info]
        super(GreensFunction, self).__init__(
            data, lead_info, out_leads, in_leads, sizes, current_conserving)

    def num_propagating(self, lead):
        """Return the number of propagating modes in the lead."""
        gamma = 1j * (self.lead_info[lead] -
                      self.lead_info[lead].conj().T)

        # The number of channels is given by the number of
        # nonzero eigenvalues of Gamma
        # rationale behind the threshold from
        # Golub; van Loan, chapter 5.5.8
        eps = np.finfo(gamma.dtype).eps * 1000
        return np.sum(np.linalg.eigvalsh(gamma) >
                      eps * np.linalg.norm(gamma, np.inf))

    def _a_ttdagger_a_inv(self, lead_out, lead_in):
        """Return t * t^dagger in a certain basis."""
        gf = self.submatrix(lead_out, lead_in)
        factors = []
        for lead, gf2 in ((lead_out, gf), (lead_in, gf.conj().T)):
            possible_se = self.lead_info[lead]
            factors.append(1j * (possible_se - possible_se.conj().T))
            factors.append(gf2)
        return reduce(np.dot, factors)

    def _transmission(self, lead_out, lead_in):
        attdagainv = self._a_ttdagger_a_inv(lead_out, lead_in)
        result = np.trace(attdagainv).real
        if lead_out == lead_in:
            # For reflection we have to be more careful
            gamma = 1j * (self.lead_info[lead_in] -
                          self.lead_info[lead_in].conj().T)
            gf = self.submatrix(lead_out, lead_in)

            # The number of channels is given by the number of
            # nonzero eigenvalues of Gamma
            # rationale behind the threshold from
            # Golub; van Loan, chapter 5.5.8
            eps = np.finfo(gamma.dtype).eps * 1000
            N = np.sum(np.linalg.eigvalsh(gamma) >
                       eps * np.linalg.norm(gamma, np.inf))

            result += 2 * np.trace(np.dot(gamma, gf)).imag + N

        return result

    def __repr__(self):
        return ("GreensFunction(data=%r, lead_info=%r, "
                "out_leads=%r, in_leads=%r)" %
                (self.data, self.lead_info, self.out_leads, self.in_leads))
