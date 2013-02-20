# Copyright 2011-2013 kwant authors.
#
# This file is part of kwant.  It is subject to the license terms in the
# LICENSE file found in the top-level directory of this distribution and at
# http://kwant-project.org/license.  A list of kwant authors can be found in
# the AUTHORS file at the top-level directory of this distribution and at
# http://kwant-project.org/authors.

__all__ = ['SparseSolver', 'BlockResult']

from collections import namedtuple
import abc
import numpy as np
import scipy.sparse as sp
from .. import physics, system

# Currently, scipy.sparse does not support matrices with one dimension being
# zero: http://projects.scipy.org/scipy/ticket/1602 We use NumPy dense matrices
# as a replacement.
# TODO: Once this issue is fixed, code for the special cases can be removed
# from _make_linear_sys, _solve_linear_sys and possibly other places marked by
# the line "See comment about zero-shaped sparse matrices at the top of
# common.py".


LinearSys = namedtuple('LinearSys', ['lhs', 'rhs', 'kept_vars'])


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
        b : a sequence of matrices.
            Sizes of these matrices may be smaller than needed, the missing
            entries at the end are padded with zeros.
        kept_vars : slice object or sequence of integers
            A sequence of numbers of variables to keep in the solution

        Returns
        -------
        output : NumPy matrix
            Solution to the system of equations.
        """
        pass

    def _make_linear_sys(self, sys, out_leads, in_leads, energy=0,
                        force_realspace=False, check_hermiticity=True):
        """
        Make a sparse linear system of equations defining a scattering
        problem.

        Parameters
        ----------
        sys : kwant.system.FiniteSystem
            Low level system, containing the leads and the Hamiltonian of a
            scattering region.
        out_leads : sequence of integers
            Numbers of leads where current or wave function is extracted
        in_leads : sequence of integers
            Numbers of leads in which current or wave function is injected.
        energy : number
            Excitation energy at which to solve the scattering problem.
        force_realspace : bool
            Calculate Green's function between the outermost lead
            sites, instead of lead modes. This is almost always
            more computationally expensive and less stable.
        check_hermiticity : bool
            Check if Hamiltonian matrices are in fact Hermitian.

        Returns
        -------
        (lhs, rhs, kept_vars) : LinearSys
            `lhs` is a numpy.sparse.csc_matrix, containing the left hand
            side of the system of equations.  `rhs` is a list of matrices
            with the right hand side, with each matrix corresponding to one
            lead mentioned in `in_leads`. `kept_vars` is a list of numbers
            of variables in the solution that have to be stored (typically
            a small part of the complete solution).
        lead_info : list of objects
            Contains one entry for each lead.  For a lead defined as a
            tight-binding system, this is an instance of `kwant.physics.Modes`
            (as returned by `kwant.physics.modes`), otherwise the lead
            self-energy matrix.

        Notes
        -----
        Both incoming and outgoing leads can be defined via either self-energy,
        or a low-level translationally invariant system.
        The system of equations that is created is described in
        kwant/doc/other/linear_system.pdf
        """

        splhsmat = getattr(sp, self.lhsformat + '_matrix')
        sprhsmat = getattr(sp, self.rhsformat + '_matrix')

        if not sys.lead_interfaces:
            raise ValueError('System contains no leads.')
        lhs, norb = sys.hamiltonian_submatrix(sparse=True,
                                              return_norb=True)[:2]
        lhs = getattr(lhs, 'to' + self.lhsformat)()
        lhs = lhs - energy * sp.identity(lhs.shape[0], format=self.lhsformat)

        if check_hermiticity:
            if np.any(abs((lhs - lhs.T.conj()).data) > 1e-13):
                raise ValueError('System Hamiltonian is not Hermitian.')

        offsets = np.empty(norb.shape[0] + 1, int)
        offsets[0] = 0
        offsets[1 :] = np.cumsum(norb)

        # Process the leads, generate the eigenvector matrices and lambda
        # vectors. Then create blocks of the linear system and add them
        # step by step.
        kept_vars = []
        rhs = []
        lead_info = []
        for leadnum, interface in enumerate(sys.lead_interfaces):
            lead = sys.leads[leadnum]
            if isinstance(lead, system.InfiniteSystem) and not force_realspace:
                h = lead.slice_hamiltonian()

                if check_hermiticity:
                    if not np.allclose(h, h.T.conj(), rtol=1e-13):
                        msg = "Lead number {0} has a non-Hermitian " \
                            "slice Hamiltonian."
                        raise ValueError(msg.format(leadnum))

                h -= energy * np.identity(h.shape[0])
                v = lead.inter_slice_hopping()
                modes = physics.modes(h, v)
                lead_info.append(modes)

                # Note: np.any(v) returns (at least from NumPy 1.6.1 -
                #       1.8-devel) False if v is purely imaginary
                if not (np.any(v.real) or np.any(v.imag)):
                    # See comment about zero-shaped sparse matrices at the top.
                    rhs.append(np.zeros((lhs.shape[1], 0)))
                    continue
                u, ulinv, nprop, svd = modes

                if leadnum in out_leads:
                    kept_vars.append(
                        np.arange(lhs.shape[0], lhs.shape[0] + nprop))

                u_out, ulinv_out = u[:, nprop:], ulinv[:, nprop:]
                u_in, ulinv_in = u[:, : nprop], ulinv[:, : nprop]

                # Construct a matrix of 1's that translates the
                # inter-slice hopping to a proper hopping
                # from the system to the lead.
                iface_orbs = np.r_[tuple(slice(offsets[i], offsets[i + 1])
                                        for i in interface)]
                coords = np.r_[[np.arange(iface_orbs.size)], [iface_orbs]]
                transf = sp.csc_matrix((np.ones(iface_orbs.size), coords),
                                       shape=(iface_orbs.size, lhs.shape[0]))

                if svd is not None:
                    v_sp = sp.csc_matrix(svd[2].T.conj()) * transf
                    vdaguout_sp = transf.T * \
                        sp.csc_matrix(np.dot(svd[2] * svd[1], u_out))
                    lead_mat = - ulinv_out
                else:
                    v_sp = transf
                    vdaguout_sp = transf.T * sp.csc_matrix(np.dot(v.T.conj(),
                                                                  u_out))
                    lead_mat = - ulinv_out

                lhs = sp.bmat([[lhs, vdaguout_sp], [v_sp, lead_mat]],
                              format=self.lhsformat)

                if leadnum in in_leads and nprop > 0:
                    if svd:
                        vdaguin_sp = transf.T * sp.csc_matrix(
                            -np.dot(svd[2] * svd[1], u_in))
                    else:
                        vdaguin_sp = transf.T * sp.csc_matrix(
                            -np.dot(v.T.conj(), u_in))

                    # defer formation of the real matrix until the proper
                    # system size is known
                    rhs.append((vdaguin_sp, ulinv_in))
                else:
                    # See comment about zero-shaped sparse matrices at the top.
                    rhs.append(np.zeros((lhs.shape[1], 0)))
            else:
                sigma = lead.self_energy(energy)
                lead_info.append(sigma)
                indices = np.r_[tuple(slice(offsets[i], offsets[i + 1])
                                      for i in interface)]
                assert sigma.shape == 2 * indices.shape
                y, x = np.meshgrid(indices, indices)
                sig_sparse = splhsmat((sigma.flat, [x.flat, y.flat]),
                                      lhs.shape)
                lhs = lhs + sig_sparse  # __iadd__ is not implemented in v0.7
                if leadnum in out_leads:
                    kept_vars.append(indices)
                if leadnum in in_leads:
                    # defer formation of true rhs until the proper system
                    # size is known
                    rhs.append((indices, ))

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

        kept_vars = \
            np.concatenate(kept_vars) if kept_vars else np.empty(0, int)
        return LinearSys(lhs, rhs, kept_vars), lead_info

    def solve(self, sys, energy=0, out_leads=None, in_leads=None,
              force_realspace=False, check_hermiticity=True):
        """
        Compute the scattering matrix or Green's function between leads.

        Parameters
        ----------
        sys : `kwant.system.FiniteSystem`
            Low level system, containing the leads and the Hamiltonian of a
            scattering region.
        energy : number
            Excitation energy at which to solve the scattering problem.
        out_leads : sequence of integers or ``None``
            Numbers of leads where current or wave function is extracted.  None
            is interpreted as all leads. Default is ``None`` and means "all
            leads".
        in_leads : sequence of integers or ``None``
            Numbers of leads in which current or wave function is injected.
            None is interpreted as all leads. Default is ``None`` and means
            "all leads".
        force_realspace : ``bool``
            Calculate Green's function between the outermost lead
            sites, instead of lead modes. This is almost always
            more computationally expensive and less stable.
        check_hermiticity : ``bool``
            Check if the Hamiltonian matrices are Hermitian.

        Returns
        -------
        output : `~kwant.solvers.common.BlockResult`
            See the notes below and `~kwant.solvers.common.BlockResult`
            documentation.

        Notes
        -----
        This function can be used to calculate the conductance and other
        transport properties of a system.  See the documentation for its output
        type, `~kwant.solvers.common.BlockResult`.

        It returns an object encapsulating the Green's function elements
        between the desired leads. For leads defined as a self-energy, the
        result is just the real-space retarded Green's function between the
        system sites interfacing the leads in `in_leads` and those interfacing
        the leads in `out_leads`. If, as is usually the case, the leads are
        defined as tight-binding systems, then the Green's function from
        incoming to outgoing modes is returned (more commonly known as the
        scattering matrix).  If some leads are defined via a self-energy, and
        some as tight-binding systems, the result has Green's function's
        elements between modes and sites.  The returned object also contains
        information about the modes or self-energies of the leads.

        If `force_realspace` is set to ``True`` all leads will be treated as if
        they would be defined in terms of self energies.  The returned Green's
        function will be thus the retarded Green's function between sites in
        real space, just like if the tradidional RGF algorithm would have been
        used.  Enabling this option is more computationally expensive and can
        be less stable.

        Both `in_leads` and `out_leads` must be sorted and may only contain
        unique entries.
        """

        n = len(sys.lead_interfaces)
        if in_leads is None:
            in_leads = range(n)
        if out_leads is None:
            out_leads = range(n)
        if sorted(in_leads) != in_leads or sorted(out_leads) != out_leads or \
            len(set(in_leads)) != len(in_leads) or \
            len(set(out_leads)) != len(out_leads):
            raise ValueError("Lead lists must be sorted and "
                             "with unique entries.")
        if len(in_leads) == 0 or len(out_leads) == 0:
            raise ValueError("No output is requested.")

        linsys, lead_info = self._make_linear_sys(sys, out_leads, in_leads,
                                                  energy, force_realspace,
                                                  check_hermiticity)

        flhs = self._factorized(linsys.lhs)
        data = self._solve_linear_sys(flhs, linsys.rhs, linsys.kept_vars)

        return BlockResult(data, lead_info, out_leads, in_leads)


    def ldos(self, fsys, energy=0):
        """
        Calculate the local density of states of a system at a given energy.

        Parameters
        ----------
        sys : `kwant.system.FiniteSystem`
            Low level system, containing the leads and the Hamiltonian of the
            scattering region.
        energy : number
            Excitation energy at which to solve the scattering problem.

        Returns
        -------
        ldos : a NumPy array
            Local density of states at each orbital of the system.
        """
        for lead in fsys.leads:
            if not isinstance(lead, system.InfiniteSystem):
                # TODO: fix this
                raise ValueError("ldos only works when all leads are "
                                 "tight binding systems.")

        (h, rhs, kept_vars), lead_info = \
            self._make_linear_sys(fsys, [], xrange(len(fsys.leads)), energy)

        Modes = physics.Modes
        num_extra_vars = sum(li.vecs.shape[1] - li.nmodes
                             for li in lead_info if isinstance(li, Modes))
        num_orb = h.shape[0] - num_extra_vars

        ldos = np.zeros(num_orb, float)
        factored = None

        factored = self._factorized(h)
        for mat in rhs:
            if mat.shape[1] == 0:
                continue

            for j in xrange(0, mat.shape[1], self.nrhs):
                jend = min(j + self.nrhs, mat.shape[1])
                psi = self._solve_linear_sys(factored, [mat[:, j:jend]],
                                             slice(num_orb))
                ldos += np.sum(np.square(abs(psi)), axis=1)

        return ldos * (0.5 / np.pi)

    def wave_func(self, sys, energy=0):
        """
        Return a callable object for the computation of the wave function
        inside the scattering region.

        Parameters
        ----------
        sys : `kwant.system.FiniteSystem`
            The low level system for which the wave functions are to be
            calculated.

        Notes
        -----
        The returned object can be itself called like a function.  Given a lead
        number, it returns a 2d NumPy array containing the wave function within
        the scattering region due to each mode of the given lead.  Index 0 is
        the mode number, index 1 is the orbital number.

        Examples
        --------
        >>> wf = kwant.solvers.default.wave_func(some_sys, some_energy)
        >>> wfs_of_lead_2 = wf(2)
        """
        return WaveFunc(self, sys, energy)


class WaveFunc(object):
    def __init__(self, solver, sys, energy=0):
        for lead in sys.leads:
            if not isinstance(lead, system.InfiniteSystem):
                # TODO: fix this
                msg = 'All leads must be tight binding systems.'
                raise ValueError(msg)
        (h, self.rhs, kept_vars), lead_info = \
            solver._make_linear_sys(sys, [], xrange(len(sys.leads)), energy)
        Modes = physics.Modes
        num_extra_vars = sum(li.vecs.shape[1] - li.nmodes
                             for li in lead_info if isinstance(li, Modes))
        self.solver = solver
        self.num_orb = h.shape[0] - num_extra_vars
        self.factorized_h = solver._factorized(h)

    def __call__(self, lead):
        result = self.solver._solve_linear_sys(
            self.factorized_h, [self.rhs[lead]], slice(self.num_orb))
        return result.transpose()


class BlockResult(object):
    """
    Solution of a transport problem, subblock of retarded Green's function
    or scattering matrix.

    Transport properties can be easily accessed using the
    `~BlockResult.transmission` method (don't be fooled by the name,
    it can also compute reflection, which is just transmission from one
    lead back into the same lead.)

    `BlockResult` however also allows for a more direct access to the result:
    The data stored in `BlockResult` is either a real-space Green's
    function (e.g. if ``force_realspace=True`` in
    `~kwant.solvers.default.solve`) or a scattering matrix with respect to
    lead modes. The details of this data can be directly accessed through
    the instance variables `data` and `lead_info`. Subblocks of data
    corresponding to particular leads are conveniently obtained by
    `~BlockResult.submatrix`.

    Instance Variables
    ------------------
    data : NumPy matrix
        a matrix containing all the requested matrix elements of Green's
        function.
    lead_info : list of data
        a list with output of `kwant.physics.modes` for each lead
        defined as a builder, and self-energy for each lead defined as
        self-energy term.
    out_leads, in_leads : list of integers
        indices of the leads where current is extracted (out) or injected
        (in). Only those are listed for which BlockResult contains the
        calculated result.
    """

    def __init__(self, data, lead_info, out_leads, in_leads):
        self.data = data
        self.lead_info = lead_info
        self.out_leads = out_leads
        self.in_leads = in_leads

        sizes = []
        for i in self.lead_info:
            if isinstance(i, tuple):
                sizes.append(i[2])
            else:
                sizes.append(i.shape[0])
        self._sizes = np.array(sizes)

        self._in_offsets = np.zeros(len(self.in_leads) + 1, int)
        self._in_offsets[1 :] = np.cumsum(self._sizes[self.in_leads])
        self._out_offsets = np.zeros(len(self.out_leads) + 1, int)
        self._out_offsets[1 :] = np.cumsum(self._sizes[self.out_leads])

    def block_coords(self, lead_out, lead_in):
        """
        Return slices corresponding to the block from lead_in to lead_out.
        """
        return self.out_block_coords(lead_out), self.in_block_coords(lead_in)

    def out_block_coords(self, lead_out):
        """Return a slice corresponding to the rows in the block corresponding
        to lead_out
        """
        lead_out = self.out_leads.index(lead_out)
        return slice(self._out_offsets[lead_out],
                     self._out_offsets[lead_out + 1])

    def in_block_coords(self, lead_in):
        """Return a slice corresponding to the columns in the block
        corresponding to lead_in
        """
        lead_in = self.in_leads.index(lead_in)
        return slice(self._in_offsets[lead_in],
                     self._in_offsets[lead_in + 1])

    def submatrix(self, lead_out, lead_in):
        """Return the matrix elements from lead_in to lead_out."""
        return self.data[self.block_coords(lead_out, lead_in)]

    def _a_ttdagger_a_inv(self, lead_out, lead_in):
        gf = self.submatrix(lead_out, lead_in)
        factors = []
        for lead, gf2 in ((lead_out, gf), (lead_in, gf.conj().T)):
            possible_se = self.lead_info[lead]
            if not isinstance(possible_se, tuple):
                # Lead is a "self energy lead": multiply gf2 with a gamma
                # matrix.
                factors.append(1j * (possible_se - possible_se.conj().T))
            factors.append(gf2)
        return reduce(np.dot, factors)

    def transmission(self, lead_out, lead_in):
        """Return transmission from lead_in to lead_out."""
        if isinstance(self.lead_info[lead_out], tuple) and \
           isinstance(self.lead_info[lead_in], tuple):
            return np.linalg.norm(self.submatrix(lead_out, lead_in)) ** 2
        else:
            result = np.trace(self._a_ttdagger_a_inv(lead_out, lead_in)).real
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
        return "BlockResult(data=%r, lead_info=%r, " \
            "out_leads=%r, in_leads=%r)" % (self.data, self.lead_info,
                                            self.out_leads, self.in_leads)
