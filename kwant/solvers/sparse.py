__all__ = [ 'make_linear_sys' , 'solve', 'BlockResult']

from functools import reduce
from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg as spl
from kwant import physics, system

def make_linear_sys(sys, out_leads, in_leads, energy=0,
                    force_realspace=False):
    """
    Make a sparse linear system of equations defining a scattering problem.

    Parameters
    ----------
    sys : kwant.system.FiniteSystem
        low level system, containing the leads and the Hamiltonian of a
        scattering region.
    energy : number
        excitation energy at which to solve the scattering problem.
    in_leads : list of integers
        numbers of leads in which current or wave function is injected.
    out_leads : list of integers
        numbers of leads where current or wave function is exctracted
    force_realspace : bool
        calculate Green's function between the outermost lead
        sites, instead of lead modes. This is almost always
        more computationally expensive and less stable.

    Returns
    -------
    (h_sys, rhs, keep_vars, num_modes) : tuple of inhomogeneous data
        `h_sys` is a numpy.sparse.csc_matrix, containing the right hand side
        of the system of equations, `rhs` is the list of matrices with the
        left hand side, `keep_vars` is a list with numbers of variables in the
        solution that have to be stored (typically a small part of the
        complete solution). Finally, `num_modes` is a list with number of
        propagating modes or lattice points in each lead.

    Notes
    -----
    Both incomding and outgoing leads can be defined via either self-energy,
    or a low-level translationally invariant system.
    The system of equations that is created is described in
    kwant/doc/other/linear_system.pdf
    """
    if not sys.lead_neighbor_seqs:
        raise ValueError('System contains no leads.')
    h_sys = sys.hamiltonian_submatrix(sparse=True).tocsc()
    h_sys = h_sys - energy * sp.identity(h_sys.shape[0], format='csc')

    norb, num_nodes = sys.num_orbitals, sys.graph.num_nodes
    norb_arr = np.array([norb(i) for i in xrange(num_nodes)], int)
    offsets = np.zeros(norb_arr.shape[0] + 1, int)
    offsets[1 :] = np.cumsum(norb_arr)

    # Process the leads, generate the eigenvector matrices and lambda vectors.
    # Then create blocks of the linear system and add them step by step.
    keep_vars = []
    rhs = []
    num_modes = []
    for leadnum, lead_neighbors in enumerate(sys.lead_neighbor_seqs):
        lead = sys.leads[leadnum]
        if isinstance(lead, system.InfiniteSystem) and not force_realspace:
            h = lead.slice_hamiltonian()
            h -= energy * np.identity(h.shape[0])
            v = lead.inter_slice_hopping()

            u, ulinv, nprop, svd = physics.modes(h, v)

            num_modes.append(nprop)

            if leadnum in out_leads:
                keep_vars.append(range(h_sys.shape[0], h_sys.shape[0] + nprop))

            u_out, ulinv_out = u[:, nprop:], ulinv[:, nprop:]
            u_in, ulinv_in = u[:, : nprop], ulinv[:, : nprop]

            # Construct a matrix of 1's that translates the
            # inter-slice hopping to a proper hopping
            # from the system to the lead.
            neighbors = np.r_[tuple(np.arange(offsets[i], offsets[i + 1])
                                    for i in lead_neighbors)]
            coords = np.r_[[np.arange(neighbors.size)], [neighbors]]
            tmp = sp.csc_matrix((np.ones(neighbors.size), coords),
                                (neighbors.size, h_sys.shape[0]))

            if svd is not None:
                v_sp = sp.csc_matrix(svd[2].T.conj()) * tmp
                vdaguout_sp = tmp.T * sp.csr_matrix(np.dot(svd[2] * svd[1],
                                                           u_out))
                lead_mat = - ulinv_out
            else:
                v_sp = sp.csc_matrix(v) * tmp
                vdaguout_sp = tmp.T * sp.csr_matrix(np.dot(v.T.conj(), u_out))
                lead_mat = - np.dot(v, ulinv_out)

            h_sys = sp.bmat([[h_sys, vdaguout_sp], [v_sp, lead_mat]])

            if nprop > 0 and leadnum in in_leads:
                if svd:
                    vdaguin_sp = tmp.T * sp.csr_matrix(-np.dot(svd[2] * svd[1],
                                                               u_in))
                    lead_mat_in = ulinv_in
                else:
                    vdaguin_sp = tmp.T * sp.csr_matrix(-np.dot(v.T.conj(),
                                                               u_in))
                    lead_mat_in = np.dot(v, ulinv_in)

                rhs.append(sp.bmat([[vdaguin_sp], [lead_mat_in]]))
            else:
                rhs.append(np.zeros((0, 0)))
        else:
            sigma = lead.self_energy(energy)
            num_modes.append(sigma)
            indices = np.r_[tuple(range(offsets[i], offsets[i + 1]) for i in
                                 lead_neighbors)]
            assert sigma.shape == 2 * indices.shape
            y, x = np.meshgrid(indices, indices)
            sig_sparse = sp.coo_matrix((sigma.flat, [x.flat, y.flat]),
                                       h_sys.shape)
            h_sys = h_sys + sig_sparse # __iadd__ is not implemented in v0.7
            if leadnum in out_leads:
                keep_vars.append(list(indices))
            if leadnum in in_leads:
                l = indices.shape[0]
                rhs.append(sp.coo_matrix((-np.ones(l), [indices,
                                                        np.arange(l)])))

    return h_sys, rhs, keep_vars, num_modes


def solve_linsys(a, b, keep_vars=None):
    """
    Solve matrix system of equations a x = b with sparse input.

    Parameters
    ----------
    a : a scipy.sparse.csc_matrix sparse matrix
    b : a list of matrices.
        Sizes of these matrices may be smaller than needed, the missing
        entries at the end are padded with zeros.
    keep_vars : list of lists of integers
        a list of numbers of variables to keep in the solution

    Returns
    -------
    output : a numpy matrix
        solution to the system of equations.

    Notes
    -----
    This function is largely a wrapper to a scipy.sparse.linalg.factorized.
    """
    a = sp.csc_matrix(a)
    if keep_vars == None:
        keep_vars = [range(a.shape[1])]
    slv = spl.factorized(a)
    keeptot = sum(keep_vars, [])
    sols = []
    for mat in b:
        if mat.shape[1] != 0:
            mat = sp.csr_matrix(mat)
            for j in xrange(mat.shape[1]):
                vec = np.zeros(a.shape[0], complex)
                vec[: mat.shape[0]] = mat[:, j].todense().flatten()
                sols.append(slv(vec)[keeptot])
    return np.mat(sols).T


def solve(sys, energy=0, out_leads=None, in_leads=None,
          force_realspace=False):
    """
    Calculate a Green's function of a system.

    Parameters
    ----------
    sys : `kwant.system.FiniteSystem`
        low level system, containing the leads and the Hamiltonian of a
        scattering region.
    energy : number
        excitation energy at which to solve the scattering problem.
    in_leads : list of integers
        numbers of leads in which current or wave function is injected.
    out_leads : list of integers
        numbers of leads where current or wave function is exctracted
    force_realspace : bool
        calculate Green's function between the outermost lead
        sites, instead of lead modes. This is almost always
        more computationally expensive and less stable.

    Returns
    -------
    output: `BlockResult`
        see notes below and `BlockResult` docstring for more information about
        the output format.

    Notes
    -----
    Both in_leads and out_leads should be sorted and should only contain
    unique entries.

    Returns the Green's function elements between in_leads and out_leads. If
    the leads are defined as a self-energy, the result is just the real
    space retarded Green's function between from in_leads to out_leads. If the
    leads are defined as tight-binding systems, then Green's function from
    incoming to outgoing modes is returned. Additionally a list containing
    numbers of modes in each lead is returned. Sum of these numbers equals to
    the size of the returned Green's function subblock. The Green's function
    elements between incoming and outgoing modes form the scattering matrix of
    the system. If some leads are defined via self-energy, and some as
    tight-binding systems, result has Green's function's elements between modes
    and sites.

    Alternatively, if force_realspace=True is used, G^R is returned
    always in real space, however this option is more computationally
    expensive and can be less stable.
    """
    n = len(sys.lead_neighbor_seqs)
    if in_leads is None:
        in_leads = range(n)
    if out_leads is None:
        out_leads = range(n)
    if sorted(in_leads) != in_leads or sorted(out_leads) != out_leads or \
        len(set(in_leads)) != len(in_leads) or \
        len(set(out_leads)) != len(out_leads):
        raise ValueError('Lead lists must be sorted and with unique entries.')
    if len(in_leads) == 0 or len(out_leads) == 0:
        raise ValueError('No output is requested.')
    linsys = make_linear_sys(sys, out_leads, in_leads, energy,
                             force_realspace)
    out_modes = [len(i) for i in linsys[2]]
    in_modes = [i.shape[1] for i in linsys[1]]
    result = BlockResult(solve_linsys(*linsys[: -1]), linsys[3])
    result.in_leads = in_leads
    result.out_leads = out_leads
    return result


class BlockResult(namedtuple('BlockResultTuple', ['data', 'num_modes'])):
    """
    Solution of a transport problem, subblock of retarded Green's function.

    This class is derived from ``namedtuple('BlockResultTuple', ['data',
    'num_modes'])``. In addition to direct access to `data` and `num_modes`,
    this class also supports a higher level interface via its methods.

    Instance Variables
    ------------------
    data : numpy matrix
        a matrix containing all the requested matrix elements of Green's
        function.
    num_modes : list of integers
        a list of numbers of modes (or sites if real space lead representation
        is used) in each lead.
    """
    def block_coords(self, lead_out, lead_in):
        """
        Return slices corresponding to the block from lead_in to lead_out.
        """
        lead_out = self.out_leads.index(lead_out)
        lead_in = self.in_leads.index(lead_in)
        if not hasattr(self, '_sizes'):
            sizes = []
            for i in self.num_modes:
                if np.isscalar(i):
                    sizes.append(i)
                else:
                    sizes.append(i.shape[0])
            self._sizes = np.array(sizes)
            self._in_offsets = np.zeros(len(self.in_leads) + 1, int)
            self._in_offsets[1 :] = np.cumsum(self._sizes[self.in_leads])
            self._out_offsets = np.zeros(len(self.out_leads) + 1, int)
            self._out_offsets[1 :] = np.cumsum(self._sizes[self.out_leads])
        return slice(self._out_offsets[lead_out],
                     self._out_offsets[lead_out + 1]), \
               slice(self._in_offsets[lead_in], self._in_offsets[lead_in + 1])

    def submatrix(self, lead_out, lead_in):
        """Return the matrix elements from lead_in to lead_out."""
        return self.data[self.block_coords(lead_out, lead_in)]

    def _a_ttdagger_a_inv(self, lead_out, lead_in):
        gf = np.asmatrix(self.submatrix(lead_out, lead_in))
        if np.isscalar(self.num_modes[lead_out]):
            gamma_out = np.asmatrix(np.identity(self.num_modes[lead_out]))
        else:
            gamma_out = np.matrix(self.num_modes[lead_out], dtype=complex)
            gamma_out -= gamma_out.H
            gamma_out *= 1j
        if np.isscalar(self.num_modes[lead_in]):
            gamma_in = np.asmatrix(np.identity(self.num_modes[lead_in]))
        else:
            gamma_in = np.matrix(self.num_modes[lead_in], dtype=complex)
            gamma_in -= gamma_in.H
            gamma_in *= 1j
        return gamma_out * gf * gamma_in * gf.H

    def transmission(self, lead_out, lead_in):
        """Return transmission from lead_in to lead_out."""
        if np.isscalar(self.num_modes[lead_out]) and \
           np.isscalar(self.num_modes[lead_in]):
            return la.norm(self.submatrix(lead_out, lead_in))**2
        else:
            return np.trace(self._a_ttdagger_a_inv(lead_out, lead_in)).real

    def noise(self, lead_out, lead_in):
        """Return shot noise from lead_in to lead_out."""
        ttdag = self._a_ttdagger_a_inv(lead_out, lead_in)
        ttdag -= ttdag * ttdag
        return np.trace(ttdag).real
