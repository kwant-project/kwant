__all__ = ['make_linear_sys', 'LinearSys', 'solve', 'ldos', 'BlockResult']

from functools import reduce
from collections import namedtuple
import numpy as np
import scipy.linalg as la
import scipy.sparse as sp
import scipy.sparse.linalg.dsolve.umfpack as umfpack
from kwant import physics, system

# This patches a memory leak in scipy:
# http://projects.scipy.org/scipy/ticket/1597
#
# TODO: Remove this code once it is likely that the official bug fix has
# reached all of our users.
def del_for_umfpackcontext(self):
    self.free()
if not hasattr(umfpack.UmfpackContext, '__del__'):
    umfpack.UmfpackContext.__del__ = del_for_umfpackcontext
del del_for_umfpackcontext

def factorized(A, piv_tol=1.0, sym_piv_tol=1.0):
    """
    Return a fuction for solving a sparse linear system, with A pre-factorized.

    Example:
      solve = factorized( A ) # Makes LU decomposition.
      x1 = solve( rhs1 ) # Uses the LU factors.
      x2 = solve( rhs2 ) # Uses again the LU factors.

    Parameters
    ----------
    A : csc_matrix
        matrix to be factorized
    piv_tol : float, 0 <= piv_tol <= 1.0
    sym_piv_tol : float, 0 <= piv_tol <= 1.0
        thresholds used by umfpack for pivoting. 0 means no pivoting,
        1.0 means full pivoting as in dense matrices (guaranteeing
        stability, but reducing possibly sparsity). Defaults of umfpack
        are 0.1 and 0.001 respectively. Whether piv_tol or sym_piv_tol
        are used is decided internally by umfpack, depending on whether
        the matrix is "symmetric" enough.
    """

    if not sp.isspmatrix_csc(A):
        A = sp.csc_matrix(A)

    A.sort_indices()
    A = A.asfptype()  #upcast to a floating point format

    if A.dtype.char not in 'dD':
        raise ValueError("convert matrix data to double, please, using"
                         " .astype()")

    family = {'d' : 'di', 'D' : 'zi'}
    umf = umfpack.UmfpackContext( family[A.dtype.char] )

    # adjust pivot thresholds
    umf.control[umfpack.UMFPACK_PIVOT_TOLERANCE] = piv_tol
    umf.control[umfpack.UMFPACK_SYM_PIVOT_TOLERANCE] = sym_piv_tol

    # Make LU decomposition.
    umf.numeric( A )

    def solve( b ):
        return umf.solve( umfpack.UMFPACK_A, A, b, autoTranspose = True )

    return solve


LinearSys = namedtuple('LinearSys', ['h_sys', 'rhs', 'keep_vars'])


def make_linear_sys(sys, out_leads, in_leads, energy=0, force_realspace=False):
    """
    Make a sparse linear system of equations defining a scattering problem.

    Parameters
    ----------
    sys : kwant.system.FiniteSystem
        low level system, containing the leads and the Hamiltonian of a
        scattering region.
    out_leads : list of integers
        numbers of leads where current or wave function is extracted
    in_leads : list of integers
        numbers of leads in which current or wave function is injected.
    energy : number
        excitation energy at which to solve the scattering problem.
    force_realspace : bool
        calculate Green's function between the outermost lead
        sites, instead of lead modes. This is almost always
        more computationally expensive and less stable.

    Returns
    -------
    (h_sys, rhs, keep_vars) : LinearSys
        `h_sys` is a numpy.sparse.csc_matrix, containing the left hand side
        of the system of equations, `rhs` is a list of matrices with the
        right hand side, `keep_vars` is a list of numbers of variables in the
        solution that have to be stored (typically a small part of the
        complete solution).
    lead_info : list of objects
        Contains one entry for each lead.  For a lead defined as a
        tight-binding system, this is an instance of `kwant.physics.Modes` (as
        returned by `kwant.physics.modes`), otherwise the lead self-energy
        matrix.

    Notes
    -----
    Both incoming and outgoing leads can be defined via either self-energy,
    or a low-level translationally invariant system.
    The system of equations that is created is described in
    kwant/doc/other/linear_system.pdf
    """
    if not sys.lead_neighbor_seqs:
        raise ValueError('System contains no leads.')
    h_sys, norb = sys.hamiltonian_submatrix(sparse=True)[:2]
    h_sys = h_sys.tocsc()
    h_sys = h_sys - energy * sp.identity(h_sys.shape[0], format='csc')

    # Hermiticity check.
    if np.any(np.abs((h_sys - h_sys.T.conj()).data) > 1e-13):
        raise ValueError('System Hamiltonian is not Hermitian.')

    offsets = np.zeros(norb.shape[0] + 1, int)
    offsets[1 :] = np.cumsum(norb)

    # Process the leads, generate the eigenvector matrices and lambda vectors.
    # Then create blocks of the linear system and add them step by step.
    keep_vars = []
    rhs = []
    lead_info = []
    for leadnum, lead_neighbors in enumerate(sys.lead_neighbor_seqs):
        lead = sys.leads[leadnum]
        if isinstance(lead, system.InfiniteSystem) and not force_realspace:
            h = lead.slice_hamiltonian()

            # Hermiticity check.
            if not np.allclose(h, h.T.conj(), rtol=1e-13):
                msg = 'Lead number {0} has a non-Hermitian slice Hamiltonian.'
                raise ValueError(msg.format(leadnum))

            h -= energy * np.identity(h.shape[0])
            v = lead.inter_slice_hopping()
            modes = physics.modes(h, v)
            lead_info.append(modes)
            if not np.any(v):
                continue
            u, ulinv, nprop, svd = modes

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
                v_sp = tmp
                vdaguout_sp = tmp.T * sp.csr_matrix(np.dot(v.T.conj(), u_out))
                lead_mat = - ulinv_out

            h_sys = sp.bmat([[h_sys, vdaguout_sp], [v_sp, lead_mat]])

            if nprop > 0 and leadnum in in_leads:
                if svd:
                    vdaguin_sp = tmp.T * sp.csr_matrix(-np.dot(svd[2] * svd[1],
                                                               u_in))
                    lead_mat_in = ulinv_in
                else:
                    vdaguin_sp = tmp.T * sp.csr_matrix(-np.dot(v.T.conj(),
                                                               u_in))
                    lead_mat_in = ulinv_in

                rhs.append(sp.bmat([[vdaguin_sp], [lead_mat_in]]))
        else:
            sigma = lead.self_energy(energy)
            lead_info.append(sigma)
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

    return LinearSys(h_sys, rhs, keep_vars), lead_info


def solve_linear_sys(a, b, keep_vars=None):
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
    This function is largely a wrapper to `factorized`.
    """
    a = sp.csc_matrix(a)

    if keep_vars == None:
        keep_vars = [range(a.shape[1])]
    slv = factorized(a)
    keeptot = sum(keep_vars, [])
    sols = []
    vec = np.empty(a.shape[0], complex)
    for mat in b:
        if mat.shape[1] == 0:
            continue
        mat = sp.csr_matrix(mat)
        for j in xrange(mat.shape[1]):
            vec[: mat.shape[0]] = mat[:, j].todense().flatten()
            vec[mat.shape[0] :] = 0
            sols.append(slv(vec)[keeptot])
    return np.mat(sols).T


def solve(sys, energy=0, out_leads=None, in_leads=None, force_realspace=False):
    """
    Calculate a Green's function of a system.

    Parameters
    ----------
    sys : `kwant.system.FiniteSystem`
        low level system, containing the leads and the Hamiltonian of a
        scattering region.
    energy : number
        excitation energy at which to solve the scattering problem.
    out_leads : list of integers
        numbers of leads where current or wave function is extracted
    in_leads : list of integers
        numbers of leads in which current or wave function is injected.
    force_realspace : bool
        calculate Green's function between the outermost lead
        sites, instead of lead modes. This is almost always
        more computationally expensive and less stable.

    Returns
    -------
    output : `BlockResult`
        see notes below and `BlockResult` docstring for more information about
        the output format.

    Notes
    -----
    Both in_leads and out_leads must be sorted and must only contain
    unique entries.

    Returns the Green's function elements between in_leads and out_leads. If
    the leads are defined as a self-energy, the result is just the real
    space retarded Green's function between from in_leads to out_leads. If the
    leads are defined as tight-binding systems, then Green's function from
    incoming to outgoing modes is returned. Also returned is a list containing
    the output of `kwant.physics.modes` for the leads which are defined as
    builders, and self-energies for leads defined via self-energy. This list
    allows to split the Green's function into blocks corresponding to different
    leads. The Green's function elements between incoming and outgoing modes
    form the scattering matrix of the system. If some leads are defined via
    self-energy, and some as tight-binding systems, result has Green's
    function's elements between modes and sites.

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
    linsys, lead_info = make_linear_sys(sys, out_leads, in_leads, energy,
                                        force_realspace)
    out_modes = [len(i) for i in linsys.keep_vars]
    in_modes = [i.shape[1] for i in linsys.rhs]
    result = BlockResult(solve_linear_sys(*linsys), lead_info)
    result.in_leads = in_leads
    result.out_leads = out_leads
    return result


class BlockResult(namedtuple('BlockResultTuple', ['data', 'lead_info'])):
    """
    Solution of a transport problem, subblock of retarded Green's function.

    This class is derived from ``namedtuple('BlockResultTuple', ['data',
    'lead_info'])``. In addition to direct access to `data` and `lead_info`,
    this class also supports a higher level interface via its methods.

    Instance Variables
    ------------------
    data : numpy matrix
        a matrix containing all the requested matrix elements of Green's
        function.
    lead_info : list of data
        a list with output of `kwant.physics.modes` for each lead defined as a
        builder, and self-energy for each lead defined as self-energy term.
    """
    def block_coords(self, lead_out, lead_in):
        """
        Return slices corresponding to the block from lead_in to lead_out.
        """
        lead_out = self.out_leads.index(lead_out)
        lead_in = self.in_leads.index(lead_in)
        if not hasattr(self, '_sizes'):
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
        return slice(self._out_offsets[lead_out],
                     self._out_offsets[lead_out + 1]), \
               slice(self._in_offsets[lead_in], self._in_offsets[lead_in + 1])

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
            return la.norm(self.submatrix(lead_out, lead_in))**2
        else:
            return np.trace(self._a_ttdagger_a_inv(lead_out, lead_in)).real

    def noise(self, lead_out, lead_in):
        """Return shot noise from lead_in to lead_out."""
        ttdag = self._a_ttdagger_a_inv(lead_out, lead_in)
        ttdag -= np.dot(ttdag, ttdag)
        return np.trace(ttdag).real

def ldos(fsys, e=0):
    """
    Calculate the local density of states of a system at a given energy.

    Parameters
    ----------
    sys : `kwant.system.FiniteSystem`
        low level system, containing the leads and the Hamiltonian of the
        scattering region.
    energy : number
        excitation energy at which to solve the scattering problem.

    Returns
    -------
    ldos : a numpy array
        local density of states at each orbital of the system.
    """
    Modes = physics.Modes
    linsys, lead_info = make_linear_sys(fsys, [], [], e)
    h = linsys.h_sys
    num_extra_vars = sum(li.vecs.shape[1] - li.nmodes
                         for li in lead_info if isinstance(li, Modes))
    num_orb = h.shape[0] - num_extra_vars
    vec = np.zeros(h.shape[0], complex)
    ldos = np.zeros(num_orb, complex)
    slv = factorized(h)
    for i in xrange(num_orb):
        vec[i] = 1
        ldos[i] = slv(vec)[i]
        vec[i] = 0
    return ldos.imag / np.pi
