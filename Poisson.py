#!/usr/bin/python
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg, bicgstab
from MultiGrid import MultiGrid

try:
    # Use faster FFTs from FFTW if available
    from pyfftw.interfaces import numpy_fft as np_fft
    from multiprocessing import cpu_count
    import pyfftw
    pyfftw.config.NUM_THREADS = cpu_count()
except ImportError:
    # Fallback to numpy FFT otherwise
    from numpy import fft as np_fft


#Poisson solver using finite-difference discretization:
class Poisson:
    
    def __init__(
        self,
        L,
        epsInv,
        dirichletBC=[False,False,False],
        use_multigrid=False,
        rescale=True
    ):
        """
        Set up Poisson solver in a box of size L (dimensions [3,])
        with inverse dielectric profile epsInv (3D scalar field).
        The boundary conditions in each direction are Dirichlet if corresponding
        entry of dirichletBC is True, and periodic otherwise (default).
        """
        #Check inputs:
        assert L.shape == (3,)
        assert L.dtype == float
        assert len(epsInv.shape) == 3
        S = np.array(epsInv.shape, dtype=int)
        Omega = np.prod(L) #unit cell volume

        #Remember inputs:
        self.L = L
        self.epsInv = epsInv
        self.dirichletBC = dirichletBC
        prodS = S.prod()
        h = L / S

        #Construct matrix for div(eps grad()):
        data = np.zeros(
            (prodS, 7),
            dtype=self.epsInv.dtype,  #complex/real matrix depending on epsInv
        ) #non-zero entries of Poisson operator (diagonal, followed by off diagonals)
        indices = np.tile(np.arange(prodS)[:,None], (1,7)) #corresponding column indices
        stride = np.array([S[1]*S[2], S[2], 1], dtype=int)
        iEdge = 0
        for dim in range(3):
            for pm in [-1,+1]:
                iEdge += 1
                #Determine offset to neighbor for this edge:
                neighOffset = np.full(S[dim], pm)
                if pm > 0:
                    neighOffset[-1] -= S[dim]
                else:
                    neighOffset[0] += S[dim]
                shape = [1,1,1]; shape[dim] = S[dim]
                tile = np.copy(S); tile[dim] = 1
                #Set corresponding column indices:
                indices[:,iEdge] += np.tile(neighOffset.reshape(shape), tile).reshape(-1) * stride[dim]
                #Set corresponding off-diagonal matrix element (average 1/epsilon on edge):
                data[:,iEdge] = -1./(((h[dim]**2) * 0.5) * (epsInv.reshape(-1) + epsInv.reshape(-1)[indices[:,iEdge]]))
        data[:,0] = -data.sum(axis=1) #set diagonal matrix elements:
        #--- Fix wrap-around terms, constructing rhs term:
        self.rhsSel = [[],[],[]] #Indices into sparse RHS term, per field direction
        self.rhsVal = [[],[],[]] #Corresponding values, for unit field value
        iEdge = 0
        for dim in range(3):
            for pm in [-1,+1]:
                iEdge += 1
                sel = np.where((indices[:,iEdge]-indices[:,0])*pm < 0)[0]
                self.rhsSel[dim].append(sel)
                if dirichletBC[dim]:
                    self.rhsVal[dim].append(-data[sel,iEdge] * (-((L[dim] + h[dim]) if pm>0 else 0.0))) #set phi in adjacent off-grid point
                    data[sel,iEdge] = 0.
                else:
                    self.rhsVal[dim].append(-data[sel,iEdge] * (-L[dim]*pm)) #potential difference between ends
            self.rhsSel[dim] = np.concatenate(self.rhsSel[dim])
            self.rhsVal[dim] = np.concatenate(self.rhsVal[dim])
        # --- Rescale for stability
        self.rescale = rescale
        if rescale:
            self.scale = np.sqrt(self.epsInv.flatten())
            for rhsSel_dim, rhsVal_dim in zip(self.rhsSel, self.rhsVal):
                rhsVal_dim *= self.scale[rhsSel_dim]  # Row-scale RHS
            data *= self.scale[:, None]  # Row-scale LHS
            data *= self.scale[indices]  # Col-scale LHS
        self.Lhs = csr_matrix((data.reshape(-1), indices.reshape(-1), range(0,7*prodS+1,7)), shape=(prodS, prodS))
        data = None
        indices = None
        print('\tMatrix dimensions:', self.Lhs.shape, 'with', self.Lhs.nnz, 'non-zero elements (fill', '%.2g%%)' % (self.Lhs.nnz*100./np.prod(self.Lhs.shape)))
        
        # Initialize preconditioner:
        if use_multigrid:
            mg = MultiGrid(self.Lhs, S, subtract_mean=(not np.any(dirichletBC)))
            precond_func = mg.Vcycle
        else:
            def getPrecond():
                grids1D = [np.concatenate((np.arange(s//2+1), np.arange(s//2+1-s,0))) for s in S]
                iG = np.array(np.meshgrid(*tuple(grids1D), indexing='ij')).reshape(3,-1).T
                Gsq = np.sum((iG * (2*np.pi/L[None,:]))**2, axis=-1); iG = None
                Gsq[0] = np.min(Gsq[1:])
                invGsq = np.reshape(1./Gsq, S)
                if not np.any(dirichletBC):
                    invGsq[0,0,0] = 0. #periodic G=0 projection
                return invGsq
            self.invGsq = getPrecond()
            
            def precond_func(x):
                x_scaled = (x / self.scale) if self.rescale else x
                result = np_fft.ifftn(
                    self.invGsq * np_fft.fftn(np.reshape(x_scaled, S))
                ).flatten()
                result = (result * self.scale) if self.rescale else result
                return result if (epsInv.dtype == np.complex128) else result.real
        self.precond = LinearOperator((prodS, prodS), precond_func)
    
    def solve(self, E, phi0=None, shouldPlot=False):
        """
        Compute potential due to an electric field E (dimensions [3,])
        Optionally provide starting potential phi0.
        Optionally, if shouldPlot=True, plot slices of the potential for debugging
        Returns potential (same dimensions as self.epsInv).
        """
        assert len(E) == 3

        #Create rhs and initial guess:
        S = np.array(self.epsInv.shape)
        prodS = S.prod()
        rhs = np.zeros(prodS, dtype=self.epsInv.dtype)
        phi = (
            np.zeros(prodS, dtype=self.epsInv.dtype)
            if (phi0 is None)
            else phi0.flatten()
        )
        for dim,Edim in enumerate(E):
            if Edim != 0.:
                rhs[self.rhsSel[dim]] += Edim * self.rhsVal[dim]
                if phi0 is None:
                    fieldProfile = (1 + np.arange(S[dim])) * (-Edim * self.L[dim] / S[dim])
                    shape = [1,1,1]; shape[dim] = S[dim]
                    tile = np.copy(S); tile[dim] = 1
                    phi += np.tile(fieldProfile.reshape(shape), tile).reshape(-1)
        if not np.any(self.dirichletBC):
            phi -= phi.mean()

        #Select solver:
        if (self.epsInv.dtype == np.complex128) or self.rescale:
            solver = bicgstab
            solver_name = "BiCGstab"
        else:
            solver = cg
            solver_name = "CG"

        #Solve matrix equations:
        global nIter
        nIter = 0
        def iterProgress(x):
            global nIter
            nIter += 1
            print(nIter, end=' ', flush=True)
        print(f'\t{solver_name}: ', end='', flush=True)
        if shouldPlot:
            phiPrev = np.copy(phi).reshape(S)  # to plot change below
        if self.rescale:
            phi /= self.scale  # convert to scaled potential (compensate for column scaling)
        phi, info = solver(self.Lhs, rhs, tol=1e-4, callback=iterProgress, x0=phi, M=self.precond, maxiter=100)
        if self.rescale:
            phi *= self.scale  # convert to actual potential (compensate for column scaling)
        print('done.', flush=True)
        phi = phi.reshape(S)
        rhs = None

        #Optional plot:
        if shouldPlot:
            import matplotlib.pyplot as plt
            plt.figure()
            plotSlice = (phi - phiPrev)[0, :, :]
            plt.imshow(plotSlice)
            plt.colorbar(label=r"$\Delta\phi$")
            plt.show()
        return phi

    def computeEps(self, phi0=None):
        """
        Compute dielectric tensor (must have dirichletBC = False in all directions).
        Optionally, phi0 specifies sequence of three potential profiles to
        use as initial guesses for Poisson solve in each field direction.
        Return dielectric tensor and final phi (that can be used as phi0 for related run).
        """
        assert (not np.any(self.dirichletBC)) #all directions must be periodic
        print('\tComputing dielectric tensor:')
        epsEff = np.zeros((3, 3), dtype=self.epsInv.dtype)
        S = np.array(self.epsInv.shape)
        prodS = S.prod()
        h = self.L / S
        
        #Loop over E-field perturbations:
        phi_out = []
        for dim1 in range(3):
            #Get potential for unit field applied in current direction:
            E = [0,0,0]
            E[dim1]=1
            phi0_i = None if (phi0 is None) else phi0[dim1]
            phi = self.solve(E, phi0=phi0_i)
            phi_out.append(phi)
            #Compute D, which is effectively the same as dielectric since E = 1
            for dim2 in range(3):
                phiDiff = phi - np.roll(phi, -1, axis=dim2)
                if dim1 == dim2:
                    slc = [slice(None)] * 3
                    slc[dim1] = -1
                    phiDiff[tuple(slc)] += self.L[dim2]
                epsInvMean = 0.5*(self.epsInv + np.roll(self.epsInv, -1, axis=dim2)) #mean of epsInv along edge endpoints
                epsEff[dim1,dim2] = np.mean(phiDiff / epsInvMean) / h[dim2]
                epsInvMean = None
                phiDiff = None
        return epsEff, phi_out


#---------- Test code ----------
if __name__ == "__main__":
    
    L = np.full(3, 10.)
    S = np.full(3, 100)
    
    #Create mask containing a few spheres:
    from scipy.special import erfc
    r0 = np.array([[0.,5.,8.],[5.,1.,4.]]); R = 2.8
    grids1D = tuple([ np.arange(Si, dtype=float)*(L[i]/Si) for i,Si in enumerate(S) ])
    dr = np.array(np.meshgrid(*grids1D, indexing='ij'))[None,...] - r0[:,:,None,None,None] #displacements from center of sphere
    Lbcast = np.array(L)[None,:,None,None,None]
    dr -= np.floor(0.5+dr/Lbcast)*Lbcast #wrap displacements by minimum image convention
    mask = 0.5*erfc(np.linalg.norm(dr, axis=1).min(axis=0) - R) #1-voxel smoothing
    
    #Convert to dielectric profile:
    epsIn = 40.0+170j
    epsOut = 3.0+0.1j
    epsInv = 1./epsOut + (1./epsIn - 1./epsOut) * mask

    #Calculate potential with FD:
    print('Computing potential:')
    Poisson(L, epsInv, [False,False,True]).solve([0, 0, 1.], shouldPlot=False)
    
    #Calculate dielectric constant:
    epsEff, _ = Poisson(L, epsInv).computeEps()
    print('Dielectric tensor:\n', epsEff)
    print('epsAvg:', np.trace(epsEff)/3)
    
    #Print Clausius-Mossoti estimate:
    CMterm = (
        r0.shape[0]
        * (4*np.pi * (R**3) / (3. * np.prod(L) * epsOut))
        * epsOut * (epsIn - epsOut) / (epsIn + 2.*epsOut)
    )
    epsCM = epsOut * (1. + CMterm) / (1. - 2*CMterm)
    print('epsCM:', epsCM)
