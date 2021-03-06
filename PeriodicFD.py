#!/usr/bin/python
import numpy as np
from pyfftw.interfaces import numpy_fft as np_fft
from scipy.sparse import csr_matrix
from scipy.sparse.linalg import LinearOperator, cg
from multiprocessing import cpu_count
import pyfftw; pyfftw.config.NUM_THREADS = cpu_count()

#Poisson solver using finite-difference discretization:
class PeriodicFD:
	
	def __init__(self, L, mask, epsIn, epsOut, dirichletBC=[False,False,False]):
		"""
		Set up Poisson solver in a box of size L (dimensions [3,])
		with grid geometry specified by mask (3D scalar field).
		The dielectric function is epsIn where mask=1 and epsOut where mask=0.
		The boundary conditions in each direction are Dirichlet if corresponding
		entry of dirichletBC is True, and periodic otherwise (default).
		"""
		#Check inputs:
		assert L.shape == (3,)
		assert L.dtype == float
		assert len(mask.shape) == 3
		S = np.array(mask.shape, dtype=int)
		Omega = np.prod(L) #unit cell volume
		assert isinstance(epsIn, float)
		assert isinstance(epsOut, float)
		#Remember inputs:
		self.L = L
		self.mask = mask
		self.epsIn = epsIn
		self.epsOut = epsOut
		self.dirichletBC = dirichletBC
		#Construct preconditioner:
		prodS = S.prod()
		h = L / S
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
		#Construct matrix for div(eps grad()):
		data = np.zeros((prodS, 7)) #non-zero entries of Poisson operator (diagonal, followed by off diagonals)
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
				data[:,iEdge] = -1./((h[dim]**2) * (1./epsOut + ((1./epsIn - 1./epsOut)*0.5)*(mask.reshape(-1) + mask.reshape(-1)[indices[:,iEdge]])))
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
					self.rhsVal[dim].append(-data[sel,iEdge] * (-(L[dim] if pm>0 else -h[dim]))) #set phi in adjacent off-grid point
					data[sel,iEdge] = 0.
				else:
					self.rhsVal[dim].append(-data[sel,iEdge] * (-L[dim]*pm)) #potential difference between ends
			self.rhsSel[dim] = np.concatenate(self.rhsSel[dim])
			self.rhsVal[dim] = np.concatenate(self.rhsVal[dim])
		self.Lhs = csr_matrix((data.reshape(-1), indices.reshape(-1), range(0,7*prodS+1,7)), shape=(prodS, prodS))
		data = None
		indices = None
		print('\tMatrix dimensions:', self.Lhs.shape, 'with', self.Lhs.nnz, 'non-zero elements (fill', '%.2g%%)' % (self.Lhs.nnz*100./np.prod(self.Lhs.shape)))
	
	def solve(self, E, shouldPlot=False):
		"""
		Compute potential due to an electric field E (dimensions [3,])
		Optionally, if shouldPlot=True, plot slices of the potential for debugging
		Returns potential (same dimensions as self.mask).
		"""
		assert len(E) == 3
		#Create rhs and initial guess:
		S = np.array(self.mask.shape)
		prodS = S.prod()
		rhs = np.zeros(prodS)
		phi = np.zeros(prodS)
		for dim,Edim in enumerate(E):
			if Edim != 0.:
				fieldProfile = np.arange(S[dim]) * (-Edim * self.L[dim] / S[dim])
				shape = [1,1,1]; shape[dim] = S[dim]
				tile = np.copy(S); tile[dim] = 1
				phi += np.tile(fieldProfile.reshape(shape), tile).reshape(-1)
				rhs[self.rhsSel[dim]] += Edim * self.rhsVal[dim]
		if not np.any(self.dirichletBC):
			phi -= phi.mean()
		#Create preconditioner:
		def precondFunc(x):
			return np.real(np_fft.ifftn(self.invGsq * np_fft.fftn(np.reshape(x,S))).flatten())
		precondOp = LinearOperator((prodS,prodS), precondFunc)
		#Solve matrix equations:
		global nIter
		nIter = 0
		def iterProgress(x):
			global nIter
			nIter += 1
			print(nIter, end=' ', flush=True)
		print('\tCG: ', end='', flush=True)
		phi,info = cg(self.Lhs, rhs, tol=1e-6, callback=iterProgress, x0=phi, M=precondOp, maxiter=100)
		print('done.', flush=True)
		phi = np.reshape(phi,S)
		rhs = None
		#Optional plot:
		if shouldPlot:
			import matplotlib.pyplot as plt
			plt.figure(1)
			plotSlice = phi[0,:,:]
			plt.imshow(plotSlice)
			plt.colorbar()
			plt.show()
		return phi

	def computeEps(self, deriv=False):
		"""
		Compute dielectric tensor (must have dirichletBC = False in all directions).
		If deriv=True, also return d(epsEff)/d(self.epsIn) and d(epsEff)/d(self.epsOut)
		"""
		assert (not np.any(self.dirichletBC)) #all directions must be periodic
		print('Computing dielectric tensor:')
		epsEff = np.zeros((3,3))
		epsEff_epsIn = np.zeros((3,3)) #optional derivative w.r.t. self.epsIn
		epsEff_epsOut = np.zeros((3,3)) #optional derivative w.r.t. self.epsOut
		S = np.array(self.mask.shape)
		prodS = S.prod()
		h = self.L / S
		
		#Preconditioner for first-order solve below:
		def precondFunc(x):
			return np.real(np_fft.ifftn(self.invGsq * np_fft.fftn(np.reshape(x,S))).flatten())
		precondOp = LinearOperator((prodS,prodS), precondFunc)
		#Iteration progress for first-order solve below: 
		global nIter
		def iterProgress(x):
			global nIter
			nIter += 1
			print(nIter, end=' ', flush=True)
		
		#Loop over E-field perturbations:
		for dim1 in range(3):
			#Get potential for unit field applied in current direction:
			E = [0,0,0]; E[dim1]=1
			phi = self.solve(E)
			if deriv:
				Lphi_in = np.zeros(S)  #(dL/depsIn)(phi) where L is the modified Poisson operator
				Lphi_out = np.zeros(S) #(dL/depsOut)(phi) where L is the modified Poisson operator
			#Compute D, which is effectively the same as dielectric since E = 1
			for dim2 in range(3):
				phiDiff = phi - np.roll(phi, -1, axis=dim2)
				if dim1 == dim2:
					slc = [slice(None)] * 3
					slc[dim1] = -1
					phiDiff[tuple(slc)] += self.L[dim2]
				maskMean = 0.5*(self.mask + np.roll(self.mask, -1, axis=dim2)) #mean of mask along edge endpoints
				epsEff[dim1,dim2] = np.mean(phiDiff / (1./self.epsOut + (1./self.epsIn - 1./self.epsOut)*maskMean)) / h[dim2]
				if deriv:
					#Derivative contribution in D due to first-order change in epsIn:
					integrand = phiDiff * maskMean / (1./self.epsOut + (1./self.epsIn - 1./self.epsOut)*maskMean)**2
					epsEff_epsIn[dim1,dim2] = np.mean(integrand) / (h[dim2] * self.epsIn**2)
					Lphi_in -= (integrand - np.roll(integrand, +1, axis=dim2)) * (1./(h[dim2]*self.epsIn)**2)
					integrand = None
					#Derivative contribution in D due to first-order change in epsout:
					integrand = phiDiff * (1.-maskMean) / (1./self.epsOut + (1./self.epsIn - 1./self.epsOut)*maskMean)**2
					epsEff_epsOut[dim1,dim2] = np.mean(integrand) / (h[dim2] * self.epsOut**2)
					Lphi_out -= (integrand - np.roll(integrand, +1, axis=dim2)) * (1./(h[dim2]*self.epsOut)**2)
					integrand = None
				maskMean = None
				phiDiff = None
			if deriv:
				#Compute first-order change in phi due to changes in epsIn:
				nIter = 0
				print('\tCG_in: ', end='', flush=True)
				phi_in,info = cg(self.Lhs, Lphi_in.reshape(-1), tol=1e-6, callback=iterProgress, x0=np.zeros(prodS), M=precondOp, maxiter=100)
				print('done.', flush=True)
				phi_in = phi_in.reshape(S)
				Lphi_in = None
				#Compute first-order change in phi due to changes in epsOut:
				nIter = 0
				print('\tCG_out: ', end='', flush=True)
				phi_out,info = cg(self.Lhs, Lphi_out.reshape(-1), tol=1e-6, callback=iterProgress, x0=np.zeros(prodS), M=precondOp, maxiter=100)
				print('done.', flush=True)
				phi_out = phi_out.reshape(S)
				Lphi_out = None
				#Compute derivative contributions in D due to first-order changes in phi:
				for dim2 in range(3):
					epsAvg = 1. / (1./self.epsOut + (0.5*(1./self.epsIn - 1./self.epsOut))*(self.mask + np.roll(self.mask, -1, axis=dim2)))
					epsEff_epsIn[dim1,dim2] += np.mean((phi_in - np.roll(phi_in, -1, axis=dim2)) * epsAvg) / h[dim2]
					epsEff_epsOut[dim1,dim2] += np.mean((phi_out - np.roll(phi_out, -1, axis=dim2)) * epsAvg) / h[dim2]
		if deriv:
			return epsEff, epsEff_epsIn, epsEff_epsOut
		else:
			return epsEff


#---------- Test code ----------
if __name__ == "__main__":
	
	L = np.array([10.,10.,20.])
	S = np.array([100, 100, 200])
	
	#Create mask containing a few spheres:
	from scipy.special import erfc
	r0 = np.array([[0.,5.,8.],[5.,1.,4.]]); R = 3.5
	grids1D = tuple([ np.arange(Si, dtype=float)*(L[i]/Si) for i,Si in enumerate(S) ])
	dr = np.array(np.meshgrid(*grids1D, indexing='ij'))[None,...] - r0[:,:,None,None,None] #displacements from center of sphere
	Lbcast = np.array(L)[None,:,None,None,None]
	dr -= np.floor(0.5+dr/Lbcast)*Lbcast #wrap displacements by minimum image convention
	mask = 0.5*erfc(np.linalg.norm(dr, axis=1).min(axis=0) - R) #1-voxel smoothing

	#Calculate potential with FD:
	epsIn = 7.0
	epsOut = 2.0
	print('Computing potential:')
	PeriodicFD(L, mask, epsIn, epsOut, [False,False,True]).solve([0,0,1.], False)
	
	#Calculate dielectric constant:
	epsEff, epsEff_in, epsEff_out = PeriodicFD(L, mask, epsIn, epsOut).computeEps(deriv=True)
	print('Dielectric tensor:\n', epsEff)
	print('epsAvg:', np.trace(epsEff)/3)
	
	#Print Clausius-Mossoti estimate:
	CMterm = r0.shape[0] * (4*np.pi*(R**3)/(3.*np.prod(L)*epsOut)) * epsOut*(epsIn-epsOut)/(epsIn+2.*epsOut)
	epsCM = epsOut*(1.+CMterm)/(1.-2*CMterm)
	print('epsCM:', epsCM)
	
	#Finite difference derivatives of dielectric tensor:
	deps = 0.01
	epsEff_epsIn = (0.5/deps) * (PeriodicFD(L, mask, epsIn+deps, epsOut).computeEps() - PeriodicFD(L, mask, epsIn-deps, epsOut).computeEps())
	epsEff_epsOut = (0.5/deps) * (PeriodicFD(L, mask, epsIn, epsOut+deps).computeEps() - PeriodicFD(L, mask, epsIn, epsOut-deps).computeEps())
	print('\n--- d(eps)/d(epsIn) ---\nNumeric:\n', epsEff_epsIn, '\nAnalytic:\n', epsEff_in)
	print('\n--- d(eps)/d(epsOut) ---\nNumeric:\n', epsEff_epsOut, '\nAnalytic:\n', epsEff_out)
