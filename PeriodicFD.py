#!/usr/bin/python
import numpy as np

def periodicFD(L, mask, epsIn, epsOut, Ez, shouldPlot=False, dirichletBC=True, computeEps=False):
	"""
	Calculate electrostatic potential in a box of size L (dimensions [3,])
	with grid geometry specified by mask (3D scalar field).
	The dielectric function is epsIn where mask=1 and epsOut where mask=0.
	A net electric field Ez along the third direction is the source term.
	Optionally, if shouldPlot=True, plot slices of the potential for debugging
	Returns potential (same dimensions as mask).
	The third direction is treated as non-periodic if dirichletBC=True,
	otherwise it is treated as periodic.
	If computeEps=True, then return [eps_xz, eps_yz, eps_zz] instead of phi. (Requires dirichletBC=False.)
	"""
	from common import printDuration
	#Check inputs:
	assert L.shape == (3,)
	assert len(mask.shape) == 3
	S = np.array(mask.shape, dtype=int)
	Omega = np.prod(L) #unit cell volume
	assert isinstance(epsIn, float)
	assert isinstance(epsOut, float)
	assert isinstance(Ez, float)
	if computeEps:
		assert dirichletBC==False
	#Initialize grid:
	prodS = np.prod(S)
	h = L / S
	hMax = np.max(h)
	i0,i1,i2 = np.meshgrid(np.arange(S[0]), np.arange(S[1]), np.arange(S[2]), indexing='ij')
	iMesh = np.concatenate((i0[...,None], i1[...,None], i2[...,None]), axis=-1)
	x = iMesh * h[None,None,None,:]
	#Calculate epsInv on edges:
	epsInv = np.zeros(x.shape) #extra dimension is for edge direction
	for dim in range(3):
		maskAv = 0.5*(mask + np.roll(mask, -1, axis=dim))
		epsInv[...,dim] = 1./epsOut + maskAv*(1./epsIn - 1./epsOut)
	print('\tAverage epsInv:', np.mean(epsInv))
	#Construct matrix for div(eps grad()):
	L_i = []; L_j = []; L_val = []
	iOffs = np.eye(3, dtype=int)[None,None,None,...]
	iCur = np.repeat(iMesh[...,None,:], 3, axis=-2)
	iPlus = np.reshape(iCur + iOffs, (-1,3))
	iCur = np.reshape(iCur, (-1,3))
	edgeVal = 1./(epsInv * h[None,None,None,:]**2).flatten()
	L_i = np.vstack((iCur, iPlus, iCur, iPlus)) #diagonal terms at iCur and iPlus, followed by off-diagonal terms
	L_j = np.vstack((iCur, iPlus, iPlus, iCur))
	L_val = np.concatenate((edgeVal, edgeVal, -edgeVal, -edgeVal))
	iCur = None; iPlus = None; edgeVal = None
	#--- Fix wrap-around terms, constructing rhs term:
	stride = np.array([S[1]*S[2], S[2], 1], dtype=int)
	L_iFlat = np.dot(L_i % S[None,:], stride)
	L_jFlat = np.dot(L_j % S[None,:], stride)
	rhs = np.zeros((prodS,))
	iCheck = (L_i[:,2]==S[2]); L_i = None
	jCheck = (L_j[:,2]==S[2]); L_j = None
	iWrap = np.where(np.logical_and(iCheck, np.logical_not(jCheck)))[0]
	jWrap = np.where(np.logical_and(jCheck, np.logical_not(iCheck)))[0]
	if dirichletBC:
		rhs[L_iFlat[iWrap]] = -L_val[iWrap] * (-Ez*(-h[2])); L_val[iWrap] = 0.; #Dirichlet BC on z = -dz
		rhs[L_iFlat[jWrap]] = -L_val[jWrap] * (-Ez*L[2]);    L_val[jWrap] = 0.; #Dirichlet BC on z = +Lz
	else:
		rhs[L_iFlat[iWrap]] = -L_val[iWrap] * (+Ez*L[2]) #Introduce Ez*L[2] potential difference between ends
		rhs[L_iFlat[jWrap]] = -L_val[jWrap] * (-Ez*L[2]) #Introduce Ez*L[2] potential difference between ends
	#Construct preconditioner:
	iG = np.reshape(iMesh, (-1,3))
	iG = np.where(iG>S[None,:]/2, iG-S[None,:], iG)
	Gsq = np.sum((iG * (2*np.pi/L[None,:]))**2, axis=-1)
	Gsq[0] = np.min(Gsq[1:])
	invGsq = np.reshape(1./Gsq, S)
	phi0 = -Ez*x[...,2].flatten()
	if not dirichletBC:
		invGsq[0,0,0] = 0. #periodic G=0 projection
		phi0 -= np.mean(phi0)
	def precondFunc(x):
		return np.real(np.fft.ifftn(invGsq * np.fft.fftn(np.reshape(x,S))).flatten())
	from scipy.sparse.linalg import LinearOperator, cg
	from scipy.sparse import csr_matrix
	precondOp = LinearOperator((prodS,prodS), precondFunc)
	#Solve matrix equations:
	Lhs = csr_matrix((L_val, (L_iFlat, L_jFlat)), shape=(prodS, prodS))
	L_val = None
	L_iFlat = None
	L_jFlat = None
	print('\tMatrix dimensions:', Lhs.shape, 'with', Lhs.nnz, 'non-zero elements (fill', '%.2g%%)' % (Lhs.nnz*100./np.prod(Lhs.shape)))
	global nIter
	nIter = 0
	def iterProgress(x):
		global nIter
		nIter += 1
		print(nIter, end=' ', flush=True)
	print('\tCG: ', end='', flush=True)
	phi,info = cg(Lhs, rhs, callback=iterProgress, x0=phi0, M=precondOp, maxiter=100)
	print('done.', flush=True)
	phi = np.reshape(phi,S)
	Lhs = None; rhs = None
	#Compute epsilon instead if required:
	if computeEps:
		epsEff = np.zeros((3,))
		for dim in range(3):
			phiDiff = phi - np.roll(phi, -1, axis=dim)
			if dim==2:
				phiDiff[:,:,-1] += Ez*L[2]
			epsEff[dim] = np.mean((phiDiff) / (h[dim]*epsInv[...,2]*Ez))
		return epsEff
	#Optional plot:
	if shouldPlot:
		import matplotlib.pyplot as plt
		plt.figure(1)
		plotSlice = phi[0,:,:]
		plt.imshow(plotSlice)
		plt.colorbar()
		plt.show()
	return phi


def computeEps(L, mask, epsIn, epsOut):
	"""
	Compute dielectric tensor in a box of size L (dimensions [3,])
	with grid geometry specified by mask (3D scalar field).
	The dielectric function is epsIn where mask=1 and epsOut where mask=0.
	"""
	print('Computing dielectric tensor:')
	epsEff = np.zeros((3,3))
	epsEff[0] = periodicFD(L[[2,1,0]], mask.swapaxes(0,2), epsIn, epsOut, 1., False, False, True)[[2,1,0]] #field applied along x
	epsEff[1] = periodicFD(L[[0,2,1]], mask.swapaxes(1,2), epsIn, epsOut, 1., False, False, True)[[0,2,1]] #field applied along y
	epsEff[2] = periodicFD(L, mask, epsIn, epsOut, 1., False, False, True) #field applied along z
	return epsEff

#---------- Test code ----------
if __name__ == "__main__":
	
	L = np.array([10.,10.,20.])
	S = np.array([50, 50, 100])
	
	#Create mask containing a single sphere:
	from scipy.special import erfc
	r0 = np.array([0.,5.,9.]); R = 3.5
	grids1D = tuple([ np.arange(Si, dtype=float)*(L[i]/Si) for i,Si in enumerate(S) ])
	dr = np.array(np.meshgrid(*grids1D, indexing='ij')) - r0[:,None,None,None] #displacements from center of sphere
	Lbcast = np.array(L)[:,None,None,None]
	dr -= np.floor(0.5+dr/Lbcast)*Lbcast #wrap displacements by minimum image convention
	mask = 0.5*erfc(np.linalg.norm(dr, axis=0) - R) #1-voxel smoothing

	#Calculate potential with FD:
	epsIn = 3.0
	epsOut = 2.0
	print('Computing potential:')
	phi = periodicFD(L, mask, epsIn, epsOut, 1., True)
	
	#Calculate dielectric constant:
	epsEff = computeEps(L, mask, epsIn, epsOut)
	print('Dielectric tensor:\n', epsEff)
	print('epsAvg:', np.trace(epsEff)/3)
	
	#Print Clausius-Mossoti estimate:
	CMterm = (4*np.pi*(R**3)/(3.*np.prod(L)*epsOut)) * epsOut*(epsIn-epsOut)/(epsIn+2.*epsOut)
	epsCM = epsOut*(1.+CMterm)/(1.-2*CMterm)
	print('epsCM:', epsCM)
	