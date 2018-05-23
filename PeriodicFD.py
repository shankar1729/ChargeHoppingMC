#!/usr/bin/python
import numpy as np
from scipy.special import erfc
import scipy.sparse as sparse

def periodicFD(L, S, r0, R, epsIn, epsOut, Ez, shouldPlot=False):
	"""
	Calculate electrostatic potential in a box of size L (dimensions [3,])
	with grid sample counts S (dimensions [3,]; integer) assumed to be
	periodic in the first two directions. Geometry is specified by
	spheres centered at r0 (dimensions [N,3]) of radii R (dimensions [N,]),
	with dielectric function inside the spheres = epsIn and that outside = epsOut.
	A net electric field Ez along the non-periodic third direction is the source term.
	Optionally, if shouldPlot=True, plot slices of the potential for debugging
	Returns potential (dimensions S) and mask (dimensions S) = 1 inside spheres and 0 outside.
	"""
	#Check inputs:
	assert L.shape == (3,)
	assert S.shape == (3,)
	Omega = np.prod(L) #unit cell volume
	assert r0.shape[1] == 3
	N = r0.shape[0] #number of spheres
	assert R.shape == (N,)
	assert isinstance(epsIn, float)
	assert isinstance(epsOut, float)
	assert isinstance(Ez, float)
	#Initialize grid:
	prodS = np.prod(S)
	h = L / S
	hMax = np.max(h)
	i0,i1,i2 = np.meshgrid(np.arange(S[0]), np.arange(S[1]), np.arange(S[2]), indexing='ij')
	iMesh = np.concatenate((i0[...,None], i1[...,None], i2[...,None]), axis=-1)
	x = iMesh * h[None,None,None,:]
	#Calculate mask on grid points:
	mask = np.zeros(S)
	for i in range(N):
		#Construct sphere in bounding box
		r0offset = np.round(r0[i] / h).astype(int)
		SiHlf = np.ceil(R[i] / h).astype(int)
		Si = 1+2*SiHlf
		i0,i1,i2 = np.meshgrid(np.arange(Si[0])-SiHlf[0], np.arange(Si[1])-SiHlf[1], np.arange(Si[2])-SiHlf[2], indexing='ij')
		rSq = np.sum((np.concatenate((i0[...,None], i1[...,None], i2[...,None]), axis=-1) * h[None,None,None,:])**2, axis=-1)
		iSphere = np.array(np.where(rSq < R[i]**2)) #list of indices within sphere, in bbox coordinates
		iSphere = np.mod(iSphere + (r0offset - SiHlf)[:,None], S[:,None]) #list of indices within sphere, in global coordinates
		mask[iSphere[0],iSphere[1],iSphere[2]] = 1
	#Calculate epsInv on edges:
	epsInv = np.zeros(x.shape) #extra dimension is for edge direction
	for dim in range(3):
		maskAv = 0.5*(mask + np.roll(mask, -1, axis=dim))
		epsInv[...,dim] = 1./epsOut + maskAv*(1./epsIn - 1./epsOut)
	print 'Average epsInv:', np.mean(epsInv)
	#Construct matrix for div(eps grad()):
	L_i = []; L_j = []; L_val = []
	iOffs = np.eye(3, dtype=int)[None,None,None,...]
	iCur = np.repeat(iMesh[...,None,:], 3, axis=-2)
	iPlus = np.reshape(iCur + iOffs, (-1,3))
	iCur = np.reshape(iCur, (-1,3))
	edgeVal = 1./(epsInv * h[None,None,None,:]**2).flatten()
	L_i.extend(iCur) ; L_j.extend(iCur) ; L_val.extend(edgeVal) #diagonal term at iCur
	L_i.extend(iPlus); L_j.extend(iPlus); L_val.extend(edgeVal) #diagonal term at iPlus
	L_i.extend(iCur) ; L_j.extend(iPlus); L_val.extend(-edgeVal) #off-diagonal term 1
	L_i.extend(iPlus); L_j.extend(iCur) ; L_val.extend(-edgeVal) #off-diagonal term 2
	L_i = np.array(L_i)
	L_j = np.array(L_j)
	L_val = np.array(L_val)
	#--- Fix wrap-around terms, constructing rhs term:
	stride = np.array([S[1]*S[2], S[2], 1], dtype=int)
	L_iFlat = np.dot(L_i % S[None,:], stride)
	L_jFlat = np.dot(L_j % S[None,:], stride)
	rhs = np.zeros((prodS,))
	iCheck = (L_i[:,2]==S[2])
	jCheck = (L_j[:,2]==S[2])
	iWrap = np.where(np.logical_and(iCheck, np.logical_not(jCheck)))[0]
	jWrap = np.where(np.logical_and(jCheck, np.logical_not(iCheck)))[0]
	rhs[L_iFlat[iWrap]] = -L_val[iWrap] * (-Ez*(-h[2])); L_val[iWrap] = 0.; #Dirichlet BC on z = -dz
	rhs[L_iFlat[jWrap]] = -L_val[jWrap] * (-Ez*L[2]);    L_val[jWrap] = 0.; #Dirichlet BC on z = +Lz
	#Construct preconditioner:
	iG = np.reshape(iMesh, (-1,3))
	iG = np.where(iG>S[None,:]/2, iG-S[None,:], iG)
	Gsq = np.sum((iG * (2*np.pi/L[None,:]))**2, axis=-1)
	Gsq[0] = np.min(Gsq[1:])
	invGsq = np.reshape(1./Gsq, S)
	def precondFunc(x):
		return np.real(np.fft.ifftn(invGsq * np.fft.fftn(np.reshape(x,S))).flatten())
	precondOp = sparse.linalg.LinearOperator((prodS,prodS), precondFunc)
	#Solve matrix equations:
	Lhs = sparse.csr_matrix((L_val, (L_iFlat, L_jFlat)), shape=(prodS, prodS))
	print 'Matrix dimensions:', Lhs.shape, 'with', Lhs.nnz, 'non-zero elements (fill', '%.2g%%)' % (Lhs.nnz*100./np.prod(Lhs.shape))
	global nIter
	nIter = 0
	def iterProgress(x):
		global nIter
		nIter += 1
		print 'CG iteration', nIter
	phi,info = sparse.linalg.cg(Lhs, rhs, callback=iterProgress, M=precondOp)
	phi = np.reshape(phi,S)
	#Optional plot:
	if shouldPlot:
		import matplotlib.pyplot as plt
		plt.figure(1)
		plotSlice = phi[0,:,:]
		print np.min(plotSlice), np.max(plotSlice)
		plt.imshow(plotSlice)
		plt.show()
	return phi, mask


#---------- Test code ----------
if __name__ == "__main__":
	
	L = np.array([10.,10.,20.])
	S = np.array([50, 50, 100])
	#r0 = np.random.rand(100,3) * L[None,:];	R = np.ones((r0.shape[0],)) * 0.5; r0[:,0] = 0.
	r0 = np.array([[0.,5.,10.]]); R = np.array([3.,]) #One sphere
	#r0 = np.zeros((0,3)); R = np.zeros((0,)) #No sphere
	epsIn = 1700.
	epsOut = 1.7
	
	#Calculate with FD:
	periodicFD(L, S, r0, R, epsIn, epsOut, 1., True)
