#!/usr/bin/env python

import numpy as np
import pickle

#Profiling functions
shouldProfile = True
if shouldProfile:
	import time
	import resource
	tPrev = time.clock()
	def printDuration(label):
		global tPrev
		tCur = time.clock()
		mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		print 'Time('+label+'):', tCur-tPrev, 's, Mem:', mem/1024, 'MB'
		tPrev = tCur
else:
	def printDuration(label):
		pass

#Unit definitions:
kB = 0.000086174 #in eV/K

#Wrapper to meshgrid producing flattened results:
def flattenedMesh(i1, i2, i3):
	iiArr = np.meshgrid(i1, i2, i3, indexing='ij')
	return np.hstack([ii.flatten()[:,None] for ii in iiArr])

class MinimaHoppingMC:
	
	"""
	Initialize class given a dict 'params' containing all required parameters
	and any optional parameters as detailed below:
		TODO
	"""
	def __init__(self, params):
		
		#Initialize box and grid:
		h = params["h"] #grid spacing
		S = np.round(np.array(params["L"])/h).astype(int) #number of grid points
		L = h * S #box size rounded to integer number of grid points
		kT = kB * params["T"]
		hopDistance = params["hopDistance"]
		
		#Initial energy landscape (without inter-electron repulsions):
		def initEnergyLandscape():
			#--- add polymer internal DOS
			E0 = params["dosMu"] + params["dosSigma"]*np.random.randn(*S)
			#--- add electric field contributions
			z = h * np.arange(S[2])
			E0 -= params["Efield"] * z[None,None,:]
			#TODO add nanoparticles here / options for exponential instead etc.
			return E0
		E0 = initEnergyLandscape().flatten()
		printDuration('InitE0')
		
		#Set up connectivity:
		def initNeighbors():
			neigh1d = np.arange(-1,2)
			neighOffsets = flattenedMesh(neigh1d, neigh1d, neigh1d) #3D
			#--- select close-packed (FCC) mesh neighbours:
			neighOffsets = neighOffsets[np.where(np.mod(np.sum(neighOffsets,axis=1),2)==0)]
			#--- remove half neighbours (prefer +z), as it will be covered by symmetry below:
			neighStride = np.array([1,10,100])
			return neighOffsets[np.where(np.dot(neighOffsets,neighStride)>0)[0]]
		neighOffsets = initNeighbors()
		
		#Flattened list of grid points and neighbours:
		print 'Constructing neighbours and adjacency:'
		from scipy.sparse import csr_matrix, diags
		iPosMesh = flattenedMesh(np.arange(S[0]), np.arange(S[1]), np.arange(S[2]))
		#--- switch to close-packed mesh
		fccSel = np.where(np.mod(np.sum(iPosMesh,axis=1),2)==0)[0]
		nGrid = len(fccSel)
		iPosMesh = iPosMesh[fccSel]
		E0 = E0[fccSel]
		def initAdjacency():
			adjMat = csr_matrix((nGrid,nGrid),dtype=np.int)
			jPosStride = np.array([S[1]*S[2], S[2], 1]) #indexing stride for cubic mesh
			fccSelInv = np.zeros((np.prod(S),), dtype=int) #initialize inverse of fccSel ...
			fccSelInv[fccSel] = np.arange(nGrid, dtype=int) #... for use in indexing below
			for neighOffset in neighOffsets:
				jPosMesh = iPosMesh + neighOffset[None,:]
				#Wrap indices in periodic directions:
				for iDir in range(2):
					if neighOffset[iDir]<0: jPosMesh[np.where(jPosMesh[:,iDir]<0)[0],iDir] += S[iDir]
					if neighOffset[iDir]>0: jPosMesh[np.where(jPosMesh[:,iDir]>=S[iDir])[0],iDir] -= S[iDir]
				jPosIndexAll = np.dot(jPosMesh, jPosStride) #index into original cubic mesh (wrapped to fcc below)
				#Handle finite boundaries in z:
				if neighOffset[2]>0: #only need to handle +z due to choice of half neighbour set above
					zjValid = np.where(jPosMesh[:,2]<S[2])[0]
					ijPosIndex = np.vstack((zjValid, fccSelInv[jPosIndexAll[zjValid]]))
				else:
					ijPosIndex = np.vstack((np.arange(nGrid,dtype=int), fccSelInv[jPosIndexAll]))
				#Direct each edge so that E[i] > E[j]:
				swapSel = np.where(E0[ijPosIndex[0]] < E0[ijPosIndex[1]])
				ijPosIndex[:,swapSel] = ijPosIndex[::-1,swapSel]
				adjMat += csr_matrix((np.ones(ijPosIndex.shape[1],dtype=int), (ijPosIndex[0],ijPosIndex[1])), shape=(nGrid,nGrid))
			return adjMat
		adjMat = initAdjacency()
		printDuration('InitAdj')
		
		#Identify local minima and their local domains:
		minimaIndex = np.where(adjMat.sum(axis=1)==0)[0]
		nMinima = len(minimaIndex)
		printDuration('InitMinima')
		#--- initialize minima domain matrix:
		minMat = csr_matrix((np.ones(nMinima,dtype=int), (minimaIndex,np.arange(nMinima,dtype=int))), shape=(nGrid,nMinima))
		printDuration('InitMinMat')
		#--- multiply adjacency matrix repeatedly till all points covered:
		minMatCum = minMat
		pathLength = 0
		nPoints = nMinima
		while nPoints:
			minMat = adjMat * minMat
			minMatCum += minMat
			#Stop propagating points already connected to >=2 minima:
			minimaCount = np.array(minMatCum.sign().sum(axis=1), dtype=int).flatten() #number of minima already connected to each point
			pointWeight = np.where(minimaCount>=2, 0, 1)
			minMat = diags([pointWeight], [0]) * minMat #stop propagating these rows further
			pathLength += 1
			nPoints = minMat.count_nonzero()
			print '\tPath length:', pathLength, 'nPoints:', nPoints
		printDuration('InitDomains')
		
		#Find saddle points connecting pairs of minima:
		#--- Find minima whose domains contain z=0 (electrons will be injected here):
		zSel = np.where(iPosMesh[:,2]==0, 1, 0)
		minimaStart = np.where(minMatCum.T * zSel)[0]
		#--- Find minima whose domains contain z=zmax (electron trajectories will end here):
		zSel = np.where(iPosMesh[:,2]==S[2]-1, 1, 0)
		minimaStop = np.where(minMatCum.T * zSel)[0]
		#--- Find pairs of minima connected by each point:
		iNZ,iMinNZ = minMatCum.nonzero()
		sel = np.where(iNZ[:-1]==iNZ[1:])[0] #two adjacent entries having same iNZ
		minimaPairs = np.array([iMinNZ[sel], iMinNZ[sel+1]]) #list of connected iMinima1 and iMinima2 connected
		swapSel = np.where(E0[minimaIndex[minimaPairs[0]]] < E0[minimaIndex[minimaPairs[1]]])[0]
		minimaPairs[:,swapSel] = minimaPairs[::-1,swapSel] #in each pair of minima, put higher energy first
		pairIndex = nMinima*minimaPairs[0] + minimaPairs[1]
		iConnection = iNZ[sel] #corresponding connecting point between above pairs
		printDuration('InitConnections')
		#--- Sort pairs by minima index, energy of connecting point:
		sortIndex = np.lexsort([E0[iConnection], pairIndex])
		pairIndexSorted = np.concatenate([[-1],pairIndex[sortIndex]])
		iFirstUniq = sortIndex[np.where(pairIndexSorted[:-1]!=pairIndexSorted[1:])[0]]
		minimaPairs = minimaPairs[:,iFirstUniq]
		iConnection = iConnection[iFirstUniq] #now index of saddle point
		printDuration('InitBarriers')
		print 'nMinima:', nMinima, 'with nConnections:', len(iConnection)
		#--- Calculate barrier energies and "entropies" (used to account for nearer barriers reached more easily):
		def getBarrierES():
			barrierDisp = iPosMesh[iConnection] - iPosMesh[minimaIndex[minimaPairs[0]]] #displacement to barriers
			for iDir in range(2): #wrap periodic directions
				barrierDisp[np.where(barrierDisp[:,iDir]<-0.5*S[iDir])[0],iDir] += S[iDir]
				barrierDisp[np.where(barrierDisp[:,iDir]>+0.5*S[iDir])[0],iDir] -= S[iDir]
			Sbarrier = -np.log(np.sum(barrierDisp**2, axis=1)/hopDistance**2)
			Ebarrier = E0[iConnection] - E0[minimaIndex[minimaPairs[0]]]
			return Ebarrier, Sbarrier
		Ebarrier,Sbarrier = getBarrierES()
		#--- Prune minima with too low barriers (<~ kT):
		print 'Pruning low barriers:'
		connPrune = np.where(Ebarrier < -kT*Sbarrier)[0]
		connKeep = np.where(Ebarrier >= -kT*Sbarrier)[0]
		#----- Replace higher energy minima with lower energy one along each pruned connection:
		while len(np.intersect1d(minimaPairs[0,connPrune], minimaPairs[1,connPrune])):
			replaceMap = np.arange(nMinima, dtype=int)
			replaceMap[minimaPairs[0,connPrune]] = minimaPairs[1,connPrune]
			minimaPairs[1,connPrune] = replaceMap[minimaPairs[1,connPrune]]
		minKeep = np.setdiff1d(np.arange(nMinima), minimaPairs[0,connPrune])
		replaceMap = np.arange(nMinima, dtype=int)
		replaceMap[minKeep] = np.arange(len(minKeep),dtype=int) #now with renumbering to account for removed minima
		replaceMap[minimaPairs[0,connPrune]] = replaceMap[minimaPairs[1,connPrune]] #now all in range [0,len(minKeep))
		#----- Apply replacements:
		minimaIndex = minimaIndex[minKeep]
		nMinima = len(minKeep)
		minimaStart = replaceMap[minimaStart]
		minimaStop = replaceMap[minimaStop]
		minimaPairs = replaceMap[minimaPairs[:,connKeep]]
		iConnection = iConnection[connKeep]
		printDuration('InitPrune')
		print 'nMinima:', nMinima, 'with nConnections:', len(iConnection)
		
		"""
		#Debug code to plot x=0 yz-plane slice of energies and minima:
		import matplotlib.pyplot as plt
		E0mesh = np.zeros((S[1],S[2]))
		yzPlaneSel = np.where(iPosMesh[:,0]==0)[0]
		E0mesh[iPosMesh[yzPlaneSel,1],iPosMesh[yzPlaneSel,2]] = E0[yzPlaneSel]
		plt.imshow(E0mesh, cmap='Greys_r')
		yzPlaneSel = np.where(iPosMesh[minimaIndex,0]==0)[0]
		plt.plot(iPosMesh[minimaIndex[yzPlaneSel],2], iPosMesh[minimaIndex[yzPlaneSel],1], 'r+')
		yzPlaneEndSel = np.where(iPosMesh[minimaIndex[minimaStart],0]==0)[0]
		plt.plot(iPosMesh[minimaIndex[minimaStart[yzPlaneEndSel]],2], iPosMesh[minimaIndex[minimaStart[yzPlaneEndSel]],1], 'b+')
		yzPlaneEndSel = np.where(iPosMesh[minimaIndex[minimaStop],0]==0)[0]
		plt.plot(iPosMesh[minimaIndex[minimaStop[yzPlaneEndSel]],2], iPosMesh[minimaIndex[minimaStop[yzPlaneEndSel]],1], 'g+')
		plt.show(); exit()
		"""
		
		#Reduce problem purely to graph of minima connected through energy barriers:
		#--- add opposite direction connections between minima:
		minimaPairs = np.hstack((minimaPairs,minimaPairs[::-1]))
		iConnection = np.hstack((iConnection,iConnection))
		#--- sort connections by first minima index and energy barrier:
		Ebarrier,Sbarrier = getBarrierES()
		sortIndex = np.lexsort((Ebarrier, minimaPairs[0]))
		minimaPairs = minimaPairs[:,sortIndex]
		iConnection = iConnection[sortIndex]
		#--- start indices of each first minima (now contiguous) in above array
		mPadded = np.hstack(([-1],minimaPairs[0]))
		startIndex = np.where(mPadded[:-1]!=mPadded[1:])[0]
		#--- determine degrees of connectivity of each minima:
		mDegrees = np.hstack((startIndex[1:]-startIndex[:-1], len(iConnection)-startIndex[-1]))
		print 'Degree of connectivity: min:', np.min(mDegrees), 'max:', np.max(mDegrees), 'mean:', np.mean(mDegrees)
		print 'Energy barriers [eV]: min:', np.min(Ebarrier), 'max:', np.max(Ebarrier), 'mean:', np.mean(Ebarrier)
		exit()
		#--- find maximum degree of any 
		#--- find an unused point for dummy non-connections used below
		iEmax = np.argmax(E0) #max value will not be a minima or saddle point
		E0[iEmax] = np.Inf
		iConnection = np.hstack((iConnection,[iEmax]))
		#--- energies and positions of graph nodes
		self.E0m = E0[minimaIndex] #E0 at the location of the minima
		self.E0b = E0[iConnection] #E0 at the barrier point connecting each pair of minima
		self.iPosm = iPosMesh[minimaIndex] #locations of the minima
		self.iPosb = iPosMesh[iConnection] #locations of the barrier points connecting each pair of minima
		print self.E0m.shape, self.E0b.shape, self.iPosb.shape, self.iPosm.shape
		exit()
		#TODO
		
		#Other parameters:
		self.nElectrons = params["nElectrons"]
		self.tMax = params["tMax"]
		self.beta = 1./kT
		hopFrequency = params["hopFrequency"]
		self.coulombPrefac = 1.44 / params["epsBG"] #prefactor to 1/r in energy [in eV] of two electrons separated by r [in nm]
	
	
	"""
	Run one complete MC simulation and return trajectory (jump times and positions for each electron)
	"""
	def run(self):
		
		#TODO
			
		return trajectory
	

#----- Test code -----
if __name__ == "__main__":
	params = { 
		"L": [ 50, 50, 500 ], #box size in nm
		"h": 1., #grid spacing in nm
		"Efield": 0.01, #electric field in V/nm
		"dosSigma": 0.2, #dos standard deviation in eV
		"dosMu": -0.3, #dos center in eV
		"T": 298., #temperature in Kelvin
		"hopDistance": 1., #average hop distance in nm
		"hopFrequency": 1e12, #attempt frequency in Hz
		"nElectrons": 16, #number of electrons to track
		"tMax": 1e3, #stop simulation at this time from start in seconds
		"epsBG": 2.5, #relative permittivity of polymer
		#--- Nano-particle parameters
		"epsNP": 10., #relative permittivity of nanoparticles
		"trapDepthNP": -1., #trap depth of nanoparticles in eV
		"radiusNP": 2.5, #radius of nanoparticles in nm
		"volFracNP": 0.004, #volume fraction of nanoparticles
		"nClusterMu": 30, #mean number of nanoparticles in each cluster (Poisson distribution)
		"clusterShape": "random" #one of "round", "random", "line" or "sheet"
	}
	mhmc = MinimaHoppingMC(params)
	trajectory = mhmc.run()
	print len(trajectory)

	trajFile = open("trajectory.pkl","wb")
	pickle.dump(trajectory, trajFile)
	trajFile.close()
