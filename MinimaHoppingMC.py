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
		self.h = params["h"] #grid spacing
		self.S = np.round(np.array(params["L"])/self.h).astype(int) #number of grid points
		self.L = self.h * self.S #box size rounded to integer number of grid points
		
		#Initial energy landscape (without inter-electron repulsions):
		def initEnergyLandscape():
			#--- add polymer internal DOS
			E0 = params["dosMu"] + params["dosSigma"]*np.random.randn(*self.S)
			#--- add electric field contributions
			z = self.h * np.arange(self.S[2])
			E0 -= params["Efield"] * z[None,None,:]
			#TODO add nanoparticles here / options for exponential instead etc.
			return E0
		self.E0 = initEnergyLandscape().flatten()
		printDuration('InitE0')
		
		#Set up connectivity:
		def initNeighbors():
			neigh1d = np.arange(-1,2)
			#neighOffsets = flattenedMesh([0], neigh1d, neigh1d) #2D test case
			neighOffsets = flattenedMesh(neigh1d, neigh1d, neigh1d) #3D
			#--- remove half neighbours (prefer +z), as it will be covered by symmetry below:
			neighStride = np.array([1,10,100])
			return neighOffsets[np.where(np.dot(neighOffsets,neighStride)>0)[0]]
		neighOffsets = initNeighbors()
		
		#Flattened list of grid points and neighbours:
		print 'Constructing neighbours and adjacency:'
		iPosMesh = flattenedMesh(np.arange(self.S[0]), np.arange(self.S[1]), np.arange(self.S[2]))
		iPosStride = np.array([self.S[1]*self.S[2], self.S[2], 1])
		nGrid = np.prod(self.S)
		def initEdges():
			iPosIndex = []
			jPosIndex = []
			for neighOffset in neighOffsets:
				jPosMesh = iPosMesh + neighOffset[None,:]
				#Wrap indices in periodic directions:
				for iDir in range(2):
					if neighOffset[iDir]<0: jPosMesh[np.where(jPosMesh[:,iDir]<0)[0],iDir] += self.S[iDir]
					if neighOffset[iDir]>0: jPosMesh[np.where(jPosMesh[:,iDir]>=self.S[iDir])[0],iDir] -= self.S[iDir]
				jPosIndexAll = np.dot(jPosMesh, iPosStride)
				#Handle finite boundaries in z:
				if neighOffset[2]>0: #only need to handle +z due to choice of half neighbour set above
					zjValid = np.where(jPosMesh[:,2]<self.S[2])[0]
					ijPosIndex = np.vstack((zjValid, jPosIndexAll[zjValid]))
				else:
					ijPosIndex = np.vstack((np.arange(len(jPosIndexAll)), jPosIndexAll))
				#Order edges so that E[i]<E[j]:
				swapSel = np.where(self.E0[ijPosIndex[0]] > self.E0[ijPosIndex[1]])
				ijPosIndex[:,swapSel] = ijPosIndex[::-1,swapSel]
				iPosIndex.append(ijPosIndex[0])
				jPosIndex.append(ijPosIndex[1])
			iPosIndex = np.concatenate(iPosIndex)
			jPosIndex = np.concatenate(jPosIndex)
			return iPosIndex, jPosIndex
			
		iPosIndex,jPosIndex = initEdges()
		printDuration('InitEdges')
		#print len(iPosIndex), len(jPosIndex)
		#exit()
		
		#Identify local minima and their local domains:
		minimaIndex = np.setdiff1d(np.arange(nGrid), np.unique(jPosIndex), assume_unique=True)
		nMinima = len(minimaIndex)
		printDuration('InitMinima')
		#--- adjacency matrix:
		from scipy.sparse import csr_matrix, diags
		adjMat = csr_matrix((np.ones(len(iPosIndex),dtype=int), (jPosIndex,iPosIndex)), shape=(nGrid,nGrid))
		printDuration('InitAdj')
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
		#--- Find pairs of minima connected by each point:
		iNZ,iMinNZ = minMatCum.nonzero()
		sel = np.where(iNZ[:-1]==iNZ[1:])[0] #two adjacent entries having same iNZ
		minimaPairs = np.array([iMinNZ[sel], iMinNZ[sel+1]]) #list of connected iMinima1 and iMinima2 connected
		minimaPairs.sort(axis=0)
		pairIndex = nMinima*minimaPairs[0] + minimaPairs[1]
		iConnection = iNZ[sel] #corresponding connecting point between above pairs
		printDuration('InitConnections')
		#--- Sort pairs by minima index, energy of connecting point:
		sortIndex = np.lexsort([self.E0[iConnection], pairIndex])
		pairIndexSorted = np.concatenate([[-1],pairIndex[sortIndex]])
		iFirstUniq = sortIndex[np.where(pairIndexSorted[:-1]!=pairIndexSorted[1:])[0]]
		minimaPairs = minimaPairs[:,iFirstUniq]
		iConnection = iConnection[iFirstUniq] #now index of saddle point
		nConnections = len(iConnection)
		printDuration('InitBarriers')
		print 'nMinima:', nMinima, 'with nConnections:', nConnections
		
		"""
		import matplotlib.pyplot as plt
		plt.imshow(np.reshape(self.E0,self.S)[0], cmap='Greys_r')
		plt.plot(iPosMesh[minimaIndex,2], iPosMesh[minimaIndex,1], 'k+')
		plt.show()
		"""
		exit()
		#TODO
		
		#Other parameters:
		self.nElectrons = params["nElectrons"]
		self.tMax = params["tMax"]
		self.beta = 1./(kB * params["T"])
		self.hopFrequency = params["hopFrequency"]
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
