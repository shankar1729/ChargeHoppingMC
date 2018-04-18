#!/usr/bin/env python

import numpy as np
import pickle

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
		#--- add polymer internal DOS
		self.E0 = params["dosMu"] + params["dosSigma"]*np.random.randn(*self.S)
		#--- add electric field contributions
		z = self.h * np.arange(self.S[2])
		self.E0 -= params["Efield"] * z[None,None,:]
		#TODO add nanoparticles here / optipons for exponential instead etc.
		
		#Set up connectivity:
		#neighOffsets = np.array([[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1]]) #2D test case
		#neighOffsets = np.array([[+1,0,0],[-1,0,0],[0,+1,0],[0,-1,0],[0,0,+1],[0,0,-1]]) #3D
		neigh1d = np.arange(-1,2)
		neighOffsets = flattenedMesh([0], neigh1d, neigh1d) #2D test case
		#neighOffsets = flattenedMesh(neigh1d, neigh1d, neigh1d) #3D
		#--- remove self:
		neighOffsets = neighOffsets[np.where(np.sum(neighOffsets**2,axis=1))[0]]
		
		#Flattened list of grid points and neighbours:
		iPosMesh = flattenedMesh(np.arange(self.S[0]), np.arange(self.S[1]), np.arange(self.S[2]))
		iPosStride = np.array([self.S[1]*self.S[2], self.S[2], 1])
		iPosIndex = np.arange(iPosMesh.shape[0])
		self.E0 = self.E0.flatten()
		nGrid = np.prod(self.S)
		#--- neighbours
		jPosMesh = np.reshape(iPosMesh[:,None,:] + neighOffsets[None,...], (-1,3))
		for iDir in range(2): #wrap indices in periodic directions
			jPosMesh[:,iDir] = np.mod(jPosMesh[:,iDir], self.S[iDir])
		jPosIndex = np.dot(jPosMesh, iPosStride)
		iPosIndex = np.tile(iPosIndex[:,None], [1,neighOffsets.shape[0]]).flatten() #repeat iPos to same shape
		#--- select valid neighbours that are within z range:
		zjValid = np.where(np.logical_and(jPosMesh[:,2]>=0, jPosMesh[:,2]<self.S[2])) #z of neighbour in range
		iPosIndex = iPosIndex[zjValid]
		jPosIndex = jPosIndex[zjValid]
		#--- select edges such that E[i] < E[j]
		uphillEdges = np.where(self.E0[iPosIndex] < self.E0[jPosIndex])
		iPosIndex = iPosIndex[uphillEdges]
		jPosIndex = jPosIndex[uphillEdges]
		
		#Identify local minima and their local domains:
		minimaIndex = np.setdiff1d(np.arange(nGrid), np.unique(jPosIndex), assume_unique=True)
		nMinima = len(minimaIndex)
		#--- adjacency matrix:
		from scipy.sparse import csr_matrix, diags
		adjMat = csr_matrix((np.ones(len(iPosIndex),dtype=int), (jPosIndex,iPosIndex)), shape=(nGrid,nGrid))
		#--- initialize minima domain matrix:
		minMat = csr_matrix((np.ones(nMinima,dtype=int), (minimaIndex,np.arange(nMinima,dtype=int))), shape=(nGrid,nMinima))
		#--- multiply adjacency matrix repeatedly till all points covered:
		minMatCum = minMat
		while minMat.count_nonzero():
			minMat = adjMat * minMat
			minMatCum += minMat
			#Stop propagating points already connected to >=2 minima:
			minimaCount = minMatCum.sign().sum(axis=1) #number of minima already connected to each point
			pointWeight = np.where(minimaCount.flatten()>=2, 0, 1)
			minMat = diags([pointWeight.flatten()], [0]) * minMat #stop propagating these rows further
			
		import matplotlib.pyplot as plt
		for iMinima in range(nMinima):
			plt.figure(iMinima+1)
			plt.imshow(np.reshape(self.E0,self.S)[0], cmap='Greys_r')
			plt.plot(iPosMesh[minimaIndex,2], iPosMesh[minimaIndex,1], 'r+')
			iDomain = minMatCum.getcol(iMinima).nonzero()[0]
			plt.scatter(iPosMesh[iDomain,2], iPosMesh[iDomain,1])
			
		plt.show()
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
		"L": [ 1, 10, 10 ], #box size in nm
		"h": 1., #grid spacing in nm
		"Efield": 10*0.01, #electric field in V/nm
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
