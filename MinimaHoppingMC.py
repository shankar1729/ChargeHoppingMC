#!/usr/bin/env python
from common import *
from minimaGraph import *
import pickle

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
		E0 = initEnergyLandscape()
		printDuration('InitE0')
		
		#Calculate graph of minima and connectivity based on this landscape:
		self.iPosMinima, self.iPosBarrier, self.jMinima, self.Sbarrier = minimaGraph(E0, hopDistance/h, kT)
		posStride = np.array([S[1]*S[2], S[2], 1]) #indexing stride for cubic mesh
		E0 = E0.flatten()
		self.E0minima = E0[np.dot(self.iPosMinima, posStride)]
		self.E0barrier = E0[np.tensordot(self.iPosBarrier, posStride, axes=1)]
		E0 = None #problem reduced to minima graph, no longer need mesh after this
		
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
		trajectory = []
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
