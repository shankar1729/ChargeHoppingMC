#!/usr/bin/env python
from common import *
from minimaGraph import *

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
		self.iPosMinima, self.iPosBarrier, self.jMinima, Sbarrier, self.minimaStart, self.minimaStop = minimaGraph(E0, hopDistance/h, kT)
		posStride = np.array([S[1]*S[2], S[2], 1]) #indexing stride for cubic mesh
		E0 = E0.flatten()
		self.Abarrier0 = (
			E0[np.tensordot(self.iPosBarrier, posStride, axes=1)]
			- E0[np.dot(self.iPosMinima, posStride), None]
			- kT * Sbarrier )
		E0 = None #problem reduced to minima graph, no longer need mesh after this
		
		#Other parameters:
		self.h = h
		self.S = S
		self.L = L
		self.nElectrons = params["nElectrons"]
		self.maxHops = params["maxHops"]
		self.tMax = params["tMax"]
		self.beta = 1./kT
		self.hopFrequency = params["hopFrequency"]
		self.coulombPrefac = 1.44 / params["epsBG"] #prefactor to 1/r in energy [in eV] of two electrons separated by r [in nm]
		self.useCoulomb = params["useCoulomb"]
	
	"""
	Calculate the Coulomb interaction (with regularization for 0):
		iPos1: array of integer coordinates with dimensions [...,3]
		iPos2: array of integer coordinates with dimensions [...,3]
		iSelf: optional: if provided, zero out iSelf'th sub-array(s) along first dimension of output
		Returns coulomb interactions of dimensions iPos1.shape[:-1] + iPos2.shape[:-1]
	"""
	def getCoulomb(self, iPos1, iPos2, iSelf=None):
		dim1 = iPos1.shape[:-1] + (1,)*(len(iPos2.shape)-1) + (3,) #dimensions to broadcast iPos1 to
		dim2 = (1,)*(len(iPos1.shape)-1) + iPos2.shape[:-1] + (3,) #dimensions to broadcast iPos2 to
		dim = (1,)*(len(iPos1.shape)-1 + len(iPos2.shape)-1) + (3,) #dimensions to broadcast 3-vectors to
		xDelta = (np.reshape(iPos1,dim1) - np.reshape(iPos2, dim2)) * (1./np.reshape(self.S, dim)) #separation in fractional coordinates
		xDelta[...,:2] -= np.floor(0.5 + xDelta[...,:2]) #wrap to nearest periodic neighbours in first two directions
		rDelta = np.sqrt(np.sum((xDelta * np.reshape(self.L,dim))**2, axis=-1)) #distance from reference point
		coulomb = self.coulombPrefac / np.maximum(1e-6*self.h, rDelta) #regularize small r
		if iSelf is not None:
			coulomb[iSelf] = 0.
		return coulomb
	
	"""
	Run one complete MC simulation and return trajectory (jump times and positions for each electron)
	"""
	def run(self, iRun=0):
		
		np.random.seed()	#Generate a new seed for every run
		print 'Starting MC run', iRun
		#Inject electrons in randomly shozen z=0 connected minima:
		iMinima = self.minimaStart[np.random.permutation(len(self.minimaStart))[:self.nElectrons]]
		
		#Initialize trajectory: list of electron number, time of event and grid position
		t = 0 #time of latest hop
		nElectronOffset = iRun*self.nElectrons
		trajectory = [ (iElectron+nElectronOffset, t) + tuple(self.iPosMinima[iMinimum]) for iElectron,iMinimum in enumerate(iMinima) ]
		
		#Initial energy of each electron and its neighbourhood
		Abarrier0 = self.Abarrier0[iMinima] #dimensions: [nElectrons,nConnections]
		coulomb = np.zeros(Abarrier0.shape)
		if self.useCoulomb:
			print 'Calculating initial coulomb interactions in run', iRun
			for iElectron,iPos0 in enumerate(self.iPosMinima[iMinima]): #for each minimum position with an electron
				coulomb += ( self.getCoulomb(self.iPosBarrier[iMinima], iPos0, iElectron) #coulomb at barrier
					- self.getCoulomb(self.iPosMinima[iMinima], iPos0, iElectron)[:,None] ) #coulomb at minimum
		
		#Main MC loop:
		izMax = 0
		while len(trajectory) < self.maxHops:
			#Calculate hopping probabilities for each electron:
			#--- for each electron to each neighbour's barrier:
			hopRateSub = self.hopFrequency * np.exp(-self.beta*(Abarrier0 + coulomb))
			#---calculate total for each electron
			hopRate = np.sum(hopRateSub, axis=1) #sum over second axis = neighbors
			
			#Calculate time for next hop for each electron:
			hopTime = np.random.exponential(1./hopRate) + t
			
			#Implement the soonest hop:
			iHop = np.argmin(hopTime)
			t = hopTime[iHop]
			if t > self.tMax:
				t = self.tMax
				#Finalize trajectory:
				trajectory += [ (iElectron+nElectronOffset, t) + tuple(self.iPosMinima[iMinimum]) for iElectron,iMinimum in enumerate(iMinima) ]
				break
			#--- select neighbour to hop to:
			iNeighbor = np.searchsorted(
				np.cumsum(hopRateSub[iHop]), #cumulative probability distribution
				hopRate[iHop]*np.random.rand()) #a random number till the total probability
			#--- pre-move Coulomb calculation:
			iPosOld = self.iPosMinima[iMinima[iHop]]
			if self.useCoulomb:
				#--- subtract Coulomb energy landscape contributions at all other electrons due to old position of current electron:
				coulomb -= ( self.getCoulomb(self.iPosBarrier[iMinima], iPosOld, iHop) #coulomb at barrier
					- self.getCoulomb(self.iPosMinima[iMinima], iPosOld, iHop)[:,None] ) #coulomb at minimum
			#--- update electron position:
			jMinimaHop = self.jMinima[iMinima[iHop],iNeighbor]
			iMinima[iHop] = jMinimaHop
			iPosNew = self.iPosMinima[jMinimaHop]
			trajectory.append((iHop+nElectronOffset, t) + tuple(iPosNew))
			if iPosNew[2] > izMax:
				izMax = iPosNew[2]
				print "Run", iRun, "reached", izMax*self.h, "nm at t =", t, "s"
			if np.in1d(jMinimaHop, self.minimaStop):
				break #Terminate: an electron has reached end of box
			#--- update cached energies:
			Abarrier0[iHop] = self.Abarrier0[jMinimaHop]
			#--- post-move Coulomb calculation:
			if self.useCoulomb:
				#--- add Coulomb energy landscape contributions at all other electrons due to new position of current electron:
				coulomb += ( self.getCoulomb(self.iPosBarrier[iMinima], iPosNew, iHop) #coulomb at barrier
					- self.getCoulomb(self.iPosMinima[iMinima], iPosNew, iHop)[:,None] ) #coulomb at minimum
				#--- recalculate Coulomb energy landscape of current electron at new position:
				coulomb[iHop] = np.sum( self.getCoulomb(self.iPosMinima[iMinima], self.iPosBarrier[iMinima[iHop]], iHop) #coulomb at barrier
						- self.getCoulomb(self.iPosMinima[iMinima], iPosNew, iHop)[:,None], #coulomb at minimum
					 axis=0 )
		
		print 'End MC run', iRun, ' with trajectory length:', len(trajectory), 'events'
		return np.array(trajectory, dtype=np.dtype('i8,f8,i8,i8,i8'))

#----- Test code -----
if __name__ == "__main__":
	params = { 
		"L": [ 50, 50, 1000 ], #box size in nm
		"h": 1., #grid spacing in nm
		"Efield": 0.03, #electric field in V/nm
		"dosSigma": 0.1, #dos standard deviation in eV
		"dosMu": -0.3, #dos center in eV
		"T": 298., #temperature in Kelvin
		"hopDistance": 1., #average hop distance in nm
		"hopFrequency": 1e12, #attempt frequency in Hz
		"nElectrons": 16, #number of electrons to track
		"maxHops": 2e5, #maximum hops per MC runs
		"nRuns": 16, #number of MC runs
		"tMax": 1e3, #stop simulation at this time from start in seconds
		"epsBG": 2.5, #relative permittivity of polymer
		"useCoulomb": True, #whether to include e-e Coulomb interactions
		#--- Nano-particle parameters
		"epsNP": 10., #relative permittivity of nanoparticles
		"trapDepthNP": -1., #trap depth of nanoparticles in eV
		"radiusNP": 2.5, #radius of nanoparticles in nm
		"volFracNP": 0.004, #volume fraction of nanoparticles
		"nClusterMu": 30, #mean number of nanoparticles in each cluster (Poisson distribution)
		"clusterShape": "random" #one of "round", "random", "line" or "sheet"
	}
	mhmc = MinimaHoppingMC(params)
	#mhmc.run(); exit() #uncomment for serial debugging
	
	nRuns = params["nRuns"]
	trajectory = np.concatenate(parallelMap(mhmc.run, cpu_count(), range(nRuns))) #Run in parallel and merge trajectories
	np.savetxt("trajectory.dat", trajectory, fmt="%d %e %d %d %d", header="iElectron t[s] ix iy iz") #Save trajectories together
