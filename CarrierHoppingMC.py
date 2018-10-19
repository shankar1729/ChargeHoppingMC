#!/usr/bin/env python
from common import *
from PeriodicFD import *
from NPClusters import *

class CarrierHoppingMC:
	
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

		#Initialize nano-particle parameters
		epsNP = params["epsNP"]
		epsBG = params["epsBG"]
		radiusNP = params["radiusNP"]
		volFracNP = params["volFracNP"]
		clusterShape = params["clusterShape"]
		nParticles = np.round(np.prod(S)*volFracNP/(4./3*np.pi*(radiusNP)**3)).astype(int) #total number of nano-particles
		print "Desired Number of nano-particles:", nParticles
		if nParticles:
			#Initialize nano-particle distribution in the polymer matrix
			print "Cluster Shape:", clusterShape
			shouldPlotNP = params["shouldPlotNP"]
			positionsNP = NPClusters(clusterShape, radiusNP, nParticles, params["nClusterMu"], params["nClusterSigma"], L)
			nParticles = positionsNP.shape[0]
			radiusNP = np.ones(nParticles)*radiusNP #array of NP radii
			print "Actual  Number of nano-particles:", nParticles
	
		#Initial energy landscape (without inter-electron repulsions):
		def initEnergyLandscape():
			#--- calculate polymer internal DOS
			Epoly = params["dosMu"] + params["dosSigma"]*np.random.randn(*S)
			#--- calculate electric field contributions and mask:
			Ez = params["Efield"]
			if nParticles:
				phi, mask = periodicFD(L, S, positionsNP, radiusNP, epsNP, epsBG, Ez, shouldPlotNP)
			else:
				z = h * np.arange(S[2])
				phi = -Ez * np.tile(z, (S[0],S[1],1))
				mask = np.zeros(phi.shape)
			#--- combine field and DOS contributions to energy landscape:
			return phi + np.where(mask, params["trapDepthNP"], Epoly)
		self.E0 = initEnergyLandscape()
		printDuration('InitE0')
		
		#Initialize neighbour list and distance factors:
		hopDistance = params["hopDistance"]
		drMax = 5.*hopDistance #truncate exponential distance function at exp(-5)
		self.irMax = int(np.ceil(drMax/h)) #max number of grid points electron can hop
		irGrid = np.arange(-self.irMax,+self.irMax+1) #list of relative neighbour indices in 1D
		irMesh = np.meshgrid(irGrid, irGrid, irGrid, indexing='ij') #3D mesh of neighbour indices
		self.ir = np.vstack([ irMesh_i.flatten()[None,:] for irMesh_i in irMesh ]) #3 x N array of neighbour indices
		#--- calculate distances and select neighbours within drMax:
		dr = np.sqrt(np.sum((self.ir*h)**2, axis=0))
		irSel = np.where(np.logical_and(dr<=drMax, dr>0))[0] #select indices within a sphere of radius drMax
		self.ir = self.ir[:,irSel]
		self.wr = np.exp(-dr[irSel]/hopDistance) #exponential probability factor due to length of hop
		
		#Pad energy landscape to accomodate neighbour list without branches:
		#--- pad along x by periodically repeating:
		self.E0 = np.concatenate((self.E0[-self.irMax:,:,:], self.E0, self.E0[:self.irMax,:,:]), axis=0)
		#--- pad along y by periodically repeating:
		self.E0 = np.concatenate((self.E0[:,-self.irMax:,:], self.E0, self.E0[:,:self.irMax,:]), axis=1)
		#--- pad along z by setting inaccessible energy (+infinity):
		EzPad = np.full((self.E0.shape[0], self.E0.shape[1], self.irMax), np.inf)
		self.E0 = np.concatenate((EzPad, self.E0, EzPad), axis=2)
		
		#Other parameters:
		self.h = h
		self.S = S
		self.L = L
		self.nElectrons = params["nElectrons"]
		self.maxHops = params["maxHops"]
		self.tMax = params["tMax"]
		self.beta = 1./(kB * params["T"])
		self.hopFrequency = params["hopFrequency"]
		self.coulombPrefac = 1.44 / params["epsBG"] #prefactor to 1/r in energy [in eV] of two electrons separated by r [in nm]
		self.useCoulomb = params["useCoulomb"] #prefactor to 1/r in energy [in eV] of two electrons separated by r [in nm]
	
	#Calculate the Coulomb interaction with regularization for 0, given vectors r (dimensions [3,...])
	def safeCoulomb(self, r):
		return self.coulombPrefac / np.maximum(1e-6*self.h, np.sqrt(np.sum(r**2, axis=0)))
	
	#**TODO** use GetCoulomb like that in MinimaHopping
	
	#Calculate coulomb landscape of neighbours given grid separation iPosDelta of center locations
	#Avoid interaction with iCur: the current electron for which interactions are being calculated
	def coulombLandscape(self, iPosDelta, iCur):
		xDelta = (iPosDelta) * (1./self.S[:,None]) #separation in fractional coordinates for pairs with iHop'th electron
		xDelta[:2] -= np.floor(0.5 + xDelta[:2]) #wrap to nearest periodic neighbours in first two directions
		rDelta = xDelta * self.L[:,None] #displacement in nearest-periodic image convention, dimensions: [3, nElectrons, nElectrons]
		rDelta[:,iCur] = np.Inf #Avoid interacting with self
		rDeltaNeighbor = rDelta[...,None] + self.h*self.ir[:,None,:] #dimensions: [3, nElectrons, nNeighbours]
		return ( self.safeCoulomb(rDeltaNeighbor)  #+ 1/r to the neighbour
			- self.safeCoulomb(rDelta)[:,None] ) #- 1/r to the center location
	
	"""
	Run one complete MC simulation and return trajectory (jump times and positions for each electron)
	"""
	def run(self, iRun=0):
		
		np.random.seed()	#Generate a new seed for every run
		print 'Starting MC run', iRun
		#Initialize electron positions (in grid coordinates)
		iPosElectron = np.vstack((
			np.random.randint(self.S[0], size=(1,self.nElectrons)),
			np.random.randint(self.S[1], size=(1,self.nElectrons)),
			np.zeros((1,self.nElectrons), dtype=int) #start all on xy plane at z=0
		))
		
		#Initialize trajectory: list of electron number, time of event and grid position
		t = 0 #time of latest hop
		# ---- nElectronOffset from MINIMAHOPPING -----
		# -----------(ADDED TO TRAJECTORY) ------------
		nElectronOffset = iRun*self.nElectrons
		trajectory = [ (iElectron+nElectronOffset, t) + tuple(iPos) for iElectron,iPos in enumerate(iPosElectron.T) ]
		print iElectron,iPos,iElectron,nElectronOffset,"\n"	
		#Initial energy of each electron and its neighbourhood
		#--- Fetch energy at each electron:
		E0electron = self.E0[
			self.irMax+iPosElectron[0],
			self.irMax+iPosElectron[1],
			self.irMax+iPosElectron[2]] #dimensions: [nElectrons]
		#--- Fetch energy landscape neighbourhoods for each electron:
		iPosNeighbor = iPosElectron[...,None] + self.ir[:,None,:]
		E0neighbor = self.E0[
			self.irMax+iPosNeighbor[0],
			self.irMax+iPosNeighbor[1],
			self.irMax+iPosNeighbor[2]] #dimensions: [nElectrons, nNeighbours]
		if self.useCoulomb:
			# TODO call getCoulomb
			#--- Coulomb contributions to energy difference for each electron at initial positions:
			xDelta = (iPosElectron[:,None,:] - iPosElectron[...,None]) * (1./self.S[:,None,None]) #separation in fractional coordinates for all pairs
			xDelta[:2] -= np.floor(0.5 + xDelta[:2]) #wrap to nearest periodic neighbours in first two directions
			rDelta = xDelta * self.L[:,None,None] #displacement in nearest-periodic image convention, dimensions: [3, nElectrons, nElectrons]
			rDelta[:,range(self.nElectrons),range(self.nElectrons)] = np.Inf #Avoid interacting with self
			rDeltaNeighbor = rDelta[...,None] + self.h*self.ir[:,None,None,:] #dimensions: [3, nElectrons, nElectrons, nNeighbours]
			coulomb = ( np.sum(self.safeCoulomb(rDeltaNeighbor), axis=0)  #+ 1/r to the neighbour
				- np.sum(self.safeCoulomb(rDelta), axis=0)[:,None] ) #- 1/r to the center location

		#Main MC loop:
		izMax = 0
		#------- Added limit to MC simulation to given maxHops -----
		while len(trajectory) < self.maxHops:
			#Calculate hopping probabilities for each electron:
			#--- for each electron to each neighbour:
			hopRateSub = (self.hopFrequency
				* self.wr[None,:] #precalculated distance factor
				* np.exp(-self.beta*np.maximum(0., E0neighbor - E0electron[:,None] + coulomb)) #exp(-DeltaE/kT) if DeltaE > 0; else 1
			)
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
				trajectory += [ (iElectron+nElectronOffset, t) + tuple(iPos) for iElectron,iPos in enumerate(iPosElectron.T) ]
				break
			#--- select neighbour to hop to:
			iNeighbor = np.searchsorted(
				np.cumsum(hopRateSub[iHop]), #cumulative probability distribution
				hopRate[iHop]*np.random.rand()) #a random number till the total probability
			# TODO pre-move Coulomb calculation
			#--- update electron position:
			iPosOld = iPosElectron[:,iHop]
			iPosNew = np.mod(iPosOld + self.ir[:,iNeighbor], self.S) #Wrap with periodic boundaries			
			iPosElectron[:,iHop] = iPosNew
			trajectory.append((iHop+nElectronOffset, t) + tuple(iPosNew))
			if iPosNew[2] > izMax:
				izMax = iPosNew[2]
				print "Run", iRun, "reached", izMax/self.h, "nm at t =", t, "s"
				if izMax >= self.S[2] - self.irMax:
					break #Terminate: an electron has reached end of box
			#--- update cached energies:
			iPosNeighborNew = iPosNew[:,None] + self.ir
			E0electron[iHop] = self.E0[
				self.irMax+iPosNew[0],
				self.irMax+iPosNew[1],
				self.irMax+iPosNew[2]]
			E0neighbor[iHop] = self.E0[
				self.irMax+iPosNeighborNew[0],
				self.irMax+iPosNeighborNew[1],
				self.irMax+iPosNeighborNew[2]]
			if self.useCoulomb:
				#--- update Coulomb energies of current electron:
				coulomb[iHop] = np.sum(self.coulombLandscape(iPosNew[:,None] - iPosElectron, iHop), axis=0)
				#--- update Coulomb energies of all other electrons:
				coulomb -= self.coulombLandscape(iPosElectron - iPosOld[:,None], iHop) #remove contribution from old position of iHop'th electron
				coulomb += self.coulombLandscape(iPosElectron - iPosNew[:,None], iHop) #remove contribution from old position of iHop'th electron
		print 'End MC run', iRun, 'with trajectory length:', len(trajectory), 'events'	
		return np.array(trajectory, dtype=np.dtype('i8,f8,i8,i8,i8'))
	
#----- Test code -----
if __name__ == "__main__":
	params = { 
		"L": [ 50, 50, 1e3 ], #box size in nm
		"h": 1., #grid spacing in nm
		"Efield": 0.06, #electric field in V/nm
		"dosSigma": 0.224, #dos standard deviation in eV
		"dosMu": 0.0, #dos center in eV
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
		"trapDepthNP": -1.1, #trap depth of nanoparticles in eV
		"radiusNP": 2.5, #radius of nanoparticles in nm
		"volFracNP": 0.00, #volume fraction of nanoparticles
		"nClusterMu": 30, #mean number of nanoparticles in each cluster (Gaussian distribution)
		"nClusterSigma": 5, #cluster size standard deviation in nm
		"clusterShape": "line", #one of "round", "random", "line" or "sheet"
		"shouldPlotNP": False #plot the electrostatic potential from PeriodicFD
	}
	chmc = CarrierHoppingMC(params)
	#trajectory = chmc.run()
	nRuns = params["nRuns"]
	# -------- Now runs parallel from MINIMAHOPPING -----------
	trajectory = np.concatenate(parallelMap(chmc.run, cpu_count(), range(nRuns))) #Run in parallel and merge trajectories
	np.savetxt("trajectory.dat", trajectory, fmt="%d %e %d %d %d", header="iElectron t[s] ix iy iz") #Save trajectories together
