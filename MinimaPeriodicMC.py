#!/usr/bin/env python
from common import *
from minimaGraph import *
from PeriodicFD import *
from NPClusters import *

class MinimaPeriodicMC:
	
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

		#Initialize nano-particle parameters
		epsNP = params["epsNP"]
		epsBG = params["epsBG"]
		radiusNP = params["radiusNP"]
		volFracNP = params["volFracNP"]
		clusterShape = params["clusterShape"]
		nParticles = np.round(np.prod(S)*volFracNP/(4./3*np.pi*(radiusNP)**3)).astype(int) #total number of nano-particles
		print "Desired Number of nano-particles:", nParticles
		if nParticles:
			print "Cluster Shape:", clusterShape
			shouldPlotNP = params["shouldPlotNP"]
			positionsNP, radiusArr = NPClusters(clusterShape, radiusNP, nParticles, params["nClusterMu"], params["nClusterSigma"], L)
			nParticles = positionsNP.shape[0]
			print "Actual  Number of nano-particles:", nParticles

		#Initial energy landscape (without inter-electron repulsions):
		def initEnergyLandscape():
			#--- calculate polymer internal DOS
			Epoly = params["dosMu"] + params["dosSigma"]*np.random.randn(*S)
			#--- calculate electric field contributions and mask:
			Ez = params["Efield"]
			if nParticles:
				phi, mask = periodicFD(L, S, positionsNP, radiusArr, epsNP, epsBG, Ez, shouldPlotNP, dirichletBC=False)
			else:
				z = h * np.arange(S[2])
				phi = -Ez * np.tile(z, (S[0],S[1],1))
				phi -= np.mean(phi)
				mask = np.zeros(phi.shape)
			#--- combine field and DOS contributions to energy landscape:
			return phi + np.where(mask, params["trapDepthNP"], Epoly)
		E0 = initEnergyLandscape()
		printDuration('InitE0')
		
		#Calculate graph of minima and connectivity based on this landscape:
		self.iPosMinima, self.iPosBarrier, self.jMinima, Sbarrier, _, _, self.jDisp = minimaGraph(E0, hopDistance/h, kT)
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
		self.nHops = params["nHops"]
		self.beta = 1./kT
		self.hopFrequency = params["hopFrequency"]

	"""
	Run one complete MC simulation and return displacements (per electron and hop number)
	"""
	def run(self, iRun=0):
		
		np.random.seed()	#Generate a new seed for every run
		print 'Starting MC run', iRun
		#Start electrons in random positions:
		iMinima = np.random.permutation(self.iPosMinima.shape[0])[:self.nElectrons]
		
		#Initialize trajectory: list of electron number, time of event and grid position
		disp = np.zeros((self.nElectrons, self.nHops, 3)) #displacement per electron per time step
		dt = np.zeros((self.nElectrons, self.nHops))   #change in time per electron per hop (INDEPENDENT TRAJECTORIES)
		
		#Main MC loop:
		izMax = 0
		for iHop in range(self.nHops):
			#Calculate mean hopping times for each electron to each neighbour's barrier:
			hopTau = (1./self.hopFrequency) * np.exp(self.beta * self.Abarrier0[iMinima])
			#Calculate corresponding time to next hop (for each electron to each neighbour):
			hopTime = np.random.exponential(hopTau)
			#Implement the soonest hop for each electron:
			dt[:,iHop] = np.min(hopTime, axis=1)
			iNeighbor = np.argmin(hopTime, axis=1)
			iMinima = self.jMinima[iMinima,iNeighbor]
			disp[:,iHop,:] = self.jDisp[iMinima,iNeighbor] * self.h #to real units
			
		print 'End MC run', iRun
		return np.concatenate((dt[...,None], disp), axis=2)

#----- Test code -----
if __name__ == "__main__":
	params = { 
		"L": [ 100., 100., 100. ], #box size in nm
		"h": 1., #grid spacing in nm
		"Efield": 0.0, #electric field in V/nm
		"dosSigma": 0.224, #dos standard deviation in eV
		"dosMu": 0.0, #dos center in eV
		"T": 298., #temperature in Kelvin
		"hopDistance": 1., #average hop distance in nm
		"hopFrequency": 1e12, #attempt frequency in Hz
		"nHops": 10000, #number of hops per electron
		"nElectrons": 256, #number of electrons per run (vectorized)
		"nRuns": 16, #number of MC runs (parallelized)
		"epsBG": 2.5, #relative permittivity of polymer
		#--- Nano-particle parameters
		"epsNP": 80., #relative permittivity of nanoparticles
		"trapDepthNP": -1.1, #trap depth of nanoparticles in eV
		"radiusNP": 2.5, #radius of nanoparticles in nm
		"volFracNP": 0., #volume fraction of nanoparticles
		"nClusterMu": 30, #mean number of nanoparticles in each cluster (Gaussian distribution)
		"nClusterSigma": 5, #cluster size standard deviation in nm
		"clusterShape": "random", #one of "round", "random", "line" or "sheet"
		"shouldPlotNP": True #plot the electrostatic potential from PeriodicFD
	}
	mpmc = MinimaPeriodicMC(params)
	#mpmc.run(); exit() #uncomment for serial debugging
	
	nRuns = params["nRuns"]
	trajectory = np.concatenate(parallelMap(mpmc.run, cpu_count(), range(nRuns)), axis=0) #Run in parallel and merge trajectories
	
	#Analyze trajectory:
	dt = trajectory[...,0].flatten()
	disp = trajectory[...,1:].reshape((-1,3))
	vd = np.mean(disp/dt[:,None]) #drift velocity (nm/s)
	D = np.mean(np.sum(disp**2, axis=1)/dt) #diffusion coefficient (nm^2/s)
	print 'Drift velocity:', vd*1e-9, 'm/s'
	print 'Diffusion const:', D*1e-18, 'm^2/s'
	
	#Right now trajectory not really useful sicne original position not included
	#np.savetxt("trajectory.dat", trajectory.reshape((-1,4)), fmt="%e %d %d %d", header="dt dx dy dz") #Save trajectories together
