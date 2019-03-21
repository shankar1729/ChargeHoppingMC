#!/usr/bin/env python
from common import *
from minimaGraph import *
from PeriodicFD import *
from NPClusters import *
import gc

def minimaProbModel(params):
	#Initialize box and grid:
	h = params["h"] #grid spacing
	S = np.round(np.array(params["L"])/h).astype(int) #number of grid points
	L = h * S #box size rounded to integer number of grid points
	kT = kB * params["T"]
	hopDistance = params["hopDistance"]

	#Initialize nano-particle parameters
	radiusNP = params["radiusNP"]
	volFracNP = params["volFracNP"]
	clusterShape = params["clusterShape"]
	nParticles = np.round(np.prod(S)*volFracNP/(4./3*np.pi*(radiusNP)**3)).astype(int) #total number of nano-particles
	print("Desired Number of nano-particles:", nParticles)
	if nParticles:
		print("Cluster Shape:", clusterShape)
		positionsNP, radiusArr = NPClusters(clusterShape, radiusNP, nParticles, params["nClusterMu"], params["nClusterSigma"], L)
		nParticles = positionsNP.shape[0]
		print("Actual  Number of nano-particles:", nParticles)

	#Initial energy landscape (without inter-electron repulsions):
	gc.enable()
	def initEnergyLandscape():
		#--- calculate polymer internal DOS
		Epoly = params["dosMu"] + params["dosSigma"]*np.random.randn(*S)
		#--- calculate mask:
		if nParticles:
			_, mask = periodicFD(L, S, positionsNP, radiusArr, 1., 1., 0.)
		else:
			mask = np.zeros(S)
		#--- combine particle and polymer DOS contributions to energy landscape:
		return np.where(mask, params["trapDepthNP"], Epoly)
	E0 = initEnergyLandscape()
	gc.collect()
	printDuration('InitE0')
	
	#Calculate graph of minima and connectivity based on this landscape:
	Eminima, Econn, iMinima, jMinima, disp = minimaGraph(E0, hopDistance/h, kT, zPeriodic=True)
	#--- shift energy reference to lowest energy point to avoid overflow issues
	Emin = np.min(Eminima)
	Eminima -= Emin
	Econn -= Emin
	
	#Other parameters:
	beta = 1./kT
	hopFrequency = params["hopFrequency"]

	#Calculate diffusivity:
	Z = np.sum(np.exp(-beta*Eminima)) #partition function
	D = (hopFrequency/(3.*Z)) * np.sum(np.exp(-beta*Econn) * np.sum(disp**2, axis=1)) #in nm^2/s units
	mu = beta*D #in nm^2/V.s units (since beta is in 1/eV already)
	print('Diffusion constant:', D*1e-18, 'm^2/s')
	print('Mobility:', mu*1e-18, 'm^2/Vs')

#----- Test code -----
if __name__ == "__main__":
	params = { 
		"L": [ 100., 100., 100. ], #box size in nm
		"h": 1., #grid spacing in nm
		"dosSigma": 0.224, #dos standard deviation in eV
		"dosMu": 0.0, #dos center in eV
		"T": 298., #temperature in Kelvin
		"hopDistance": 1., #average hop distance in nm
		"hopFrequency": 1e12, #attempt frequency in Hz
		#--- Nano-particle parameters
		"trapDepthNP": -1.1, #trap depth of nanoparticles in eV
		"radiusNP": 2.5, #radius of nanoparticles in nm
		"volFracNP": 0.02, #volume fraction of nanoparticles
		"nClusterMu": 30, #mean number of nanoparticles in each cluster (Gaussian distribution)
		"nClusterSigma": 5, #cluster size standard deviation in nm
		"clusterShape": "random", #one of "round", "random", "line" or "sheet"
		"shouldPlotNP": False #plot the electrostatic potential from PeriodicFD
	}
	minimaProbModel(params)
