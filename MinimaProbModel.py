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
	epsNP = params["epsNP"]
	epsBG = params["epsBG"]
	radiusNP = params["radiusNP"]
	volFracNP = params["volFracNP"]
	clusterShape = params["clusterShape"]
	nParticles = np.round(np.prod(S)*volFracNP/(4./3*np.pi*(radiusNP)**3)).astype(int) #total number of nano-particles
	print("Desired Number of nano-particles:", nParticles)
	if nParticles:
		print("Cluster Shape:", clusterShape)
		shouldPlotNP = params["shouldPlotNP"]
		positionsNP, radiusArr = NPClusters(clusterShape, radiusNP, nParticles, params["nClusterMu"], params["nClusterSigma"], L)
		nParticles = positionsNP.shape[0]
		print("Actual  Number of nano-particles:", nParticles)

	#Initial energy landscape (without inter-electron repulsions):
	gc.enable()
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
	gc.collect()
	printDuration('InitE0')
	
	#Calculate graph of minima and connectivity based on this landscape:
	iPosMinima, iPosBarrier, jMinima, Abarrier, _, _, jDisp = minimaGraph(E0, hopDistance/h, kT, zPeriodic=True, EzLz=params["Efield"]*L[2])
	
	#Other parameters:
	beta = 1./kT
	hopFrequency = params["hopFrequency"]

	#Reconstruct sparse matrix of hopping rates:
	from scipy.sparse import csr_matrix, csc_matrix
	nMinima = jMinima.shape[0]
	iMinima = np.tile(np.arange(nMinima, dtype=int)[:,None], (1, jMinima.shape[1]))
	hopRate = hopFrequency * np.exp(-beta*Abarrier)
	sel = np.where(iMinima != jMinima)
	iMinima = iMinima[sel]
	jMinima = jMinima[sel]
	hopRate = hopRate[sel]
	jDisp = jDisp[sel]
	#--- get diag terms by doing a column sum:
	hopRateDiag = np.array(csc_matrix((hopRate, (iMinima,jMinima)), shape=(nMinima,nMinima)).sum(axis=0)).flatten()
	iDiag = np.arange(nMinima, dtype=int)
	hopRate = np.concatenate((hopRate, -hopRateDiag))
	iMinima = np.concatenate((iMinima, iDiag))
	jMinima = np.concatenate((jMinima, iDiag))
	hopRateMat = csr_matrix((hopRate, (iMinima,jMinima)), shape=(nMinima,nMinima))
	
	#Solve steady state equation:
	from scipy.sparse.linalg import spsolve
	import pyamg
	#--- replace last row with probability normalization
	sel = np.where(iMinima < nMinima-1)
	hTyp = np.mean(hopRate) #to scale the last row to be on the same scale
	hopRate = np.concatenate((hopRate[sel], np.full(nMinima, hTyp)))
	iMinima = np.concatenate((iMinima[sel], np.full(nMinima, nMinima-1, dtype=int)))
	jMinima = np.concatenate((jMinima[sel], np.arange(nMinima, dtype=int)))
	Lhs = csr_matrix((hopRate, (iMinima,jMinima)), shape=(nMinima,nMinima))
	rhs = np.zeros(nMinima); rhs[-1] = hTyp #RHS for probability normalization
	printDuration('InitMatrices')
	#p = spsolve(Lhs, rhs, use_umfpack=True)
	ml = pyamg.ruge_stuben_solver(Lhs)
	print(ml)
	p = ml.solve(rhs, tol=1e-10)
	printDuration('SolveSteadyState')
	print("residual: ", np.linalg.norm(rhs - Lhs*p)) 
	print(np.sum(p))

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
		"epsBG": 2.5, #relative permittivity of polymer
		#--- Nano-particle parameters
		"epsNP": 80., #relative permittivity of nanoparticles
		"trapDepthNP": -1.1, #trap depth of nanoparticles in eV
		"radiusNP": 2.5, #radius of nanoparticles in nm
		"volFracNP": 0.0, #volume fraction of nanoparticles
		"nClusterMu": 30, #mean number of nanoparticles in each cluster (Gaussian distribution)
		"nClusterSigma": 5, #cluster size standard deviation in nm
		"clusterShape": "random", #one of "round", "random", "line" or "sheet"
		"shouldPlotNP": False #plot the electrostatic potential from PeriodicFD
	}
	minimaProbModel(params)
