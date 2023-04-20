#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.special import erfc
from CarrierHoppingMC import CarrierHoppingMC
from PeriodicFD import PeriodicFD
from common import *
import sys

print("This old code no longer works. Switch to ellipsoidInterface.")
exit()


if len(sys.argv)>1:
    suffix = sys.argv[1]
    if suffix != 'SwapXZ':
        print('If specified, argument must be "SwapXZ" and x and z axis will be swapped.')
        exit(1)
    swapXZ = True
else:
    suffix = ''
    swapXZ = False

#Extract information from matlab file:
matFile = loadmat('structure.mat')
aspectRatio = float(matFile['Aspect_Ratio'])
centers = np.array(matFile['Center_list']) - 1. #Octave to python convention
axisDir = np.array(matFile['Orientation_list'])
volFrac = float(matFile['VF'])
L = np.array(matFile['Side_length'], dtype=float).flatten()
hTarget = 1. #grid spacing
a = float(matFile['semi_major_axis_length'])
b = a/aspectRatio
axisDir = axisDir * (1./np.linalg.norm(axisDir,axis=1)[:,None]) #normalize
matFile = None

#Apply x <-> z swap if required:
if swapXZ:
    centers = centers[:,::-1]
    axisDir = axisDir[:,::-1]

#Construct mask (1 inside particles, 0 outside):
def constructMask():
    S = np.round(L/hTarget).astype(int)
    grids1D = [ np.arange(Si)*(L[i]/Si) for i,Si in enumerate(S) ]
    mask = np.zeros(S)
    #Split into loop over first direction to save memory:
    print('Creating mask in', S[0], 'slices:', end=' ', flush=True)
    for i0 in range(S[0]):
        dr = ( np.array(np.meshgrid(grids1D[0][i0:i0+1], grids1D[1], grids1D[2], indexing='ij'))[None,...]
            - centers[...,None,None,None] ) #vectors from each center to each grid point
        Lbcast = L[None,:,None,None,None]
        dr -= np.floor(0.5+dr/Lbcast)*Lbcast #wrap displacements by minimum image convention
        dist = np.maximum(np.sqrt(np.sum(dr**2, axis=1)), 1e-6) #corresponding length (regularized to avoide division by zero below) 
        cosTheta = np.sum(dr * axisDir[...,None,None,None], axis=1) / dist #corresponding cosTheta's to major axis direction
        dist -= b/np.sqrt(1 + ((b/a)**2 - 1)*(cosTheta**2)) #distance outside surface of ellipsoid
        mask[i0:i0+1] = 0.5*erfc(dist.min(axis=0)) #1-voxel smoothing
        print(i0+1, end=' ', flush=True)
    print('done.\n')
    return mask
mask = constructMask()
print('Volume fraction: expected:', volFrac, 'actual:', mask.mean())
printDuration('MaskDone')

#Initialize carrier hopping parameters:
params = { 
    "L": np.array(L), #box size in nm
    "h": hTarget, #grid spacing in nm
    "Efield": 0.06, #electric field in V/nm
    "dosSigma": 0.224, #dos standard deviation in eV
    "dosMu": 0.0, #dos center in eV
    "T": 298., #temperature in Kelvin
    "hopDistance": 1., #average hop distance in nm
    "hopFrequency": 1e12, #attempt frequency in Hz
    "nElectrons": 16, #number of electrons to track
    "maxHops": 1e5, #maximum hops per MC runs
    "tMax": 1e3, #stop simulation at this time from start in seconds
    "epsBG": 2.3, #relative permittivity of polymer (set to PP)
    "useCoulomb": True, #whether to include e-e Coulomb interactions
    #--- Nano-particle parameters
    "mask": mask, #1 where particles are, 0 outside
    "epsNP": 41., #relative permittivity of nanoparticles (set to TiO2)
    "trapDepthNP": -1.1, #trap depth of nanoparticles in eV
    "shouldPlotNP": False, #plot the electrostatic potential from PeriodicFD
}

#Run hopping simulation:
nRuns = 32
np.random.seed(0)
model = CarrierHoppingMC(params)
trajectory = np.concatenate(parallelMap(model.run, cpu_count(), range(nRuns))) #Run in parallel and merge trajectories
np.savetxt("trajectory"+suffix+".dat.gz", trajectory, fmt="%d %e %d %d %d", header="iElectron t[s] ix iy iz") #Save trajectories together
printDuration('HoppingDone')

#Extract mobility:
print(trajectory.shape)
nElectrons = 1 + int(np.max(trajectory['f0']))
v = np.zeros(nElectrons)
print('Analyzing trajectory with ', nElectrons, 'electrons: ', end='', flush=True)
for i in range(nElectrons):
    finalEntry = trajectory[np.where(trajectory['f0']==i)[0][-1]] #final state of current electron
    v[i] = finalEntry[-1] / finalEntry[1] #average velocity over entire trajectory
print('done.')
mu = v / (params["Efield"]*1e9) # mobility [m^2/(V.s)] for each electron
muMean = mu.mean() #average moibility
muErr = mu.std() / np.sqrt(nElectrons) #standard error in average mobility
print('Mobility:', muMean, '+/-', muErr)

#Bypass dielectric calculation in x <-> z swap case (as results will be redundant with unswapped case):
if swapXZ:
    results = np.array([[muMean, muErr]])
    np.savetxt('results'+suffix+'.csv', results, delimiter=',', fmt='%g')
    exit(0)

#Compute dielectric function:
epsEff, epsEff_NP, epsEff_BG = PeriodicFD(np.array(L), mask, params['epsNP'], params['epsBG']).computeEps(deriv=True)
epsAvg = np.trace(epsEff)/3
epsAvg_NP = np.trace(epsEff_NP)/3
epsAvg_BG = np.trace(epsEff_BG)/3
print('epsAvg:', epsAvg)
print('d(epsAvg)/d(epsFiller):', epsAvg_NP)
print('d(epsAvg)/d(epsMatrix):', epsAvg_BG)
print('epsTensor:\n', epsEff)
print('d(epsTensor)/d(epsFiller):\n', epsEff_NP)
print('d(epsTensor)/d(epsMatrix):\n', epsEff_BG)
printDuration('DielDone')

#Write results file:
def unpack(mat):
    return [ mat[0,0], mat[1,1], mat[2,2], mat[1,2], mat[2,0], mat[0,1] ]
results = np.concatenate(([ muMean, muErr, epsAvg, epsAvg_NP, epsAvg_BG], unpack(epsEff), unpack(epsEff_NP), unpack(epsEff_BG)))[None,:]
np.savetxt('results'+suffix+'.csv', results, delimiter=',', fmt='%g')
