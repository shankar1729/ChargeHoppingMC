#!/usr/bin/env python
from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
from scipy.special import erfc
from CarrierHoppingMC import CarrierHoppingMC
from common import *

#Extract information from matlab file:
matFile = loadmat('structure.mat')
aspectRatio = float(matFile['Aspect_Ratio'])
centers = np.array(matFile['Center_list']) - 1. #Octave to python convention
axisDir = np.array(matFile['Orientation_list'])
volFrac = float(matFile['VF'])
img = np.array(matFile['img'])
L = img.shape
a = 20 #semi-major axis length TODO: get from matlab file after revisions
b = a/aspectRatio
axisDir = axisDir * (1./np.linalg.norm(axisDir,axis=1)[:,None]) #normalize

print(img.min(), img.max(), img.mean())

#Construct mask (1 inside particles, 0 outside):
def constructMask():
	grids1D = tuple([ np.arange(Li, dtype=float) for Li in L ])
	r = np.array(np.meshgrid(*grids1D, indexing='ij'))
	dr = r[None,...] - centers[...,None,None,None] #vectors from each center to each grid point
	Lbcast = np.array(L)[None,:,None,None,None]
	dr -= np.floor(0.5+dr/Lbcast)*Lbcast #wrap displacements by minimum image convention
	dist = np.maximum(np.sqrt(np.sum(dr**2, axis=1)), 1e-6) #corresponding length (regularized to avoide division by zero below) 
	cosTheta = np.sum(dr * axisDir[...,None,None,None], axis=1) / dist #corresponding cosTheta's to major axis direction
	dist -= b/np.sqrt(1 + ((b/a)**2 - 1)*(cosTheta**2)) #distance outside surface of ellipsoid
	return 0.5*erfc(dist.min(axis=0)) #1-voxel smoothing
mask = constructMask()
print('Volume fraction: expected:', volFrac, 'actual:', mask.mean())

#Initialize carrier hopping parameters:
params = { 
    "L": np.array(L), #box size in nm
    "h": 1., #grid spacing in nm
    "Efield": 0.06, #electric field in V/nm
    "dosSigma": 0.224, #dos standard deviation in eV
    "dosMu": 0.0, #dos center in eV
    "T": 298., #temperature in Kelvin
    "hopDistance": 1., #average hop distance in nm
    "hopFrequency": 1e12, #attempt frequency in Hz
    "nElectrons": 16, #number of electrons to track
    "maxHops": 1e5, #maximum hops per MC runs
    "tMax": 1e3, #stop simulation at this time from start in seconds
    "epsBG": 2.4, #relative permittivity of polymer
    "useCoulomb": True, #whether to include e-e Coulomb interactions
    #--- Nano-particle parameters
    "mask": mask, #1 where particles are, 0 outside
    "epsNP": 3.8, #relative permittivity of nanoparticles
    "trapDepthNP": -1.1, #trap depth of nanoparticles in eV
    "shouldPlotNP": False, #plot the electrostatic potential from PeriodicFD
}


#Run hopping simulation:
nRuns = 32
model = CarrierHoppingMC(params)
trajectory = np.concatenate(parallelMap(model.run, cpu_count(), range(nRuns))) #Run in parallel and merge trajectories
np.savetxt("trajectory.dat", trajectory, fmt="%d %e %d %d %d", header="iElectron t[s] ix iy iz") #Save trajectories together

#TODO

#Extract mobility:
#TODO


## Model parameters

"""
## Execute the model
return analyze.plotTrajectory(trajFile)/(params["Efield"]*1e9) # mobility [m^2/(V.s)]

import time
data = loadmat('../DOE_part3.mat')['images']
mu = np.zeros(data.shape)
totalImg = np.prod(data.shape)
count=1
tPrev = time.time()
print("Start time:", time.ctime(tPrev))
for i, doe in enumerate(data):
        for j, img in enumerate(doe):
                print("===", count,"/",totalImg,"==="); count=count+1
                mask = convolveTEM(img[0])
                print("Convolved 2D image to 3D mask")
                suffix = str(i+51)+"-"+str(j+1)
                mu[i][j] = hoppingMu('CHMC', mask, 32, "t-"+suffix+".dat")
                print("mu: {:.4e} m^2/(V.s)".format(mu[i][j]))
                #print("Time taken: {:.2f} s".format(tCur-tPrev))
print("End time:", time.ctime(time.time()))
np.savetxt("../mu3.dat", mu)


#------------- copied from Abhishek's analyze.py ----------------
#!/usr/bin/env python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import sys

def plotTrajectory(fname='trajectory.dat'):
        trajectory = np.loadtxt(fname)
        #print("Reading trajectories from", fname)
        nElectrons = 1 + int(np.max(trajectory[:,0]))
        #print("Read trajectory with", len(trajectory), "hops and", nElectrons, "electrons")

        # Create time grid
        tMax = trajectory[:,1].max()
        nSteps = 1e2
        deltaTime = tMax / nSteps
        timeGrid = np.linspace(0, tMax, nSteps+1)
        zPos = np.zeros((nElectrons, len(timeGrid))) #z-poistions of electrons in time grid
        #print("Divided", tMax, "s into", nSteps, "intervals")

        #--- Trajectories --
        #print("Analyzing Trajectories...")
        #plt.figure(1, figsize=(10,6))
        # Iterate over all electrons and plot its displacement in z-direction
        for i in range(nElectrons):
                sel = np.where(trajectory[:,0]==i)[0]
                time = trajectory[sel,1]
                dz = trajectory[sel,4]
                zPos[i] = dz[np.minimum(np.searchsorted(time, timeGrid), len(dz)-1)]
                #plt.plot(time, dz)
        #plt.xlabel('Time [s]')
        #plt.ylabel('Displacement in z-dir [nm]')
        #plt.savefig('trajectory.pdf', bbox_inches='tight') # uncomment to save the figure

        #--- Avg. Displacement and Avg. Velocity ---
        avgZpos = np.mean(zPos, axis=0)
        avgVel = (avgZpos[1:] - avgZpos[:-1])/deltaTime
        timeGridMid = 0.5*(timeGrid[1:] + timeGrid[:-1])

        #plt.figure(2, figsize=(5,6))
        #plt.subplot(211)
        #plt.semilogx(timeGrid, np.array(avgZpos))
        #plt.ylabel("Average Displacement in z-dir [nm]")
        #plt.xlabel("Time [s]")

        #plt.subplot(212)
        avgVel = np.array(avgVel)*1e-9 # convert from nm/s to m/s
        #plt.semilogx(timeGridMid, avgVel)
        #plt.ylabel("Average Velocity in z-dir [m/s]")
        #plt.xlabel("Time [s]")
        v = avgZpos[-1]/timeGrid[-1]
        #print("Velocity: {:.2e} [m/s]".format(v))
        #plt.axhline(v, c='k', ls='--')
        #plt.savefig("avgDist.pdf", bbox_inches='tight') # uncomment to save the figure
        #plt.show()

        # uncomment to save the data
        #np.savetxt("avgZpos.dat", np.array([timeGrid, avgZpos]).T)
        #np.savetxt("avgVel.dat", np.array([timeGridMid, avgVel]).T)
        return v

"""