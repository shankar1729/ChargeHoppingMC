#!/opt/miniconda3/bin/python
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat
import glob
from convolveTEM import *
from common import *
from CarrierHoppingMC import *
from MinimaHoppingMC import *

sampleImg = loadmat('sampleTEM.mat')['img_out'][:200,:200]
microStr = convolveTEM(sampleImg, 16, 7, 7, 0.0143)
print(microStr.shape, sampleImg.shape)

# plot input microstructures
#fig, ax = plt.subplots(1,2, figsize=(12,5))
#ax[0].imshow(sampleImg, origin='lower')
#ax[0].set_title("Orginal")
#proj = microStr.sum(axis=0)
#proj = np.where(proj>0, 1, 0)
#ax[1].imshow(proj, origin='lower')
#ax[1].set_title("Convoluted")
#plt.show()

params = { 
    "L": microStr.shape, #box size in pixels
    "h": 1., #grid spacing in pixels
    "Efield": 0.06, #electric field in V/nm
    "dosSigma": 0.224, #dos standard deviation in eV
    "dosMu": 0.0, #dos center in eV
    "T": 298., #temperature in Kelvin
    "hopDistance": 1., #average hop distance in nm
    "hopFrequency": 1e12, #attempt frequency in Hz
    "nElectrons": 16, #number of electrons to track
    "maxHops": 1e5, #maximum hops per MC runs
    "tMax": 1e3, #stop simulation at this time from start in seconds
    "epsBG": 2.0, # PS: 2.0, PMMA: 3.2
    "useCoulomb": True, #whether to include e-e Coulomb interactions
    #--- Nano-particle parameters
    "epsNP": 3.9, #Silica: 3.9, BariumTitanate: 
    "trapDepthNP": -1.1, #trap depth of nanoparticles in eV
    "shouldPlotNP": False, #plot the electrostatic potential from PeriodicFD
    "mask": microStr
}
mhmc = MinimaHoppingMC(params)
mhmc.run(); exit() 
#chmc = CarrierHoppingMC(params)
#chmc.run(); exit()

#trajectory = chmc.run()
#nRuns = 1
# -------- Now runs parallel from MINIMAHOPPING -----------
#Run in parallel and merge trajectories
#trajectory = np.concatenate(parallelMap(chmc.run, cpu_count(), range(nRuns)))
#Save trajectories together
#np.savetxt("tempTraj.dat", trajectory, fmt="%d %e %d %d %d", header="iElectron t[s] ix iy iz")
