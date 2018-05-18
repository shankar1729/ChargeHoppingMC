import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
import matplotlib.ticker as mtick

trajectory = np.loadtxt('trajectory.dat')
nElectrons = 1 + int(np.max(trajectory[:,0]))
print "Read trajectory with", len(trajectory), "hops and", nElectrons, "electrons"

# Create time grid
tMax = trajectory[:,1].max()
nSteps = 1e2
deltaTime = tMax / nSteps
timeGrid = np.linspace(0, tMax, nSteps+1)
zPos = np.zeros((nElectrons, len(timeGrid))) #z-poistions of electrons in time grid
print "Divided", tMax, "s into", nSteps, "intervals"

######### Trajectories #########
print "Analyzing Trajectories...",
plt.figure(1, figsize=(10,6))
# Iterate over all electrons and plot its displacement in z-direction
for i in range(nElectrons):
	sel = np.where(trajectory[:,0]==i)[0]
	time = trajectory[sel,1]
	dz = trajectory[sel,4]
	zPos[i] = dz[np.minimum(np.searchsorted(time, timeGrid), len(dz)-1)]
	plt.plot(time, dz)
plt.title('Charge Hopping Monte Carlo in Pure Polymer')
plt.xlabel('Time [s]')
plt.ylabel('Displacement in z-dir [nm]')
plt.savefig('trajectory.pdf', bbox_inches='tight')
print "Done"

######### Avg. Displacement and Avg. Velocity #########
avgZpos = np.mean(zPos, axis=0)
avgVel = (avgZpos[1:] - avgZpos[:-1])/deltaTime
timeGridMid = 0.5*(timeGrid[1:] + timeGrid[:-1])

plt.figure(2, figsize=(10,6))
avgVel = np.array(avgVel)*1e-9 # convert from nm/s to m/s
plt.plot(timeGridMid, avgVel)
plt.title('Charge Hopping Monte Carlo in Pure Polymer')
plt.ylabel("Average Velocity in z-dir [m/s]")
plt.xlabel("Time [s]")
plt.savefig("avgVel.pdf", bbox_inches='tight')

plt.figure(3, figsize=(10,6))
plt.plot(timeGrid, np.array(avgZpos))
plt.title('Charge Hopping Monte Carlo in Pure Polymer')
plt.ylabel("Average Displacement in z-dir [nm]")
plt.xlabel("Time [s]")
plt.savefig("avgDist.pdf", bbox_inches='tight')
print "Done"

###### Mobility ######
mu = avgVel/0.01

plt.show()

