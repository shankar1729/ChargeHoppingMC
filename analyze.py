import numpy as np
import matplotlib.pyplot as plt
# import mpl_toolkits.mplot3d.axes3d as p3
# import matplotlib.animation as animation
import matplotlib.ticker as mtick

trajectory = np.loadtxt('trajectory.dat')
nElectrons = 1 + int(np.max(trajectory[:,0]))
print "Read trajectory with", len(trajectory), "hops and", nElectrons, "electrons"


######### Trajectories #########
print "Analyzing Trajectories...",
plt.figure(1, figsize=(10,6))
# Iterate over all electrons and plot its displacement in z-direction
for i in range(nElectrons):
	sel = np.where(trajectory[:,0]==i)[0]
	time = trajectory[sel,1]
	dz = trajectory[sel,4]
	plt.plot(time, dz)
plt.title('Charge Hopping Monte Carlo in Pure Polymer')
plt.xlabel('Time [s]')
plt.ylabel('Displacement in z-dir [nm]')
plt.savefig('trajectory.pdf', bbox_inches='tight')
print "Done"

######### Avg. Displacement and Avg. Velocity #########
# initialize arrays
velZ = np.zeros(nElectrons)
zPos = np.zeros(nElectrons)
avgZpos = []
avgVel = []

# Create time grid
tMax = trajectory[:,1].max()
nSteps = 1e2
deltaTime = tMax / nSteps
timeGrid = np.linspace(0, tMax, nSteps+1)
print "Divided", tMax, "s into", nSteps, "intervals"

print "Analyzing Displacement and Velocity...",
# Find displacement in regular time intervals, and average over all electrons
for t in timeGrid:
	for i in range(nElectrons):
		sel = np.where(np.logical_and(trajectory[:,0]==i,
			trajectory[:,1] >= t, trajectory[:,1] < t+deltaTime))[0]
		iZpos = trajectory[sel,4]	# select z-coordinates of ith electron
		deltaZpos = (iZpos.max() - iZpos.min()) if iZpos.size > 1 else 0 # find displacement in this time interval
		velZ[i]  = deltaZpos/deltaTime # calculate velocity in this time interval
		zPos[i] += deltaZpos # z-coordinate at the end of this time interval
	avgVel.append(np.average(velZ))
	avgZpos.append(np.average(zPos))

plt.figure(2, figsize=(10,6))
avgVel = np.array(avgVel)*1e-9 # convert from nm/s to m/s
plt.plot(timeGrid, avgVel)
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

