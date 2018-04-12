import pickle
import numpy as np
import matplotlib.pyplot as plt

chmcFile = open('chmc.pkl')
trajFile = open('trajectory.pkl')
chmc = pickle.load(chmcFile)
trajectory = pickle.load(trajFile)
chmcFile.close()
trajFile.close()

print len(trajectory)

nElectrons = chmc["nElectrons"]

plt.figure(1, figsize=(10,6))

# Iterate over all electrons and plot its displacement in z-direction
for i in range(nElectrons):
	time = [event[1] for event in trajectory if event[0]==i]
	dz = [event[2][2] for event in trajectory if event[0]==i]
	plt.plot(time, dz)

plt.title('Charge Hopping Monte Carlo in Pure Polymer')
plt.xlabel('Time[s]')
plt.ylabel('Displacement in z-dir [nm]')
plt.savefig('MCdata.pdf', bbox_inches='tight')
plt.show()