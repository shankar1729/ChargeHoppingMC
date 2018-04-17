import pickle
import numpy as np
import matplotlib.pyplot as plt
import mpl_toolkits.mplot3d.axes3d as p3
import matplotlib.animation as animation

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

# Attaching 3D axis to the figure
#fig = plt.figure(2)
#ax = p3.Axes3D(fig)

#elecData = np.zeros(nElectrons, dtype=[('position', float, 3)])
##for i in range(nElectrons):
	##time = [event[1] for event in trajectory if event[0]==i]
	##dz = [event[2][2] for event in trajectory if event[0]==i]
	##dy = [event[2][1] for event in trajectory if event[0]==i]
	##dx = [event[2][0] for event in trajectory if event[0]==i]
	###plt.plot(time, dz)
	##ax.scatter(dx, dy, dz)

## Setting the axes properties
#ax.set_xlim3d([0.0, chmc["S"][0]])
#ax.set_xlabel('X')

#ax.set_ylim3d([0.0, chmc["S"][1]])
#ax.set_ylabel('Y')

#ax.set_zlim3d([0.0, chmc["S"][2]])
#ax.set_zlabel('Z')

#ax.set_title('Charge Hopping Monte Carlo')

# Creating the Animation object
#line_ani = animation.FuncAnimation(fig, update_lines, 25, fargs=(data, lines),
                                   #interval=50, blit=False)

plt.show()