import numpy as np
from scipy.stats import ortho_group

def NPClusters(clusterShape, radiusNP, nParticles, nClusterMu, nClusterSigma, L):
	if clusterShape=="random":
		r0 = np.random.randn(nParticles, 3) * L[None,:]
		radiusArr = np.ones(nParticles)*radiusNP
	
	elif clusterShape!="file":
		r0 = np.array([[0,0,0]])
		nClusters = nParticles/nClusterMu # fix number of clusters
		clusterCenters = np.random.rand(nClusters, 3) * L[None,:] # randomly place cluster centers
		clusterSize = np.random.normal(nClusterMu, nClusterSigma, size=nClusters).astype(int) # randomly fix the cluster size (Gaussian distribution)
		if clusterShape=="round":
			# Round - generate random parameters in spherical coordinates (r, theta, phi) for each particle in each cluster
			for i in range(nClusters):  # for each cluster
				radius = radiusNP * np.cbrt(clusterSize[i])
				for j in range(clusterSize[i]):  # for each particle
					r, theta, phi = np.random.random(3) * [radius, np.pi, 2*np.pi]
					x = r*np.sin(theta)*np.cos(phi) + clusterCenters[i,0]
					y = r*np.sin(theta)*np.sin(phi) + clusterCenters[i,1]
					z = r*np.cos(theta)             + clusterCenters[i,2]
					r0 = np.append(r0, [[x,y,z]], axis=0)

		elif clusterShape=="sheet":
			# Sheet - fix a random orientation - generate random coordinates within a thin slab in the local coordinate system
			for i in range(nClusters):	# for each cluster
				orientation = ortho_group.rvs(3) # random orientation
				radius = radiusNP * np.sqrt(clusterSize[i]*4./3.) # radius of cylindrical sheet of 1 NP thickness
				for j in range(clusterSize[i]): # for each particle
					r, phi = np.random.random(2) * [radius, 2*np.pi]
					coords = r * np.array([np.cos(phi), np.sin(phi), 0.]) + clusterCenters[i,:]
					coords = np.dot(orientation, coords)
					r0 = np.append(r0, [coords], axis=0)
		elif clusterShape=="line":
			# Line - fix a random orientation - add equal number of particles about the cluster center
			for i in range(nClusters):	# for each cluster
				N = clusterSize[i]
				orientation = ortho_group.rvs(3) # random orientation
				coords = np.zeros((3,N))
				coords[2,:] = np.arange(-N/2, N/2)*radiusNP*2 # linearly arranged NP along z-direction
				coords = np.dot(orientation, coords + clusterCenters[i,:,None]) # coordination transformation
				r0 = np.append(r0, coords.T, axis=0)
		r0 = np.delete(r0, 0, 0)
		radiusArr = np.ones(r0.shape[0])*radiusNP
	
	elif clusterShape=="file":
		data = np.loadtxt("np.dat")
		r0 = np.ones((data.shape[0],3))
		r0[:,1:3] = data[:,:2]
		radiusArr = data[:,2]
	
	return r0, radiusArr