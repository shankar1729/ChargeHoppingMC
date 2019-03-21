#!/usr/bin/python
from common import *

"""
Construct a graph of minima connected by energy barriers given inputs:
	E0: energy landscape on a regular 3D mesh (dimensions: S[0] x S[1] x S[2])
	hopDistGrid: hop distance in grid units (used for hopping 'entropy' calculation)
	kT: kB * T at current temperature T (in same units as E0)
	zPeriodic: whether z direction is bounded (True) or periodic (False)
Outputs:
	if zPeriodic:
		Eminima: Free energies per minima (dimensions: nMinima)
		Econn: Free energies of transition state positions per connection (dimensions: nConnections)
		iMinima: list of first minima index in connections (dimensions: nConnections)
		jMinima: list of second minima index in connections (dimensions: nConnections)
		disp: edge displacement in grid coordinates (dimensions: nConnections x 3)
	else:
		iPosMinima: Grid coordinates of minima (dimensions: nMinima x 3)
		iPosBarrier: Grid coordinates of barrier positions per minima per connection (dimensions: nMinima x maxDegree x 3)
		jMinima: Minima index at the other end of each connection (dimensions: nMinima x maxDegree)
		Abarrier: Hopping free energy for each connection (dimensions: nMinima x maxDegree)
		minimaStart: Set of minima connected to z=0 (where electrons should start)
		minimaStop: Set of minima connected to z=zMax (where electron trajectories should end)
"""
def minimaGraph(E0, hopDistGrid, kT, zPeriodic=False):
	from scipy.sparse import csr_matrix, diags
	
	#Get shape and then flatten for easy indexing:
	S = np.array(E0.shape)
	E0 = E0.flatten()
	
	#Set up connectivity:
	def initNeighbors():
		neigh1d = np.arange(-1,2)
		neighOffsets = flattenedMesh(neigh1d, neigh1d, neigh1d)
		#--- select close-packed (FCC) mesh neighbours:
		neighOffsets = neighOffsets[np.where(np.mod(np.sum(neighOffsets,axis=1),2)==0)]
		#--- remove half neighbours (prefer +z), as it will be covered by symmetry below:
		neighStride = np.array([1,10,100])
		return neighOffsets[np.where(np.dot(neighOffsets,neighStride)>0)[0]]
	neighOffsets = initNeighbors()
	
	#Flattened list of grid points and neighbours:
	print('Constructing neighbours and adjacency:')
	iPosMesh = flattenedMesh(np.arange(S[0]), np.arange(S[1]), np.arange(S[2]))
	#--- switch to close-packed mesh
	fccSel = np.where(np.mod(np.sum(iPosMesh,axis=1),2)==0)[0]
	nGrid = len(fccSel)
	iPosMesh = iPosMesh[fccSel]
	E0 = E0[fccSel]
	def getEdiff(i1, i2):
		return E0[i1] - E0[i2]
	nPeriodic = 3 if zPeriodic else 2
	def initAdjacency():
		adjMat = csr_matrix((nGrid,nGrid),dtype=np.int)
		posStride = np.array([S[1]*S[2], S[2], 1]) #indexing stride for cubic mesh
		fccSelInv = np.zeros((np.prod(S),), dtype=int) #initialize inverse of fccSel ...
		fccSelInv[fccSel] = np.arange(nGrid, dtype=int) #... for use in indexing below
		for neighOffset in neighOffsets:
			jPosMesh = iPosMesh + neighOffset[None,:]
			#Wrap indices in periodic directions:
			for iDir in range(nPeriodic):
				if neighOffset[iDir]<0: jPosMesh[np.where(jPosMesh[:,iDir]<0)[0],iDir] += S[iDir]
				if neighOffset[iDir]>0: jPosMesh[np.where(jPosMesh[:,iDir]>=S[iDir])[0],iDir] -= S[iDir]
			jPosIndexAll = np.dot(jPosMesh, posStride) #index into original cubic mesh (wrapped to fcc below)
			#Handle finite boundaries in z:
			if (not zPeriodic) and (neighOffset[2]>0): #only need to handle +z due to choice of half neighbour set above
				zjValid = np.where(jPosMesh[:,2]<S[2])[0]
				ijPosIndex = np.vstack((zjValid, fccSelInv[jPosIndexAll[zjValid]]))
			else:
				ijPosIndex = np.vstack((np.arange(nGrid,dtype=int), fccSelInv[jPosIndexAll]))
			#Direct each edge so that E[i] > E[j]:
			swapSel = np.where(getEdiff(ijPosIndex[0], ijPosIndex[1]) < 0.)
			ijPosIndex[:,swapSel] = ijPosIndex[::-1,swapSel]
			adjMat += csr_matrix((np.ones(ijPosIndex.shape[1],dtype=int), (ijPosIndex[0],ijPosIndex[1])), shape=(nGrid,nGrid))
		return adjMat
	adjMat = initAdjacency()
	printDuration('InitAdj')
	
	#Identify local minima and their local domains:
	minimaIndex = np.where(adjMat.sum(axis=1)==0)[0]
	nMinima = len(minimaIndex)
	printDuration('InitMinima')
	#--- initialize minima domain matrix:
	minMat = csr_matrix((np.ones(nMinima,dtype=int), (minimaIndex,np.arange(nMinima,dtype=int))), shape=(nGrid,nMinima))
	printDuration('InitMinMat')
	#--- multiply adjacency matrix repeatedly till all points covered:
	minMatCum = minMat
	pathLength = 0
	nPoints = nMinima
	while nPoints:
		minMat = adjMat * minMat
		minMatCum += minMat
		#Stop propagating points already connected to >=2 minima:
		minimaCount = np.array(minMatCum.sign().sum(axis=1), dtype=int).flatten() #number of minima already connected to each point
		pointWeight = np.where(minimaCount>=2, 0, 1)
		minMat = diags([pointWeight], [0]) * minMat #stop propagating these rows further
		pathLength += 1
		nPoints = minMat.count_nonzero()
		print('\tPath length:', pathLength, 'nPoints:', nPoints)
	printDuration('InitDomains')
	
	#Find saddle points connecting pairs of minima:
	if zPeriodic:
		minimaStart = []
		minimaStop = []
	else:
		#Find minima whose domains contain z=0 (electrons will be injected here):
		zSel = np.where(iPosMesh[:,2]==0, 1, 0)
		minimaStart = np.where(minMatCum.T * zSel)[0]
		#Find minima whose domains contain z=zmax (electron trajectories will end here):
		zSel = np.where(iPosMesh[:,2]==S[2]-1, 1, 0)
		minimaStop = np.where(minMatCum.T * zSel)[0]
	#--- Find pairs of minima connected by each point:
	iNZ,iMinNZ = minMatCum.nonzero()
	sel = np.where(iNZ[:-1]==iNZ[1:])[0] #two adjacent entries having same iNZ
	minimaPairs = np.array([iMinNZ[sel], iMinNZ[sel+1]]) #list of connected iMinima1 and iMinima2 connected
	swapSel = np.where(getEdiff(minimaIndex[minimaPairs[0]], minimaIndex[minimaPairs[1]]) < 0.)[0]
	minimaPairs[:,swapSel] = minimaPairs[::-1,swapSel] #in each pair of minima, put higher energy first
	pairIndex = nMinima*minimaPairs[0] + minimaPairs[1]
	iConnection = iNZ[sel] #corresponding connecting point between above pairs
	printDuration('InitConnections')
	#--- Sort pairs by minima index, barrier:
	sortIndex = np.lexsort([getEdiff(iConnection, minimaIndex[minimaPairs[0]]), pairIndex])
	pairIndexSorted = np.concatenate([[-1],pairIndex[sortIndex]])
	iFirstUniq = sortIndex[np.where(pairIndexSorted[:-1]!=pairIndexSorted[1:])[0]]
	minimaPairs = minimaPairs[:,iFirstUniq]
	iConnection = iConnection[iFirstUniq] #now index of saddle point
	printDuration('InitBarriers')
	print('nMinima:', nMinima, 'with nConnections:', len(iConnection))
	#--- Calculate barrier energies and "entropies" (used to account for nearer barriers reached more easily):
	def getBarrierES():
		barrierDisp = iPosMesh[iConnection] - iPosMesh[minimaIndex[minimaPairs[0]]] #displacement to barriers
		for iDir in range(nPeriodic): #wrap periodic directions
			barrierDisp[np.where(barrierDisp[:,iDir]<-0.5*S[iDir])[0],iDir] += S[iDir]
			barrierDisp[np.where(barrierDisp[:,iDir]>+0.5*S[iDir])[0],iDir] -= S[iDir]
		Sbarrier = -np.log(np.sum(barrierDisp**2, axis=1)/hopDistGrid**2)
		Ebarrier = getEdiff(iConnection, minimaIndex[minimaPairs[0]])
		return Ebarrier, Sbarrier
	Ebarrier,Sbarrier = getBarrierES()
	
	if zPeriodic:
		#Return sparse output directly for probabilitic model:
		Eminima = E0[minimaIndex]
		Econn = E0[iConnection]
		iMinima = minimaPairs[0]
		jMinima = minimaPairs[1]
		disp = iPosMesh[minimaIndex[jMinima]] - iPosMesh[minimaIndex[iMinima]]
		return Eminima, Econn, iMinima, jMinima, disp
	
	#Prune minima with too low barriers (<~ kT):
	print('Pruning low barriers')
	connPrune = np.where(Ebarrier < -kT*Sbarrier)[0]
	connKeep = np.where(Ebarrier >= -kT*Sbarrier)[0]
	#--- Replace higher energy minima with lower energy one along each pruned connection:
	while len(np.intersect1d(minimaPairs[0,connPrune], minimaPairs[1,connPrune])):
		replaceMap = np.arange(nMinima, dtype=int)
		replaceMap[minimaPairs[0,connPrune]] = minimaPairs[1,connPrune]
		minimaPairs[1,connPrune] = replaceMap[minimaPairs[1,connPrune]]
	minKeep = np.setdiff1d(np.arange(nMinima), minimaPairs[0,connPrune])
	replaceMap = np.arange(nMinima, dtype=int)
	replaceMap[minKeep] = np.arange(len(minKeep),dtype=int) #now with renumbering to account for removed minima
	replaceMap[minimaPairs[0,connPrune]] = replaceMap[minimaPairs[1,connPrune]] #now all in range [0,len(minKeep))
	#--- Apply replacements:
	minimaIndex = minimaIndex[minKeep]
	nMinima = len(minKeep)
	minimaStart = replaceMap[minimaStart]
	minimaStop = replaceMap[minimaStop]
	minimaPairs = replaceMap[minimaPairs[:,connKeep]]
	iConnection = iConnection[connKeep]
	printDuration('InitPrune')
	print('nMinima:', nMinima, 'with nConnections:', len(iConnection))
	
	"""
	#Debug code to plot x=0 yz-plane slice of energies and minima:
	import matplotlib.pyplot as plt
	E0mesh = np.zeros((S[1],S[2]))
	yzPlaneSel = np.where(iPosMesh[:,0]==0)[0]
	E0mesh[iPosMesh[yzPlaneSel,1],iPosMesh[yzPlaneSel,2]] = E0[yzPlaneSel]
	plt.imshow(E0mesh, cmap='Greys_r')
	yzPlaneSel = np.where(iPosMesh[minimaIndex,0]==0)[0]
	plt.plot(iPosMesh[minimaIndex[yzPlaneSel],2], iPosMesh[minimaIndex[yzPlaneSel],1], 'r+')
	yzPlaneEndSel = np.where(iPosMesh[minimaIndex[minimaStart],0]==0)[0]
	plt.plot(iPosMesh[minimaIndex[minimaStart[yzPlaneEndSel]],2], iPosMesh[minimaIndex[minimaStart[yzPlaneEndSel]],1], 'b+')
	yzPlaneEndSel = np.where(iPosMesh[minimaIndex[minimaStop],0]==0)[0]
	plt.plot(iPosMesh[minimaIndex[minimaStop[yzPlaneEndSel]],2], iPosMesh[minimaIndex[minimaStop[yzPlaneEndSel]],1], 'g+')
	plt.show(); exit()
	"""
	
	#Reduce problem purely to graph of minima connected through energy barriers:
	#--- add opposite direction connections between minima:
	minimaPairs = np.hstack((minimaPairs,minimaPairs[::-1]))
	iConnection = np.hstack((iConnection,iConnection))
	#--- sort connections by first minima index and energy barrier:
	Ebarrier,Sbarrier = getBarrierES()
	Abarrier = Ebarrier - kT*Sbarrier
	sortIndex = np.lexsort((Abarrier, minimaPairs[0]))
	minimaPairs = minimaPairs[:,sortIndex]
	iConnection = iConnection[sortIndex]
	Abarrier = Abarrier[sortIndex]
	Ebarrier = Ebarrier[sortIndex]
	Sbarrier = Sbarrier[sortIndex]
	#--- start indices of each first minima (now contiguous) in above array
	mPadded = np.hstack(([-1],minimaPairs[0]))
	startIndex = np.where(mPadded[:-1]!=mPadded[1:])[0]
	#--- remove connections which are extremely improbable from each minima:
	typDegree = len(np.where(Abarrier - Abarrier[startIndex[minimaPairs[0]]] < 7*kT)[0])*(1./nMinima)  #rest will have probability < exp(-7) ~ 0.1%
	maxDegree = int(2*typDegree)
	connKeep = np.where(np.arange(len(iConnection),dtype=int) - startIndex[minimaPairs[0]] < maxDegree)[0] #truncate degree of connectivity to maxDegree
	minimaPairs = minimaPairs[:,connKeep]
	iConnection = iConnection[connKeep]
	Abarrier = Abarrier[connKeep]
	Ebarrier = Ebarrier[connKeep]
	Sbarrier = Sbarrier[connKeep]
	mPadded = np.hstack(([-1],minimaPairs[0]))
	startIndex = np.where(mPadded[:-1]!=mPadded[1:])[0]
	#--- determine degrees of connectivity of each minima:
	mDegrees = np.hstack((startIndex[1:]-startIndex[:-1], len(iConnection)-startIndex[-1]))
	maxDegree = np.max(mDegrees)
	print('Degree of connectivity:    min:', np.min(mDegrees), 'max:', maxDegree, 'mean:', np.mean(mDegrees))
	print('Energy (0K) barriers [eV]: min:', np.min(Ebarrier), 'max:', np.max(Ebarrier), 'mean:', np.mean(Ebarrier))
	print('Free energy barriers [eV]: min:', np.min(Abarrier), 'max:', np.max(Abarrier), 'mean:', np.mean(Abarrier))
	#--- make regular mesh of connections by adding dummy connections up to maxDegree if needed:
	jMinima = np.tile(np.arange(nMinima, dtype=int)[:,None], (1,maxDegree)) #each minima connects to itself
	iConnMesh = minimaIndex[jMinima] #each trivial connection point is the minima itself
	Amesh = np.full(jMinima.shape, np.Inf) #each such connection will have zero probability of selection
	connNumber = np.arange(len(iConnection),dtype=int) - startIndex[minimaPairs[0]] #index of each connection from global list in set of connections from minimaPairs[0]
	jMinima[minimaPairs[0], connNumber] = minimaPairs[1]
	iConnMesh[minimaPairs[0], connNumber] = iConnection
	Amesh[minimaPairs[0], connNumber] = Abarrier
	printDuration('InitGraph')
	
	#Collect and return outputs:
	iPosMinima = iPosMesh[minimaIndex] #locations of the minima (grid coordinates)
	iPosBarrier = iPosMesh[iConnMesh] #locations of barrier points per minima per connection
	return iPosMinima, iPosBarrier, jMinima, Amesh, minimaStart, minimaStop
