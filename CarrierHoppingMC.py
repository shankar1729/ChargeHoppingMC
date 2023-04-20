#!/usr/bin/env python
from common import flattenedMesh


class CarrierHoppingMC:
    
    def __init__(
        self,
        *,
        L: np.ndarray  #: box-size in each dimension
        E0: np.ndarray,  #: energy landscape on grid
        epsBG: float, #: background dielectric constant
        hopDistance: float, #: distance per hop
        hopFrequency: float,  #: hops per unit time
        maxHops: int, #: number of hops per electron
        nElectrons: int, #: number of electrons to propagate at once
        tMax: float,  #: maximum time in sumulation
        T: float, #: temperature
    ):
        
        #Initialize box and grid:
        S = E0.shape
        h = L / S   # grid spacing
                
        #Initialize neighbour list and distance factors:
        drMax = 5.*hopDistance #truncate exponential distance function at exp(-5)
        self.irMax = int(np.ceil(drMax/h)) #max number of grid points electron can hop
        irGrid = np.arange(-self.irMax,+self.irMax+1) #list of relative neighbour indices in 1D
        self.ir = flattenedMesh(irGrid, irGrid, irGrid).T  #3 x N array of neighbour indices
        #--- calculate distances and select neighbours within drMax:
        dr = np.sqrt(np.sum((self.ir*h)**2, axis=0))
        irSel = np.where(np.logical_and(dr<=drMax, dr>0))[0] #select indices within a sphere of radius drMax
        self.ir = self.ir[:,irSel]
        self.wr = np.exp(-dr[irSel]/hopDistance) #exponential probability factor due to length of hop
        
        #Pad energy landscape to accomodate neighbour list without branches:
        self.E0 = E0
        #--- pad along x by periodically repeating:
        self.E0 = np.concatenate((self.E0[-self.irMax:,:,:], self.E0, self.E0[:self.irMax,:,:]), axis=0)
        #--- pad along y by periodically repeating:
        self.E0 = np.concatenate((self.E0[:,-self.irMax:,:], self.E0, self.E0[:,:self.irMax,:]), axis=1)
        #--- pad along z by setting inaccessible energy (+infinity):
        EzPad = np.full((self.E0.shape[0], self.E0.shape[1], self.irMax), np.inf)
        self.E0 = np.concatenate((EzPad, self.E0, EzPad), axis=2)
        
        #Other parameters:
        self.h = h
        self.S = S
        self.L = L
        self.nElectrons = nElectrons
        self.maxHops = maxHops
        self.tMax = tMax
        self.beta = 1./(kB * T)
        self.hopFrequency = hopFrequency
        self.coulombPrefac = 1.44 / epsBG #prefactor to 1/r in energy [in eV] of two electrons separated by r [in nm]
    
    #Calculate the Coulomb interaction with regularization for 0, given vectors r (dimensions [3,...])
    def safeCoulomb(self, r):
        return self.coulombPrefac / np.maximum(1e-6*self.h, np.sqrt(np.sum(r**2, axis=0)))
    
    #Calculate coulomb landscape of neighbours given grid separation iPosDelta of center locations
    #Avoid interaction with iCur: the current electron for which interactions are being calculated
    def coulombLandscape(self, iPosDelta, iCur):
        xDelta = (iPosDelta) * (1./self.S[:,None]) #separation in fractional coordinates for pairs with iHop'th electron
        xDelta[:2] -= np.floor(0.5 + xDelta[:2]) #wrap to nearest periodic neighbours in first two directions
        rDelta = xDelta * self.L[:,None] #displacement in nearest-periodic image convention, dimensions: [3, nElectrons, nElectrons]
        rDelta[:,iCur] = np.Inf #Avoid interacting with self
        rDeltaNeighbor = rDelta[...,None] + self.h*self.ir[:,None,:] #dimensions: [3, nElectrons, nNeighbours]
        return ( self.safeCoulomb(rDeltaNeighbor)  #+ 1/r to the neighbour
            - self.safeCoulomb(rDelta)[:,None] ) #- 1/r to the center location
    
    def run(self, iRun=0):
        """Run one complete MC simulation and return trajectory
        (jump times and positions for each electron)."""       
        np.random.seed(iRun) #Generate a new seed for every run
        print('Starting MC run', iRun)
        #Initialize electron positions (in grid coordinates)
        iPosElectron = np.vstack((
            np.random.randint(self.S[0], size=(1,self.nElectrons)),
            np.random.randint(self.S[1], size=(1,self.nElectrons)),
            np.zeros((1,self.nElectrons), dtype=int) #start all on xy plane at z=0
        ))
        
        #Initialize trajectory: list of electron number, time of event and grid position
        t = 0 #time of latest hop
        nElectronOffset = iRun*self.nElectrons
        trajectory = [ (iElectron+nElectronOffset, t) + tuple(iPos) for iElectron,iPos in enumerate(iPosElectron.T) ]
        #Initial energy of each electron and its neighbourhood
        #--- Fetch energy at each electron:
        E0electron = self.E0[
            self.irMax+iPosElectron[0],
            self.irMax+iPosElectron[1],
            self.irMax+iPosElectron[2]] #dimensions: [nElectrons]
        #--- Fetch energy landscape neighbourhoods for each electron:
        iPosNeighbor = iPosElectron[...,None] + self.ir[:,None,:]
        E0neighbor = self.E0[
            self.irMax+iPosNeighbor[0],
            self.irMax+iPosNeighbor[1],
            self.irMax+iPosNeighbor[2]] #dimensions: [nElectrons, nNeighbours]
        #--- Coulomb contributions to energy difference for each electron at initial positions:
        xDelta = (iPosElectron[:,None,:] - iPosElectron[...,None]) * (1./self.S[:,None,None]) #separation in fractional coordinates for all pairs
        xDelta[:2] -= np.floor(0.5 + xDelta[:2]) #wrap to nearest periodic neighbours in first two directions
        rDelta = xDelta * self.L[:,None,None] #displacement in nearest-periodic image convention, dimensions: [3, nElectrons, nElectrons]
        rDelta[:,range(self.nElectrons),range(self.nElectrons)] = np.Inf #Avoid interacting with self
        rDeltaNeighbor = rDelta[...,None] + self.h*self.ir[:,None,None,:] #dimensions: [3, nElectrons, nElectrons, nNeighbours]
        coulomb = ( np.sum(self.safeCoulomb(rDeltaNeighbor), axis=0)  #+ 1/r to the neighbour
            - np.sum(self.safeCoulomb(rDelta), axis=0)[:,None] ) #- 1/r to the center location

        #Main MC loop:
        izMax = 0
        #------- Added limit to MC simulation to given maxHops -----
        while len(trajectory) < self.maxHops:
            #Calculate hopping probabilities for each electron:
            #--- for each electron to each neighbour:
            hopRateSub = (self.hopFrequency
                * self.wr[None,:] #precalculated distance factor
                * np.exp(-self.beta*np.maximum(0., E0neighbor - E0electron[:,None] + coulomb)) #exp(-DeltaE/kT) if DeltaE > 0; else 1
            )
            #---calculate total for each electron
            hopRate = np.sum(hopRateSub, axis=1) #sum over second axis = neighbors
            
            #Calculate time for next hop for each electron:
            hopTime = np.random.exponential(1./hopRate) + t
            
            #Implement the soonest hop:
            iHop = np.argmin(hopTime)
            t = hopTime[iHop]
            if t > self.tMax:
                t = self.tMax
                #Finalize trajectory:
                trajectory += [ (iElectron+nElectronOffset, t) + tuple(iPos) for iElectron,iPos in enumerate(iPosElectron.T) ]
                break
            #--- select neighbour to hop to:
            iNeighbor = np.searchsorted(
                np.cumsum(hopRateSub[iHop]), #cumulative probability distribution
                hopRate[iHop]*np.random.rand()) #a random number till the total probability
            #--- update electron position:
            iPosOld = iPosElectron[:,iHop]
            iPosNew = np.mod(iPosOld + self.ir[:,iNeighbor], self.S) #Wrap with periodic boundaries            
            iPosElectron[:,iHop] = iPosNew
            trajectory.append((iHop+nElectronOffset, t) + tuple(iPosNew))
            if iPosNew[2] > izMax:
                izMax = iPosNew[2]
                print("Run", iRun, "reached", izMax/self.h, "nm at t =", t, "s")
                if izMax >= self.S[2] - self.irMax:
                    break #Terminate: an electron has reached end of box
            #--- update cached energies:
            iPosNeighborNew = iPosNew[:,None] + self.ir
            E0electron[iHop] = self.E0[
                self.irMax+iPosNew[0],
                self.irMax+iPosNew[1],
                self.irMax+iPosNew[2]]
            E0neighbor[iHop] = self.E0[
                self.irMax+iPosNeighborNew[0],
                self.irMax+iPosNeighborNew[1],
                self.irMax+iPosNeighborNew[2]]
            #--- update Coulomb energies of current electron:
            coulomb[iHop] = np.sum(self.coulombLandscape(iPosNew[:,None] - iPosElectron, iHop), axis=0)
            #--- update Coulomb energies of all other electrons:
            coulomb -= self.coulombLandscape(iPosElectron - iPosOld[:,None], iHop) #remove contribution from old position of iHop'th electron
            coulomb += self.coulombLandscape(iPosElectron - iPosNew[:,None], iHop) #add contribution from new position of iHop'th electron
        print('End MC run', iRun, 'with trajectory length:', len(trajectory), 'events')
        return np.array(trajectory, dtype=np.dtype('i8,f8,i8,i8,i8'))
