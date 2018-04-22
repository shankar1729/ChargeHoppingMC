#!/usr/bin/python
import numpy as np

#Units:
kB = 0.000086174 #in eV/K

#Profiling functions
shouldProfile = True
if shouldProfile:
	import time
	import resource
	tPrev = time.clock()
	def printDuration(label):
		global tPrev
		tCur = time.clock()
		mem = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
		print 'Time('+label+'):', tCur-tPrev, 's, Mem:', mem/1024, 'MB'
		tPrev = tCur
else:
	def printDuration(label):
		pass


#Wrapper to meshgrid producing flattened results:
def flattenedMesh(i1, i2, i3):
	iiArr = np.meshgrid(i1, i2, i3, indexing='ij')
	return np.hstack([ii.flatten()[:,None] for ii in iiArr])
