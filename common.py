#!/usr/bin/python
from __future__ import print_function
import numpy as np
from multiprocessing import Process, Queue, cpu_count

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
		print('Time('+label+'):', tCur-tPrev, 's, Mem:', mem/1024, 'MB')
		tPrev = tCur
else:
	def printDuration(label):
		pass


#Wrapper to meshgrid producing flattened results:
def flattenedMesh(i1, i2, i3):
	iiArr = np.meshgrid(i1, i2, i3, indexing='ij')
	return np.hstack([ii.flatten()[:,None] for ii in iiArr])

#Parallel map function: run func for each arg in argArr in nProcesses in parallel and return a list
def parallelMap(func, nProcesses, argArr):
	#Start process objects that will put in a common queue:
	queue = Queue()
	nArg = len(argArr)
	procArr = []
	for iProcess in range(nProcesses):
		proc = Process(target=spawn(func), args=(queue,
			argArr[int(np.floor((nArg*iProcess)/nProcesses)):int(np.floor((nArg*(iProcess+1))/nProcesses))] #split arguments among processes
		) )
		proc.daemon = True #required for Ctrl+C
		proc.start()
		procArr.append(proc)
	#Get all results:
	result = []
	while len(result) < nArg:
		result.append(queue.get())
	#Join processes and return results:
	[proc.join() for proc in procArr]
	return result
#--- Helper function for above
def spawn(func):
	def funcTmp(queue, argArr):
		for arg in argArr:
			queue.put(func(arg))
	return funcTmp
