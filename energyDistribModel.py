#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.integrate import solve_ivp

#For now hop frequency and distance are 1
#Putting in actual numbers will only change the time scale and the overall D magnitude

#All energies in eV
E = np.arange(-1., 0.5, 0.01)
dE = E[1] - E[0]

#Normalized density of states:
Esigma = 0.1
g = np.exp(-0.5*(E/Esigma)**2) + 0.001*np.exp(-0.5*((E+0.8)/0.05)**2)
g *= 1./(dE*np.sum(g)) #normalize

#Transfer matrix of hopping with dE multiplied:
kT = 0.026 #300 K
beta = 1./kT
nu = dE*np.exp(np.minimum(beta*(E[:,None]-E[None,:]), 0.))

#Differential equation matrix and function:
deqMat = g[:,None]*nu.T - np.diag(np.dot(nu, g))
def fDot(t, f):
	return np.dot(deqMat, f)

#Solve and plot
t = np.concatenate((np.arange(10.), np.logspace(1,5)))
f0 = np.copy(g)
res = solve_ivp(fDot, (t.min(),t.max()), f0, t_eval=t)

DofE = np.dot(nu, g)
D = dE*np.dot(DofE, res.y) #as a functoin of time

fEq = g * np.exp(-beta*E)
fEq *= 1./(dE*np.sum(fEq)) #normalize
Deq = dE*np.dot(DofE, fEq) #equilibrium diffusion constant
print('Equilibrium diffusion constant', Deq)

plt.figure(1)
plt.plot(E, res.y[:,::5])
plt.xlabel('E [eV]')
plt.ylabel('f(E) [eV$^{-1}$]')


plt.figure(2)
plt.loglog(t, D)
plt.loglog(t, 0.6*t**-0.6 + 0.0074)
plt.xlabel('t [a.u.]')
plt.ylabel('D [a.u.]')

plt.show()
