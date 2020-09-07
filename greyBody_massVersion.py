'''
This script is written for fitting multi-band far-IR to
sub-mm fluxes of cores/clumps to modified-blackbodies.

In default, the free parameters are the dust temperature 
and source mass. It the betaVary is set to True, 
the emissivity index (beta) can also be fit. 

Originally written by Jinghua Yuan
Revised by Xu Fengwei

Version Sept. 7, 2020
'''

import os
import time
import numpy as np
from astropy import constants as cons
from astropy import units as u
from astropy.io import fits
from matplotlib import pyplot as plt
import matplotlib as mpl
from lmfit import minimize, Parameters, Model

# Initiate the parameters
iMass  = 1000*u.Msun # in cm^-2
iTdust = 22.0 # in K.
ibeta  = 2. # 
betaVary = False # If True, Beta will be fitted.
                 # If False, beta will be fixed.
betaMin = 1.0
betaMax = 3.0

# Constants
h = cons.h.cgs.value   # Planck constant in CGS unit
k = cons.k_B.cgs.value # Boltzmann constant in CGS unit
c = cons.c.cgs.value # speed of light in CGS unit
mH = cons.m_n.cgs.value # mass of an neutron
muh2 = 2.8 # mean molecular weight adopted from Kauffmann et al. (2008)
rGD = 100.0 # gas-to-dust mass ratio
nu0 = 599.584916E9 # Reference frequency in Hz.
kappa0 = 5.0 # Dust emissivity at reference frequency
wavelengths = np.array([70,160,250,350,500,870])*u.um
frequencies = wavelengths.to(u.Hz, u.spectral()).value.copy()

# define the model
def greybody(nu, mass=iMass.cgs.value, Tdust=iTdust, beta=ibeta):
    blackBody = 2*h*nu**3/c**2/(np.exp(h*nu/k/Tdust)-1)
    tau = mass*kappa0*(nu/nu0)**beta/d**2/rGD
    return blackBody*tau

gbMod = Model(greybody)
gbMod.set_param_hint('beta', min = betaMin, max = betaMax, vary = betaVary)
gbMod.set_param_hint('Tdust',  max = 80.0)
pars = gbMod.make_params(mass = iMass.cgs.value, Tdust = iTdust)

# source parameter of your own
distance = 2.92*u.kpc # distance towards to your source
d = distance.cgs.value # distance in unit of cm

# the data you should derive in your ds9
# circle the specific region and get the statistic data that ds9 show
# the unit of 'smregrid45.fits' is MJy/sr, should be converted into Jy like below shows
flatfield = np.array([66263,198491,59008,27758,11148,7277])*10**6*(14/3600*np.pi/180)**2*u.Jy
fluxData = np.array([338421,760148,272224,143021,58027,16642])*10**6*(14/3600*np.pi/180)**2*u.Jy
fluxErr = fluxData*0.2

# sed fit result
sedResult = gbMod.fit((fluxData).cgs.value, nu = frequencies)
T = sedResult.params['Tdust'].value
errT = sedResult.params['Tdust'].stderr
B = sedResult.params['beta'].value
errB = sedResult.params['beta'].stderr
M = sedResult.params['mass'].value*u.g
errM = sedResult.params['mass'].stderr*u.g

# plot the result
x = np.array(np.logspace(1,3.5,300))*u.um
xFreq = x.to(u.Hz, u.spectral()).value.copy()
y = greybody(xFreq,Tdust=sedResult.params['Tdust'].value,
             mass=sedResult.params['mass'].value)*u.g/u.s/u.s

fig = plt.figure(figsize = (5,5))
ax  = fig.add_axes([0.1,0.1,0.8,0.6])
ax.plot(x,y.to(u.Jy).value, '-k')
# ax.scatter(wavelengths.value, fluxData.value, color = 'black')
ax.errorbar(wavelengths.value, fluxData.value, fluxErr.value, fmt='o', ecolor='r', color='k')

ax.set_xscale('log')
ax.set_yscale('log')
ax.set_xlim(20, 5000)
ax.set_ylim(10,10**4)

xmin,xmax = ax.get_xlim()
ymin,ymax = ax.get_ylim()
ax.text(10**2.7, 10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.04), 
        'ID: I16272-4837', horizontalalignment='left')

mLabel = ('${ =\ '+'%d'%M.to(u.Msun).value+
          '\pm'+'%d'%errM.to(u.Msun).value+'\ M_{\odot}}$')
ax.text(10**2.7, 10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.075), 
        '$M_{env}$'+mLabel, horizontalalignment='left')

temLabel = ("${=\ "+'%.1f'%T+"\pm"+'%.1f'%errT+"\ K}$")
ax.text(10**2.7, 10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.11),
        '$T_{dust}$'+temLabel, horizontalalignment='left')

modelFlux = greybody(frequencies, Tdust = sedResult.params['Tdust'].value,
                     mass= sedResult.params['mass'].value)*u.g/u.s/u.s
chi2 = (np.sum(((modelFlux.to(u.Jy)-
             fluxData)/fluxErr)**2)/(len(fluxData)-1))
chi2Label = ("${=\ "+'%.1f'%chi2+"}$")
ax.text(10**2.7, 10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.145), 
        '$\chi^2$'+chi2Label , horizontalalignment='left')

ax.set_xlabel('Wavelength ($\mu m$)')
ax.set_ylabel('Flux (Jy)')


