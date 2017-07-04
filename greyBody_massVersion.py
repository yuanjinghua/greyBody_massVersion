'''
This script is written for fitting multi-band far-IR to
sub-mm fluxes of cores/clumps to modified-blackbodies.

In default, the free parameters are the dust temperature 
and source mass. It the betaVary is set to True, 
the emissivity index (beta) can also be fit. 

Originally written by Jinghua Yuan

Version 0.1  July 4th, 2017
'''

import os
import time
import myblack
import numpy as np
from astropy import wcs
from astropy import constants as cons
from astropy import units as u
from astropy.table import Column
from astropy.io import ascii, fits
from matplotlib import pyplot as plt
import matplotlib as mpl
from astropy.table import Column, Table
from lmfit import minimize, Parameters, Model

mpl.rc("font", family="serif", size=15)
mpl.rc("axes", linewidth =  1 )
mpl.rc("lines", linewidth = 1 )
mpl.rc("xtick.major", pad = 8, size = 8, width = 1)
mpl.rc("ytick.major", pad = 8, size = 8, width = 1)
mpl.rc("xtick.minor", size = 4, width = 1 )
mpl.rc("ytick.minor", size = 4, width = 1 )

sourList = ascii.read('./sourList.txt')

# Initiate the parameters
iMass  = 1000*u.Msun # in cm^-2
iTdust = 22.0 # in K.
ibeta  = 2.0 # 
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

def greybody(nu, mass=iMass.cgs.value, Tdust=iTdust, beta=ibeta):
    
    blackBody = 2*h*nu**3/c**2/(np.exp(h*nu/k/Tdust)-1)
    tau = mass*kappa0*(nu/nu0)**beta/d**2/rGD
    return blackBody*tau

gbMod = Model(greybody)

gbMod.set_param_hint('beta', min = betaMin, max = betaMax, 
        vary = betaVary)
gbMod.set_param_hint('Tdust',  max = 80.0)

pars = gbMod.make_params(mass = iMass.cgs.value, Tdust = iTdust)

pfmt = '%6i %18s %6.2f%% finished.'

Tdust_col = []
errTdust_col = []
Menv_col = []
errMenv_col = []

for isour in range(len(sourList)):
    sName = sourList['Name'][isour]
    dis   = sourList['dis'][isour]*u.kpc
    d   = dis.cgs.value # distance in cm
    fluxData = np.array([sourList['S160'][isour], sourList['S250'][isour], 
                         sourList['S350'][isour], sourList['S500'][isour],
                         sourList['S870'][isour]])*u.Jy

    fluxErr = fluxData*0.2

    wavelengths = np.array([160,250,350, 500, 870]) * u.um
    frequencies = wavelengths.to(u.Hz, u.spectral()).value.copy()
    
    sedResult = gbMod.fit(fluxData.cgs.value, nu = frequencies)
    
    T    = sedResult.params['Tdust'].value
    errT = sedResult.params['Tdust'].stderr
    B    = sedResult.params['beta'].value
    errB = sedResult.params['beta'].stderr
    M    = sedResult.params['mass'].value*u.g
    errM = sedResult.params['mass'].stderr*u.g
    
    Tdust_col.append(T)
    errTdust_col.append(errT)
    Menv_col.append(M.to(u.Msun).value)
    errMenv_col.append(errM.to(u.Msun).value)
    
    # plotting the SED


    x = np.array(np.logspace(1,3.5, 300))*u.um
    xFreq = x.to(u.Hz, u.spectral()).value.copy()
    y = greybody(xFreq,Tdust = sedResult.params['Tdust'].value,
                 mass= sedResult.params['mass'].value)*u.g/u.s/u.s

    fig = plt.figure(figsize = (5,5))
    ax  = fig.add_axes([0.1,0.1, 0.8, 0.6])
    ax.plot(x,y.to(u.Jy).value, '-k')
    ax.scatter(wavelengths, fluxData, color = 'black')
    ax.set_xscale('log')
    ax.set_yscale('log')
    ax.set_xlim(8, 5000)
    ax.set_ylim(np.e**(np.log(np.min(fluxData.value))-
                     0.2*(np.log(np.max(fluxData.value))
                          -np.log(np.min(fluxData.value)))),
                np.e**(np.log(np.max(fluxData.value))+
                     0.5*(np.log(np.max(fluxData.value))
                          -np.log(np.min(fluxData.value)))))

    xmin,xmax = ax.get_xlim()
    ymin,ymax = ax.get_ylim()
    
    ax.text(10**(np.log10(xmin)+(np.log(xmax)-np.log(xmin))*0.03), 
            10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.04), 
            sName, horizontalalignment='left')
    
    mLabel = ("${ =\ "+'%d'%M.to(u.Msun).value+
              "\pm"+'%d'%errM.to(u.Msun).value+"\ M_{\odot}}$")
    ax.text(10**(np.log10(xmin)+(np.log(xmax)-np.log(xmin))*0.03), 
            10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.075), 
            "$M_{env}$", horizontalalignment='left')
    ax.text(10**(np.log10(xmin)+(np.log(xmax)-np.log(xmin))*0.075), 
            10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.075), 
            mLabel, horizontalalignment='left')
    temLabel = ("${=\ "+'%.1f'%T+
              "\pm"+'%.1f'%errT+"\ K}$")
    ax.text(10**(np.log10(xmin)+(np.log(xmax)-np.log(xmin))*0.03), 
            10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.11),
            "$T_{dust}$", horizontalalignment='left')
    ax.text(10**(np.log10(xmin)+(np.log(xmax)-np.log(xmin))*0.075), 
            10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.11),
            temLabel, horizontalalignment='left')
    
    modelFlux = greybody(frequencies, Tdust = sedResult.params['Tdust'].value,
                         mass= sedResult.params['mass'].value)*u.g/u.s/u.s
    chi2 = (np.sum(((modelFlux.to(u.Jy)-
                 fluxData)/fluxErr)**2)/(len(fluxData)-1))
    chi2Label = ("${=\ "+'%.1f'%chi2+"}$")
    ax.text(10**(np.log10(xmin)+(np.log(xmax)-np.log(xmin))*0.03), 
            10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.145), 
            "$\chi^2$" , horizontalalignment='left')
    ax.text(10**(np.log10(xmin)+(np.log(xmax)-np.log(xmin))*0.075), 
            10**(np.log10(ymax)-(np.log(ymax)-np.log(ymin))*0.145), 
            chi2Label, horizontalalignment='left')
    ax.set_xlabel("Wavelength ($\mu m$)")
    ax.set_ylabel("Flux (Jy)")
    fig.savefig('./figDir/'+sName+'_singleSED.pdf' ,
                dpi = 300, bbox_inches='tight')
    #fig.savefig('./figDir/'+sName+'_singleSED.eps' ,
    #            dpi = 300, bbox_inches='tight')
    
    fig.clf()
    print(pfmt %(isour+1, sName, ((isour+1.0) / len(sourList)*100)))
    
# writing out the fitting results

Tdust_col = Column(Tdust_col, name = 'Tdust')
Menv_col  = Column(Menv_col, name = 'Mass_env')
errTdust_col = Column(errTdust_col, name = 'errTdust')
errMenv_col  = Column(errMenv_col, name = 'errMass_env')

sourList.add_columns([Tdust_col, errTdust_col, 
                      Menv_col, errMenv_col])

sourList.write('./sourList._with_sedResults.txt',
               format = 'ascii.ipac',
               overwrite=True)