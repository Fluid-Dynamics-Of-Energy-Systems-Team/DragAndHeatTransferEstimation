#!/usr/bin/env python
# coding: utf-8

# # <center>Estimating mean profiles and fluxes in high-speed turbulent boundary layers using inner/outer-layer transformations</center>
# 
#              Created on: July 6, 2023
#                 Authors: Rene Pecnik (r.pecnik@tudelft.nl)
#                          Asif M. Hasan (a.m.hasan@tudelft.nl)
#                          Process & Energy Department, Faculty of 3mE
#                          Delft University of Technology, the Netherlands.
# 
#        Last modified on: July 9, 2023 (Rene Pecnik) 
# 
# 


# # Required functions 
# 
# ### 1. Mean velocity (in non-dimensional form)
# The mean velocity is obtained by 
# 
def meanVelocity(ReTheta, ReTau, MTau, y_ye, r_rw, mu_muw):
    
    # Semi local Reynolds number and scaled wall distances 
    ReTauSt = ReTau*np.sqrt(r_rw)/mu_muw
    ypl     = y_ye*ReTau
    yst     = y_ye*ReTauSt
    
    # eddy viscosity model
    D   = (1-np.exp(-yst/(Apl + 19.3*MTau)))**2
    mut = kappa*mu_muw*yst*D
    
    # wake parameter
    z1   = ReTheta/425-1
    Pi   = 0.69*(1 - np.exp(-0.243*z1**0.5 - 0.150*z1))
    Wake = Pi/kappa*np.pi*np.sin(np.pi*y_ye)
    
    # velocity 
    upl  = cumtrapz(1/(mu_muw + mut) + 1/ReTau/np.sqrt(r_rw)*Wake, ypl, initial=0)
    
    upl_inf = upl[-1]/0.99 # calculate upl_inf

    return ypl, yst, upl, upl_inf


# ### 2. Temperature velocity relationship (Zhang et al. (2014), JFM)
# 
def temperature(u_uinf, Minf, Tw_Tr):
    
    r       = Pr**(1/3)
    Tr_Tinf = 1 + r*(gamma - 1)/2*Minf**2
    Tinf_Tw = 1/(Tw_Tr*Tr_Tinf)
    
    sPr  = 0.8
    T_Tw = 1 + (1/Tw_Tr-1)*u_uinf*((1-sPr)*(u_uinf) + sPr) + (Tinf_Tw-1/Tw_Tr)*(u_uinf)**2

    dTduinf_wall = (1/Tw_Tr - 1)*sPr # Derivate of T with respect to u. 
                                     # Used for calculation of heat transfer coefficient ch
    
    return T_Tw, Tinf_Tw, dTduinf_wall


# ### 3. Density profile (using ideal gas equation of state)
# 
def density(T_Tw):
    return 1/T_Tw


# ### 4. Viscosity profile (using Sutherland's law)
# $$\frac{\bar\mu}{\mu_w}=\left(\frac{\bar T}{T_w}\right)^{3 / 2} \frac{T_w+S}{\bar T+S},$$
# 
def viscosity(T_Tw, Tinf_dim, Tinf_Tw, viscLaw):

    if viscLaw == "Sutherland":
        S     = 110.56/Tinf_dim * Tinf_Tw
        mu_muw= T_Tw**(1.5)*(1 + S)/(T_Tw + S)

    elif viscLaw == "PowerLaw":
        mu_muw = T_Tw**0.75

    else:
        print('Viscosity law ', viscLaw, " not available")
    
    return mu_muw


# ### 5. Computing $Re_\tau$ and $M_\tau$ using the inputs $Re_\theta$ and $M_\infty$
# $$Re_\tau = {Re_\theta}\frac{ \mu_\infty/\mu_w}{(\rho_\infty/\rho_w) u_\infty^+ (\theta/\delta)}$$
# 
def calcParameters(ReTheta, Minf, y_ye, r_rw, mu_muw, upl, uinf, 
                   T_Tw, Tw_Tr, Tinf_Tw, Tinf_dim, dTduinf_wall, viscLaw):
    
    rinf  = density(Tinf_Tw)
    muinf = viscosity(Tinf_Tw, Tinf_dim, Tinf_Tw, viscLaw)    

    Theta         = trapz(r_rw/rinf*upl/uinf*(1 - upl/uinf), y_ye)
    ReTheta_ReTau = rinf*uinf*Theta/muinf
    ReTau         = ReTheta/ReTheta_ReTau
    cf            = 2/(rinf*uinf**2)
    MTau          = Minf*(cf/2)**0.5    

    ch = np.nan     # ch=nan for adiabatic boundary layers
    if Tw_Tr != 1:
        # Calculate temperature gradeitn: dT/dy = dT/du * du+/dy+.
        # Since solver is based on viscous scales: du+/dy+ = 1
        dTdy_Wall = dTduinf_wall / uinf
        ch        = 1/Pr*dTdy_Wall/(rinf*uinf*(1/Tw_Tr - 1))   
    
    return ReTau, MTau, cf, ch


# # Iterative solver

# Required inputs are $Re_\theta$, $M_\infty$, $T_w/T_r$ and (optionally) the dimensional wall or free-stream temperature for Sutherland's law.  It is important to note that all solver inputs are based on the quantities in the free-stream, and not at the boundary layer edge.

# in case the notebook is executed on binder make sure that modules are installed.
# get_ipython().system('pip install numpy')
# get_ipython().system('pip install scipy')
# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install pandas')



import numpy as np
from scipy.integrate import cumtrapz, trapz



# n    ... number of points
# fact ... stretching/clustering
def grid(nPoints = 100, stretch = 5):
    H = 1.0       # --> y/y_e = 1
    tanhyp = 0.5  # half hyp tangens
    i = tanhyp*(np.arange(0,nPoints))/(nPoints-1) - 0.5
    y = 1./tanhyp*H * (1.0 + np.tanh(stretch*i)/np.tanh(stretch/2))/2.0
    return y



def solver(y_ye    = grid(200, 4), 
           ReTheta = 1000, Minf = 1.0, Tw_Tr = 1.0,    
           viscLaw = "Sutherland", Tinf_dim = 300):
    
    # set initial values for ReTau, MTau and upl
    ReTau = 100
    MTau  = 0.0
    upl   = np.ones_like(y_ye)*0.01
    uinf  = upl[-1]/0.99

    niter = 0
    err   = 1e10
    
    while(err > 1e-10 and niter < 1000):

        ReTauOld = ReTau
        
        T_Tw, Tinf_Tw, dTdu_inf = temperature(upl/uinf, Minf, Tw_Tr)
        mu_muw              = viscosity(T_Tw, Tinf_dim,Tinf_Tw, viscLaw)
        r_rw                = density(T_Tw)
        ypl, yst, upl, uinf = meanVelocity(ReTheta, ReTau, MTau, y_ye, r_rw, mu_muw)
        
        ReTau, MTau, cf, ch = calcParameters(ReTheta, Minf, y_ye, r_rw, mu_muw, upl, uinf, 
                                             T_Tw, Tw_Tr, Tinf_Tw, Tinf_dim, dTdu_inf, viscLaw)

        err = abs(ReTauOld-ReTau)
        niter += 1

    return cf, ch, ReTau, MTau, ypl, yst, upl, T_Tw, niter


# ### Import plotting modules


import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import matplotlib.ticker as ticker
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
rc('text', usetex=False)  # switch to True for latex font (might be slow)
rcParams.update({'font.size': 16})


# # An example with $M_\infty=15$, $Re_\theta =10^7$, $T_w/T_r= 0.01$, and $T_\infty=100$ K


########################################
# Constants
#
gamma     = 1.4    # ratio of specific heat capacities
Pr        = 0.72   # Prandtl number
kappa     = 0.41   # Karman constant 
Apl       = 17     # Van Driest damping constant 

########################################
# Run a test case
#
cf,ch,ReTau,MTau,ypl,yst,upl,T,niter = solver(y_ye     = grid(100000, 9),
                                              ReTheta  = 1.0e7, 
                                              Minf     = 15.0, 
                                              Tw_Tr    = 0.01,
                                              viscLaw  = "Sutherland", 
                                              Tinf_dim = 100.0)

print('Convergence reached after {0} iterations.'.format(niter))
print('\nSkin friction coefficient cf = {0:.5e} \nHeat transfer coefficient ch = {1:.5e}'.format(cf, ch))
print('ReTau = {0:.5e} \nMtau  = {1:.5e}'.format(ReTau,MTau))

################################################
# plot profiles
#
fig, ax = plt.subplots(1,2,figsize=(12,5))
ax[0].semilogx(ypl[1:],upl[1:], color='tab:red', lw=2)
ax[1].semilogx(ypl[1:],T[1:],   color='tab:red', lw=2)
ax[0].set_ylabel(r"$\bar u^+$",  fontsize = 18)
ax[1].set_ylabel(r"$\bar T/T_w$",fontsize = 18)
for a in ax:
    a.tick_params(axis='both', which='both', direction='in',labelsize=16,right=True,top=True)
    a.tick_params(which='major', length=10, width=1)
    a.tick_params(which='minor', length=5,  width=1)
    a.set_xticks(10.0**np.arange(-1, 8, 1))
    a.set_xlabel(r"$y^+$",fontsize = 18)
ax[1].set_yticks(np.arange(0,25,5))
plt.tight_layout()


# # Compare $c_f$ and $c_h$ estimates with various DNS cases from literature

import pandas as pd
DNS = pd.read_csv("DataForDragAndHeatTransfer.csv")
groups = DNS.groupby('Author', as_index=True)

fig, ax = plt.subplots(1,2,figsize=(16,5))

cf_rms = 0.0
ch_rms = 0.0

for group_name, group in groups:
    for row_index, row in group.reset_index().iterrows():
        
        Minf      = row['Minf']
        ReTheta   = row['ReTheta']
        Tw_Tr     = row['Tw_Tr']
        viscLaw   = row['ViscLaw']
        Tinf_dim  = row['Tinf']
        cf_DNS    = row['cf_DNS']
        ch_DNS    = row['ch_DNS']

        label = None
        if row_index == 0:
            label = row['Author']

        cf,ch,_,_,_,_,_,_,_ = solver(ReTheta=ReTheta, Minf=Minf, Tw_Tr=Tw_Tr,
                                     viscLaw=viscLaw, Tinf_dim=Tinf_dim)

        cf_err  = (cf-cf_DNS)/cf_DNS*100
        ch_err  = (ch-ch_DNS)/ch_DNS*100
        
        cf_rms += cf_err**2
        ch_rms  = np.nansum([ch_rms,ch_err**2])

        ax[0].plot(Minf, cf_err, marker = row['Symbol'], color = row['Color'], ms=10*row['Size'], 
             mew=2, fillstyle='none', linestyle='None', label=label)
        ax[1].plot(Minf, ch_err, marker = row['Symbol'], color = row['Color'], ms=10*row['Size'], 
             mew=2, fillstyle='none', linestyle='None', label=label)

cf_rms = np.sqrt(cf_rms/DNS["cf_DNS"].count())
ch_rms = np.sqrt(ch_rms/DNS["ch_DNS"].count())

ax[0].text(1.7,5, "rms = " + str(round(cf_rms,2)))
ax[1].text(1.7,10,"rms = " + str(round(ch_rms,2)))
ax[0].set_ylabel(r"Error $\varepsilon_{c_f}~[\%]$",fontsize = 18)
ax[1].set_ylabel(r"Error $\varepsilon_{c_h}~[\%]$",fontsize = 18)

ax[0].set_ylim([-6, 6])
ax[1].set_ylim([-12, 12])

for a in ax:
    a.tick_params(axis='both', which='both', direction='out',labelsize=16,right=True,top=True)
    a.tick_params(which='major', length=7, width=1)
    a.tick_params(which='minor', length=4, width=1)
    a.set_xlabel(r"$M_\infty$",fontsize = 18)

ax[0].axhline(y=0, color="gray", linestyle="-")
ax[1].axhline(y=0, color="gray", linestyle="-")

ax[1].legend(bbox_to_anchor=(1.05, 1.035))

plt.tight_layout()


# ## Plot estimated velocity and temperature profiles for these cases


fig, ax = plt.subplots(2,1,figsize=(14,9))

mult_x = 0.1

for group_name, group in groups:
    for row_index, row in group.reset_index().iterrows():

        label = None
        if row_index == 0:
            label = row['Author']
            mult_x *= 10

        _,_,_,_,ypl, yst, upl, T_Tw,_ = solver(ReTheta  = row['ReTheta'], 
                                               Minf     = row['Minf'], 
                                               Tw_Tr    = row['Tw_Tr'],
                                               viscLaw  = row['ViscLaw'], 
                                               Tinf_dim = row['Tinf'])

        ax[0].semilogx(ypl[1:]*mult_x, upl[1:],  color = row['Color'], label=label)
        ax[1].semilogx(ypl[1:]*mult_x, T_Tw[1:], color = row['Color'], label=label)

ax[0].set_ylabel(r"$\bar u^+$",   fontsize = 18)
ax[1].set_ylabel(r"$\bar T/T_w$", fontsize = 18)

for a in ax:
    a.tick_params(axis='both', which='both', direction='in',labelsize=16,right=True,top=True)
    a.tick_params(which='major', length=7, width=1)
    a.tick_params(which='minor', length=4, width=1)
    a.set_xticks(10.0**np.arange(-1, 9, 1))
    a.set_xlabel(r"$y^+$",      fontsize = 18)

ax[0].legend(bbox_to_anchor=(1.05, 1.035))

plt.tight_layout()
plt.show()
