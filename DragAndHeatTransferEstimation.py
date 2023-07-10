#!/usr/bin/env python
# coding: utf-8

# # <center>Estimating mean profiles and fluxes in high-speed turbulent boundary layers using inner/outer-layer transformations</center>
# 
#              Created on: July 6, 2023
#            Developed by: Rene Pecnik (r.pecnik@tudelft.nl)
#                          Asif M. Hasan (a.m.hasan@tudelft.nl)
#                          Process & Energy Department, Faculty of 3mE
#                          Delft University of Technology, the Netherlands.
# 
#        Last modified on: July 10, 2023 (Rene Pecnik) 
# 
# 
# The following python code in this notebook is based on the publication: https://doi.org/10.48550/arXiv.2307.02199.
# 
# 

# # 1. Required functions 
# 
# ### Mean velocity (in non-dimensional form)
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


# ### Temperature velocity relationship (Zhang et al. (2014), JFM)
# 
def temperature(u_uinf, Minf, Tw_Tr):
    
    r       = Pr**(1/3)
    Tr_Tinf = 1 + r*(gamma - 1)/2*Minf**2
    Tinf_Tw = 1/(Tw_Tr*Tr_Tinf)
    T_Tw    = 1 + (1/Tw_Tr-1)*u_uinf*((1-sPr)*(u_uinf) + sPr) + (Tinf_Tw-1/Tw_Tr)*(u_uinf)**2
    
    return T_Tw, Tinf_Tw


# ### Density profile (using ideal gas equation of state)
# 
def density(T_Tw):
    return 1/T_Tw


# ### Viscosity profile (using Sutherland's law)
# $$\frac{\bar\mu}{\mu_w}=\left(\frac{\bar T}{T_w}\right)^{3 / 2} \frac{T_w+S}{\bar T+S},$$
# 
def viscosity(T_Tw, Tinf_dim, Tinf_Tw, viscLaw):

    if viscLaw == "Sutherland":
        S      = 110.56/Tinf_dim * Tinf_Tw
        mu_muw = T_Tw**(1.5)*(1 + S)/(T_Tw + S)

    elif viscLaw == "PowerLaw":
        mu_muw = T_Tw**0.75

    else:
        print('Viscosity law ', viscLaw, " not available")
    
    return mu_muw


# ### Compute $c_f$ and $c_h$ $-$  also $Re_\tau$,  and $M_\tau$ based on the inputs $Re_\theta$ and $M_\infty$
# 
# 
def calcParameters(ReTheta, Minf, y_ye, r_rw, mu_muw, upl, uinf, 
                   T_Tw, Tw_Tr, Tinf_Tw, Tinf_dim, viscLaw):
    
    rinf  = density(Tinf_Tw)
    muinf = viscosity(Tinf_Tw, Tinf_dim, Tinf_Tw, viscLaw)

    cf = 2/(rinf*uinf**2)
    ch = cf/2*sPr/Pr if Tw_Tr != 1 else np.nan  # set ch to NaN for adiabatic boundary layers
        
    Theta = trapz(r_rw/rinf*upl/uinf*(1 - upl/uinf), y_ye)
    ReTau = ReTheta/(rinf*uinf*Theta/muinf)
    MTau  = Minf*np.sqrt(cf/2)

    return cf, ch, ReTau, MTau


# # 2. Iterative solver

# ### Import modules

# in case the notebook is executed on binder, make sure that these modules are installed:
# get_ipython().system('pip install numpy')
# get_ipython().system('pip install scipy')
# get_ipython().system('pip install matplotlib')
# get_ipython().system('pip install pandas')


# In[8]:


import numpy as np
from scipy.integrate import cumtrapz, trapz


# ### Grid, here we use a tanhyp function with clustering/stretching at the wall
# n    ... number of points
# fact ... stretching/clustering
def grid(nPoints = 100, stretch = 5):
    H = 1.0       # --> y/y_e = 1
    tanhyp = 0.5  # half hyp tangens
    i = tanhyp*(np.arange(0,nPoints))/(nPoints-1) - 0.5
    y = 1./tanhyp*H * (1.0 + np.tanh(stretch*i)/np.tanh(stretch/2))/2.0
    return y


# ### Main solver
# 
# Iterate velocity and temperature profiles until friction Reynolds number converges
# 
# Required inputs are $Re_\theta$, $M_\infty$, $T_w/T_r$ and (optionally) the dimensional wall or free-stream temperature for Sutherland's law.  It is important to note that all solver inputs are based on the quantities in the free-stream, and not at the boundary layer edge.
def solver(y_ye    = grid(nPoints = 200, stretch = 4), 
           ReTheta = 1000, Minf = 1.0, Tw_Tr = 1.0,    
           viscLaw = "Sutherland", Tinf_dim = 300, 
           verbose = False):
    
    # set initial values for ReTau = 500, MTau = 0.1, and upl = 0.01
    ReTau = 500
    MTau  = 0.1
    upl   = np.ones_like(y_ye)*0.01
    uinf  = upl[-1]/0.99

    niter = 0
    err   = 1e10
    
    while(err > 1e-4 and niter < 10000):

        ReTauOld = ReTau
        
        T_Tw, Tinf_Tw       = temperature(upl/uinf, Minf, Tw_Tr)
        mu_muw              = viscosity(T_Tw, Tinf_dim, Tinf_Tw, viscLaw)
        r_rw                = density(T_Tw)
        ypl, yst, upl, uinf = meanVelocity(ReTheta, ReTau, MTau, y_ye, r_rw, mu_muw)
        cf, ch, ReTau, MTau = calcParameters(ReTheta, Minf, y_ye, r_rw, mu_muw, upl, uinf, 
                                             T_Tw, Tw_Tr, Tinf_Tw, Tinf_dim, viscLaw)

        err = abs(ReTauOld-ReTau)
        niter += 1
        
        if verbose == True:
            print('iter = {0}, err = {1:.5e}, cf = {2:.5e}, ch = {3:.5e}, ReTau = {4:.5f}, MTau = {5:.3e}'.
                  format(niter, err, cf, ch, ReTau, MTau))


    return cf, ch, ReTau, MTau, ypl, yst, upl, T_Tw


# # 3. Example with $M_\infty=15$, $Re_\theta =10^6$, $T_w/T_r= 0.2$, and $T_\infty=100$ K

# In[11]:


import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.ticker as ticker
# get_ipython().run_line_magic('matplotlib', 'inline')
# get_ipython().run_line_magic('config', "InlineBackend.figure_format = 'retina'")
rc('text', usetex=False)  # switch to True for latex font (might be slow)
rcParams.update({"xtick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 6, "ytick.minor.size": 3, 
                 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 
                 'font.size': 16})


########################################
# Constants (they are global variables!)
#
gamma     = 1.4    # ratio of specific heat capacities
Pr        = 0.72   # Prandtl number
sPr       = 0.8    # see Zhang et al. JFM, 2014
kappa     = 0.41   # Karman constant 
Apl       = 17     # Van Driest damping constant 


########################################
# Run a test case
#
cf,ch,ReTau,MTau,ypl,yst,upl,T = solver(y_ye     = grid(nPoints = 10000, stretch = 4),
                                        ReTheta  = 1.0e6, 
                                        Minf     = 15.0, 
                                        Tw_Tr    = 0.2,
                                        viscLaw  = "Sutherland", 
                                        Tinf_dim = 100.0, 
                                        verbose  = True) 


################################################
# plot profiles
#
fig, ax = plt.subplots(1,2,figsize=(12,4.5))
ax[0].semilogx(ypl[1:],upl[1:], color='tab:red', lw=2)
ax[1].semilogx(ypl[1:],T[1:],   color='tab:red', lw=2)
ax[0].set_ylabel(r"$\bar u^+$",  fontsize = 18)
ax[1].set_ylabel(r"$\bar T/T_w$",fontsize = 18)
for a in ax:
    a.set_xticks(10.0**np.arange(-1, 6, 1))
    a.set_xlabel(r"$y^+$",fontsize = 18)
plt.tight_layout()



ReTheta = 10**np.linspace(np.log10(1e4), np.log10(1e6), 20)

cf = np.zeros_like(ReTheta)
ch = np.zeros_like(ReTheta)

for i, Re in enumerate(ReTheta):
    cf_,ch_,_,_,_,_,_,_ = solver(y_ye     = grid( nPoints = 10000, stretch = 4),
                                 ReTheta  = Re, 
                                 Minf     = 15.0, Tw_Tr = 0.2, viscLaw  = "Sutherland", 
                                 Tinf_dim = 100.0) 
    cf[i] = cf_
    ch[i] = ch_

# plot cf and ch as a function of ReTheta
fig, ax = plt.subplots(1,2,figsize=(12,4.5))
ax[0].plot(ReTheta, cf, color='tab:red', lw = 2)
ax[1].plot(ReTheta, ch, color='tab:red', lw = 2)
ax[0].set_ylabel(r"$c_f$", fontsize = 18)
ax[1].set_ylabel(r"$c_h$", fontsize = 18)
for a in ax:
    a.set_xlabel(r"$Re_\theta$",fontsize = 18)
    a.ticklabel_format(axis='y', style='scientific', scilimits=(0,0))
plt.tight_layout()


# # 4. Validate $c_f$ and $c_h$ estimates with various DNS cases from literature


import pandas as pd
DNS = pd.read_csv("DataForDragAndHeatTransfer.csv")
groups = DNS.groupby('Author', as_index=True)

cf_rms = 0.0
ch_rms = 0.0

fig, ax = plt.subplots(2,3, figsize=(16,8), sharex='col', sharey='row')

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

        cf,ch,_,_,_,_,_,_ = solver(ReTheta = ReTheta, Minf = Minf, Tw_Tr = Tw_Tr,
                                   viscLaw = viscLaw, Tinf_dim = Tinf_dim)

        cf_err  = (cf-cf_DNS)/cf_DNS*100
        ch_err  = (ch-ch_DNS)/ch_DNS*100
        
        cf_rms += cf_err**2
        ch_rms  = np.nansum([ch_rms,ch_err**2])

        params = {"marker": row['Symbol'], "color": row['Color'], "ms": 10*row['Size'], 
                  "mew": 2, "fillstyle": 'none', "linestyle": 'None', "label": label}
        
        ax[0,0].plot(Minf,    cf_err, **params)
        ax[0,1].plot(ReTheta, cf_err, **params)
        ax[0,2].plot(Tw_Tr,   cf_err, **params)
        
        ax[1,0].plot(Minf,    ch_err, **params)
        ax[1,1].plot(ReTheta, ch_err, **params)
        ax[1,2].plot(Tw_Tr,   ch_err, **params)

cf_rms = np.sqrt(cf_rms/DNS["cf_DNS"].count())
ch_rms = np.sqrt(ch_rms/DNS["ch_DNS"].count())

ax[0,0].text(8,-5.5, "rms = " + str(round(cf_rms,2)))
ax[1,0].text(8,-11,  "rms = " + str(round(ch_rms,2)))
ax[0,0].set_ylabel(r"Skin friction $\varepsilon_{c_f}~[\%]$",fontsize = 18)
ax[1,0].set_ylabel(r"Heat trans. $\varepsilon_{c_h}~[\%]$",fontsize = 18)
ax[1,0].set_xlabel(r"$M_\infty$",fontsize = 18)
ax[1,1].set_xlabel(r"$Re_\theta$",fontsize = 18)
ax[1,2].set_xlabel(r"$T_w/T_r$",fontsize = 18)

ax[0,0].set_ylim([-6, 6])
ax[1,0].set_ylim([-12, 12])

for a in ax.reshape(-1):
    a.axhline(y=0, color="gray", linestyle=":")

ax[0,2].legend(bbox_to_anchor=(1.05, 1.035))

fig.suptitle(r"Error in $c_f$ and $c_h$ estimation compared to DNS", fontsize = 18)
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

        _,_,_,_,ypl, yst, upl, T_Tw = solver(ReTheta  = row['ReTheta'], 
                                             Minf     = row['Minf'], 
                                             Tw_Tr    = row['Tw_Tr'],
                                             viscLaw  = row['ViscLaw'], 
                                             Tinf_dim = row['Tinf'])

        ax[0].semilogx(ypl[1:]*mult_x, upl[1:],  color = row['Color'], label=label)
        ax[1].semilogx(ypl[1:]*mult_x, T_Tw[1:], color = row['Color'], label=label)

ax[0].set_ylabel(r"$\bar u^+$",   fontsize = 18)
ax[1].set_ylabel(r"$\bar T/T_w$", fontsize = 18)

for a in ax:
    a.set_xticks(10.0**np.arange(-1, 9, 1))
    a.set_xlabel(r"$y^+$",      fontsize = 18)

ax[0].legend(bbox_to_anchor=(1.05, 1.035))

plt.tight_layout()
plt.show()


