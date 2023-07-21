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
# This is the bibtex entry in case you use the code: 
# 
#     @article{hasan2023estimating,
#       title={Estimating mean profiles and fluxes in high-speed turbulent boundary layers using inner/outer-layer transformations},
#       author={Hasan, Asif Manzoor and Larsson, Johan and Pirozzoli, Sergio and Pecnik, Rene},
#       journal={arXiv preprint arXiv:2307.02199},
#       year={2023}
#     }




import numpy as np
from scipy.integrate import cumtrapz, trapz




def meanVelocity(ReTheta, ReTau, MTau, y_ye, r_rw, mu_muw, kappa, Apl):
    
    # Semi local Reynolds number and scaled wall distances 
    ReTauSt = ReTau*np.sqrt(r_rw)/mu_muw
    ypl     = y_ye*ReTau
    yst     = y_ye*ReTauSt
    
    # eddy viscosity model
    D    = (1-np.exp(-yst/(Apl + 19.3*MTau)))**2
    mut  = kappa*mu_muw*yst*D
    
    # wake parameter
    z1   = ReTheta/425-1
    Pi   = 0.69*(1 - np.exp(-0.243*z1**0.5 - 0.150*z1))
    Wake = Pi/kappa*np.pi*np.sin(np.pi*y_ye)
    
    # velocity 
    upl  = cumtrapz(1/(mu_muw + mut) + 1/ReTau/np.sqrt(r_rw)*Wake, ypl, initial=0)
    
    upl_inf = upl[-1]/0.99 # calculate upl_inf

    return ypl, yst, upl, upl_inf





def temperature(u_uinf, Minf, Tw_Tr, Pr, sPr, gamma):
    
    r       = Pr**(1/3)
    Tr_Tinf = 1 + r*(gamma - 1)/2*Minf**2
    Tinf_Tw = 1/(Tw_Tr*Tr_Tinf)
    T_Tw    = 1 + (1/Tw_Tr-1)*u_uinf*((1-sPr)*(u_uinf) + sPr) + (Tinf_Tw-1/Tw_Tr)*(u_uinf)**2
    
    return T_Tw, Tinf_Tw





def density(T_Tw):
    return 1/T_Tw





def viscosity(T_Tw, Tinf_dim, Tinf_Tw, viscLaw):

    if viscLaw == "Sutherland":
        S      = 110.56/Tinf_dim * Tinf_Tw
        mu_muw = T_Tw**(1.5)*(1 + S)/(T_Tw + S)

    elif viscLaw == "PowerLaw":
        mu_muw = T_Tw**0.75

    else:
        print('Viscosity law ', viscLaw, " not available")
    
    return mu_muw





def calcParameters(ReTheta, Minf, y_ye, r_rw, mu_muw, upl, uinf, 
                   T_Tw, Tw_Tr, Tinf_Tw, Tinf_dim, viscLaw, Pr, sPr):
    
    rinf  = density(Tinf_Tw)
    muinf = viscosity(Tinf_Tw, Tinf_dim, Tinf_Tw, viscLaw)

    cf    = 2/(rinf*uinf**2)
    ch    = cf/2*sPr/Pr if Tw_Tr != 1 else np.nan  # set ch to NaN for adiabatic boundary layers
        
    Theta = trapz(r_rw/rinf*upl/uinf*(1 - upl/uinf), y_ye)
    ReTau = ReTheta/(rinf*uinf*Theta/muinf)
    MTau  = Minf*np.sqrt(cf/2)

    return cf, ch, ReTau, MTau






def grid(nPoints = 1000, stretch = 4):
    
    H = 1.0       # --> y/y_e = 1
    tanhyp = 0.5  # half hyp tangens
    i = tanhyp*(np.arange(0,nPoints))/(nPoints-1) - 0.5
    y = 1./tanhyp*H * (1.0 + np.tanh(stretch*i)/np.tanh(stretch/2))/2.0
    
    return y






def solver(ReTheta   = 1000,          #   Reynolds number based on momentum thickness
           Minf      = 1.0,           #   Mach number
           Tw_Tr     = 1.0,           #   wall to recovery temperature ratio
           viscLaw   = "Sutherland",  #   viscosity law
           Tinf_dim  = 300,           #   dimensional value of free stream temperature in K
           gamma     = 1.4,           #   ratio of specific heat capacities gamma=cp/cv
           Pr        = 0.72,          #   Prandtl number
           sPr       = 0.8,           #   sPr number, see Zhang et al. JFM, 2014
           kappa     = 0.41,          #   Karman constant 
           Apl       = 17.0,          #   Van Driest damping constant 
           y_ye      = grid(),        #   grid points in y/ye (from 0 to 1)
           verbose   = False):        #   if True: print iteration residuals
    
    # set initial values for ReTau = 500, MTau = 0.1, and upl = 0.01
    ReTau = 500
    MTau  = 0.1
    upl   = np.ones_like(y_ye)*0.01
    uinf  = upl[-1]/0.99

    niter = 0
    err   = 1e10
    
    while(err > 1e-4 and niter < 10000):

        ReTauOld = ReTau
        
        T_Tw, Tinf_Tw       = temperature(upl/uinf, Minf, Tw_Tr, Pr, sPr, gamma)
        mu_muw              = viscosity(T_Tw, Tinf_dim, Tinf_Tw, viscLaw)
        r_rw                = density(T_Tw)
        ypl, yst, upl, uinf = meanVelocity(ReTheta, ReTau, MTau, y_ye, r_rw, mu_muw, kappa, Apl)
        cf, ch, ReTau, MTau = calcParameters(ReTheta, Minf, y_ye, r_rw, mu_muw, upl, uinf, 
       T_Tw, Tw_Tr, Tinf_Tw, Tinf_dim, viscLaw, Pr, sPr)

        err = abs(ReTauOld-ReTau)
        niter += 1
        
        if verbose == True:
            if niter==1:
                print('{0:>6}{1:>14}{2:>14}{3:>14}{4:>14}{5:>14}'.format('iter','err','cf','ch','ReTau','MTau'))
            print('{0:>6}{1:14.4e}{2:14.4e}{3:14.4e}{4:14.4f}{5:14.4e}'.format(niter, err, cf, ch, ReTau, MTau))

    return cf, ch, ReTau, MTau, ypl, yst, upl, T_Tw

