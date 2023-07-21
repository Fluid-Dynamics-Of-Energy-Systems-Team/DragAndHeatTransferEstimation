from DragAndHeatEstimate import *




cf,ch,ReTau,MTau,ypl,yst,upl,T = solver(ReTheta  = 1.0e6,        #   Reynolds number based on momentum thickness
                                        Minf     = 15.0,         #   Mach number
                                        Tw_Tr    = 0.2,          #   wall to recovery temperature ratio
                                        viscLaw  = "Sutherland", #   viscosity law
                                        Tinf_dim = 100.0,        #   dimensional value of free stream temperature in K
                                        gamma    = 1.4,          #   ratio of specific heat capacities gamma=cp/cv
                                        Pr       = 0.72,         #   Prandtl number
                                        sPr      = 0.8,          #   sPr number, see Zhang et al. JFM, 2014
                                        kappa    = 0.41,         #   Karman constant 
                                        Apl      = 17.0,         #   Van Driest damping constant 
                                        y_ye     = grid(nPoints = 15000, stretch = 4),  #   grid points in y/ye (from 0 to 1)
                                        verbose  = True)         #   if True: print iteration residuals

print('\nResult: cf = {0:.5e}, ch = {1:.5e}, ReTau = {2:.5f}, MTau = {3:.3e}'.
      format(cf, ch, ReTau, MTau))



#################################################################################
#
# plot profiles
#
import matplotlib.pyplot as plt
from matplotlib import rc, rcParams
import matplotlib.ticker as ticker
rc('text', usetex=False)  # switch to True for latex font (might be slow)
rcParams.update({"xtick.major.size": 6, "xtick.minor.size": 3, "ytick.minor.size": 6, "ytick.minor.size": 3, 
                 'xtick.direction': 'in', 'ytick.direction': 'in', 'xtick.top': True, 'ytick.right': True, 
                 'font.size': 16})


fig, ax = plt.subplots(1,2,figsize=(12,4.5))
ax[0].semilogx(ypl[1:],upl[1:], color='tab:red', lw=2, label="Velocity, $c_f$ = {:.3e}".format(cf))
ax[1].semilogx(ypl[1:],T[1:],   color='tab:red', lw=2, label="Temperature, $c_h$ = {:.2e}".format(ch))
ax[0].set_ylabel(r"$\bar u^+$",  fontsize = 18)
ax[1].set_ylabel(r"$\bar T/T_w$",fontsize = 18)
for a in ax:
    a.set_xticks(10.0**np.arange(-1, 6, 1))
    a.set_xlabel(r"$y^+$",fontsize = 18)
    a.legend()
plt.tight_layout()
plt.show()
