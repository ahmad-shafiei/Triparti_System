
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
from numpy import sin, cos, sqrt, array, linspace, pi


def calculate_H(w_m, w_a, w_L, Delta_aL, g_cm, g_am, g_ac, g_eff,gamma, kappa, F_L, n_th):
    N = 5                  # number of cavity fock states
    M = 10                  # number of phonon fock states 

    #w_L = w_c  - (g_ac**2)/Delta_aL     # zero detuning  
    alpha = g_am * (1-(g_ac/w_a)**2) 
    #w_c = w_L  - (2*alpha)*g_cm/w_m
    #w_c = w_L #+(g_ac**2)/Delta_aL - (2*alpha)*g_eff/w_m
    #w_c = w_L +(g_ac**2)/Delta_aL -(2*alpha)*g_eff/w_m
    w_c = 9950.901
        
    a = tensor(destroy(N), qeye(M), qeye(2))
    b = tensor(qeye(N), destroy(M), qeye(2))
    sigma_z = tensor(qeye(N), qeye(M), sigmaz())
    sigma_p = tensor(qeye(N), qeye(M), sigmap())  # raising and lowering ops for atom
    sigma_m = tensor(qeye(N), qeye(M), sigmam())
    
    num_a = a.dag() * a 
    num_b = b.dag() * b
    gamma_a = gamma
    c_ops = []
    rate = kappa                              # kappa is cavity decay rate
    if rate > 0.0 : 
        c_ops.append(a * sqrt(rate))
    rate = n_th * gamma                       # gamma is mechanical resonator
    if rate > 0.0: 
        c_ops.append(b.dag() * sqrt(rate))
    rate = (n_th+1) * gamma
    if rate > 0.0 : 
        c_ops.append(b * sqrt(rate))
    # Effiective coupling   
    H_Cav  = - (w_L- w_c - (2*alpha)*g_cm/w_m) *num_a            # - (2*alpha)*g_cm/w_m - (2*alpha)*g_cm/w_m
    H_Atm  =  0.5 * (Delta_aL + 4*alpha*g_am/w_m ) *sigma_z      #### + 4*alpha*g_am/w_m 
    #### Mechanical effective Hamiltonian 
    H_Mech = w_m * num_b - alpha * (b + b.dag())  ###
    #### Interaction effective Hamiltonian
    H_Int   = 1j * g_ac * (sigma_p * a - sigma_m * a.dag())- g_cm * num_a *(b.dag()+ b)- g_am * sigma_z * (b.dag()+ b)
    # Laser drive 
    H_drive =  - 1j* F_L *(a.dag()-a)     
    ##### Total H : 

    H_0 =  (alpha**2)/w_m
    H = 0*H_0 + H_Cav + H_Atm + H_Mech + H_Int + H_drive
    # at resonance and for g=0
    H0 = 0*H_0 + 0 * H_Cav + H_Atm + H_Mech + 0* H_Int + H_drive   # H at resonance
    
    return H, H0, c_ops, a # , n_0


w_m  = 1                       # mechanical frequency
b3 = False
if b3== True:
    w_a_j  = [ 1500 ]
    w_L_j =  [1000  ]
    g_am_j = [  50   ]
    g_ac_j = [1.0 * 50]
else:
    w_a_j  = [ 15000 ]
    w_L_j =  [10000  ]
    g_am_j = [  50   ]
    g_ac_j = [1.0 * 500]
    
n_th_i   = [0, 1, 0 ]
gamma    = w_m/20.                                ## gamma is decay for mechanical resonator 
kappa_i  = [2. * w_m, w_m/2., w_m/2.]     # kappa is cavity decay rate
g_cm = 0.001 * w_m               # cavity-resonator couplng  10 ^-5


fig, ax= plt.subplots(1,1,figsize=(6,4))
solvers = ['direct','power','eigen','iterative-gmres','iterative-bicgstab']  # methods 
# only for ['iterative-gmres','iterative-bicgstab']---  use_rcm = True
use_rcm = False

tlist = np.linspace(0, 200, 3000)
wlist = np.linspace(-6, 4, 150) * w_m

linestyle =['--','--','-']
color=['black', 'r', 'b']
j=0
opts=Options(nsteps=5000)

xplot = []       #######=====###########====####
pllot4b  = np.zeros((len(wlist),3))       ############
for j in [0]:
    w_a  = w_a_j[j]  
    w_L  = w_L_j[j]
    Delta_aL = w_a - w_L 
    
    g_am = g_am_j[j] 
    g_ac = g_ac_j[j]
    g_eff = g_cm + 2 * g_am * (g_ac/Delta_aL)**2

    for i in tqdm([0,1,2]): 
        n_th            = n_th_i[i]
        kappa = kappa_i[i]
        F_L = sqrt(kappa) * 0.01

        H, H0, c_ops, a = calculate_H(w_m, w_a, w_L, Delta_aL, g_cm, g_am, g_ac, g_eff,gamma, kappa, F_L, n_th)    
        # calculate |<a(t)>|^2
        #rho      = steadystate(H, c_ops, method=solvers[0] , tol=1e-14)
        #n        = mesolve(H, rho, tlist, c_ops, [a], options=opts).expect[0]
        #mean_a_t = np.abs(n)**2 
        ##########################
        spec = spectrum(H, wlist, c_ops, a.dag(), a)
        ########################
        #corr      = correlation_ss(H, tlist, c_ops, a.dag(), a)
        #corr       = correlation_2op_1t(H, rho, tlist, c_ops, a.dag(), a)
        #spect_tot = corr - mean_a_t
        ### n_0 
        rho0     = steadystate(H0, c_ops, method=solvers[0] , tol=1e-14)
        n0       = mesolve(H0, rho0, tlist, c_ops, [a.dag() * a]).expect[0]  #, options=opt
        n_0      = n0[-1]
        # fft 
        #wlist, spec = spectrum_correlation_fft(tlist, spect_tot) 
        
        xplot = wlist                             ###########
        #pllot3b[: , i] = spec/n_0                 ############
        pllot4b[: , i] = spec/n_0                 ############
        
        ax.plot(wlist / w_m, spec/n_0, linestyle =linestyle[i], color=color[i])  # /n_0
    
        ax.set_xlabel(r'$\omega/ \omega_M $', fontsize=14)
        ax.set_ylabel(r'$ S(\omega) /n_0 $', fontsize=14)    # (r'$ e $',fontsize=16,color='red')
        ax.set_title('Spectrum')
        ax.set_xlim([-7*w_m,3*w_m])
        ax.set_ylim([0,2])   # ([0,max(spec)])

#plt.savefig("Spectrum_Hyb3b.pdf", dpi=150)      #####   ##### ###### #####
plt.savefig("Spectrum_Hyb4b.pdf", dpi=150)      #####   ##### ###### #####
#plt.show()


xarray = xplot
bblack = pllot4b[: , 0]
red =    pllot4b[: , 1]
blue = pllot4b[: , 2]

data4b = np.column_stack([xarray, bblack,red ,blue])
np.savetxt("Fig4b.txt" , data4b, fmt=['%s','%s','%s','%s'])

linestyle =['-','--','--','-']
color=['black', 'black', 'r', 'b']
fig, ax= plt.subplots(1,1,figsize=(9,4))
for i in [1,2,3]:
    plt.plot(xarray, data4b[:,i], linestyle = linestyle[i], color = color[i])
    
ax.set_xlim([-7*w_m,3*w_m])
ax.set_ylim([0,5])


