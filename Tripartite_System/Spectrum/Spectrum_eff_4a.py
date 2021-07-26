import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from qutip import *
from tqdm import tqdm
from numpy import sin, cos, sqrt, array, linspace, pi

def calculate_H(w_m, w_a, w_L , Delta_aL, g_cm, g_am, g_ac, g_eff,gamma, kappa, F_L, n_th):
    N = 5                  
    M = 10                  

    alpha = g_am * (1-(g_ac/Delta_aL)**2) 
    Delta_aL = w_a - w_L
    #w_c = w_L +(g_ac**2)/Delta_aL - (2*alpha)*g_eff/w_m   
    w_c = 9950.901
    
    a = tensor(destroy(N), qeye(M))
    b = tensor(qeye(N), destroy(M))
    num_a = a.dag() * a 
    num_b = b.dag() * b
    # H in rotating frame with derive laser F_L with frequency omega_L
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
    Heff_Cav  = - (w_L- w_c + (g_ac**2)/Delta_aL- (2*alpha)*g_eff/w_m) *num_a   # 
    #### Mechanical effective Hamiltonian 
    Heff_Mech = w_m * num_b  #+ alpha * (b + b.dag())  ###
    #### Interaction effective Hamiltonian
    Heff_Int  =  - g_eff * num_a *(b + b.dag())
    # Laser drive 
    H_drive   =  -1j* F_L *(a.dag()-a)     
    ##### Total H : 

    H_0 =  (alpha**2)/w_m   # - w_a/2. - (g_ac**2.)/(2.*w_a) -
    H = 0*H_0 + Heff_Cav + Heff_Mech + Heff_Int + H_drive
    # at resonance and for g=0
    H0 = 0*H_0 + 0 * Heff_Cav + Heff_Mech + 0* Heff_Int + H_drive   # H at resonance
    
    return H, H0, c_ops, a # , n_0

w_m  = 1                       # mechanical frequency
w_a_j  = [15000 ]#[1500 ]
w_L_j =  [10000 ]#[1000 ]
g_am_j = [ 50   ]#[ 50   ]
g_ac_j = [1.0 * 500 ]#[ 1.0 * 50 ]

n_th_i   = [0, 1, 0 ]
gamma    = w_m/20.                                ## gamma is decay for mechanical resonator 
kappa_i  = [2. * w_m, w_m/2., w_m/2.]     # kappa is cavity decay rate
g_cm = 0.001 * w_m               # cavity-resonator couplng  10 ^-5


fig, ax= plt.subplots(1,1,figsize=(6,4))
solvers = ['direct','power','eigen','iterative-gmres','iterative-bicgstab']  # methods 
# only for ['iterative-gmres','iterative-bicgstab']---  use_rcm = True
use_rcm = False
tlist = np.linspace(0, 200, 2000)
linestyle =['--','--','-']
color=['black', 'r', 'b']
j=0
xplot = []       #######=====###########====####
#pllot3a  = np.zeros((len(tlist),3))       ############
pllot4a  = np.zeros((len(tlist),3))       ############

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
        
        H, H0, c_ops, a = calculate_H(w_m, w_a,w_L , Delta_aL, g_cm, g_am, g_ac, g_eff,gamma, kappa, F_L, n_th)    
        # calculate |<a(t)>|^2
        rho      = steadystate(H, c_ops, method=solvers[2] , tol=1e-15)
        n        = mesolve(H, rho, tlist, c_ops, [a]).expect[0]
        mean_a_t = np.abs(n)**2 
        ##########################
        #corr      = correlation_ss(H, tlist, c_ops, a.dag(), a) #####0000 solver='me'
        corr       = correlation_2op_1t(H, rho, tlist, c_ops, a.dag(), a)
        spect_tot = corr - mean_a_t
        ### n_0 
        rho0     = steadystate(H0, c_ops, method=solvers[2] , tol=1e-15)
        n0       = mesolve(H0, rho0, tlist, c_ops, [a.dag() * a]).expect[0]  #, options=opt
        n_0      = n0[-1]
        # fft 
        wlist, spec = spectrum_correlation_fft(tlist, spect_tot)  
        
        xplot = wlist                             ###########
        #pllot3a[: , i] = spec/n_0                 ############
        pllot4a[: , i] = spec/n_0                 ############
        
        ax.plot(wlist / w_m, spec/n_0, linestyle =linestyle[i], color=color[i])  # /n_0    
        ax.set_xlabel(r'$\omega/ \omega_M $', fontsize=14)
        ax.set_ylabel(r'$ S(\omega) /n_0 $', fontsize=14)    # (r'$ e $',fontsize=16,color='red')
        ax.set_title('Spectrum')
        ax.set_xlim([-7*w_m,4*w_m])
        ax.set_ylim([0,3])   # ([0,max(spec)])

#plt.savefig("Spectrum_eff_3a.pdf", dpi=150)
plt.savefig("Spectrum_eff_4a.pdf", dpi=150)      ####
#plt.show()


xarray = xplot
bblack = pllot4a[: , 0]        ####
red = pllot4a[: , 1]           ####
blue = pllot4a[: , 2]          #####

data4a = np.column_stack([xarray, bblack,red ,blue])        ####
np.savetxt("Fig4a.txt" , data4a, fmt=['%s','%s','%s','%s'])    ###

linestyle =['-','--','--','-']
color=['black', 'black', 'r', 'b']
fig, ax= plt.subplots(1,1,figsize=(6,4))
for i in [1,2,3]:
    plt.plot(xarray, data4a[:,i], linestyle = linestyle[i], color = color[i]) #####
    
ax.set_xlim([-7*w_m,4*w_m])
ax.set_ylim([0,3])
