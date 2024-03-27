import matplotlib.pyplot as plt
import numpy as np  
import pandas as pd

data = pd.read_excel('polars.xlsx')
cl = data['Cl'].to_numpy()
cd = data['Cd'].to_numpy()
cm = data['Cm'].to_numpy()  
alpha = data['Alfa'].to_numpy()

TSR = 6 #[6, 8, 10] # tip speed ratio THAT"S WHAT WE VARY 

# 
Uinf = 10 # m/s free stream velocity
Nb = 3 # number of blades
R = 50 # m radius of the rotor
Omega = Uinf*TSR/R

Rstart = 0.2
Rtip=1

delta_r_R = 0.01
r_R = np.arange(0.2, 1+delta_r_R/2, delta_r_R)

pitch = -2 # degrees
chord_distribution = 3*(1-r_R)+1 # meters
twist_distribution = 14*(1-r_R)+pitch # degrees

def ainduction(CT):
    a = np.zeros(np.shape(CT))
    CT1=1.816
    CT2=2*np.sqrt(CT1)-CT1
    a[CT>=CT2] = 1 + (CT[CT>=CT2]-CT1)/(4*(np.sqrt(CT1)-1))
    a[CT<CT2] = 0.5-0.5*np.sqrt(1-CT[CT<CT2])
    return a

def PrandtlTipRootCorrection(r_R, rootradius_R, tipradius_R, TSR, NBlades, axial_induction):
    # Prandtl tip correction
    temp1 = -NBlades/2*(tipradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Ftip = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Ftip[np.isnan(Ftip)] = 0
    temp1 = NBlades/2*(rootradius_R-r_R)/r_R*np.sqrt( 1+ ((TSR*r_R)**2)/((1-axial_induction)**2))
    Froot = np.array(2/np.pi*np.arccos(np.exp(temp1)))
    Froot[np.isnan(Froot)] = 0
    return Froot*Ftip, Ftip, Froot

def loadsPerElement(Vaxial, Vtan, chord, twist, polar_alpha, polar_cl, polar_cd):
    Vp = np.sqrt(Vaxial**2 + Vtan**2)

    phi = np.arctan2(Vaxial,Vtan)
    alpha = phi*180/np.pi + twist 

    Cl = np.interp(alpha, polar_alpha, polar_cl)
    Cd = np.interp(alpha, polar_alpha, polar_cd)

    L = 0.5*Vp**2*Cl*chord
    D = 0.5*Vp**2*Cd*chord

    Fnormal = L*np.cos(phi) + D*np.sin(phi)
    Fazim = L*np.sin(phi) - D*np.cos(phi)
    gamma = 0.5*Vp*Cl*chord

    return Fnormal, Fazim, gamma

def solver(Uinf, r1_R, r2_R, Rstart, Rtip , Omega, R, Nb, chord, twist, alpha, cl, cd ):
    A = np.pi*((r2_R*R)**2 - (r1_R*R)**2) # area of the annulus
    r_R = (r1_R + r2_R)/2 # mean radius

    a  = 0.3
    aprime = 0.0

    Nit = 100 # number of iterations
    error = 1e-5

    for i in range(Nit):

        Vaxial = Uinf*(1-a)
        Vtan = (1+aprime)*Omega*r_R*R

        Fnorm, Fazim, gamma = loadsPerElement(Vaxial, Vtan, chord, twist, alpha, cl, cd)
        axialLoad = Fnorm*R*(r2_R-r1_R)*Nb
        # We have just calculated the loads on the blade element

        CT = axialLoad/(0.5*Uinf**2*A)

        a_new = ainduction(CT)
        a_new = 0.25*a_new + 0.75*a

        Prandtl, Prandtltip, Prandtlroot = PrandtlTipRootCorrection(r_R, Rstart, Rtip, TSR, Nb, a_new)
        if (Prandtl < 0.0001): 
            Prandtl = 0.0001 # avoid divide by zero
        aprime = Fazim*Nb/(2*np.pi*Uinf*(1-a)*Omega*2*(r_R*R)**2) #why?????!!!!
        aprime= aprime/Prandtl


        # we have calculated the new value of a and a' for the current iteration

        if np.abs(a-a_new)<error:
            print('iterations: ', i)
            break

    return [a, aprime, r_R, Fnorm, Fazim, gamma]


results =np.zeros([len(r_R)-1,6]) 
for i in range(len(r_R)-1):
    chord = np.interp((r_R[i]+r_R[i+1])/2, r_R, chord_distribution)
    twist = np.interp((r_R[i]+r_R[i+1])/2, r_R, twist_distribution)
    results[i,:]=solver(Uinf, r_R[i], r_R[i+1], Rstart, Rtip , Omega, R, Nb, chord, twist, alpha, cl, cd )
    break

print(results)